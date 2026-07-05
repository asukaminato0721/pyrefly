/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::Context as _;
use pyrefly_util::fs_anyhow;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;

use crate::config::ConfigFile;

static DEPENDENCY_SPEC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\s*\[[^\]]+\])?\s*(.*)$")
        .expect("dependency regex must be valid")
});
static EXACT_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*==\s*([^,;\s]+)").expect("version regex must be valid"));

/// A dependency declaration at a byte range in a pyproject.toml file.
#[derive(Debug, PartialEq, Eq)]
pub struct PyProjectDependency {
    pub name: String,
    pub configured_version: Option<String>,
    pub range: Range<usize>,
}

/// Find the dependency declaration containing `offset` in standard dependency sections.
pub fn dependency_at_offset(source: &str, offset: usize) -> Option<PyProjectDependency> {
    let document = toml_edit::Document::parse(source).ok()?;
    if let Some(project) = document.get("project").and_then(toml_edit::Item::as_table) {
        if let Some(dependencies) = project
            .get("dependencies")
            .and_then(toml_edit::Item::as_array)
            && let Some(dependency) = dependency_in_array(dependencies, source, offset)
        {
            return Some(dependency);
        }
        if let Some(groups) = project
            .get("optional-dependencies")
            .and_then(toml_edit::Item::as_table)
        {
            for (_, group) in groups {
                if let Some(dependencies) = group.as_array()
                    && let Some(dependency) = dependency_in_array(dependencies, source, offset)
                {
                    return Some(dependency);
                }
            }
        }
    }
    if let Some(groups) = document
        .get("dependency-groups")
        .and_then(toml_edit::Item::as_table)
    {
        for (_, group) in groups {
            if let Some(dependencies) = group.as_array()
                && let Some(dependency) = dependency_in_array(dependencies, source, offset)
            {
                return Some(dependency);
            }
        }
    }
    None
}

fn dependency_in_array(
    dependencies: &toml_edit::Array,
    source: &str,
    offset: usize,
) -> Option<PyProjectDependency> {
    for value in dependencies {
        let span = value.span()?;
        if offset < span.start || offset > span.end {
            continue;
        }
        let spec = value.as_str()?;
        let captures = DEPENDENCY_SPEC.captures(spec.trim())?;
        let name = captures.get(1)?.as_str();
        let literal = source.get(span.clone())?;
        let name_start = span.start + literal.find(name)?;
        return Some(PyProjectDependency {
            name: name.to_owned(),
            configured_version: EXACT_VERSION
                .captures(captures.get(2)?.as_str())
                .and_then(|captures| captures.get(1))
                .map(|version| version.as_str().to_owned()),
            range: name_start..name_start + name.len(),
        });
    }
    None
}

/// Known Python tool names whose presence in `[tool.*]` indicates
/// this directory is a Python project root.
const PYTHON_TOOL_NAMES: &[&str] = &["ruff", "mypy", "pyright"];

/// Wrapper used to (de)serialize pyrefly configs from pyproject.toml files.
#[derive(Debug, Serialize, Deserialize)]
struct Tool {
    pyrefly: Option<ConfigFile>,
    /// Catch-all for other `[tool.*]` sections. We check this for known
    /// Python tool names (see `PYTHON_TOOL_NAMES`) to detect Python project roots.
    #[serde(flatten)]
    other_tools: HashMap<String, toml::Value>,
}

/// Wrapper used to (de)serialize pyrefly configs from pyproject.toml files.
#[derive(Debug, Serialize, Deserialize)]
pub struct PyProject {
    tool: Option<Tool>,
}

/// Recursively finds the maximum position among a table and all its nested sub-tables.
/// This is needed because pyproject.toml files often have deeply nested tool sections
/// like `[tool.ruff.lint.pydocstyle]`, and we need to find the position of the last
/// nested section to insert `[tool.pyrefly]` after all of them.
fn max_position_recursive(table: &toml_edit::Table) -> isize {
    let own_pos = table.position().unwrap_or(0);
    let child_max = table
        .iter()
        .filter_map(|(_, v)| v.as_table().map(max_position_recursive))
        .max()
        .unwrap_or(0);
    own_pos.max(child_max)
}

impl PyProject {
    /// Wrap the given ConfigFile in a `PyProject { Tool { ... }}`
    pub fn new(cfg: ConfigFile) -> Self {
        Self {
            tool: Some(Tool {
                pyrefly: Some(cfg),
                other_tools: HashMap::new(),
            }),
        }
    }

    pub(crate) fn pyrefly(self) -> Option<ConfigFile> {
        self.tool.and_then(|t| t.pyrefly)
    }

    /// Whether this pyproject.toml has sections for Python tools like ruff,
    /// mypy, or pyright. This is a strong signal that the directory is a Python
    /// project root, even without an explicit `[tool.pyrefly]` section.
    pub(crate) fn has_python_tools(&self) -> bool {
        self.tool.as_ref().is_some_and(|t| {
            PYTHON_TOOL_NAMES
                .iter()
                .any(|name| t.other_tools.contains_key(*name))
        })
    }

    pub fn update(pyproject_path: &Path, config: ConfigFile) -> anyhow::Result<()> {
        const ERR_WRITE_CONFIG: &str =
            "While trying to write Pyrefly config to pyproject.toml file";
        let config_pyproject = PyProject::new(config);
        if pyproject_path.exists() {
            let original_content = fs_anyhow::read_to_string(pyproject_path)?;
            let mut doc = original_content
                .parse::<toml_edit::DocumentMut>()
                .with_context(|| {
                    format!(
                        "Failed to parse {} as TOML document",
                        pyproject_path.display()
                    )
                })?;
            let toml_string = toml::to_string_pretty(&config_pyproject)?;
            let config_doc = toml_string.parse::<toml_edit::DocumentMut>()?;
            if let Some(tool_table) = config_doc.get("tool")
                && let Some(pyrefly_table) = tool_table.get("pyrefly")
            {
                let is_new_tool_table = !doc.contains_key("tool");
                let tool_entry = doc
                    .entry("tool")
                    .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
                if let Some(tool_table_mut) = tool_entry.as_table_mut() {
                    if is_new_tool_table {
                        tool_table_mut.set_implicit(true);
                    }
                    tool_table_mut.remove("pyrefly");
                    // Use recursive position finding to account for deeply nested
                    // tool sections like [tool.ruff.lint.pydocstyle]
                    let max_tool_pos = tool_table_mut
                        .iter()
                        .filter_map(|(_, v)| v.as_table().map(max_position_recursive))
                        .max()
                        .unwrap_or(0);
                    tool_table_mut.insert("pyrefly", pyrefly_table.clone());
                    if !original_content.is_empty()
                        && let Some(pyrefly_item) = tool_table_mut.get_mut("pyrefly")
                        && let Some(pyrefly_table_mut) = pyrefly_item.as_table_mut()
                    {
                        pyrefly_table_mut.decor_mut().set_prefix("\n");
                        pyrefly_table_mut.set_position(Some(max_tool_pos + 1));
                    }
                }
            }
            fs_anyhow::write(pyproject_path, doc.to_string()).with_context(|| ERR_WRITE_CONFIG)
        } else {
            let mut serialized_toml = toml::to_string_pretty(&config_pyproject)?;
            if !serialized_toml.contains("[tool.pyrefly]") {
                serialized_toml = String::from("[tool.pyrefly]\n");
            }
            fs_anyhow::write(pyproject_path, serialized_toml).with_context(|| ERR_WRITE_CONFIG)
        }
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::globs::Globs;

    use super::*;

    #[test]
    fn find_project_dependency() {
        let source = r#"[project]
dependencies = ["requests==2.32.0"]
"#;
        let offset = source.find("requests").unwrap();
        assert_eq!(
            dependency_at_offset(source, offset),
            Some(PyProjectDependency {
                name: "requests".to_owned(),
                configured_version: Some("2.32.0".to_owned()),
                range: offset..offset + "requests".len(),
            })
        );
    }

    #[test]
    fn find_dependency_groups() {
        for source in [
            r#"[project.optional-dependencies]
dev = ["pytest>=8"]
"#,
            r#"[dependency-groups]
dev = ["pytest>=8"]
"#,
        ] {
            let offset = source.find("pytest").unwrap();
            assert_eq!(
                dependency_at_offset(source, offset),
                Some(PyProjectDependency {
                    name: "pytest".to_owned(),
                    configured_version: None,
                    range: offset..offset + "pytest".len(),
                })
            );
        }
    }

    #[test]
    fn ignore_strings_outside_dependency_sections() {
        let source = r#"[project]
description = "requests"
"#;
        assert_eq!(
            dependency_at_offset(source, source.find("requests").unwrap()),
            None
        );
    }

    #[test]
    fn test_replace_existing_pyrefly_config() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let pyproject_path = tmp.path().join("pyproject.toml");

        let existing_content = r#"[project]
name = "test-project"
version = "0.1.0"

[tool.poetry]
dependencies = { python = "^3.8" }

[tool.pyrefly]
project_includes = ["old/path/**/*.py"]
project_excludes = ["should/be/removed.py"]

[tool.black]
line-length = 88
"#;
        fs_anyhow::write(&pyproject_path, existing_content)?;

        let config = ConfigFile {
            project_includes: Globs::new(vec!["new/path/**/*.py".to_owned()]).unwrap(),
            ..Default::default()
        };
        PyProject::update(&pyproject_path, config)?;

        let updated_content = fs_anyhow::read_to_string(&pyproject_path)?;

        assert!(updated_content.contains("[project]"));
        assert!(updated_content.contains("name = \"test-project\""));
        assert!(updated_content.contains("[tool.poetry]"));
        assert!(updated_content.contains("[tool.black]"));

        // Make sure we add a blank line between the pyrefly section and the previous one
        assert!(updated_content.contains("\n\n[tool.pyrefly]"));
        assert!(updated_content.contains("project-includes = [\"new/path/**/*.py\"]"));
        assert!(!updated_content.contains("project_includes = [\"old/path/**/*.py\"]"));
        assert!(!updated_content.contains("project_excludes"));

        Ok(())
    }

    #[test]
    fn test_add_pyrefly_config_to_existing_pyproject() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let pyproject_path = tmp.path().join("pyproject.toml");

        let existing_content = "";
        fs_anyhow::write(&pyproject_path, existing_content)?;

        let config = ConfigFile {
            project_includes: Globs::new(vec!["new/path/**/*.py".to_owned()]).unwrap(),
            ..Default::default()
        };
        PyProject::update(&pyproject_path, config)?;

        let updated_content = fs_anyhow::read_to_string(&pyproject_path)?;

        assert!(updated_content.contains("[tool.pyrefly]"));
        assert!(updated_content.contains("project-includes = [\"new/path/**/*.py\"]"));

        // Regression test for bug where we would insert an unnecessary [tool] section
        assert!(!updated_content.contains("[tool]"));

        // Make sure we don't add an extra blank line
        assert!(!updated_content.starts_with("\n"));

        Ok(())
    }

    #[test]
    fn test_pyrefly_section_ordering() -> anyhow::Result<()> {
        // This test verifies that [tool.pyrefly] is placed after ALL tool sections,
        // including deeply nested ones like [tool.ruff.lint.pydocstyle].
        let tmp = tempfile::tempdir()?;
        let ordering_path = tmp.path().join("ordering_test.toml");
        let existing_content = r#"[project]
name = "test-project"
version = "0.1.0"

# Comment before tool section
[tool]
# Comment within tool section

[tool.black]
line-length = 88

[tool.pytest]
testpaths = ["tests"]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["C4", "LOG"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
"#;
        fs_anyhow::write(&ordering_path, existing_content)?;

        let config = ConfigFile {
            project_includes: Globs::new(vec!["ordering_test.py".to_owned()]).unwrap(),
            ..Default::default()
        };
        PyProject::update(&ordering_path, config)?;

        let toml_content = fs_anyhow::read_to_string(&ordering_path)?;

        let toml_expected = concat!(
            "[project]\n",
            "name = \"test-project\"\n",
            "version = \"0.1.0\"\n",
            "\n",
            "# Comment before tool section\n",
            "[tool]\n",
            "# Comment within tool section\n",
            "\n",
            "[tool.black]\n",
            "line-length = 88\n",
            "\n",
            "[tool.pytest]\n",
            "testpaths = [\"tests\"]\n",
            "\n",
            "[tool.ruff]\n",
            "line-length = 88\n",
            "\n",
            "[tool.ruff.lint]\n",
            "select = [\"C4\", \"LOG\"]\n",
            "\n",
            "[tool.ruff.lint.pydocstyle]\n",
            "convention = \"google\"\n",
            "\n",
            "[tool.pyrefly]\n",
            "project-includes = [\"ordering_test.py\"]\n",
            "\n",
            "[build-system]\n",
            "requires = [\"setuptools\"]\n",
            "build-backend = \"setuptools.build_meta\"\n",
        );

        assert_eq!(toml_content.trim(), toml_expected.trim());

        Ok(())
    }
}
