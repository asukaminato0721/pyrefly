/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use pyrefly_build::handle::Handle;
use pyrefly_python::keywords::is_valid_identifier;
use pyrefly_python::module_path::ModulePathDetails;
use toml::Value;

/// Installed modules that do not belong to a declared dependency.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AutoImportFilter {
    excluded: HashSet<String>,
    site_packages: Vec<PathBuf>,
}

/// Normalize a distribution name according to the Python packaging specification.
fn normalize_distribution(name: &str) -> String {
    name.split(['-', '_', '.'])
        .filter(|part| !part.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join("-")
}

/// Read all declared extras and dependency groups, regardless of environment markers.
fn direct_dependencies(root: &Path) -> Option<HashSet<String>> {
    let contents = fs::read_to_string(root.join("pyproject.toml")).ok()?;
    let project: Value = toml::from_str(&contents).ok()?;
    let metadata = project.get("project");
    if metadata
        .and_then(|value| value.get("dynamic"))
        .and_then(Value::as_array)
        .is_some_and(|fields| {
            fields.iter().any(|field| {
                matches!(
                    field.as_str(),
                    Some("dependencies" | "optional-dependencies")
                )
            })
        })
    {
        return None;
    }
    let dependencies = metadata.and_then(|value| value.get("dependencies"));
    let optional = metadata.and_then(|value| value.get("optional-dependencies"));
    let groups = project.get("dependency-groups");
    // Legacy manifests and missing declarations do not establish which packages are direct.
    if dependencies.is_none() && optional.is_none() && groups.is_none() {
        return None;
    }
    let mut requirements = Vec::new();
    if let Some(dependencies) = dependencies {
        requirements.extend(dependencies.as_array()?);
    }
    for table in [optional, groups].into_iter().flatten() {
        for group in table.as_table()?.values() {
            requirements.extend(group.as_array()?);
        }
    }
    let mut direct = HashSet::new();
    for requirement in requirements {
        // Every group is included, so group references need no recursive expansion.
        if requirement
            .get("include-group")
            .and_then(Value::as_str)
            .is_some()
        {
            continue;
        }
        let requirement = requirement.as_str()?.trim_start();
        let end = requirement
            .find(|ch: char| !ch.is_ascii_alphanumeric() && !matches!(ch, '-' | '_' | '.'))
            .unwrap_or(requirement.len());
        if end == 0 {
            return None;
        }
        direct.insert(normalize_distribution(&requirement[..end]));
    }
    Some(direct)
}

impl AutoImportFilter {
    /// Cache package metadata when the project configuration is loaded.
    pub(crate) fn from_project(root: &Path, site_packages: Vec<PathBuf>) -> Self {
        let Some(direct) = direct_dependencies(root) else {
            return Self::default();
        };
        let mut included = HashSet::new();
        let mut excluded = HashSet::new();
        for site in &site_packages {
            let Ok(entries) = fs::read_dir(site) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_none_or(|ext| ext != "dist-info") {
                    continue;
                }
                let Ok(metadata) = fs::read_to_string(path.join("METADATA")) else {
                    continue;
                };
                let Some(name) = metadata
                    .lines()
                    .find_map(|line| line.strip_prefix("Name: "))
                else {
                    continue;
                };
                let name = normalize_distribution(name.trim());
                // Stub distributions can provide the same import names as their runtime package.
                let is_direct = direct.contains(&name)
                    || name
                        .strip_prefix("types-")
                        .is_some_and(|name| direct.contains(name))
                    || name
                        .strip_suffix("-stubs")
                        .is_some_and(|name| direct.contains(name));
                let modules = if is_direct {
                    &mut included
                } else {
                    &mut excluded
                };
                if let Ok(top_level) = fs::read_to_string(path.join("top_level.txt")) {
                    modules.extend(
                        top_level
                            .lines()
                            .map(str::trim)
                            .filter(|name| is_valid_identifier(name))
                            .map(str::to_owned),
                    );
                }
                // Wheels need not contain top_level.txt. RECORD supplies installed import paths.
                if let Ok(record) = fs::read_to_string(path.join("RECORD")) {
                    for line in record.lines() {
                        let Some((file, _)) = line.split_once(',') else {
                            continue;
                        };
                        let file = file.trim_matches('"');
                        if ![".py", ".pyi", ".so", ".pyd"]
                            .iter()
                            .any(|ext| file.ends_with(ext))
                        {
                            continue;
                        }
                        let module = if let Some((directory, _)) = file.split_once('/') {
                            directory.strip_suffix("-stubs").unwrap_or(directory)
                        } else {
                            file.split_once('.')
                                .expect("Python source and extension files have a suffix")
                                .0
                        };
                        if is_valid_identifier(module) {
                            modules.insert(module.to_owned());
                        }
                    }
                }
            }
        }
        // A namespace can be shared by several distributions. Keep it if any owner is direct.
        excluded.retain(|module| !included.contains(module));
        Self {
            excluded,
            site_packages,
        }
    }

    pub(crate) fn allows(&self, handle: &Handle) -> bool {
        if !self
            .excluded
            .contains(handle.module().first_component().as_str())
        {
            return true;
        }
        match handle.path().details() {
            ModulePathDetails::FileSystem(path) | ModulePathDetails::Namespace(path) => {
                !self.site_packages.iter().any(|site| path.starts_with(site))
            }
            ModulePathDetails::BundledThirdParty(_)
            | ModulePathDetails::BundledTypeshedThirdParty(_) => false,
            ModulePathDetails::Memory(_) | ModulePathDetails::BundledTypeshed(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use tempfile::TempDir;

    use super::*;
    use crate::config::ConfigFile;

    fn distribution(site: &Path, name: &str, file: &str, contents: &str) {
        let path = site.join(format!("{name}-1.0.dist-info"));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("METADATA"), format!("Name: {name}\n")).unwrap();
        fs::write(path.join(file), contents).unwrap();
    }

    #[test]
    fn declarations_include_extras_groups_and_normalized_names() {
        let root = TempDir::new().unwrap();
        fs::write(
            root.path().join("pyproject.toml"),
            r#"
[project]
dependencies = [
    "My_Package.name[extra]>=1; python_version >= '3.10'",
    "Pillow @ https://example.com/pillow.whl",
]
[project.optional-dependencies]
test = ["pytest>=8"]
[dependency-groups]
lint = ["ruff"]
dev = [{include-group = "lint"}, "mypy"]
"#,
        )
        .unwrap();
        assert_eq!(
            direct_dependencies(root.path()).unwrap(),
            ["my-package-name", "pillow", "pytest", "ruff", "mypy"]
                .map(str::to_owned)
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn unavailable_declarations_do_not_filter() {
        let root = TempDir::new().unwrap();
        assert!(direct_dependencies(root.path()).is_none());
        for contents in [
            "not valid toml",
            "[tool.poetry.dependencies]\nrequests = '*'",
            "[project]\ndynamic = ['dependencies']\n[dependency-groups]\ndev = ['pytest']",
            "[project]\ndependencies = []\ndynamic = ['optional-dependencies']",
            "[project]\ndependencies = [42]",
        ] {
            fs::write(root.path().join("pyproject.toml"), contents).unwrap();
            assert!(direct_dependencies(root.path()).is_none(), "{contents}");
        }
        fs::write(
            root.path().join("pyproject.toml"),
            "[project]\ndependencies = []",
        )
        .unwrap();
        assert_eq!(direct_dependencies(root.path()), Some(HashSet::new()));
    }

    #[test]
    fn installed_metadata_maps_distribution_names_to_modules() {
        let root = TempDir::new().unwrap();
        let site = root.path().join("site-packages");
        fs::write(
            root.path().join("pyproject.toml"),
            "[project]\ndependencies = ['Pillow', 'direct']",
        )
        .unwrap();
        distribution(&site, "Pillow", "top_level.txt", "PIL\n");
        distribution(&site, "direct", "RECORD", "shared/direct/__init__.py,,\n");
        distribution(
            &site,
            "types-Pillow",
            "RECORD",
            "PIL-stubs/__init__.pyi,,\n",
        );
        distribution(
            &site,
            "indirect",
            "RECORD",
            "hidden/__init__.py,,\n\
             shared/indirect/__init__.py,,\n\
             single.py,,\n\
             native.cpython-313.so,,\n\
             indirect-1.0.dist-info/METADATA,,\n\
             ../../../bin/script.py,,\n",
        );
        distribution(&site, "unknown", "RECORD", "");
        let filter = AutoImportFilter::from_project(root.path(), vec![site.clone()]);
        assert_eq!(
            filter.excluded,
            ["hidden", "single", "native"]
                .map(str::to_owned)
                .into_iter()
                .collect()
        );
        for (path, expected) in [
            (
                ModulePath::filesystem(site.join("hidden/submodule.py")),
                false,
            ),
            (ModulePath::namespace(site.join("hidden")), false),
            (
                ModulePath::bundled_third_party("hidden/submodule.pyi".into()),
                false,
            ),
            (
                ModulePath::bundled_typeshed_third_party("hidden/submodule.pyi".into()),
                false,
            ),
            (
                ModulePath::filesystem(root.path().join("hidden/submodule.py")),
                true,
            ),
            (
                ModulePath::bundled_typeshed("hidden/submodule.pyi".into()),
                true,
            ),
            (ModulePath::memory("hidden/submodule.py".into()), true),
        ] {
            let handle = Handle::new(
                ModuleName::from_str("hidden.submodule"),
                path,
                Default::default(),
            );
            assert_eq!(filter.allows(&handle), expected, "{handle:?}");
        }
    }

    #[test]
    fn configuration_reload_refreshes_dependencies() {
        let root = TempDir::new().unwrap();
        let site = root.path().join("site-packages");
        distribution(&site, "indirect", "top_level.txt", "hidden");
        let mut config = ConfigFile::default();
        config.interpreters.skip_interpreter_query = true;
        config.auto_import_exclude_transitive_dependencies = true;
        config.python_environment.site_package_path = Some(vec![site.clone()]);
        let handle = Handle::new(
            ModuleName::from_str("hidden"),
            ModulePath::filesystem(site.join("hidden.py")),
            Default::default(),
        );
        for (dependencies, expected) in [("[]", false), ("['indirect']", true)] {
            fs::write(
                root.path().join("pyproject.toml"),
                format!("[project]\ndependencies = {dependencies}"),
            )
            .unwrap();
            config.configure_at(Some(root.path()));
            assert_eq!(config.allows_auto_import(&handle), expected);
        }
    }
}
