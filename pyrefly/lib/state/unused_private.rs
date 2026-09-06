/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::module::TextRangeWithModule;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::Stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::error::collector::ErrorCollector;
use crate::state::errors::Errors;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::ReferenceOptions;
use crate::state::require::Require;
use crate::state::state::Transaction;

struct Candidate {
    name: Identifier,
    /// References inside a function body depend on that function being live. Initializers,
    /// class bodies, decorators, and default arguments execute when the definition is evaluated.
    body: Option<TextRange>,
    class_member: bool,
}

#[derive(Default)]
struct Candidates {
    items: Vec<Candidate>,
    in_class: bool,
}

impl Candidates {
    fn add(&mut self, name: Identifier, body: Option<TextRange>) {
        let text = name.as_str();
        if text.starts_with('_') && text != "_" && !(text.starts_with("__") && text.ends_with("__"))
        {
            self.items.push(Candidate {
                name,
                body,
                class_member: self.in_class,
            });
        }
    }

    fn assignment(&mut self, target: &Expr) {
        match target {
            Expr::Name(name) => self.add(Identifier::new(name.id.clone(), name.range), None),
            Expr::Tuple(tuple) => {
                for element in &tuple.elts {
                    self.assignment(element);
                }
            }
            Expr::List(list) => {
                for element in &list.elts {
                    self.assignment(element);
                }
            }
            Expr::Starred(starred) => self.assignment(&starred.value),
            _ => {}
        }
    }
}

impl<'a> Visitor<'a> for Candidates {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(function) => {
                // Decorators can register functions without any other reference to their name.
                if function.decorator_list.is_empty() {
                    self.add(
                        function.name.clone(),
                        function
                            .body
                            .first()
                            .zip(function.body.last())
                            .map(|(first, last)| TextRange::new(first.start(), last.end())),
                    );
                }
                // Underscores on function-local bindings conventionally mean intentionally unused.
            }
            Stmt::ClassDef(class) => {
                if class.decorator_list.is_empty() {
                    self.add(class.name.clone(), None);
                    let in_class = self.in_class;
                    self.in_class = true;
                    for stmt in &class.body {
                        self.visit_stmt(stmt);
                    }
                    self.in_class = in_class;
                }
            }
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    self.assignment(target);
                }
            }
            Stmt::AnnAssign(assign) => self.assignment(&assign.target),
            Stmt::TypeAlias(alias) => self.assignment(&alias.name),
            _ => walk_stmt(self, stmt),
        }
    }
}

impl Transaction<'_> {
    /// Collect diagnostics after checking all supplied project files. Reference data is retained
    /// only when the private-symbol diagnostic is enabled in at least one of those files.
    pub(crate) fn get_project_errors(&mut self, handles: &[Handle]) -> Errors {
        let enabled: Vec<_> = handles
            .iter()
            .filter(|handle| {
                !handle.path().is_interface()
                    && self.get_config(handle).is_some_and(|config| {
                        config
                            .get_error_config(handle.path().as_path())
                            .display_config
                            .severity(ErrorKind::UnusedPrivateSymbol)
                            != Severity::Ignore
                    })
            })
            .collect();
        if enabled.is_empty() {
            return self.get_errors(handles);
        }
        // A file with the diagnostic disabled can still keep another file's private symbol live.
        self.run(handles, Require::Everything, None);
        let mut candidates = Vec::new();
        for handle in enabled {
            let Some(ast) = self.get_ast(handle) else {
                continue;
            };
            let exports = self.get_exports_data(handle);
            let mut visitor = Candidates::default();
            for stmt in &ast.body {
                visitor.visit_stmt(stmt);
            }
            for candidate in visitor.items {
                if !candidate.class_member
                    && (exports.unresolvable_dunder_all_range().is_some()
                        || exports
                            .explicit_dunder_all_names()
                            .is_some_and(|names| names.contains(&candidate.name.id)))
                {
                    continue;
                }
                // An override can be invoked through its base class's interface.
                if candidate.class_member
                    && self
                        .get_solutions(handle)
                        .and_then(|s| s.get_index())
                        .is_some_and(|index| {
                            index
                                .lock()
                                .parent_methods_map
                                .contains_key(&candidate.name.range)
                        })
                {
                    continue;
                }
                candidates.push((handle, candidate));
            }
        }

        let mut owners = SmallMap::new();
        for (index, (handle, candidate)) in candidates.iter().enumerate() {
            if let Some(body) = candidate.body {
                owners
                    .entry(handle.path())
                    .or_insert_with(Vec::new)
                    .push((body, index));
            }
        }
        for bodies in owners.values_mut() {
            bodies.sort_unstable_by_key(|(body, _)| body.start());
        }
        let mut live = vec![false; candidates.len()];
        let mut edges = vec![Vec::new(); candidates.len()];
        for (target, (handle, candidate)) in candidates.iter().enumerate() {
            let module = self
                .get_module_info(handle)
                .expect("A private symbol candidate has a parsed module");
            let Ok(references) = self.find_global_references_from_definition(
                *handle.sys_info(),
                DefinitionMetadata::VariableOrAttribute(None),
                TextRangeWithModule::new(module, candidate.name.range),
                ReferenceOptions::all(false),
            ) else {
                // A cancelled reference search cannot establish that a symbol is unused.
                return self.get_errors(handles);
            };
            for (module, ranges) in references {
                for range in ranges {
                    // Candidate function bodies do not overlap: function-local definitions
                    // are excluded. The last body starting before this reference is its only
                    // possible owner.
                    let owner = owners.get(module.path()).and_then(|bodies| {
                        let position =
                            bodies.partition_point(|(body, _)| body.start() <= range.start());
                        let (body, index) = bodies.get(position.checked_sub(1)?)?;
                        body.contains_range(range).then_some(*index)
                    });
                    if let Some(owner) = owner {
                        edges[owner].push(target);
                    } else {
                        live[target] = true;
                    }
                }
            }
        }
        // References from unused private definitions, including recursive cycles, are not roots.
        let mut pending: Vec<_> = live
            .iter()
            .enumerate()
            .filter_map(|(index, live)| live.then_some(index))
            .collect();
        while let Some(owner) = pending.pop() {
            for &target in &edges[owner] {
                if !live[target] {
                    live[target] = true;
                    pending.push(target);
                }
            }
        }
        let mut errors = self.get_errors(handles);
        for ((handle, candidate), live) in candidates.into_iter().zip(live) {
            if !live {
                let load = self
                    .get_load(handle)
                    .expect("A private symbol candidate has a loaded module");
                let collector = errors
                    .project_errors
                    .entry(handle.path().dupe())
                    .or_insert_with(|| {
                        ErrorCollector::new(load.module_info.dupe(), load.errors.style())
                    });
                collector
                    .error_builder(
                        candidate.name.range,
                        ErrorKind::UnusedPrivateSymbol,
                        format!(
                            "Private symbol `{}` is not used in the checked project",
                            candidate.name
                        ),
                    )
                    .emit();
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pyrefly_config::error::ErrorDisplayConfig;
    use pyrefly_config::error_kind::ErrorKind;
    use pyrefly_config::error_kind::Severity;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_util::arc_id::ArcId;
    use pyrefly_util::thread_pool::ThreadCount;
    use tempfile::TempDir;

    use super::*;
    use crate::config::config::ConfigFile;
    use crate::config::finder::ConfigFinder;
    use crate::state::require::Require;
    use crate::state::state::State;

    /// Check real files so imports and reverse dependencies use the same paths as the CLI.
    fn project(sources: &[(&str, &str)], severity: Severity) -> (TempDir, State, Vec<Handle>) {
        let root = TempDir::new().unwrap();
        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        config.interpreters.skip_interpreter_query = true;
        config.search_path_from_file = vec![root.path().to_path_buf()];
        config.disable_search_path_heuristics = true;
        config.root.errors = Some(ErrorDisplayConfig::new(
            [
                (ErrorKind::UnusedPrivateSymbol, severity),
                (ErrorKind::UnusedIgnore, Severity::Error),
            ]
            .into(),
        ));
        config.configure();
        let sys_info = config.get_sys_info();
        let state = State::new(
            ConfigFinder::new_constant(ArcId::new(config)),
            ThreadCount::Inline,
        );
        let handles = sources
            .iter()
            .map(|(filename, source)| {
                let path = root.path().join(filename);
                fs::write(&path, source).unwrap();
                Handle::new(
                    ModuleName::from_str(path.file_stem().unwrap().to_str().unwrap()),
                    ModulePath::filesystem(path),
                    sys_info,
                )
            })
            .collect();
        (root, state, handles)
    }

    fn assert_unused(errors: &Errors, expected: &[&str]) {
        let collected = errors.collect_errors();
        let mut names: Vec<_> = collected
            .ordinary
            .iter()
            .map(|error| {
                assert_eq!(
                    error.error_kind(),
                    ErrorKind::UnusedPrivateSymbol,
                    "{error:?}"
                );
                error.module().code_at(error.range())
            })
            .collect();
        names.sort();
        assert_eq!(names, expected);
    }

    #[test]
    fn test_unused_private_project() {
        let (_root, state, handles) = project(
            &[
                (
                    "library.py",
                    r#"
def _dead(): pass
def _used(): pass
def _via_module(): pass
def _via_reexport(): pass
class Public:
    def _dead_method(self): pass
    def _used_method(self): pass
"#,
                ),
                (
                    "reexport.py",
                    "from library import _via_reexport as public\n",
                ),
                (
                    "consumer.py",
                    r#"
from library import _used as alias, Public
from reexport import public
import library
alias()
public()
library._via_module()
Public()._used_method()
"#,
                ),
            ],
            Severity::Error,
        );
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.run(&handles, Require::Errors, None);
        assert_unused(
            &transaction.get_project_errors(&handles),
            &["_dead", "_dead_method"],
        );
    }

    #[test]
    fn test_unused_private_same_attribute_in_different_modules() {
        let definition = "class Public:\n    def _method(self): pass\n";
        let consumer = format!("{definition}\nimport library\nlibrary.Public()._method()\n");
        let (_root, state, handles) = project(
            &[("library.py", definition), ("consumer.py", &consumer)],
            Severity::Error,
        );
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.run(&handles, Require::Errors, None);
        let errors = transaction.get_project_errors(&handles);
        assert_unused(&errors, &["_method"]);
        assert_eq!(
            errors.collect_errors().ordinary[0].module().name(),
            ModuleName::from_str("consumer"),
        );
    }

    #[test]
    fn test_unused_private_reachability() {
        let (_root, state, handles) = project(
            &[(
                "main.py",
                r#"
def _recursive():
    _recursive()
def _first():
    _second()
def _second():
    _first()
def _live():
    _helper()
def _helper():
    pass
def public():
    _live()
def _initializer():
    return 1
_unused_value = _initializer()
def _default():
    return 1
def _unused_function(value=_default()):
    pass
"#,
            )],
            Severity::Error,
        );
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.run(&handles, Require::Errors, None);
        assert_unused(
            &transaction.get_project_errors(&handles),
            &[
                "_first",
                "_recursive",
                "_second",
                "_unused_function",
                "_unused_value",
            ],
        );
    }

    #[test]
    fn test_unused_private_exemptions() {
        let (_root, state, handles) = project(
            &[
                (
                    "main.py",
                    r#"
from abc import abstractmethod
__all__ = ["_exported"]
def _exported(): pass
def public(): pass
def __dunder__(): pass
_ = 1
def register(f): return f
@register
def _registered(): pass
@register
class _Decorated:
    _field: int
class Base:
    def _method(self): pass
    @abstractmethod
    def _abstract(self): pass
class Child(Base):
    def _method(self): pass
    def _abstract(self): pass
    def _unused(self): pass
def local():
    _variable = 1
    def _function(): pass
"#,
                ),
                (
                    "dynamic.py",
                    "def exports() -> list[str]: return []\n__all__ = exports()  # pyrefly: ignore[unresolvable-dunder-all]\ndef _unknown(): pass\n",
                ),
                ("interface.pyi", "def _stub() -> None: ...\n"),
            ],
            Severity::Error,
        );
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.run(&handles, Require::Errors, None);
        assert_unused(&transaction.get_project_errors(&handles), &["_unused"]);
    }

    #[test]
    fn test_unused_private_variables_and_shadowing() {
        let (_root, state, handles) = project(
            &[(
                "main.py",
                r#"
_dead: int = 1
_used = 1
_pair, _other = (1, 2)
class _Unused: pass
class _Forward: pass
type _Alias = int
type _UsedAlias = int
def takes(value: "_Forward", other: _UsedAlias): pass
class Public:
    class _Inner: pass
    _field: int = 1
    _used_field = 1
    def read(self):
        return self._used_field
print(_used, _other)
def takes_nested(value: "Public._Inner"): pass
def shadow(_dead: int):
    print(_dead)
"#,
            )],
            Severity::Error,
        );
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.run(&handles, Require::Errors, None);
        assert_unused(
            &transaction.get_project_errors(&handles),
            &["_Alias", "_Unused", "_dead", "_field", "_pair"],
        );
    }

    #[test]
    fn test_unused_private_suppression() {
        let (_root, state, handles) = project(
            &[(
                "main.py",
                r#"
def _suppressed():  # pyrefly: ignore[unused-private-symbol]
    pass
def _reported(): pass
"#,
            )],
            Severity::Warn,
        );
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.run(&handles, Require::Errors, None);
        let errors = transaction.get_project_errors(&handles);
        assert_unused(&errors, &["_reported"]);
        let collected = errors.collect_errors();
        assert_eq!(collected.ordinary[0].severity(), Severity::Warn);
        assert_eq!(collected.suppressed.len(), 1);
        assert!(errors.collect_unused_ignore_errors(&collected).is_empty());
    }

    #[test]
    fn test_unused_private_disabled() {
        let (_root, state, handles) =
            project(&[("main.py", "def _unused(): pass\n")], Severity::Ignore);
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.run(&handles, Require::Errors, None);
        assert_unused(&transaction.get_project_errors(&handles), &[]);
        assert_eq!(transaction.get_require(&handles[0]), Some(Require::Errors));
    }

    #[test]
    fn test_unused_private_incremental() {
        let (root, state, handles) = project(
            &[
                ("library.py", "def _helper(): pass\n"),
                ("consumer.py", "import library\n"),
            ],
            Severity::Error,
        );
        let mut transaction = state.new_committable_transaction(Require::Exports, None);
        transaction.as_mut().run(&handles, Require::Errors, None);
        let old_errors = transaction.as_mut().get_project_errors(&handles);
        assert_unused(&old_errors, &["_helper"]);
        state.commit_transaction(transaction, None);

        for (source, expected) in [
            ("import library\nlibrary._helper()\n", &[][..]),
            ("import library\n", &["_helper"][..]),
        ] {
            let path = root.path().join("consumer.py");
            fs::write(&path, source).unwrap();
            let mut transaction = state.new_committable_transaction(Require::Exports, None);
            transaction.as_mut().invalidate_disk(&[path]);
            transaction.as_mut().run(&handles, Require::Errors, None);
            assert_unused(&transaction.as_mut().get_project_errors(&handles), expected);
            state.commit_transaction(transaction, None);
        }
        assert_unused(&old_errors, &["_helper"]);
    }
}
