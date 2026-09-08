/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::slice;

use itertools::Itertools;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::TextRangeWithModule;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use tempfile::TempDir;

use crate::state::lsp::FindPreference;
use crate::state::lsp::ReferenceOptions;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::TestEnv;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

#[test]
fn test_rename_inherited_method_parameter_across_files() {
    let files = [
        (
            "base",
            r#"
class Base:
    def process(self, *, value: int) -> int:
        return value
"#,
        ),
        (
            "child",
            r#"
from typing import override
from base import Base
class Child(Base):
    @override
    def process(self, value: int) -> int:
        return super().process(value=value)
"#,
        ),
        (
            "sibling",
            r#"
from base import Base
class Sibling(Base):
    def process(self, *, value: int) -> int:
        return value
"#,
        ),
        (
            "grandchild",
            r#"
from child import Child
from sibling import Sibling
class Grandchild(Child, Sibling):
    def process(self, value: int) -> int:
        return value
"#,
        ),
        (
            "usage",
            r#"
from base import Base
from child import Child
from sibling import Sibling
from grandchild import Grandchild
Base().process(value=1)
Child().process(value=2)
Sibling().process(value=3)
Grandchild().process(value=4)
"#,
        ),
        (
            "unrelated",
            r#"
from base import Base
class Other:
    def process(self, value: int) -> int:
        return value
def process(value: int) -> int:
    return value
Other().process(value=1)
process(value=2)
"#,
        ),
    ];
    assert_inherited_parameter_rename(&files);
}

#[test]
fn test_rename_inherited_method_parameter_same_file() {
    assert_inherited_parameter_rename(&[(
        "main",
        r#"
class Base:
    def process(self, value: int) -> int:
        return value

class Child(Base):
    def process(self, value: int) -> int:
        return value

class Inherited(Child):
    pass

Base().process(value=1)
Child().process(value=2)
Inherited().process(value=3)
"#,
    )]);
}

/// Check every rename entry point against all occurrences in the related modules.
fn assert_inherited_parameter_rename(files: &[(&str, &str)]) {
    let temp = TempDir::new().unwrap();
    let mut env = TestEnv::new()
        .with_default_require_level(Require::Indexing)
        .with_run_require(Require::Indexing);
    for (name, code) in files {
        let path = temp.path().join(format!("{name}.py"));
        fs::write(&path, code).unwrap();
        env.add_real_path(name, path);
    }
    let (state, get_handle) = env.to_state();
    let expected = files
        .iter()
        .filter(|(name, _)| *name != "unrelated")
        .flat_map(|(name, code)| {
            code.match_indices("value").map(move |(offset, _)| {
                (
                    name.to_string(),
                    TextRange::at(TextSize::new(offset as u32), TextSize::new(5)),
                )
            })
        })
        .sorted_by_key(|(name, range)| (name.clone(), range.start()))
        .collect::<Vec<_>>();
    // A rename from any declaration, body reference, or keyword argument must cover the
    // entire inheritance hierarchy, including siblings in separate modules.
    for (name, range) in &expected {
        let handle = get_handle(name);
        let mut transaction = state.cancellable_transaction();
        transaction
            .run(slice::from_ref(&handle), Require::Everything, None)
            .unwrap();
        let definition = transaction
            .as_ref()
            .find_definition(&handle, range.start(), FindPreference::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let references = transaction
            .find_global_references_from_definition(
                *handle.sys_info(),
                definition.metadata,
                TextRangeWithModule::new(definition.module, definition.definition_range),
                ReferenceOptions::textual_only(true),
            )
            .unwrap();
        let actual = references
            .iter()
            .flat_map(|(module, ranges)| {
                ranges
                    .iter()
                    .map(move |range| (module.name().to_string(), *range))
            })
            .sorted_by_key(|(name, range)| (name.clone(), range.start()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "Rename from {name}:{range:?}");
    }
}

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let transaction = state.transaction();
    // Mirror the reference-collection half of `textDocument/rename`: only ranges whose text
    // is the symbol being renamed, since every returned range is rewritten in place.
    let ranges =
        transaction.find_local_references(handle, position, ReferenceOptions::textual_only(true));
    let module_info = transaction.get_module_info(handle).unwrap();
    format!(
        "Rename locations:\n{}",
        ranges
            .into_iter()
            .map(|range| code_frame_of_source_at_range(module_info.contents(), range))
            .join("\n")
    )
}

#[test]
fn test_rename_parameter_updates_keyword_arguments() {
    let code = r#"
def greet(name, message):
    """Greet someone with a message."""
    print(f"{message}, {name}!")
    return name

def another_func():
    result = greet(name="Alice", message="Hello")
#                  ^
    return result
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
8 |     result = greet(name="Alice", message="Hello")
                       ^
Rename locations:
2 | def greet(name, message):
              ^^^^
4 |     print(f"{message}, {name}!")
                            ^^^^
5 |     return name
               ^^^^
8 |     result = greet(name="Alice", message="Hello")
                       ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_parameter_only_updates_correct_function() {
    let code = r#"
def func1(name, message):
    """First function with name parameter."""
    print(f"{message}, {name}!")
    return name

def func2(name, value):
    """Second function with same name parameter."""
    print(f"Value: {value}, Name: {name}")
    return name

def caller():
    result1 = func1(name="Alice", message="Hello")
#                   ^
    result2 = func2(name="Bob", value=42)
    return result1, result2
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
13 |     result1 = func1(name="Alice", message="Hello")
                         ^
Rename locations:
2 | def func1(name, message):
              ^^^^
4 |     print(f"{message}, {name}!")
                            ^^^^
5 |     return name
               ^^^^
13 |     result1 = func1(name="Alice", message="Hello")
                         ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_function_parameter_updates_call_sites() {
    let code = r#"
def greet(name, message):
#         ^
    """Greet someone with a message."""
    print(f"{message}, {name}!")
    return name

def caller():
    result1 = greet(name="Alice", message="Hello")
    result2 = greet(message="Hi", name="Bob")
    result3 = greet(name="Charlie", message="Hey")
    return result1, result2, result3
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def greet(name, message):
              ^
Rename locations:
2 | def greet(name, message):
              ^^^^
5 |     print(f"{message}, {name}!")
                            ^^^^
6 |     return name
               ^^^^
9 |     result1 = greet(name="Alice", message="Hello")
                        ^^^^
10 |     result2 = greet(message="Hi", name="Bob")
                                       ^^^^
11 |     result3 = greet(name="Charlie", message="Hey")
                         ^^^^
"#
        .trim(),
        report.trim(),
    );
}

/// Find-references reports `Foo()` as a reference to `Foo.__init__`, but that range spells the
/// class name. Rename must skip it, or renaming `__init__` would rewrite the constructor call.
#[test]
fn test_rename_dunder_init_skips_constructor_call_sites() {
    let code = r#"
class Foo:
    def __init__(self): ...
    #   ^

Foo()
Foo().__init__()
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     def __init__(self): ...
            ^
Rename locations:
3 |     def __init__(self): ...
            ^^^^^^^^
7 | Foo().__init__()
          ^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_legacy_type_parameters_updates_constructor_names() {
    let code = r#"
from typing import Callable, ParamSpec, TypeVar, TypeVarTuple

T = TypeVar("T")
P = ParamSpec(name="P")
Ts = TypeVarTuple("Ts")
unrelated = "T"

def f(value: T) -> T:
#            ^
    return value

def g(func: Callable[P, None]) -> None:
#                    ^
    pass

def h(value: tuple[*Ts]) -> None:
#                   ^
    pass
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 | def f(value: T) -> T:
                 ^
Rename locations:
4 | T = TypeVar("T")
    ^
4 | T = TypeVar("T")
                 ^
9 | def f(value: T) -> T:
                 ^
9 | def f(value: T) -> T:
                       ^

13 | def g(func: Callable[P, None]) -> None:
                          ^
Rename locations:
5 | P = ParamSpec(name="P")
    ^
5 | P = ParamSpec(name="P")
                        ^
13 | def g(func: Callable[P, None]) -> None:
                          ^

17 | def h(value: tuple[*Ts]) -> None:
                         ^
Rename locations:
6 | Ts = TypeVarTuple("Ts")
    ^^
6 | Ts = TypeVarTuple("Ts")
                       ^^
17 | def h(value: tuple[*Ts]) -> None:
                         ^^
"#
        .trim(),
        report.trim(),
    );
}
