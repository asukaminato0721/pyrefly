/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CompletionItem;
use lsp_types::CompletionItemTextEdit;
use lsp_types::Range;
use lsp_types::TextEdit;
use pretty_assertions::assert_eq;
use pyrefly_python::ast::Ast;
use pyrefly_python::sys_info::PythonVersion;
use ruff_python_ast::PySourceType;
use ruff_text_size::TextSize;

use crate::state::lsp::ImportFormat;
use crate::state::require::Require;
use crate::test::util::TestEnv;
use crate::test::util::mk_multi_file_state;
use crate::test::util::mk_multi_file_state_with_env;

/// Applies the completion and all additional edits to the original buffer.
fn complete(code: &str, dependencies: &[(&'static str, &str)]) -> (CompletionItem, String) {
    let mut files = dependencies.to_vec();
    files.push(("main", code));
    let (handles, state) = mk_multi_file_state(&files, Require::Everything, false);
    let handle = &handles["main"];
    let txn = state.transaction();
    let module = txn.get_module_info(handle).unwrap();
    let name_start = code.rfind("def met").unwrap() + "def ".len();
    let position = TextSize::try_from(name_start + "met".len()).unwrap();
    let item = txn
        .completion(handle, position, ImportFormat::Absolute, true, None)
        .into_iter()
        .find(|item| item.label == "method")
        .expect("expected a method-name completion");
    let main_edit = match &item.text_edit {
        Some(CompletionItemTextEdit::TextEdit(edit)) => edit.clone(),
        Some(CompletionItemTextEdit::InsertReplaceEdit(_)) => {
            panic!("unexpected insert/replace edit")
        }
        None => TextEdit {
            range: Range::new(
                module.to_lsp_position(TextSize::try_from(name_start).unwrap()),
                module.to_lsp_position(position),
            ),
            new_text: item
                .insert_text
                .clone()
                .unwrap_or_else(|| item.label.clone()),
        },
    };
    let mut edits = item.additional_text_edits.clone().unwrap_or_default();
    edits.push(main_edit);
    let mut edits = edits
        .into_iter()
        .map(|edit| {
            let start = module.from_lsp_position(edit.range.start, None).to_usize();
            let end = module.from_lsp_position(edit.range.end, None).to_usize();
            (start..end, edit.new_text)
        })
        .collect::<Vec<_>>();
    edits.sort_by_key(|(range, _)| range.start);
    for pair in edits.windows(2) {
        assert!(pair[0].0.end <= pair[1].0.start, "completion edits overlap");
    }
    let mut result = code.to_owned();
    for (range, text) in edits.into_iter().rev() {
        result.replace_range(range, &text);
    }
    (item, result)
}

/// Checks Python syntax and type errors after all completion edits apply.
fn check_result(code: &str, dependencies: &[(&'static str, &str)]) {
    let (_, errors, unsupported) = Ast::parse(code, PySourceType::Python);
    assert!(
        errors.is_empty(),
        "invalid completed source:\n{code}\n{errors:?}"
    );
    assert!(unsupported.is_empty(), "{unsupported:?}");
    let mut files = dependencies.to_vec();
    files.push(("main", code));
    mk_multi_file_state(&files, Require::Everything, true);
}

#[test]
fn instance_method() {
    let code = "class Base:\n    def method(this, value: str, /, *args: int, flag: bool = False, **kwargs: str) -> int:\n        return 0\n\nclass Child(Base):\n    def met\n";
    let (item, result) = complete(code, &[]);
    assert!(item.text_edit.is_some());
    assert_eq!(
        result,
        "from typing import override\nclass Base:\n    def method(this, value: str, /, *args: int, flag: bool = False, **kwargs: str) -> int:\n        return 0\n\nclass Child(Base):\n    @override\n    def method(this, value: str, /, *args: int, flag: bool = False, **kwargs: str) -> int:\n        return super().method(value, *args, flag=flag, **kwargs)\n"
    );
    check_result(&result, &[]);
}

#[test]
fn imported_annotations_and_future_import() {
    let deps = [(
        "base",
        "class Payload: pass\nclass Base:\n    def method(self, value: Payload) -> Payload:\n        return value\n",
    )];
    let code = "\"\"\"Module documentation.\"\"\"\nfrom __future__ import annotations\nfrom base import Base\n\nclass Child(Base):\n    def met\n";
    let (item, result) = complete(code, &deps);
    assert!(item.text_edit.is_some());
    assert!(result.starts_with("\"\"\"Module documentation.\"\"\"\nfrom __future__ import annotations\nfrom base import Payload\nfrom typing import override\n"), "{result}");
    check_result(&result, &deps);
}

#[test]
fn existing_import_aliases() {
    let deps = [(
        "base",
        "class Payload: pass\nclass Base:\n    def method(self, value: Payload) -> Payload:\n        return value\n",
    )];
    let code = "from base import Base, Payload as Data\nfrom typing import override as overrides\nclass Child(Base):\n    def met\n";
    let (_, result) = complete(code, &deps);
    assert!(
        result.contains("@overrides\n    def method(self, value: Data) -> Data:"),
        "{result}"
    );
    assert_eq!(result.matches("from typing").count(), 1);
    check_result(&result, &deps);
}

#[test]
fn crlf_and_tabs() {
    let code = "class Base:\r\n\tdef method(this, value: int = 7) -> int:\r\n\t\treturn value\r\nclass Child(Base):\r\n\tdef met\r\n";
    let (_, result) = complete(code, &[]);
    assert!(!result.replace("\r\n", "").contains('\n'));
    assert!(result.contains("\t@override\r\n\tdef method(this, value: int = 7) -> int:\r\n\t\treturn super().method(value)"), "{result}");
    check_result(&result, &[]);
}

#[test]
fn existing_declaration_stays_name_only() {
    let code = "class Base:\n    def method(this, value: int) -> int:\n        return value\nclass Child(Base):\n    def met(this, value: int) -> int:\n        return value + 1\n";
    let (item, result) = complete(code, &[]);
    assert!(item.text_edit.is_none());
    assert!(item.additional_text_edits.is_none());
    assert_eq!(result, code.replace("def met(", "def method("));
    check_result(&result, &[]);
}

#[test]
fn unsupported_methods_stay_name_only() {
    for (base, child) in [
        (
            "    @staticmethod\n    def method(self: int) -> int: return self\n",
            "    def met\n",
        ),
        (
            "    @staticmethod\n    def method(cls: int) -> int: return cls\n",
            "    @staticmethod\n    def met\n",
        ),
        (
            "    @classmethod\n    def method(cls, value: int) -> int: return value\n",
            "    @classmethod\n    def met\n",
        ),
        (
            "    @classmethod\n    def method(cls, value: int) -> int: return value\n",
            "    def met\n",
        ),
        (
            "    @property\n    def method(self) -> int: return 0\n",
            "    @property\n    def met\n",
        ),
        (
            "    async def method(self) -> int: return 0\n",
            "    async def met\n",
        ),
        (
            "    async def method(self):\n        yield 1\n",
            "    async def met\n",
        ),
        (
            "    def method[T](self, value: T) -> T: return value\n",
            "    def met\n",
        ),
        (
            "    def method(self, callback: Callable[[int], str]) -> str: return callback(0)\n",
            "    def met\n",
        ),
        (
            "    def method(self, values: tuple[int, ...] = ()) -> tuple[int, ...]: return values\n",
            "    def met\n",
        ),
        (
            "    @overload\n    def method(self, value: int) -> int: ...\n    @overload\n    def method(self, value: str) -> str: ...\n    def method(self, value: int | str) -> int | str: return value\n",
            "    def met\n",
        ),
        (
            "    def method(self) -> None: pass\n",
            "    @override\n    def met\n",
        ),
        (
            "    def method(super, value: int) -> int: return value\n",
            "    def met\n",
        ),
        (
            "    def method(self) -> None: pass\n",
            "    override = 0\n    def met\n",
        ),
    ] {
        let code = format!(
            "from typing import Callable, overload, override\nclass Base:\n{base}\nclass Child(Base):\n{child}"
        );
        let (item, _) = complete(&code, &[]);
        assert!(item.text_edit.is_none(), "unexpected stub for:\n{code}");
        assert!(
            item.additional_text_edits.is_none(),
            "unexpected edits for:\n{code}"
        );
    }
}

#[test]
fn current_class_method_stays_name_only() {
    let code = "class Child:\n    def method(this, value: int) -> int:\n        return value\n    def met\n";
    let (item, _) = complete(code, &[]);
    assert!(item.text_edit.is_none());
    assert!(item.additional_text_edits.is_none());
}

#[test]
fn unresolved_override_binding_stays_name_only() {
    let code = "override = 0\nclass Base:\n    def method(self) -> None: pass\nclass Child(Base):\n    def met\n";
    let (item, _) = complete(code, &[]);
    assert!(item.text_edit.is_none());
    assert!(item.additional_text_edits.is_none());
}

#[test]
fn missing_override_export_stays_name_only() {
    let code = "class Base:\n    def method(self) -> None: pass\nclass Child(Base):\n    def met\n";
    let env = TestEnv::new().with_version(PythonVersion {
        major: 3,
        minor: 11,
        micro: 0,
    });
    let (handles, state) =
        mk_multi_file_state_with_env(env, &[("main", code)], Require::Everything, false);
    let position = TextSize::try_from(code.rfind("met").unwrap() + "met".len()).unwrap();
    let item = state
        .transaction()
        .completion(
            &handles["main"],
            position,
            ImportFormat::Absolute,
            true,
            None,
        )
        .into_iter()
        .find(|item| item.label == "method")
        .expect("expected inherited method completion");
    assert!(item.text_edit.is_none());
    assert!(item.additional_text_edits.is_none());
}
