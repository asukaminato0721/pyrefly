/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[cfg(unix)]
use std::os::unix::fs::symlink;
use std::time::Instant;

use lsp_types::PrepareRenameRequest;
use lsp_types::RenameRequest;
use lsp_types::Uri;
use pyrefly_lsp_test::IndexingMode;
use pyrefly_lsp_test::LspArgs;
use pyrefly_lsp_test::Message;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use serde_json::json;
use tempfile::TempDir;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_rename_constructor_call_renames_class() {
    let root = TempDir::new().unwrap();
    let path = root.path().join("main.py");
    std::fs::write(
        &path,
        r#"from typing import Protocol

class WorkbookAction(Protocol):
    def __call__(self, value: int) -> None: ...

class FooAction(WorkbookAction):
    def __init__(self, sheet_name: str) -> None:
        self.sheet_name = sheet_name

    def __call__(self, value: int) -> None:
        print(f"FooAction: {value} on sheet {self.sheet_name}")

FooAction("test")
"#,
    )
    .unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "test".to_owned(),
                Uri::from_file_path(root.path()).unwrap(),
            )]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("main.py");

    interaction
        .client
        .send_request::<RenameRequest>(json!({
            "textDocument": {"uri": Uri::from_file_path(&path).unwrap().to_string()},
            "position": {"line": 12, "character": 1},
            "newName": "BarAction"
        }))
        .expect_response(json!({
            "changes": {
                Uri::from_file_path(&path).unwrap().to_string(): [
                    {
                        "newText": "BarAction",
                        "range": {
                            "start": {"line": 5, "character": 6},
                            "end": {"line": 5, "character": 15}
                        }
                    },
                    {
                        "newText": "BarAction",
                        "range": {
                            "start": {"line": 12, "character": 0},
                            "end": {"line": 12, "character": 9}
                        }
                    }
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_prepare_rename() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    let path = root.path().join("basic/foo.py");

    interaction
        .client
        .send_request::<PrepareRenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&path).unwrap().to_string()
            },
            "position": {
                "line": 6,
                "character": 16
            }
        }))
        .expect_response(json!({
            "start": {"line": 6, "character": 16},
            "end": {"line": 6, "character": 19},
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename_third_party_symbols_in_venv_is_not_allowed() {
    let root = get_test_files_root();
    let root_path = root.path().join("rename_third_party");
    let scope_uri = Uri::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let user_code = root_path.join("user_code.py");

    interaction.client.did_open("user_code.py");

    // Verify that prepareRename returns null, indicating that renaming third party symbols is not allowed
    interaction
        .client
        .send_request::<PrepareRenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "external_result = external_function()"
                "character": 25  // Position on "external_function"
            }
        }))
        .expect_response(serde_json::Value::Null)
        .unwrap();

    // Verify that attempting to rename a third party symbol returns an error
    interaction
        .client
        .send_request::<RenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "external_result = external_function()"
                "character": 25  // Position on "external_function"
            },
            "newName": "new_external_function"
        }))
        .expect_response_error(json!({
            "code": -32600,
            "message": "Third-party symbols cannot be renamed",
            "data": null,
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename_editable_package_symbols_is_allowed() {
    // This test verifies that symbols from editable packages CAN be renamed.
    // Editable packages (installed via `pip install -e .`) appear in both site-packages
    // AND the search_path, and should be treated like first-party code for renaming purposes.
    let root = get_test_files_root();
    let root_path = root.path().join("rename_editable_package");
    let scope_uri = Uri::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let user_code = root_path.join("user_code.py");
    let editable_module = root_path.join("editable_module.py");

    interaction.client.did_open("user_code.py");
    interaction.client.did_open("editable_module.py");

    // Verify that prepareRename returns a range for editable package symbols
    interaction
        .client
        .send_request::<PrepareRenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "editable_result = editable_function()"
                "character": 25  // Position on "editable_function"
            }
        }))
        .expect_response(json!({
            "start": {"line": 14, "character": 22},
            "end": {"line": 14, "character": 39},
        }))
        .unwrap();

    // Verify that renaming an editable package symbol succeeds
    interaction
        .client
        .send_request::<RenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "editable_result = editable_function()"
                "character": 25  // Position on "editable_function"
            },
            "newName": "new_editable_function"
        }))
        .expect_response(json!({
            "changes": {
                Uri::from_file_path(&user_code).unwrap().to_string(): [
                    {
                        "newText": "new_editable_function",
                        "range": {"start": {"line": 5, "character": 28}, "end": {"line": 5, "character": 45}}
                    },
                    {
                        "newText": "new_editable_function",
                        "range": {"start": {"line": 14, "character": 22}, "end": {"line": 14, "character": 39}}
                    },
                ],
                Uri::from_file_path(&editable_module).unwrap().to_string(): [
                    {
                        "newText": "new_editable_function",
                        "range": {"start": {"line": 6, "character": 4}, "end": {"line": 6, "character": 21}}
                    },
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename_kwarg_in_unopened_file() {
    let root = get_test_files_root();
    let root_path = root.path().join("rename_kwargs_across_files");
    let scope_uri = Uri::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: LspArgs {
            indexing_mode: IndexingMode::LazyBlocking,
            ..LspInteractionArgs::default().args
        },
        ..Default::default()
    });
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let defs = root_path.join("defs.py");
    let uses = root_path.join("uses.py");

    interaction.client.did_open("defs.py");

    interaction
        .client
        .send_request::<RenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&defs).unwrap().to_string()
            },
            "position": {
                "line": 6,
                "character": 16
            },
            "newName": "note"
        }))
        .expect_response(json!({
            "changes": {
                Uri::from_file_path(&defs).unwrap().to_string(): [
                    {
                        "newText":"note",
                        "range":{"start":{"line":6,"character":16},"end":{"line":6,"character":23}}
                    },
                    {
                        "newText":"note",
                        "range":{"start":{"line":7,"character":14},"end":{"line":7,"character":21}}
                    },
                ],
                Uri::from_file_path(&uses).unwrap().to_string(): [
                    {
                        "newText":"note",
                        "range":{"start":{"line":13,"character":31},"end":{"line":13,"character":38}}
                    },
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Find-references reports `Person(...)` in usage.py as a reference to `Person.__init__`, but
/// that range spells the class name. RenameRequest must not rewrite it into the new method name.
#[test]
fn test_rename_dunder_init_skips_constructor_call_sites_across_files() {
    let root = get_test_files_root();
    let root_path = root.path().join("constructor_references");
    let scope_uri = Uri::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let person = root_path.join("person.py");

    interaction.client.did_open("person.py");
    interaction.client.did_open("usage.py");

    interaction
        .client
        .send_request::<RenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&person).unwrap().to_string()
            },
            "position": {
                "line": 7,
                "character": 10
            },
            "newName": "__init2__"
        }))
        .expect_response(json!({
            "changes": {
                Uri::from_file_path(&person).unwrap().to_string(): [
                    {
                        "newText":"__init2__",
                        "range":{"start":{"line":7,"character":8},"end":{"line":7,"character":16}}
                    },
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Uri::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.client.did_open("bar.py");
    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");

    interaction
        .client
        .send_request::<RenameRequest>(json!({
            "textDocument": {
                "uri": Uri::from_file_path(&bar).unwrap().to_string()
            },
            "position": {
                "line": 10,
                "character": 1
            },
            "newName": "Baz"
        }))
        .expect_response(json!({
            "changes": {
                Uri::from_file_path(&foo).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":6,"character":16},"end":{"line":6,"character":19}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":8,"character":0},"end":{"line":8,"character":3}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":9,"character":4},"end":{"line":9,"character":7}}
                    },
                ],
                Uri::from_file_path(&various_imports).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":16},"end":{"line":5,"character":19}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":26},"end":{"line":5,"character":29}}
                    },
                ],
                Uri::from_file_path(&with_synthetic_bindings).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":16},"end":{"character":19,"line":5}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":10,"character":4},"end":{"character":7,"line":10}}
                    },
                ],
                Uri::from_file_path(&bar).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":6,"character":6},"end":{"character":9,"line":6}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":10,"character":0},"end":{"character":3,"line":10}}
                    },
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename_cross_file_attribute_assignment_and_deletion() {
    let root = TempDir::new().unwrap();
    let a_path = root.path().join("a.py");
    let b_path = root.path().join("b.py");

    std::fs::write(
        &a_path,
        r#"class R:
    history: list[int]
    def __init__(self):
        self.history = []
"#,
    )
    .unwrap();

    std::fs::write(
        &b_path,
        r#"from a import R

def f(r: R, h: list[int]):
    r.history = h
    del r.history
    print(r.history)
"#,
    )
    .unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "test".to_owned(),
                Uri::from_file_path(root.path()).unwrap(),
            )]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("a.py");
    interaction.client.did_open("b.py");

    interaction
        .client
        .send_request::<RenameRequest>(json!({
            "textDocument": {"uri": Uri::from_file_path(&a_path).unwrap().to_string()},
            "position": {"line": 1, "character": 5},
            "newName": "zz"
        }))
        .expect_response(json!({
            "changes": {
                Uri::from_file_path(&a_path).unwrap().to_string(): [
                    {
                        "newText": "zz",
                        "range": {
                            "start": {"line": 1, "character": 4},
                            "end": {"line": 1, "character": 11}
                        }
                    },
                    {
                        "newText": "zz",
                        "range": {
                            "start": {"line": 3, "character": 13},
                            "end": {"line": 3, "character": 20}
                        }
                    }
                ],
                Uri::from_file_path(&b_path).unwrap().to_string(): [
                    {
                        "newText": "zz",
                        "range": {
                            "start": {"line": 3, "character": 6},
                            "end": {"line": 3, "character": 13}
                        }
                    },
                    {
                        "newText": "zz",
                        "range": {
                            "start": {"line": 4, "character": 10},
                            "end": {"line": 4, "character": 17}
                        }
                    },
                    {
                        "newText": "zz",
                        "range": {
                            "start": {"line": 5, "character": 12},
                            "end": {"line": 5, "character": 19}
                        }
                    }
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Rename must use the synchronized Python buffer and leave arbitrary text files alone.
#[test]
fn test_rename_comments_characterization() {
    for (rename, confirmation, filename) in [
        (json!({}), true, "main.py"),
        (json!({"commentsAndStrings": false}), true, "main.py"),
        (json!({"commentsAndStrings": true}), true, "main.py"),
        (json!({"commentsAndStrings": true}), false, "main.py"),
        (json!({"commentsAndStrings": true}), true, "main.cinc"),
        (json!({"commentsAndStrings": true}), true, "excluded.py"),
        (json!({"commentsAndStrings": true}), true, "generated.py"),
    ] {
        let root = TempDir::new().unwrap();
        let main = root.path().join(filename);
        std::fs::write(
            root.path().join("pyrefly.toml"),
            "extra-file-extensions = [\"cinc\"]\nproject-excludes = [\"excluded.py\"]\n",
        )
        .unwrap();
        std::fs::write(root.path().join("unrelated.cinc"), "foo = 1\nprint(foo)\n").unwrap();
        let notes = root.path().join("notes.txt");
        std::fs::write(&main, "foo = 1\nprint(foo)\n").unwrap();
        std::fs::write(&notes, "foo\n").unwrap();
        #[cfg(unix)]
        symlink(&notes, root.path().join("linked_notes.txt")).unwrap();
        let uri = Uri::from_file_path(&main).unwrap();
        let mut interaction = LspInteraction::new();
        interaction.set_root(root.path().to_path_buf());
        interaction
            .initialize(InitializeSettings {
                capabilities: confirmation.then(|| {
                    json!({
                        "workspace": {"workspaceEdit": {
                            "documentChanges": true,
                            "changeAnnotationSupport": {"groupsOnLabel": true}
                        }},
                        "textDocument": {"rename": {"honorsChangeAnnotations": true}}
                    })
                }),
                configuration: Some(Some(json!([{"pyrefly": {"rename": rename}}]))),
                ..Default::default()
            })
            .unwrap();
        interaction
            .client
            .did_open_uri(&uri, "python", "foo = 1\nprint(foo)\n");
        interaction.client.did_open("unrelated.cinc");
        interaction.client.did_change(
            filename,
            if filename == "generated.py" {
                "# @generated\nfoo = 1\nprint(foo)\n# foo\ntext = 'foo'\n"
            } else {
                "prefix = 0\nfoo = 1\nprint(foo)\n# foo\ntext = 'foo'\n"
            },
        );
        interaction.client.did_open_uri(
            &Uri::from_file_path(&notes).unwrap(),
            "plaintext",
            "prefix\nfoo\n",
        );
        let semantic = json!([
            {"newText": "bar", "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 3}}},
            {"newText": "bar", "range": {"start": {"line": 2, "character": 6}, "end": {"line": 2, "character": 9}}}
        ]);
        let expected = if rename["commentsAndStrings"] == true
            && confirmation
            && !["excluded.py", "generated.py"].contains(&filename)
        {
            let mut edits = semantic.as_array().unwrap().clone();
            edits.extend([
                json!({"annotationId": "pyrefly.rename.commentsAndStrings", "newText": "bar", "range": {"start": {"line": 3, "character": 2}, "end": {"line": 3, "character": 5}}}),
                json!({"annotationId": "pyrefly.rename.commentsAndStrings", "newText": "bar", "range": {"start": {"line": 4, "character": 8}, "end": {"line": 4, "character": 11}}}),
            ]);
            json!({
                "documentChanges": [{"textDocument": {"uri": uri, "version": 2}, "edits": edits}],
                "changeAnnotations": {"pyrefly.rename.commentsAndStrings": {
                    "label": "Review comment and string matches", "needsConfirmation": true,
                    "description": "These text matches may refer to a different symbol."
                }}
            })
        } else {
            json!({"changes": {uri.to_string(): semantic}})
        };
        let request = interaction.client.send_request::<RenameRequest>(json!({
            "textDocument": {"uri": uri},
            "position": {"line": 1, "character": 0},
            "newName": "bar"
        }));
        let id = request.id().clone();
        // Compare wire JSON because an untagged TextEdit decoder can discard annotation IDs.
        interaction
            .client
            .expect_message("annotated rename response", |message| {
                if let Message::Response(response) = message
                    && response.id == id
                {
                    assert!(response.error.is_none(), "{:?}", response.error);
                    assert_eq!(response.result, Some(expected.clone()));
                    Some(Ok(()))
                } else {
                    None
                }
            })
            .unwrap();
        interaction.shutdown().unwrap();
    }
}

#[test]
fn test_rename_comments_configuration_changes() {
    let root = TempDir::new().unwrap();
    let path = root.path().join("main.py");
    std::fs::write(&path, "foo = 1\n# foo\n").unwrap();
    let uri = Uri::from_file_path(&path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        capabilities: Some(json!({
            "workspace": {"workspaceEdit": {"documentChanges": true, "changeAnnotationSupport": {}}},
            "textDocument": {"rename": {"honorsChangeAnnotations": true}}
        })),
        configuration: Some(None),
        ..Default::default()
    }).unwrap();
    interaction.client.did_open("main.py");
    for (config, enabled) in [
        (
            json!({"pyrefly": {"rename": {"commentsAndStrings": true}}}),
            true,
        ),
        (
            json!({"pyrefly": {"rename": {"commentsAndStrings": false}}}),
            false,
        ),
        (
            json!({"pyrefly": {"rename": {"commentsAndStrings": true}}}),
            true,
        ),
        (json!({}), false),
        (
            json!({"pyrefly": {"rename": {"commentsAndStrings": true}}}),
            true,
        ),
        (json!({"pyrefly": {"rename": "invalid"}}), false),
    ] {
        interaction.client.did_change_configuration();
        interaction
            .client
            .expect_configuration_request(None)
            .unwrap()
            .send_configuration_response(json!([config]));
        let request = interaction.client.send_request::<RenameRequest>(json!({
            "textDocument": {"uri": uri}, "position": {"line": 0, "character": 0}, "newName": "bar"
        }));
        let id = request.id().clone();
        interaction.client.expect_message("rename after configuration update", |message| {
            if let Message::Response(response) = message && response.id == id {
                assert!(response.error.is_none(), "{:?}", response.error);
                let result = response.result.unwrap();
                if enabled {
                    assert!(result.get("changes").is_none());
                    assert_eq!(result["documentChanges"][0]["textDocument"], json!({"uri": uri, "version": 1}));
                    assert_eq!(result["documentChanges"][0]["edits"].as_array().unwrap().len(), 2);
                    assert_eq!(result["changeAnnotations"]["pyrefly.rename.commentsAndStrings"]["needsConfirmation"], true);
                } else {
                    assert!(result.get("documentChanges").is_none());
                    assert!(result.get("changeAnnotations").is_none());
                    assert_eq!(result["changes"][uri.to_string()].as_array().unwrap().len(), 1);
                }
                Some(Ok(()))
            } else { None }
        }).unwrap();
    }
    interaction.shutdown().unwrap();
}

/// Compare complete LSP requests in the same synthetic workspace, including serialization.
#[test]
#[ignore = "Manual benchmark: run with --profile bench --ignored --nocapture"]
fn benchmark_rename_comments() {
    let root = TempDir::new().unwrap();
    for i in 0..64 {
        std::fs::write(
            root.path().join(format!("unrelated_{i}.py")),
            "# unrelated\n".repeat(1024),
        )
        .unwrap();
    }
    let source = format!(
        "foo = 1\n{}",
        (0..200)
            .map(|i| format!(
                "value_{i} = foo\n# foo {}\nlabel_{i} = 'foo'\n",
                "padding ".repeat(20)
            ))
            .collect::<String>()
    );
    let path = root.path().join("main.py");
    std::fs::write(&path, &source).unwrap();
    let uri = Uri::from_file_path(&path).unwrap();
    eprintln!(
        "rename workspace: 65 Python files, initiating document {} bytes, 201 semantic references, 400 heuristic matches; budgets: 1 MiB / 1000 matches",
        source.len()
    );
    let mut interaction = LspInteraction::new();
    interaction.client.set_message_logging(false);
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        capabilities: Some(json!({
            "workspace": {"workspaceEdit": {"documentChanges": true, "changeAnnotationSupport": {}}},
            "textDocument": {"rename": {"honorsChangeAnnotations": true}}
        })),
        configuration: Some(None),
        ..Default::default()
    }).unwrap();
    interaction.client.did_open("main.py");
    for enabled in [false, true, false] {
        interaction.client.did_change_configuration();
        interaction
            .client
            .expect_configuration_request(None)
            .unwrap()
            .send_configuration_response(
                json!([{"pyrefly": {"rename": {"commentsAndStrings": enabled}}}]),
            );
        let mut elapsed = Vec::new();
        for sample in 0..23 {
            let start = Instant::now();
            interaction.client.send_request::<RenameRequest>(json!({
                "textDocument": {"uri": uri}, "position": {"line": 0, "character": 0}, "newName": "bar"
            })).expect_response_with(|response| {
                let response = serde_json::to_value(response.unwrap()).unwrap();
                if enabled {
                    assert_eq!(response["documentChanges"][0]["edits"].as_array().unwrap().len(), 601);
                } else {
                    assert_eq!(response["changes"][uri.to_string()].as_array().unwrap().len(), 201);
                }
                true
            }).unwrap();
            if sample >= 3 {
                elapsed.push(start.elapsed());
            }
        }
        elapsed.sort_unstable();
        let median = (elapsed[9] + elapsed[10]) / 2;
        eprintln!(
            "commentsAndStrings={enabled}: median={:.3} ms, p95={:.3} ms (20 requests)",
            median.as_secs_f64() * 1000.0,
            elapsed[18].as_secs_f64() * 1000.0
        );
    }
    interaction.shutdown().unwrap();
}
