/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeActionKind;
use lsp_types::CodeActionOrCommand;
use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_local_to_field_code_action() {
    let root = get_test_files_root();
    let root_path = root.path().join("local_to_field");
    let file = "main.py";
    let file_path = root_path.join(file);
    let uri = Url::from_file_path(&file_path).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open(file);

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 3, "character": 14 },
                "end": { "line": 3, "character": 14 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionOrCommand>>| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return false;
                };
                if code_action.title != "Convert local variable to field" {
                    return false;
                }
                if code_action.kind != Some(CodeActionKind::REFACTOR_REWRITE) {
                    return false;
                }
                let Some(text_edits) = code_action
                    .edit
                    .as_ref()
                    .and_then(|edit| edit.changes.as_ref())
                    .and_then(|changes| changes.get(&uri))
                else {
                    return false;
                };
                text_edits.iter().any(|edit| {
                    edit.range.start.line == 2
                        && edit.range.start.character == 8
                        && edit.new_text == "self.local_var"
                }) && text_edits.iter().any(|edit| {
                    edit.range.start.line == 3
                        && edit.range.start.character == 14
                        && edit.new_text == "self.local_var"
                })
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
