/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use lsp_types::CodeActionOrCommand;
use lsp_types::Range;
use lsp_types::TextDocumentIdentifier;
use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use lsp_types::request::Request;
use pyrefly::commands::lsp::IndexingMode;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use tempfile::TempDir;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;

const ADDITIONAL_IMPORT_MATCH_CODE_ACTION_TITLE: &str = "Search for additional matching imports";

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AdditionalImportMatchesParams {
    text_document: TextDocumentIdentifier,
    range: Range,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AdditionalImportMatch {
    title: String,
    text_document: TextDocumentIdentifier,
    range: Range,
    new_text: String,
}

enum AdditionalImportMatchesRequest {}

impl Request for AdditionalImportMatchesRequest {
    type Params = AdditionalImportMatchesParams;
    type Result = Vec<AdditionalImportMatch>;
    const METHOD: &'static str = "pyrefly/textDocument/additionalImportMatches";
}

#[test]
fn test_additional_import_matches_are_deferred() {
    let root = TempDir::with_prefix("pyrefly_code_action").unwrap();
    fs::write(root.path().join("main.py"), "foo\n").unwrap();
    fs::write(root.path().join("a.py"), "foo = 1\n").unwrap();
    fs::write(root.path().join("b.py"), "foobar = 1\n").unwrap();

    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("main.py");
    interaction.client.did_open("a.py");
    interaction.client.did_open("b.py");

    let main_uri = Url::from_file_path(root.path().join("main.py")).unwrap();
    let range = json!({
        "start": {"line": 0, "character": 0},
        "end": {"line": 0, "character": 3}
    });

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": {
                "uri": main_uri.clone()
            },
            "range": range,
            "context": {
                "diagnostics": []
            }
        }))
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return false;
                };
                code_action.title == ADDITIONAL_IMPORT_MATCH_CODE_ACTION_TITLE
                    && code_action.command.as_ref().is_some_and(|command| {
                        command.command == "pyrefly.searchAdditionalImportMatches"
                    })
            }) && !actions.iter().any(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return false;
                };
                code_action.title == "Insert import: `from b import foobar`"
            })
        })
        .unwrap();

    interaction
        .client
        .send_request::<AdditionalImportMatchesRequest>(json!({
            "textDocument": {
                "uri": main_uri.clone()
            },
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 3}
            }
        }))
        .expect_response_with(|matches| {
            matches.iter().any(|item| {
                item.title == "Insert import: `from b import foobar`"
                    && item.text_document.uri == main_uri
                    && item.new_text == "from b import foobar\n"
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
