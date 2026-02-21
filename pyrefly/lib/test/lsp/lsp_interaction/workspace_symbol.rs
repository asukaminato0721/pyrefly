/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::WorkspaceSymbolResponse;
use lsp_types::request::WorkspaceSymbolRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_workspace_symbol() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("autoimport_provider.py");

    interaction
        .client
        .send_request::<WorkspaceSymbolRequest>(
            json!({
                "query": "this_is_a_very_long_function_name_so_we_can"
            }),
        )
        .expect_response(json!([
            {
                "kind": 12,
                "location": {
                    "range": {
                        "start": {"line": 6, "character": 4},
                        "end": {"line": 6, "character": 99}
                    },
                    "uri": Url::from_file_path(root_path.join("autoimport_provider.py")).unwrap().to_string()
                },
                "name": "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search"
            }
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_workspace_symbol_ranks_init_reexports_lower() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("pkg/__init__.py");

    interaction
        .client
        .send_request::<WorkspaceSymbolRequest>(
            json!({
                "query": "reexported_symbol_for_workspace_search"
            }),
        )
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(items)) = result else {
                return false;
            };
            let mut init_indices = Vec::new();
            let mut defs_indices = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let path = item.location.uri.path();
                if path.ends_with("/pkg/__init__.py") {
                    init_indices.push(index);
                } else if path.ends_with("/pkg/defs.py") {
                    defs_indices.push(index);
                }
            }
            !init_indices.is_empty()
                && !defs_indices.is_empty()
                && init_indices
                    .iter()
                    .all(|init_index| defs_indices.iter().all(|defs_index| defs_index < init_index))
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
