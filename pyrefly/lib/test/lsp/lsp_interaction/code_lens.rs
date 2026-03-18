/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeLens;
use lsp_types::CodeLensOptions;
use lsp_types::Url;
use pyrefly::commands::lsp::IndexingMode;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_initialize_advertises_code_lens_with_indexing() {
    let interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);

    interaction
        .client
        .send_initialize(
            interaction
                .client
                .get_initialize_params(&InitializeSettings::default()),
        )
        .expect_response_with(|response| {
            response.capabilities.code_lens_provider
                == Some(CodeLensOptions {
                    resolve_provider: Some(false),
                })
        })
        .unwrap();
    interaction.client.send_initialized();
    interaction.shutdown().unwrap();
}

#[test]
fn test_code_lens_shows_reference_counts() {
    let root = get_test_files_root();
    let root_path = root.path().join("code_lens_references");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path);
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("symbols.py");
    interaction.client.did_open("usage.py");

    interaction
        .client
        .code_lens("symbols.py")
        .expect_response_with(|response| {
            let Some(lenses) = response else {
                return false;
            };
            has_reference_lens(&lenses, 0, "3 references", 3)
                && has_reference_lens(&lenses, 1, "2 references", 2)
                && has_reference_lens(&lenses, 4, "2 references", 2)
                && lenses.len() == 3
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

fn has_reference_lens(
    lenses: &[CodeLens],
    line: u32,
    expected_title: &str,
    expected_locations: usize,
) -> bool {
    lenses.iter().any(|lens| {
        let Some(command) = &lens.command else {
            return false;
        };
        if lens.range.start.line != line
            || command.title != expected_title
            || command.command != "editor.action.showReferences"
        {
            return false;
        }
        let Some(arguments) = &command.arguments else {
            return false;
        };
        arguments
            .get(2)
            .and_then(|value| value.as_array())
            .is_some_and(|locations| locations.len() == expected_locations)
    })
}
