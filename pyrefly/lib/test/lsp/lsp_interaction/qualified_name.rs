/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_qualified_name_command() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("provide_type"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();
    interaction.client.did_open("bar.py");

    interaction
        .client
        .qualified_name("bar.py", 18, 9)
        .expect_response(json!("bar.Bar.Baz.f"))
        .unwrap();

    interaction.shutdown().unwrap();
}
