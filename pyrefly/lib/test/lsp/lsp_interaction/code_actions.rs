use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_types::Url;
use serde_json::Value;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::object_model::ValidationResult;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_move_class_code_action() {
    let root = get_test_files_root();
    let workspace_root = root.path().join("move_class");
    let mut interaction = LspInteraction::new();
    interaction.set_root(workspace_root.clone());

    let workspace_uri = Url::from_file_path(&workspace_root).unwrap();
    let settings = InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), workspace_uri.clone())]),
        capabilities: Some(serde_json::json!({
            "workspace": {
                "workspaceEdit": {
                    "documentChanges": true
                }
            }
        })),
        ..Default::default()
    };
    interaction.initialize(settings);
    interaction.server.did_open("pkg/foo.py");

    let file_uri = Url::from_file_path(workspace_root.join("pkg/foo.py")).unwrap();
    let request_id = RequestId::from(2);
    interaction.server.send_message(Message::Request(Request {
        id: request_id.clone(),
        method: "textDocument/codeAction".to_owned(),
        params: serde_json::json!({
            "textDocument": {"uri": file_uri.to_string()},
            "range": {
                "start": {"line": 0, "character": 6},
                "end": {"line": 0, "character": 12}
            },
            "context": {"diagnostics": []}
        }),
    }));

    interaction.client.expect_message_helper(
        |msg| match msg {
            Message::Response(response) if response.id == request_id => {
                let result = response
                    .result
                    .clone()
                    .unwrap_or_else(|| Value::Array(vec![]));
                let actions: Vec<Value> = serde_json::from_value(result).unwrap();
                let move_action = actions
                    .iter()
                    .find(|action| action.get("kind").and_then(|k| k.as_str()) == Some("refactor"))
                    .expect("missing move-class refactor");
                assert!(
                    move_action
                        .get("title")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .contains("Sample")
                );
                let workspace_edit = move_action.get("edit").expect("missing workspace edit");
                let document_changes = workspace_edit
                    .get("documentChanges")
                    .and_then(|dc| dc.as_array())
                    .expect("missing documentChanges");
                assert_eq!(document_changes.len(), 3);
                let create_entry = &document_changes[0];
                assert_eq!(
                    create_entry.get("kind").and_then(|k| k.as_str()),
                    Some("create")
                );
                assert!(
                    create_entry
                        .get("uri")
                        .and_then(|u| u.as_str())
                        .expect("create uri")
                        .ends_with("sample.py")
                );
                ValidationResult::Pass
            }
            _ => ValidationResult::Skip,
        },
        "waiting for code action response",
    );

    interaction.shutdown();
}
