/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;

use lsp_types::SemanticTokenType;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::symbol_kind::SymbolKind;
use ruff_source_file::PositionEncoding as RuffPositionEncoding;
use ruff_text_size::TextRange;
use scip::symbol::format_symbol;
use scip::types::Descriptor;
use scip::types::Document;
use scip::types::Index;
use scip::types::Metadata;
use scip::types::MultiLineRange;
use scip::types::Occurrence;
use scip::types::PositionEncoding;
use scip::types::SingleLineRange;
use scip::types::Symbol;
use scip::types::SymbolInformation;
use scip::types::SymbolRole;
use scip::types::SyntaxKind;
use scip::types::TextEncoding;
use scip::types::ToolInfo;
use scip::types::descriptor;
use scip::types::symbol_information;

use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::semantic_tokens::SemanticTokenBuilder;
use crate::state::state::Transaction;

/// Build a SCIP index for the checked files rooted at `project_root`.
pub fn index(
    transaction: &Transaction,
    handles: &[Handle],
    project_root: &Path,
    version: &str,
) -> anyhow::Result<Index> {
    let mut documents = handles
        .iter()
        .map(|handle| document(transaction, handle, project_root))
        .collect::<anyhow::Result<Vec<_>>>()?;
    documents.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let project_root = lsp_types::Url::from_directory_path(project_root)
        .map_err(|()| anyhow::anyhow!("cannot convert project root to a file URL"))?;
    Ok(Index {
        metadata: Some(Metadata {
            tool_info: Some(ToolInfo {
                name: "pyrefly".to_owned(),
                version: version.to_owned(),
                ..Default::default()
            })
            .into(),
            project_root: project_root.to_string(),
            text_document_encoding: TextEncoding::UTF8.into(),
            ..Default::default()
        })
        .into(),
        documents,
        ..Default::default()
    })
}

fn document(
    transaction: &Transaction,
    handle: &Handle,
    project_root: &Path,
) -> anyhow::Result<Document> {
    let module = transaction
        .get_module_info(handle)
        .ok_or_else(|| anyhow::anyhow!("no module information for `{}`", handle.path()))?;
    let source_path = handle.path().as_path();
    let relative_path = if source_path.is_absolute() {
        source_path.strip_prefix(project_root).map_err(|_| {
            anyhow::anyhow!(
                "source file `{}` is outside project root `{}`",
                source_path.display(),
                project_root.display()
            )
        })?
    } else {
        source_path
    };
    let relative_path = relative_path.to_string_lossy().replace('\\', "/");
    let ast = transaction
        .get_ast(handle)
        .ok_or_else(|| anyhow::anyhow!("no syntax tree for `{}`", source_path.display()))?;
    let mut builder = SemanticTokenBuilder::new(None, Vec::new());
    builder.process_ast(&ast, &|_| None, &|_| None);

    let preference = FindPreference {
        import_behavior: ImportBehavior::JumpThroughEverything,
        prefer_pyi: false,
        resolve_call_dunders: false,
        disable_style_fallback: false,
    };
    let mut symbols = Vec::new();
    let mut seen_definitions = HashSet::new();
    let mut seen_occurrences = HashSet::new();
    let mut occurrences = Vec::new();
    for token in builder.all_tokens_sorted() {
        if !seen_occurrences.insert(token.range) {
            continue;
        }
        let Ok(definitions) = transaction.find_definition(handle, token.range.start(), preference)
        else {
            continue;
        };
        let definition = definitions
            .into_iter()
            .next()
            .expect("definitions is nonempty");
        let symbol = symbol(
            &definition.module,
            definition.definition_range,
            &definition.metadata,
        );
        let is_definition =
            definition.module.path() == module.path() && definition.definition_range == token.range;
        if is_definition && seen_definitions.insert(symbol.clone()) {
            symbols.push(SymbolInformation {
                symbol: symbol.clone(),
                kind: symbol_information_kind(&definition.metadata).into(),
                display_name: definition
                    .module
                    .code_at(definition.definition_range)
                    .to_owned(),
                ..Default::default()
            });
        }
        occurrences.push(Occurrence {
            symbol,
            symbol_roles: if is_definition {
                SymbolRole::Definition as i32
            } else {
                SymbolRole::ReadAccess as i32
            },
            syntax_kind: syntax_kind(&token.token_type, is_definition).into(),
            typed_range: Some(typed_range(&module, token.range)),
            ..Default::default()
        });
    }

    Ok(Document {
        language: "python".to_owned(),
        relative_path,
        occurrences,
        symbols,
        position_encoding: PositionEncoding::UTF8CodeUnitOffsetFromLineStart.into(),
        ..Default::default()
    })
}

fn symbol(module: &Module, range: TextRange, metadata: &DefinitionMetadata) -> String {
    let name = module.code_at(range).to_owned();
    let symbol_kind = metadata.symbol_kind();
    let suffix = match symbol_kind {
        Some(SymbolKind::Module) => descriptor::Suffix::Namespace,
        Some(SymbolKind::Class | SymbolKind::TypeAlias) => descriptor::Suffix::Type,
        Some(SymbolKind::Function | SymbolKind::Method) => descriptor::Suffix::Method,
        Some(SymbolKind::TypeParameter) => descriptor::Suffix::TypeParameter,
        Some(SymbolKind::Parameter) => descriptor::Suffix::Parameter,
        _ => descriptor::Suffix::Term,
    };
    let mut descriptors = vec![Descriptor {
        name: module.name().as_str().to_owned(),
        suffix: descriptor::Suffix::Namespace.into(),
        ..Default::default()
    }];
    if symbol_kind != Some(SymbolKind::Module) || !name.is_empty() {
        descriptors.extend([
            Descriptor {
                name: if name.is_empty() {
                    module.name().as_str().to_owned()
                } else {
                    name
                },
                suffix: suffix.into(),
                ..Default::default()
            },
            // Python permits repeated definitions with the same qualified name. The
            // source offset keeps their identities distinct within one index.
            Descriptor {
                name: range.start().to_u32().to_string(),
                suffix: descriptor::Suffix::Meta.into(),
                ..Default::default()
            },
        ]);
    }
    format_symbol(Symbol {
        scheme: "pyrefly".to_owned(),
        descriptors,
        ..Default::default()
    })
}

fn symbol_information_kind(metadata: &DefinitionMetadata) -> symbol_information::Kind {
    match metadata.symbol_kind() {
        Some(SymbolKind::Module) => symbol_information::Kind::Module,
        Some(SymbolKind::Attribute) => symbol_information::Kind::Attribute,
        Some(SymbolKind::Variable) => symbol_information::Kind::Variable,
        Some(SymbolKind::Constant) => symbol_information::Kind::Constant,
        Some(SymbolKind::Parameter) => symbol_information::Kind::Parameter,
        Some(SymbolKind::TypeParameter) => symbol_information::Kind::TypeParameter,
        Some(SymbolKind::TypeAlias) => symbol_information::Kind::TypeAlias,
        Some(SymbolKind::Function) => symbol_information::Kind::Function,
        Some(SymbolKind::Method) => symbol_information::Kind::Method,
        Some(SymbolKind::Class) => symbol_information::Kind::Class,
        None => symbol_information::Kind::UnspecifiedKind,
    }
}

fn syntax_kind(token_type: &SemanticTokenType, is_definition: bool) -> SyntaxKind {
    if token_type == &SemanticTokenType::FUNCTION || token_type == &SemanticTokenType::METHOD {
        if is_definition {
            SyntaxKind::IdentifierFunctionDefinition
        } else {
            SyntaxKind::IdentifierFunction
        }
    } else if token_type == &SemanticTokenType::CLASS
        || token_type == &SemanticTokenType::TYPE
        || token_type == &SemanticTokenType::INTERFACE
        || token_type == &SemanticTokenType::STRUCT
        || token_type == &SemanticTokenType::ENUM
        || token_type == &SemanticTokenType::TYPE_PARAMETER
    {
        SyntaxKind::IdentifierType
    } else if token_type == &SemanticTokenType::PARAMETER {
        SyntaxKind::IdentifierParameter
    } else if token_type == &SemanticTokenType::PROPERTY {
        SyntaxKind::IdentifierAttribute
    } else if token_type == &SemanticTokenType::NAMESPACE {
        SyntaxKind::IdentifierNamespace
    } else {
        SyntaxKind::Identifier
    }
}

fn typed_range(module: &Module, range: TextRange) -> scip::types::occurrence::Typed_range {
    let source = module.contents();
    let lines = module.lined_buffer().line_index();
    let start = lines.source_location(range.start(), source, RuffPositionEncoding::Utf8);
    let end = lines.source_location(range.end(), source, RuffPositionEncoding::Utf8);
    let start_line = start.line.to_zero_indexed() as i32;
    let start_character = start.character_offset.to_zero_indexed() as i32;
    let end_line = end.line.to_zero_indexed() as i32;
    let end_character = end.character_offset.to_zero_indexed() as i32;
    if start_line == end_line {
        scip::types::occurrence::Typed_range::SingleLineRange(SingleLineRange {
            line: start_line,
            start_character,
            end_character,
            ..Default::default()
        })
    } else {
        scip::types::occurrence::Typed_range::MultiLineRange(MultiLineRange {
            start_line,
            start_character,
            end_line,
            end_character,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::require::Require;
    use crate::test::util::mk_multi_file_state_assert_no_errors;

    #[test]
    fn indexes_cross_module_references() {
        let files = [
            ("foo", "def answer() -> int:\n    return 42\n"),
            ("bar", "from foo import answer\nresult = answer()\n"),
        ];
        let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Everything);
        let transaction = state.transaction();
        let handles = handles.into_values().collect::<Vec<_>>();
        let index = index(
            &transaction,
            &handles,
            &std::env::current_dir().unwrap(),
            "test",
        )
        .unwrap();
        let foo = index
            .documents
            .iter()
            .find(|document| document.relative_path == "foo.py")
            .unwrap();
        let bar = index
            .documents
            .iter()
            .find(|document| document.relative_path == "bar.py")
            .unwrap();
        let answer = foo
            .symbols
            .iter()
            .find(|symbol| symbol.display_name == "answer")
            .unwrap();

        assert!(foo.occurrences.iter().any(|occurrence| {
            occurrence.symbol == answer.symbol
                && occurrence.symbol_roles & SymbolRole::Definition as i32 != 0
        }));
        assert!(bar.occurrences.iter().any(|occurrence| {
            occurrence.symbol == answer.symbol
                && occurrence.symbol_roles & SymbolRole::ReadAccess as i32 != 0
        }));
        for document in &index.documents {
            for (index, occurrence) in document.occurrences.iter().enumerate() {
                assert!(
                    !document.occurrences[..index]
                        .iter()
                        .any(|other| other.typed_range == occurrence.typed_range)
                );
            }
        }
        for occurrence in index
            .documents
            .iter()
            .flat_map(|document| &document.occurrences)
        {
            scip::symbol::parse_symbol(&occurrence.symbol).unwrap();
        }
    }

    #[test]
    fn uses_utf8_byte_offsets() {
        let files = [("foo", "rocket = '🚀'; answer = rocket\n")];
        let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Everything);
        let transaction = state.transaction();
        let handles = handles.into_values().collect::<Vec<_>>();
        let index = index(
            &transaction,
            &handles,
            &std::env::current_dir().unwrap(),
            "test",
        )
        .unwrap();
        let answer = index.documents[0].occurrences.iter().find(|occurrence| {
            matches!(
                occurrence.typed_range,
                Some(scip::types::occurrence::Typed_range::SingleLineRange(
                    SingleLineRange {
                        start_character: 17,
                        ..
                    }
                ))
            )
        });
        assert!(answer.is_some());
    }
}
