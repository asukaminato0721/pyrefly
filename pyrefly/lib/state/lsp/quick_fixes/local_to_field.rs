/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::symbol_kind::SymbolKind;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::first_parameter_name;
use super::extract_shared::function_has_decorator;
use super::extract_shared::is_disallowed_scope_expr;
use super::extract_shared::reference_in_disallowed_scope;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::lsp::Transaction;

struct MethodContext<'a> {
    function_def: &'a StmtFunctionDef,
    receiver_name: String,
}

/// Converts a simple local variable assignment inside a method into a field assignment.
pub(crate) fn local_to_field_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let position = selection.start();
    let identifier = transaction.identifier_at(handle, position)?;
    if !matches!(identifier.context, IdentifierContext::Expr(_)) {
        return None;
    }
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let method = find_enclosing_method(ast.as_ref(), position)?;
    let definition = transaction
        .find_definition(handle, position, FindPreference::default())
        .into_iter()
        .find(|definition| {
            definition.module.path() == module_info.path()
                && method
                    .function_def
                    .range()
                    .contains_range(definition.definition_range)
                && matches!(
                    definition.metadata.symbol_kind(),
                    Some(SymbolKind::Variable | SymbolKind::Constant)
                )
        })?;
    if !is_supported_local_definition(&method.function_def.body, definition.definition_range) {
        return None;
    }
    let references =
        transaction.find_local_references(handle, definition.definition_range.start(), true);
    if references.is_empty() {
        return None;
    }
    if references
        .iter()
        .any(|range| !method.function_def.range().contains_range(*range))
    {
        return None;
    }
    if references
        .iter()
        .any(|range| range.start() < definition.definition_range.start())
    {
        return None;
    }
    if references.iter().any(|range| {
        *range != definition.definition_range && reference_in_disallowed_scope(ast.as_ref(), *range)
    }) {
        return None;
    }
    if references_in_nested_scope(
        method.function_def,
        &references,
        definition.definition_range,
    ) {
        return None;
    }
    if has_other_store(
        &method.function_def.body,
        identifier.identifier.id.as_str(),
        definition.definition_range,
    ) {
        return None;
    }
    let replacement = format!("{}.{}", method.receiver_name, identifier.identifier.id);
    let edits = references
        .into_iter()
        .map(|range| (module_info.dupe(), range, replacement.clone()))
        .collect();
    Some(vec![LocalRefactorCodeAction {
        title: "Convert local variable to field".to_owned(),
        edits,
        kind: CodeActionKind::REFACTOR_REWRITE,
    }])
}

fn find_enclosing_method(ast: &ModModule, position: TextSize) -> Option<MethodContext<'_>> {
    let covering_nodes = Ast::locate_node(ast, position);
    for (idx, node) in covering_nodes.iter().enumerate() {
        let AnyNodeRef::StmtFunctionDef(function_def) = node else {
            continue;
        };
        let Some(AnyNodeRef::StmtClassDef(_)) = covering_nodes.get(idx + 1) else {
            return None;
        };
        if function_has_decorator(function_def, "staticmethod") {
            return None;
        }
        return Some(MethodContext {
            function_def,
            receiver_name: first_parameter_name(&function_def.parameters)?,
        });
    }
    None
}

fn is_supported_local_definition(stmts: &[Stmt], definition_range: TextRange) -> bool {
    struct Finder {
        definition_range: TextRange,
        found: bool,
    }

    impl Visitor<'_> for Finder {
        fn visit_stmt(&mut self, stmt: &Stmt) {
            if self.found {
                return;
            }
            match stmt {
                Stmt::Assign(assign) => {
                    if assign.targets.len() == 1
                        && let Expr::Name(name) = &assign.targets[0]
                        && name.range() == self.definition_range
                    {
                        self.found = true;
                    }
                }
                Stmt::AnnAssign(assign) => {
                    if assign.value.is_some()
                        && let Expr::Name(name) = assign.target.as_ref()
                        && name.range() == self.definition_range
                    {
                        self.found = true;
                    }
                }
                Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
                _ => walk_stmt(self, stmt),
            }
        }
    }

    let mut finder = Finder {
        definition_range,
        found: false,
    };
    finder.visit_body(stmts);
    finder.found
}

fn has_other_store(stmts: &[Stmt], name: &str, definition_range: TextRange) -> bool {
    struct StoreVisitor<'a> {
        name: &'a str,
        definition_range: TextRange,
        found: bool,
    }

    impl Visitor<'_> for StoreVisitor<'_> {
        fn visit_stmt(&mut self, stmt: &Stmt) {
            if self.found {
                return;
            }
            match stmt {
                Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
                _ => walk_stmt(self, stmt),
            }
        }

        fn visit_expr(&mut self, expr: &Expr) {
            if self.found {
                return;
            }
            if is_disallowed_scope_expr(expr) {
                return;
            }
            if let Expr::Name(expr_name) = expr
                && expr_name.id.as_str() == self.name
                && matches!(expr_name.ctx, ExprContext::Store | ExprContext::Del)
                && expr_name.range() != self.definition_range
            {
                self.found = true;
                return;
            }
            walk_expr(self, expr);
        }
    }

    let mut visitor = StoreVisitor {
        name,
        definition_range,
        found: false,
    };
    visitor.visit_body(stmts);
    visitor.found
}

fn references_in_nested_scope(
    function_def: &StmtFunctionDef,
    references: &[TextRange],
    definition_range: TextRange,
) -> bool {
    struct NestedScopeCollector {
        ranges: Vec<TextRange>,
    }

    impl Visitor<'_> for NestedScopeCollector {
        fn visit_stmt(&mut self, stmt: &Stmt) {
            match stmt {
                Stmt::FunctionDef(function_def) => {
                    self.ranges.push(function_def.range());
                }
                Stmt::ClassDef(class_def) => {
                    self.ranges.push(class_def.range());
                }
                _ => walk_stmt(self, stmt),
            }
        }
    }

    let mut collector = NestedScopeCollector { ranges: Vec::new() };
    collector.visit_body(&function_def.body);
    references
        .iter()
        .filter(|range| **range != definition_range)
        .any(|range| {
            collector
                .ranges
                .iter()
                .any(|nested| nested.contains_range(*range))
        })
}
