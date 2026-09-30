/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::slice;

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::dedent_block_preserving_layout;
use pyrefly_util::visit::Visit;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::Vec1;

use super::extract_shared::MethodInfo;
use super::extract_shared::first_parameter_name;
use super::extract_shared::is_static_or_class_method;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::validate_non_empty_selection;
use crate::state::lsp::FindPreference;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

const HELPER_INDENT: &str = "    ";

/// Builds extract-function quick fix code actions for the supplied selection.
pub(crate) fn extract_function_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let module_source = module_info.contents();
    let ast = transaction.get_ast(handle)?;
    let selection_text = validate_non_empty_selection(selection, module_info.code_at(selection))?;
    let module_len = TextSize::try_from(module_info.contents().len()).unwrap_or(TextSize::new(0));
    let module_stmt_range =
        find_enclosing_module_statement_range(ast.as_ref(), selection, module_len);
    let control_flow = SelectionControlFlow::collect(ast.as_ref(), selection)?;
    let (load_refs, store_refs) = collect_identifier_refs(ast.as_ref(), selection);
    if load_refs.is_empty() && store_refs.is_empty() && control_flow.exits.is_empty() {
        return None;
    }
    let post_loads = collect_post_selection_loads(
        ast.as_ref(),
        module_stmt_range,
        selection,
        &control_flow.enclosing_loops,
    );
    let block_indent = detect_block_indent(selection_text);

    let function_helper_name = generate_name(module_source, "extracted_function");
    let mut params = Vec::new();
    let mut seen_params = HashSet::new();
    for ident in load_refs {
        if seen_params.contains(&ident.name) {
            continue;
        }
        if ident.synthetic_load {
            let defined_earlier_in_selection = store_refs
                .iter()
                .any(|store| store.name == ident.name && store.position < ident.position);
            if !defined_earlier_in_selection {
                seen_params.insert(ident.name.clone());
                params.push(ident.name.clone());
            }
            continue;
        }
        let defs = transaction
            .find_definition(handle, ident.position, FindPreference::default())
            .map(Vec1::into_vec)
            .unwrap_or_default();
        let Some(def) = defs.first() else {
            continue;
        };
        if def.module.path() != module_info.path() {
            continue;
        }
        if !module_stmt_range.contains_range(def.definition_range)
            || selection.contains_range(def.definition_range)
            || def.definition_range.start() >= selection.start()
        {
            continue;
        }
        seen_params.insert(ident.name.clone());
        params.push(ident.name.clone());
    }

    let mut returns = Vec::new();
    let mut seen_returns = HashSet::new();
    for ident in store_refs {
        if seen_returns.contains(&ident.name) || !post_loads.contains(&ident.name) {
            continue;
        }
        seen_returns.insert(ident.name.clone());
        returns.push(ident.name.clone());
    }

    // Escaping exits carry their kind, return value, and live outputs back to the
    // caller. The caller restores the outputs before performing the exit.
    let mut body = selection_text.to_owned();
    let mut helper_returns = returns.clone();
    let mut call_returns = returns.clone();
    let mut dispatch = String::new();
    if let Some((first_exit, _)) = control_flow.exits.first() {
        // Every output must exist at each exit. Parameters and unconditional
        // assignments before the first exit provide that guarantee.
        let mut bound: HashSet<String> = params.iter().cloned().collect();
        for (range, name) in &control_flow.assignments {
            if range.end() <= first_exit.start() {
                bound.insert(name.clone());
            }
        }
        if !returns.is_empty()
            && (control_flow.has_finally || returns.iter().any(|name| !bound.contains(name)))
        {
            // A finally block can change outputs after a return tuple is evaluated.
            return None;
        }
        let control_name = generate_name(module_source, "extracted_control");
        let value_name = generate_name(module_source, "extracted_value");
        helper_returns.splice(0..0, ["'fallthrough'".to_owned(), "None".to_owned()]);
        if control_flow.always_exits {
            helper_returns.clear();
        }
        call_returns.splice(0..0, [control_name.clone(), value_name.clone()]);
        // Replace from the end so that all ranges still refer to the original source.
        for (range, exit) in control_flow.exits.iter().rev() {
            let value = match exit {
                SelectionExit::Return(Some(value)) => {
                    format!("({})", module_info.code_at(*value))
                }
                _ => "None".to_owned(),
            };
            let mut values = vec![format!("'{}'", exit.keyword()), value];
            values.extend(returns.iter().cloned());
            body.replace_range(
                (range.start() - selection.start()).to_usize()
                    ..(range.end() - selection.start()).to_usize(),
                &format!("return {}", values.join(", ")),
            );
        }
        for keyword in ["return", "break", "continue"] {
            if control_flow
                .exits
                .iter()
                .any(|(_, exit)| exit.keyword() == keyword)
            {
                let statement = if keyword == "return" {
                    format!("return {value_name}")
                } else {
                    keyword.to_owned()
                };
                dispatch.push_str(&format!(
                    "{block_indent}if {control_name} == '{keyword}':\n{block_indent}{HELPER_INDENT}{statement}\n"
                ));
            }
        }
    }
    let mut dedented_body = dedent_block_preserving_layout(&body)?;
    if dedented_body.ends_with('\n') {
        dedented_body.pop();
        if dedented_body.ends_with('\r') {
            dedented_body.pop();
        }
    }
    let helper_text = build_helper_text(
        &function_helper_name,
        &params,
        &helper_returns,
        &dedented_body,
        "",
    );
    let call_expr = build_call_expr(&function_helper_name, None, &params);
    let replacement_line =
        build_call_replacement(&block_indent, &call_expr, &call_returns) + &dispatch;
    let helper_edit = (
        module_info.dupe(),
        TextRange::at(module_stmt_range.start(), TextSize::new(0)),
        helper_text,
    );
    let call_edit = (module_info.dupe(), selection, replacement_line);
    let mut actions = vec![LocalRefactorCodeAction {
        title: format!("Extract into helper `{function_helper_name}`"),
        edits: vec![helper_edit, call_edit],
        kind: CodeActionKind::RefactorExtract,
    }];
    if let Some(method_ctx) = find_enclosing_method(ast.as_ref(), selection, module_source) {
        let method_helper_name = generate_name(module_source, "extracted_method");
        let mut signature_params = Vec::new();
        signature_params.push(method_ctx.info.receiver_name.clone());
        let method_params = filter_params_excluding(&params, &method_ctx.info.receiver_name);
        signature_params.extend(method_params.iter().cloned());
        let method_helper_text = build_helper_text(
            &method_helper_name,
            &signature_params,
            &helper_returns,
            &dedented_body,
            &method_ctx.method_indent,
        );
        let method_call_expr = build_call_expr(
            &method_helper_name,
            Some(&method_ctx.info.receiver_name),
            &method_params,
        );
        let method_replacement =
            build_call_replacement(&block_indent, &method_call_expr, &call_returns) + &dispatch;
        let method_helper_edit = (
            module_info.dupe(),
            TextRange::at(method_ctx.insert_position, TextSize::new(0)),
            method_helper_text,
        );
        let method_call_edit = (module_info.dupe(), selection, method_replacement);
        actions.push(LocalRefactorCodeAction {
            title: format!(
                "Extract into method `{}` on `{}`",
                method_helper_name, method_ctx.info.class_name
            ),
            edits: vec![method_helper_edit, method_call_edit],
            kind: CodeActionKind::RefactorExtract,
        });
    }

    Some(actions)
}

#[derive(Clone, Debug)]
struct IdentifierRef {
    /// Identifier string.
    name: String,
    /// Byte offset where the identifier was observed.
    position: TextSize,
    /// True when this "load" came from reading the left-hand side of an augmented assignment.
    synthetic_load: bool,
}

fn collect_identifier_refs(
    ast: &ModModule,
    selection: TextRange,
) -> (Vec<IdentifierRef>, Vec<IdentifierRef>) {
    struct IdentifierCollector {
        selection: TextRange,
        loads: Vec<IdentifierRef>,
        stores: Vec<IdentifierRef>,
    }

    impl<'a> Visitor<'a> for IdentifierCollector {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if self.selection.contains_range(expr.range())
                && let Expr::Name(name) = expr
            {
                let ident = IdentifierRef {
                    name: name.id.to_string(),
                    position: name.range.start(),
                    synthetic_load: false,
                };
                match name.ctx {
                    ExprContext::Load => self.loads.push(ident),
                    ExprContext::Store => self.stores.push(ident),
                    ExprContext::Del | ExprContext::Invalid => {}
                }
            }
            walk_expr(self, expr);
        }

        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            if self.selection.contains_range(stmt.range())
                && let Stmt::AugAssign(aug) = stmt
                && let Expr::Name(name) = aug.target.as_ref()
            {
                self.loads.push(IdentifierRef {
                    name: name.id.to_string(),
                    position: name.range.start(),
                    synthetic_load: true,
                });
            }
            walk_stmt(self, stmt);
        }
    }

    let mut collector = IdentifierCollector {
        selection,
        loads: Vec::new(),
        stores: Vec::new(),
    };
    collector.visit_body(&ast.body);
    (collector.loads, collector.stores)
}

#[derive(Clone, Debug)]
/// Context information for extracting a method from a class.
///
/// Contains details about where and how to insert the extracted method,
/// as well as relevant naming and formatting information.
struct MethodContext {
    /// Core method information (class name, receiver name).
    info: MethodInfo,
    /// Byte offset in the source code where the extracted method should be inserted.
    insert_position: TextSize,
    /// Indentation string to use for the method definition line.
    method_indent: String,
}

enum SelectionExit {
    Return(Option<TextRange>),
    Break,
    Continue,
}

impl SelectionExit {
    fn keyword(&self) -> &'static str {
        match self {
            Self::Return(_) => "return",
            Self::Break => "break",
            Self::Continue => "continue",
        }
    }
}

#[derive(Default)]
struct SelectionControlFlow {
    exits: Vec<(TextRange, SelectionExit)>,
    enclosing_loops: Vec<TextRange>,
    assignments: Vec<(TextRange, String)>,
    has_finally: bool,
    always_exits: bool,
    disallowed: bool,
}

impl SelectionControlFlow {
    fn collect(ast: &ModModule, selection: TextRange) -> Option<Self> {
        let mut result = Self::default();
        for stmt in &ast.body {
            result.visit_stmt(stmt, selection, 0, false);
        }
        result.exits.sort_by_key(|(range, _)| range.start());
        if !result.exits.is_empty() {
            fn has_suspension(expr: &Expr) -> bool {
                if matches!(expr, Expr::Yield(_) | Expr::YieldFrom(_) | Expr::Await(_)) {
                    return true;
                }
                let mut found = false;
                expr.recurse(&mut |child| found |= has_suspension(child));
                found
            }
            ast.visit(&mut |expr: &Expr| {
                if selection.contains_range(expr.range()) {
                    result.disallowed |= has_suspension(expr);
                }
            });
        }
        if result.disallowed {
            None
        } else {
            Some(result)
        }
    }

    /// Only jumps targeting a loop outside the selection need a caller-side jump.
    fn visit_stmt(
        &mut self,
        stmt: &Stmt,
        selection: TextRange,
        loop_depth: usize,
        inside_selection: bool,
    ) {
        if self.disallowed || stmt.range().intersect(selection).is_none() {
            return;
        }
        let selected = selection.contains_range(stmt.range());
        if selected && !inside_selection {
            self.always_exits |= statements_always_exit(slice::from_ref(stmt));
        }
        if selected {
            match stmt {
                Stmt::Return(ret) => self.exits.push((
                    stmt.range(),
                    SelectionExit::Return(ret.value.as_ref().map(|value| value.range())),
                )),
                Stmt::Break(_) if loop_depth == 0 => {
                    self.exits.push((stmt.range(), SelectionExit::Break));
                }
                Stmt::Continue(_) if loop_depth == 0 => {
                    self.exits.push((stmt.range(), SelectionExit::Continue));
                }
                Stmt::Raise(_) | Stmt::FunctionDef(_) | Stmt::ClassDef(_) | Stmt::Delete(_) => {
                    self.disallowed = true;
                    return;
                }
                Stmt::For(loop_stmt) if loop_stmt.is_async => self.disallowed = true,
                Stmt::With(block) if block.is_async => self.disallowed = true,
                Stmt::Try(block) if !block.finalbody.is_empty() => self.has_finally = true,
                Stmt::Assign(assign) if !inside_selection => {
                    for target in &assign.targets {
                        if let Expr::Name(name) = target {
                            self.assignments.push((stmt.range(), name.id.to_string()));
                        }
                    }
                }
                Stmt::AnnAssign(assign) if !inside_selection && assign.value.is_some() => {
                    if let Expr::Name(name) = assign.target.as_ref() {
                        self.assignments.push((stmt.range(), name.id.to_string()));
                    }
                }
                _ => {}
            }
        }
        let loop_bodies = match stmt {
            Stmt::For(loop_stmt) => Some((&loop_stmt.body, &loop_stmt.orelse)),
            Stmt::While(loop_stmt) => Some((&loop_stmt.body, &loop_stmt.orelse)),
            _ => None,
        };
        if let Some((body, orelse)) = loop_bodies {
            if !selected {
                self.enclosing_loops.push(stmt.range());
            }
            for child in body {
                self.visit_stmt(
                    child,
                    selection,
                    loop_depth + usize::from(selected),
                    selected,
                );
            }
            // A loop's else suite is outside that loop's break/continue target.
            for child in orelse {
                self.visit_stmt(child, selection, loop_depth, selected);
            }
        } else {
            stmt.recurse(&mut |child| self.visit_stmt(child, selection, loop_depth, selected));
        }
    }
}

/// Recognizes unconditional exits without assuming that a loop executes or a
/// context manager propagates exceptions from its body.
fn statements_always_exit(body: &[Stmt]) -> bool {
    body.last().is_some_and(|stmt| match stmt {
        Stmt::Return(_) | Stmt::Break(_) | Stmt::Continue(_) => true,
        Stmt::If(branch) => {
            statements_always_exit(&branch.body)
                && branch
                    .elif_else_clauses
                    .last()
                    .is_some_and(|clause| clause.test.is_none())
                && branch
                    .elif_else_clauses
                    .iter()
                    .all(|clause| statements_always_exit(&clause.body))
        }
        Stmt::Try(block) => {
            statements_always_exit(&block.finalbody)
                || (statements_always_exit(&block.body)
                    && block
                        .handlers
                        .iter()
                        .all(|ExceptHandler::ExceptHandler(handler)| {
                            statements_always_exit(&handler.body)
                        }))
        }
        _ => false,
    })
}

fn find_enclosing_module_statement_range(
    ast: &ModModule,
    selection: TextRange,
    module_len: TextSize,
) -> TextRange {
    for stmt in &ast.body {
        if stmt.range().contains_range(selection) {
            return stmt.range();
        }
    }
    TextRange::new(TextSize::new(0), module_len)
}

fn collect_post_selection_loads(
    ast: &ModModule,
    module_stmt_range: TextRange,
    selection: TextRange,
    enclosing_loops: &[TextRange],
) -> HashSet<String> {
    struct LoadCollector<'a> {
        module_stmt_range: TextRange,
        selection: TextRange,
        enclosing_loops: &'a [TextRange],
        loads: HashSet<String>,
    }
    impl<'a> Visitor<'a> for LoadCollector<'_> {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Name(name) = expr
                && matches!(name.ctx, ExprContext::Load)
                && self.module_stmt_range.contains_range(name.range)
                && !self.selection.contains_range(name.range)
                && (name.range.start() >= self.selection.end()
                    // An enclosing loop can read the output on the next iteration.
                    || self.enclosing_loops.iter().any(|range| range.contains_range(name.range)))
            {
                self.loads.insert(name.id.to_string());
            }
            walk_expr(self, expr);
        }
    }
    let mut collector = LoadCollector {
        module_stmt_range,
        selection,
        enclosing_loops,
        loads: HashSet::new(),
    };
    collector.visit_body(&ast.body);
    collector.loads
}

fn find_enclosing_method(
    ast: &ModModule,
    selection: TextRange,
    source: &str,
) -> Option<MethodContext> {
    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && let Some(ctx) = method_context_in_class(class_def, selection, source)
        {
            return Some(ctx);
        }
    }
    None
}

fn method_context_in_class(
    class_def: &StmtClassDef,
    selection: TextRange,
    source: &str,
) -> Option<MethodContext> {
    for stmt in &class_def.body {
        match stmt {
            Stmt::FunctionDef(function_def) if function_def.range().contains_range(selection) => {
                if let Some(ctx) = method_context_from_function(class_def, function_def, source) {
                    return Some(ctx);
                }
            }
            Stmt::ClassDef(inner_class) => {
                if let Some(ctx) = method_context_in_class(inner_class, selection, source) {
                    return Some(ctx);
                }
            }
            _ => {}
        }
    }
    None
}

fn method_context_from_function(
    class_def: &StmtClassDef,
    function_def: &StmtFunctionDef,
    source: &str,
) -> Option<MethodContext> {
    if is_static_or_class_method(function_def) {
        return None;
    }
    let receiver_name = first_parameter_name(&function_def.parameters)?;
    let (method_indent, insert_position) =
        line_indent_and_start(source, function_def.range().start())?;
    Some(MethodContext {
        info: MethodInfo {
            class_name: class_def.name.id.to_string(),
            receiver_name,
        },
        insert_position,
        method_indent,
    })
}

fn detect_block_indent(selection_text: &str) -> String {
    for line in selection_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        return line
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect::<String>();
    }
    String::new()
}

fn build_helper_text(
    helper_name: &str,
    params: &[String],
    returns: &[String],
    dedented_body: &str,
    definition_indent: &str,
) -> String {
    let mut helper_text = if params.is_empty() {
        format!("{definition_indent}def {helper_name}():\n")
    } else {
        let helper_params = params.join(", ");
        format!("{definition_indent}def {helper_name}({helper_params}):\n")
    };
    let body_indent = format!("{definition_indent}{HELPER_INDENT}");
    helper_text.push_str(&prefix_lines_with(dedented_body, &body_indent));
    if !returns.is_empty() && !returns.iter().all(|name| name.is_empty()) {
        let return_expr = if returns.len() == 1 {
            returns[0].clone()
        } else {
            returns.join(", ")
        };
        helper_text.push_str(&format!("{body_indent}return {return_expr}\n"));
    }
    helper_text.push('\n');
    helper_text
}

fn build_call_expr(helper_name: &str, receiver: Option<&str>, params: &[String]) -> String {
    let callee = if let Some(receiver) = receiver {
        format!("{receiver}.{helper_name}")
    } else {
        helper_name.to_owned()
    };
    let call_args = params.join(", ");
    format!("{callee}({call_args})")
}

fn build_call_replacement(block_indent: &str, call_expr: &str, returns: &[String]) -> String {
    if returns.is_empty() {
        format!("{block_indent}{call_expr}\n")
    } else {
        let lhs = if returns.len() == 1 {
            returns[0].clone()
        } else {
            returns.join(", ")
        };
        format!("{block_indent}{lhs} = {call_expr}\n")
    }
}

fn filter_params_excluding(params: &[String], excluded: &str) -> Vec<String> {
    params
        .iter()
        .filter(|name| name.as_str() != excluded)
        .cloned()
        .collect()
}

fn prefix_lines_with(block: &str, indent: &str) -> String {
    let mut result = String::new();
    for line in block.lines() {
        result.push_str(indent);
        result.push_str(line);
        result.push('\n');
    }
    result
}

/// Generated names must not shadow any existing reference or binding.
fn generate_name(source: &str, prefix: &str) -> String {
    let mut candidate = prefix.to_owned();
    let mut counter = 2;
    while source.contains(&candidate) {
        candidate = format!("{prefix}_{counter}");
        counter += 1;
    }
    candidate
}
