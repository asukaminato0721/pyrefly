/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_util::visit::Visit;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtIf;
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::token::Tokens;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::reindent_block;
use super::extract_shared::selection_anchor;
use super::extract_shared::statement_removal_range_from_range;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

struct UnwrapBlockTarget {
    outer_range: TextRange,
    body_range: TextRange,
    from_indent: String,
    to_indent: String,
}

/// Builds unwrap-block refactor actions for the block under the cursor.
pub(crate) fn unwrap_block_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let source = module_info.contents();
    let parsed = transaction.get_parsed_module(handle)?;
    let tokens = parsed.tokens()?;
    let position = selection_anchor(source, selection);
    let target = find_unwrap_block_target_in_body(&parsed.body, source, &tokens, position)?;
    let removal_range = statement_removal_range_from_range(source, target.outer_range)?;
    let replacement = reindent_block(
        source,
        target.body_range,
        &tokens,
        &target.from_indent,
        &target.to_indent,
    )?;
    Some(vec![LocalRefactorCodeAction {
        title: "Unwrap block".to_owned(),
        edits: vec![(module_info.dupe(), removal_range, replacement)],
        kind: CodeActionKind::RefactorRewrite,
    }])
}

fn find_unwrap_block_target_in_body(
    body: &[Stmt],
    source: &str,
    tokens: &Tokens,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    for stmt in body {
        if !range_contains_position(stmt.range(), position) {
            continue;
        }
        if let Some(target) = find_unwrap_block_target_in_stmt(stmt, source, tokens, position) {
            return Some(target);
        }
    }
    None
}

fn find_unwrap_block_target_in_stmt(
    stmt: &Stmt,
    source: &str,
    tokens: &Tokens,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    match stmt {
        Stmt::FunctionDef(function_def) => {
            find_unwrap_block_target_in_body(&function_def.body, source, tokens, position)
        }
        Stmt::ClassDef(class_def) => {
            find_unwrap_block_target_in_body(&class_def.body, source, tokens, position)
        }
        Stmt::If(if_stmt) => find_unwrap_if_target(if_stmt, source, tokens, position),
        Stmt::For(for_stmt) => {
            if let Some(target) =
                find_unwrap_block_target_in_body(&for_stmt.body, source, tokens, position)
            {
                return Some(target);
            }
            if let Some(target) =
                find_unwrap_block_target_in_body(&for_stmt.orelse, source, tokens, position)
            {
                return Some(target);
            }
            if !for_stmt.orelse.is_empty() || for_stmt.body.iter().any(loop_control_escapes) {
                return None;
            }
            build_unwrap_block_target(
                stmt.range(),
                stmt.range().start(),
                &for_stmt.body,
                None,
                source,
                tokens,
                position,
            )
        }
        Stmt::While(while_stmt) => {
            if let Some(target) =
                find_unwrap_block_target_in_body(&while_stmt.body, source, tokens, position)
            {
                return Some(target);
            }
            if let Some(target) =
                find_unwrap_block_target_in_body(&while_stmt.orelse, source, tokens, position)
            {
                return Some(target);
            }
            if !while_stmt.orelse.is_empty() || while_stmt.body.iter().any(loop_control_escapes) {
                return None;
            }
            build_unwrap_block_target(
                stmt.range(),
                stmt.range().start(),
                &while_stmt.body,
                None,
                source,
                tokens,
                position,
            )
        }
        Stmt::With(with_stmt) => {
            if let Some(target) =
                find_unwrap_block_target_in_body(&with_stmt.body, source, tokens, position)
            {
                return Some(target);
            }
            build_unwrap_block_target(
                stmt.range(),
                stmt.range().start(),
                &with_stmt.body,
                None,
                source,
                tokens,
                position,
            )
        }
        Stmt::Try(try_stmt) => {
            if let Some(target) =
                find_unwrap_block_target_in_body(&try_stmt.body, source, tokens, position)
            {
                return Some(target);
            }
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                if let Some(target) =
                    find_unwrap_block_target_in_body(&handler.body, source, tokens, position)
                {
                    return Some(target);
                }
            }
            if let Some(target) =
                find_unwrap_block_target_in_body(&try_stmt.orelse, source, tokens, position)
            {
                return Some(target);
            }
            if let Some(target) =
                find_unwrap_block_target_in_body(&try_stmt.finalbody, source, tokens, position)
            {
                return Some(target);
            }
            let else_start = find_try_clause_start(tokens, &try_stmt.orelse, TokenKind::Else);
            let finally_start =
                find_try_clause_start(tokens, &try_stmt.finalbody, TokenKind::Finally);
            for (index, handler) in try_stmt.handlers.iter().enumerate() {
                let ExceptHandler::ExceptHandler(handler) = handler;
                let next_clause_start = try_stmt
                    .handlers
                    .get(index + 1)
                    .map(|next| next.range().start())
                    .or(else_start)
                    .or(finally_start);
                if let Some(target) = build_unwrap_block_target(
                    stmt.range(),
                    handler.range().start(),
                    &handler.body,
                    next_clause_start,
                    source,
                    tokens,
                    position,
                ) {
                    return Some(target);
                }
            }
            if let Some(else_start) = else_start
                && let Some(target) = build_unwrap_block_target(
                    stmt.range(),
                    else_start,
                    &try_stmt.orelse,
                    finally_start,
                    source,
                    tokens,
                    position,
                )
            {
                return Some(target);
            }
            build_unwrap_block_target(
                stmt.range(),
                finally_start?,
                &try_stmt.finalbody,
                None,
                source,
                tokens,
                position,
            )
        }
        Stmt::Match(match_stmt) => {
            for case in &match_stmt.cases {
                if let Some(target) =
                    find_unwrap_block_target_in_body(&case.body, source, tokens, position)
                {
                    return Some(target);
                }
            }
            None
        }
        _ => None,
    }
}

fn find_unwrap_if_target(
    if_stmt: &StmtIf,
    source: &str,
    tokens: &Tokens,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    if let Some(target) = find_unwrap_block_target_in_body(&if_stmt.body, source, tokens, position)
    {
        return Some(target);
    }
    for clause in &if_stmt.elif_else_clauses {
        if let Some(target) =
            find_unwrap_block_target_in_body(&clause.body, source, tokens, position)
        {
            return Some(target);
        }
    }

    let next_clause_start = if_stmt
        .elif_else_clauses
        .first()
        .map(|clause| clause.range.start());
    if let Some(target) = build_unwrap_block_target(
        if_stmt.range(),
        if_stmt.range().start(),
        &if_stmt.body,
        next_clause_start,
        source,
        tokens,
        position,
    ) {
        return Some(target);
    }

    for (index, clause) in if_stmt.elif_else_clauses.iter().enumerate() {
        let next_clause_start = if_stmt
            .elif_else_clauses
            .get(index + 1)
            .map(|next_clause| next_clause.range.start());
        if let Some(target) = build_unwrap_block_target(
            if_stmt.range(),
            clause.range.start(),
            &clause.body,
            next_clause_start,
            source,
            tokens,
            position,
        ) {
            return Some(target);
        }
    }

    None
}

fn build_unwrap_block_target(
    outer_range: TextRange,
    header_start: TextSize,
    body: &[Stmt],
    next_clause_start: Option<TextSize>,
    source: &str,
    tokens: &Tokens,
    position: TextSize,
) -> Option<UnwrapBlockTarget> {
    let first_stmt = body.first()?;
    let (to_indent, _) = line_indent_and_start(source, outer_range.start())?;
    let (_, header_line_start) = line_indent_and_start(source, header_start)?;
    let (from_indent, _) = line_indent_and_start(source, first_stmt.range().start())?;
    // The suite colon is the last significant token before its first statement.
    let colon = tokens
        .in_range(TextRange::new(header_start, first_stmt.range().start()))
        .iter()
        .rev()
        .find(|token| {
            !matches!(
                token.kind(),
                TokenKind::Comment
                    | TokenKind::Newline
                    | TokenKind::NonLogicalNewline
                    | TokenKind::Indent
            )
        })?;
    if colon.kind() != TokenKind::Colon {
        return None;
    }
    let colon_position = colon.start();
    if position < header_line_start || position > colon_position + TextSize::new(1) {
        return None;
    }
    let block_start = line_end_position(source, colon_position);
    if block_start > first_stmt.range().start() {
        return None;
    }
    let block_end = if let Some(next_clause_start) = next_clause_start {
        line_indent_and_start(source, next_clause_start)?.1
    } else {
        line_end_position(source, outer_range.end())
    };
    if block_start >= block_end {
        return None;
    }
    Some(UnwrapBlockTarget {
        outer_range,
        body_range: TextRange::new(block_start, block_end),
        from_indent,
        to_indent,
    })
}

fn find_try_clause_start(tokens: &Tokens, body: &[Stmt], keyword: TokenKind) -> Option<TextSize> {
    tokens
        .before(body.first()?.range().start())
        .iter()
        .rev()
        .find(|token| token.kind() == keyword)
        .map(Ranged::start)
}

/// Checks whether a control statement targets the loop being removed.
fn loop_control_escapes(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Break(_) | Stmt::Continue(_) => true,
        Stmt::FunctionDef(_) | Stmt::ClassDef(_) => false,
        // A nested loop's else suite still belongs to the enclosing loop.
        Stmt::For(stmt) => stmt.orelse.iter().any(loop_control_escapes),
        Stmt::While(stmt) => stmt.orelse.iter().any(loop_control_escapes),
        _ => {
            let mut escapes = false;
            stmt.recurse(&mut |child| escapes |= loop_control_escapes(child));
            escapes
        }
    }
}

fn range_contains_position(range: TextRange, position: TextSize) -> bool {
    range.start() <= position && position < range.end()
}
