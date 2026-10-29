/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Reverse;
use std::sync::Arc;

use lsp_types::AnnotatedTextEdit;
use lsp_types::Edit;
use lsp_types::TextEdit;
use lsp_types::Uri;
use pyrefly_python::module_name::is_python_identifier;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::ModuleInfo;
use crate::state::load::LspFile;

/// Budgets apply only to optional heuristic edits. Semantic rename has its own cancellation.
pub(super) const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_MATCHES: usize = 1000;
pub(super) const COMMENT_ANNOTATION: &str = "pyrefly.rename.commentsAndStrings";

/// Immutable input captured from one synchronized document before reference collection.
pub(super) struct CommentRename {
    pub module: ModuleInfo,
    pub ast: Arc<ModModule>,
    pub identifier: TextRange,
    pub uri: Uri,
    pub version: i32,
    pub document: Arc<LspFile>,
}

impl CommentRename {
    /// Return heuristic edits separately so semantic references never require heuristic confirmation.
    pub fn edits(
        &self,
        new_name: &str,
        semantic: &[TextEdit],
        cancelled: &impl Fn() -> bool,
    ) -> Vec<Edit> {
        let source = self.module.contents();
        let name = &source[self.identifier.start().to_usize()..self.identifier.end().to_usize()];
        occurrences(
            &self.module,
            &self.ast,
            self.identifier.start(),
            name,
            cancelled,
        )
        .into_iter()
        .filter_map(|range| {
            let range = self.module.to_lsp_range(range);
            if semantic
                .iter()
                .any(|edit| edit.range.start < range.end && range.start < edit.range.end)
            {
                return None;
            }
            Some(Edit::AnnotatedTextEdit(AnnotatedTextEdit {
                annotation_id: COMMENT_ANNOTATION.to_owned(),
                text_edit: TextEdit {
                    range,
                    new_text: new_name.to_owned(),
                },
            }))
        })
        .collect()
    }
}

/// Search source spellings in the initiating lexical scope. Nested scopes are excluded.
fn occurrences(
    module: &ModuleInfo,
    ast: &ModModule,
    position: TextSize,
    name: &str,
    cancelled: &impl Fn() -> bool,
) -> Vec<TextRange> {
    let source = module.contents();
    if source.len() > MAX_SOURCE_BYTES || name.is_empty() || cancelled() {
        return Vec::new();
    }
    let mut collector = ContentRanges {
        ranges: Vec::new(),
        scopes: Vec::new(),
        cancelled,
        source,
    };
    for stmt in &ast.body {
        collector.visit_stmt(stmt);
    }
    let scope = collector
        .scopes
        .iter()
        .filter(|(range, declaration)| range.contains(position) && !declaration.contains(position))
        .map(|(range, _)| *range)
        .min_by_key(|range| range.len())
        .unwrap_or(TextRange::new(
            TextSize::new(0),
            TextSize::new(source.len() as u32),
        ));
    let mut nested: Vec<_> = collector
        .scopes
        .into_iter()
        .map(|(range, _)| range)
        .filter(|range| *range != scope && scope.contains_range(*range))
        .collect();
    nested.sort_by_key(|range| (range.start(), Reverse(range.end())));
    let mut excluded: Vec<TextRange> = Vec::new();
    for range in nested {
        if let Some(last) = excluded.last_mut()
            && last.end() >= range.start()
        {
            *last = last.cover(range);
        } else {
            excluded.push(range);
        }
    }
    let mut hits = Vec::new();
    let strings = collector.ranges.into_iter().filter(|range| {
        // Escaped literals need a decoded-to-source map. This feature excludes them entirely.
        source
            .get(range.start().to_usize()..range.end().to_usize())
            .is_some_and(|text| !text.contains('\\'))
    });
    for range in strings.chain(module.ignore().comment_ranges()) {
        if cancelled() {
            return Vec::new();
        }
        let index = excluded.partition_point(|nested| nested.start() <= range.start());
        if !scope.contains_range(range)
            || index
                .checked_sub(1)
                .is_some_and(|index| excluded[index].end() > range.start())
        {
            continue;
        }
        let Some(text) = source.get(range.start().to_usize()..range.end().to_usize()) else {
            continue;
        };
        for (offset, _) in text.match_indices(name) {
            if cancelled() {
                return Vec::new();
            }
            // Prefixing a character with '_' tests Python's identifier continuation rules,
            // including combining marks, through the existing identifier validator.
            let continues = |ch| is_python_identifier(&format!("_{ch}"));
            if text[..offset].chars().next_back().is_some_and(continues)
                || text[offset + name.len()..]
                    .chars()
                    .next()
                    .is_some_and(continues)
            {
                continue;
            }
            if hits.len() == MAX_MATCHES {
                return Vec::new();
            }
            let start = range.start() + TextSize::new(offset as u32);
            hits.push(TextRange::new(
                start,
                start + TextSize::new(name.len() as u32),
            ));
        }
    }
    hits.sort_unstable_by_key(|range| (range.start(), range.end()));
    hits.dedup();
    hits
}

struct ContentRanges<'a, F> {
    ranges: Vec<TextRange>,
    /// Each entry contains a scope and its declaration name, which belongs to the outer scope.
    scopes: Vec<(TextRange, TextRange)>,
    cancelled: &'a F,
    source: &'a str,
}

impl<'a, F: Fn() -> bool> Visitor<'a> for ContentRanges<'_, F> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if (self.cancelled)() {
            return;
        }
        let declaration = match stmt {
            Stmt::FunctionDef(def) => Some(def.name.range),
            Stmt::ClassDef(def) => Some(def.name.range),
            _ => None,
        };
        if let Some(declaration) = declaration {
            // The AST ends before a trailing inline comment on the final statement.
            let end = stmt.end().to_usize();
            let line_end = self.source[end..]
                .find(['\r', '\n'])
                .map_or(self.source.len(), |offset| end + offset);
            self.scopes.push((
                TextRange::new(stmt.start(), TextSize::new(line_end as u32)),
                declaration,
            ));
        }
        walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        if (self.cancelled)() {
            return;
        }
        match expr {
            Expr::StringLiteral(expr) => {
                self.ranges
                    .extend(expr.value.iter().map(|literal| literal.content_range()));
            }
            Expr::FString(expr) => {
                self.ranges.extend(
                    expr.value
                        .elements()
                        .filter_map(|element| element.as_literal().map(|literal| literal.range)),
                );
            }
            Expr::TString(expr) => {
                self.ranges.extend(
                    expr.value
                        .elements()
                        .filter_map(|element| element.as_literal().map(|literal| literal.range)),
                );
            }
            Expr::Lambda(_)
            | Expr::ListComp(_)
            | Expr::SetComp(_)
            | Expr::DictComp(_)
            | Expr::Generator(_) => {
                self.scopes
                    .push((expr.range(), TextRange::empty(expr.start())));
            }
            // Bytes literals are excluded because a Python identifier can contain non-ASCII text.
            _ => {}
        }
        walk_expr(self, expr);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::PathBuf;

    use pyrefly_python::ast::Ast;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_python_ast::PySourceType;

    use super::*;
    use crate::state::require::Require;
    use crate::test::util::mk_state;

    fn fixture(source: &str) -> (ModuleInfo, ModModule) {
        let module = Module::new(
            ModuleName::from_str("test"),
            ModulePath::filesystem(PathBuf::from("test.py")),
            Arc::new(source.to_owned()),
        );
        let (ast, errors, _) = Ast::parse(source, PySourceType::Python);
        assert!(errors.is_empty(), "{errors:?}");
        (module, ast)
    }

    #[test]
    fn comments_use_python_identifier_boundaries() {
        let (module, ast) =
            fixture("foo = 1\n# foo foo\u{301} foobar _foo foo_ caféfoo\ns = 'foo'\n");
        let hits = occurrences(&module, &ast, TextSize::new(0), "foo", &|| false);
        assert_eq!(hits.len(), 2);
        assert_eq!(module.to_lsp_range(hits[0]).start.line, 1);
        assert_eq!(module.to_lsp_range(hits[1]).start.line, 2);
    }

    #[test]
    fn comments_skip_escaped_and_bytes_literals_and_interpolations() {
        let (module, ast) = fixture(
            "foo = 1\nr'foo'\n'\\nfoo'\n'foo\\u1234'\nb'foo'\nf'{foo} foo'\nt'{foo} foo'\n",
        );
        let hits = occurrences(&module, &ast, TextSize::new(0), "foo", &|| false);
        assert_eq!(
            hits.iter()
                .map(|range| module.to_lsp_range(*range).start.line)
                .collect::<Vec<_>>(),
            [1, 5, 6]
        );
    }

    #[test]
    fn comments_stay_in_the_initiating_scope() {
        let source = "foo = 1\n# foo\ns = 'foo'\ndef other():\n    foo = 2\n    # foo\n    s = 'foo'\n    return foo\n";
        let (module, ast) = fixture(source);
        let outer = occurrences(&module, &ast, TextSize::new(0), "foo", &|| false);
        assert_eq!(
            outer
                .iter()
                .map(|range| module.to_lsp_range(*range).start.line)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        let position = TextSize::new(source.find("foo = 2").unwrap() as u32);
        let inner = occurrences(&module, &ast, position, "foo", &|| false);
        assert_eq!(
            inner
                .iter()
                .map(|range| module.to_lsp_range(*range).start.line)
                .collect::<Vec<_>>(),
            [5, 6]
        );
    }

    #[test]
    fn comments_skip_nested_expression_scopes() {
        let (module, ast) =
            fixture("foo = 1\ns = 'foo'\nx = lambda: 'foo'\ny = ['foo' for foo in []]\n");
        assert_eq!(
            occurrences(&module, &ast, TextSize::new(0), "foo", &|| false).len(),
            1
        );
    }

    #[test]
    fn comments_include_the_last_statement_inline_comment() {
        let source = "def f():\n    foo = 1\n    return foo  # foo\n";
        let (module, ast) = fixture(source);
        let position = TextSize::new(source.find("foo = 1").unwrap() as u32);
        let hits = occurrences(&module, &ast, position, "foo", &|| false);
        assert_eq!(hits.len(), 1);
        assert_eq!(module.to_lsp_range(hits[0]).start.line, 2);
    }

    #[test]
    fn comments_preserve_crlf_and_utf16_positions() {
        let (module, ast) = fixture("foo = 1\r\n# 😀 foo\r\n");
        let hits = occurrences(&module, &ast, TextSize::new(0), "foo", &|| false);
        assert_eq!(hits.len(), 1);
        let range = module.to_lsp_range(hits[0]);
        assert_eq!(
            (range.start.line, range.start.character, range.end.character),
            (1, 5, 8)
        );
    }

    #[test]
    fn comments_tolerate_incomplete_python() {
        for source in ["foo = 1\ns = 'foo\n", "foo = 1\ns = \"a\" b'𝐁"] {
            let module = Module::new(
                ModuleName::from_str("test"),
                ModulePath::filesystem(PathBuf::from("test.py")),
                Arc::new(source.to_owned()),
            );
            let (ast, _, _) = Ast::parse(source, PySourceType::Python);
            for range in occurrences(&module, &ast, TextSize::new(0), "foo", &|| false) {
                assert_eq!(
                    source.get(range.start().to_usize()..range.end().to_usize()),
                    Some("foo")
                );
            }
        }
    }

    #[test]
    fn comments_never_insert_non_ascii_identifiers_in_bytes_literals() {
        let (module, ast) = fixture("foo = 1\ns = b'foo'\n");
        let rename = CommentRename {
            document: Arc::new(LspFile::from_source(
                module.contents().to_string(),
                "python".to_owned(),
            )),
            uri: Uri::parse("file:///test.py").unwrap(),
            version: 1,
            identifier: TextRange::new(TextSize::new(0), TextSize::new(3)),
            module,
            ast: Arc::new(ast),
        };
        assert!(rename.edits("café", &[], &|| false).is_empty());
    }

    #[test]
    fn comments_enforce_search_budgets() {
        let (module, ast) = fixture(&format!("foo = 1\n# {}", "foo ".repeat(MAX_MATCHES + 1)));
        assert!(occurrences(&module, &ast, TextSize::new(0), "foo", &|| false).is_empty());
        let (module, ast) = fixture(&format!("foo = 1\n# foo {}", " ".repeat(MAX_SOURCE_BYTES)));
        assert!(occurrences(&module, &ast, TextSize::new(0), "foo", &|| false).is_empty());
    }

    #[test]
    fn comments_cancel_during_matching() {
        let (module, ast) = fixture(&format!("foo = 1\n# {}", "foo ".repeat(100)));
        let checks = Cell::new(0);
        let hits = occurrences(&module, &ast, TextSize::new(0), "foo", &|| {
            checks.set(checks.get() + 1);
            checks.get() >= 50
        });
        assert!(hits.is_empty());
        assert_eq!(checks.get(), 50);
    }

    #[test]
    fn comments_allow_a_recheck_to_commit_during_matching() {
        let source = format!("foo = 1\n# {}", "foo ".repeat(100));
        let (handle, state) = mk_state(&source);
        let transaction = state.cancellable_transaction();
        let module = transaction.as_ref().get_module_info(&handle).unwrap();
        let ast = transaction.as_ref().get_ast(&handle).unwrap();
        let cancellation = transaction.get_cancellation_handle();
        // The worker retains only immutable results while it constructs optional edits.
        drop(transaction);
        let checks = Cell::new(0);
        let committed = Cell::new(false);
        let hits = occurrences(&module, &ast, TextSize::new(0), "foo", &|| {
            checks.set(checks.get() + 1);
            if checks.get() == 50 {
                let mut recheck = state.new_committable_transaction(Require::Everything, None);
                recheck
                    .as_mut()
                    .invalidate_disk(&[handle.path().as_path().to_path_buf()]);
                recheck
                    .as_mut()
                    .run(std::slice::from_ref(&handle), Require::Everything, None);
                state.commit_transaction(recheck, None);
                committed.set(true);
                cancellation.cancel();
            }
            cancellation.is_cancelled()
        });
        assert!(committed.get());
        assert!(hits.is_empty());
    }

    #[test]
    fn comments_do_not_duplicate_semantic_string_references() {
        let (module, ast) = fixture("foo = 1\n__all__ = ['foo']\n");
        let range = occurrences(&module, &ast, TextSize::new(0), "foo", &|| false)[0];
        let rename = CommentRename {
            document: Arc::new(LspFile::from_source(
                module.contents().to_string(),
                "python".to_owned(),
            )),
            uri: Uri::parse("file:///test.py").unwrap(),
            version: 1,
            identifier: TextRange::new(TextSize::new(0), TextSize::new(3)),
            module,
            ast: Arc::new(ast),
        };
        assert!(
            rename
                .edits(
                    "café",
                    &[TextEdit {
                        range: rename.module.to_lsp_range(range),
                        new_text: "café".to_owned(),
                    }],
                    &|| false
                )
                .is_empty()
        );
    }
}
