/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use anyhow::Context;
use clap::Parser;
use pyrefly_config::args::EnvironmentArgs;
use pyrefly_types::types::Type;
use pyrefly_util::thread_pool::ThreadCount;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::TypeParams;
use ruff_python_ast::visitor::source_order::SourceOrderVisitor;
use ruff_python_ast::visitor::source_order::walk_expr;
use ruff_python_ast::visitor::source_order::walk_stmt;
use ruff_text_size::Ranged;
use serde::Serialize;

use crate::alt::answers::Answers;
use crate::commands::check::Handles;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::config::config::ConfigScope;
use crate::state::require::Require;
use crate::state::state::State;

/// Report inferred expression types independently of annotation coverage.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct ExpressionsArgs {
    /// Which files to analyze.
    #[command(flatten)]
    files: FilesArgs,

    #[command(flatten)]
    config_override: EnvironmentArgs,
}

/// Each analyzed expression is counted once, even if inference visits it repeatedly.
#[derive(Debug, Default, Serialize)]
struct ExpressionCounts {
    n_expressions: usize,
    n_any: usize,
    n_unanalyzed: usize,
}

impl ExpressionCounts {
    fn coverage(&self) -> f64 {
        if self.n_expressions == 0 {
            100.0
        } else {
            (self.n_expressions - self.n_any) as f64 / self.n_expressions as f64 * 100.0
        }
    }
}

#[derive(Debug, Serialize)]
struct ModuleReport {
    name: String,
    path: String,
    #[serde(flatten)]
    counts: ExpressionCounts,
    coverage: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    n_modules: usize,
    #[serde(flatten)]
    counts: ExpressionCounts,
    coverage: f64,
}

#[derive(Debug, Serialize)]
struct ExpressionReport {
    schema_version: &'static str,
    module_reports: Vec<ModuleReport>,
    summary: Summary,
}

impl ExpressionReport {
    fn new(mut module_reports: Vec<ModuleReport>) -> Self {
        module_reports.sort_by(|a, b| a.path.cmp(&b.path).then(a.name.cmp(&b.name)));
        let mut counts = ExpressionCounts::default();
        for module in &module_reports {
            counts.n_expressions += module.counts.n_expressions;
            counts.n_any += module.counts.n_any;
            counts.n_unanalyzed += module.counts.n_unanalyzed;
        }
        let summary = Summary {
            n_modules: module_reports.len(),
            coverage: counts.coverage(),
            counts,
        };
        Self {
            schema_version: "0.1",
            module_reports,
            summary,
        }
    }
}

/// Walk value expressions, excluding annotations and store/delete targets themselves.
fn collect_expressions(ast: &ModModule, answers: &Answers) -> ExpressionCounts {
    struct Collector<'a> {
        answers: &'a Answers,
        counts: ExpressionCounts,
    }

    impl<'a> SourceOrderVisitor<'a> for Collector<'_> {
        fn visit_annotation(&mut self, _expr: &'a Expr) {}

        fn visit_type_params(&mut self, _params: &'a TypeParams) {}

        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            if !matches!(stmt, Stmt::TypeAlias(_)) {
                walk_stmt(self, stmt);
            }
        }

        fn visit_expr(&mut self, expr: &'a Expr) {
            let is_value = match expr {
                Expr::Name(x) => x.ctx.is_load(),
                Expr::Attribute(x) => x.ctx.is_load(),
                Expr::Subscript(x) => x.ctx.is_load(),
                Expr::Starred(x) => x.ctx.is_load(),
                Expr::List(x) => x.ctx.is_load(),
                Expr::Tuple(x) => x.ctx.is_load(),
                _ => true,
            };
            if is_value {
                if let Some(ty) = self.answers.get_type_trace(expr.range()) {
                    // Traces may still contain solver variables for inferred containers or returns.
                    let ty = self.answers.solver().expand(ty);
                    self.counts.n_expressions += 1;
                    if ty.any(Type::is_any) {
                        self.counts.n_any += 1;
                    }
                } else {
                    self.counts.n_unanalyzed += 1;
                }
            }
            // A store target can contain evaluated expressions, such as a subscript's index.
            walk_expr(self, expr);
        }
    }

    let mut collector = Collector {
        answers,
        counts: ExpressionCounts::default(),
    };
    collector.visit_body(&ast.body);
    collector.counts
}

impl ExpressionsArgs {
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files, config_finder, _) = self.files.resolve_scoped(
            self.config_override.into(),
            wrapper,
            ConfigScope::Coverage,
        )?;
        let expanded = config_finder.checkpoint(files.files_iter())?;
        let (handles, _, sourcedb_errors) = Handles::new(expanded).all(&config_finder);
        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            anyhow::bail!("Failed to query sourcedb.");
        }
        let state = State::new(config_finder, thread_count);
        let mut transaction = state.new_transaction(Require::Exports, None);
        // Only selected files need expression traces; dependencies provide exported types.
        transaction.run(&handles, Require::Everything, None);
        let mut module_reports = Vec::new();
        for handle in &handles {
            let ast = transaction
                .get_ast(handle)
                .with_context(|| format!("No syntax tree for {}", handle.path()))?;
            let answers = transaction
                .get_answers(handle)
                .with_context(|| format!("No inferred types for {}", handle.path()))?;
            let counts = collect_expressions(&ast, &answers);
            module_reports.push(ModuleReport {
                name: handle.module().to_string(),
                path: handle.path().as_path().display().to_string(),
                coverage: counts.coverage(),
                counts,
            });
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&ExpressionReport::new(module_reports))?
        );
        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::util::TestEnv;

    fn coverage(code: &str) -> ExpressionCounts {
        let (state, handle) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let transaction = state.transaction();
        let handle = handle("test");
        collect_expressions(
            &transaction.get_ast(&handle).unwrap(),
            &transaction.get_answers(&handle).unwrap(),
        )
    }

    #[test]
    fn test_expression_coverage_inferred() {
        let counts = coverage("x = 1\ny = x + 2\n");
        assert_eq!(counts.n_expressions, 4);
        assert_eq!(counts.n_any, 0);
        assert_eq!(counts.n_unanalyzed, 0);
        assert_eq!(counts.coverage(), 100.0);
    }

    #[test]
    fn test_expression_coverage_any_in_private_function() {
        let counts = coverage("from typing import Any\ndef _f(x: Any):\n    return x + 1\n");
        assert_eq!(counts.n_expressions, 3);
        assert_eq!(counts.n_any, 2);
        assert_eq!(counts.n_unanalyzed, 0);
        assert!((counts.coverage() - 100.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_expression_coverage_nested_any() {
        let counts = coverage("from typing import Any\ndef f(x: list[Any]):\n    return x\n");
        assert_eq!(counts.n_expressions, 1);
        assert_eq!(counts.n_any, 1);
        assert_eq!(counts.coverage(), 0.0);
    }

    #[test]
    fn test_expression_coverage_implicit_and_inferred_any() {
        for code in [
            "def f(x):\n    return [x]\n",
            "from typing import Any\ndef f(x: Any):\n    return [x]\n",
        ] {
            let counts = coverage(code);
            assert_eq!(counts.n_expressions, 2);
            assert_eq!(counts.n_any, 2);
            assert_eq!(counts.n_unanalyzed, 0);
        }
    }

    #[test]
    fn test_expression_coverage_store_and_delete() {
        let counts = coverage("def f(xs: list[int]):\n    xs[0] = 1\n    del xs[2]\n");
        assert_eq!(counts.n_expressions, 5);
        assert_eq!(counts.n_any, 0);
        assert_eq!(counts.n_unanalyzed, 0);
    }

    #[test]
    fn test_expression_coverage_annotations_and_empty_module() {
        for code in [
            "",
            "from typing import Any\nx: Any\ntype Alias = list[Any]\n",
        ] {
            let counts = coverage(code);
            assert_eq!(counts.n_expressions, 0);
            assert_eq!(counts.n_unanalyzed, 0);
            assert_eq!(counts.coverage(), 100.0);
        }
    }

    #[test]
    fn test_expression_coverage_unreachable() {
        let counts = coverage("if False:\n    x = 1\n");
        // The constant condition is folded during binding, so neither literal has a trace.
        assert_eq!(counts.n_expressions, 0);
        assert_eq!(counts.n_any, 0);
        assert_eq!(counts.n_unanalyzed, 2);
    }

    #[test]
    fn test_expression_coverage_summary() {
        let report = ExpressionReport::new(
            [
                ("b", "x = 1\ny = x + 2\n"),
                (
                    "a",
                    "from typing import Any\ndef f(x: Any):\n    return x\n",
                ),
            ]
            .into_iter()
            .map(|(name, code)| {
                let counts = coverage(code);
                ModuleReport {
                    name: name.to_owned(),
                    path: format!("{name}.py"),
                    coverage: counts.coverage(),
                    counts,
                }
            })
            .collect(),
        );
        assert_eq!(report.module_reports[0].name, "a");
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["schema_version"], "0.1");
        assert_eq!(
            json["summary"],
            serde_json::json!({
                "n_modules": 2,
                "n_expressions": 5,
                "n_any": 1,
                "n_unanalyzed": 0,
                "coverage": 80.0,
            }),
        );
        assert_eq!(ExpressionReport::new(Vec::new()).summary.coverage, 100.0);
    }
}
