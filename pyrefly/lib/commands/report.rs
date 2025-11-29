/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::fs::File;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::includes::Includes;
use regex::Regex;
use ruff_python_ast::Parameters;
use ruff_text_size::Ranged;
use serde::Serialize;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::binding::ReturnTypeKind;
use crate::binding::bindings::Bindings;
use crate::commands::check::Handles;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::state::require::Require;
use crate::state::state::State;

/// Location information for code elements
#[derive(Debug, Serialize)]
struct Location {
    start: Position,
    end: Position,
}

/// Position with line and column
#[derive(Debug, Serialize)]
struct Position {
    line: usize,
    column: usize,
}

/// Parameter information
#[derive(Debug, Serialize)]
struct Parameter {
    name: String,
    annotation: Option<String>,
    location: Location,
}

/// Suppression information
#[derive(Debug, Serialize)]
struct Suppression {
    kind: String,
    codes: Vec<String>,
    location: Location,
}

/// Function information
#[derive(Debug, Serialize)]
struct Function {
    name: String,
    return_annotation: Option<String>,
    parameters: Vec<Parameter>,
    location: Location,
    is_method: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FunctionCoverageStatus {
    FullyTyped,
    PartiallyTyped,
    Untyped,
}

/// File report
#[derive(Debug, Serialize)]
struct FileReport {
    line_count: usize,
    functions: Vec<Function>,
    suppressions: Vec<Suppression>,
    coverage: CoverageStatsOutput,
}

/// Top-level JSON structure returned by the report command
#[derive(Debug, Serialize)]
struct CoverageReport {
    files: HashMap<String, FileReport>,
    summary: CoverageStatsOutput,
}

/// Serialized coverage statistics for a file or the overall project
#[derive(Debug, Serialize, Clone)]
struct CoverageStatsOutput {
    total_functions: usize,
    fully_typed_functions: usize,
    partially_typed_functions: usize,
    untyped_functions: usize,
    total_parameters: usize,
    annotated_parameters: usize,
    total_returns: usize,
    annotated_returns: usize,
    slot_coverage_percent: f64,
    fully_typed_function_percent: f64,
}

/// Internal accumulator for coverage counting before serialization
#[derive(Debug, Default, Clone)]
struct CoverageCounters {
    total_functions: usize,
    fully_typed_functions: usize,
    partially_typed_functions: usize,
    untyped_functions: usize,
    total_parameters: usize,
    annotated_parameters: usize,
    total_returns: usize,
    annotated_returns: usize,
}

impl CoverageCounters {
    fn record_function(
        &mut self,
        total_parameters: usize,
        annotated_parameters: usize,
        has_return_annotation: bool,
    ) {
        self.total_functions += 1;
        self.total_parameters += total_parameters;
        self.annotated_parameters += annotated_parameters;
        self.total_returns += 1;
        if has_return_annotation {
            self.annotated_returns += 1;
        }

        let missing_parameters = total_parameters.saturating_sub(annotated_parameters);
        let missing_return = usize::from(!has_return_annotation);
        if missing_parameters == 0 && missing_return == 0 {
            self.fully_typed_functions += 1;
        } else if annotated_parameters == 0 && !has_return_annotation {
            self.untyped_functions += 1;
        } else {
            self.partially_typed_functions += 1;
        }
    }

    fn absorb(&mut self, other: &CoverageCounters) {
        self.total_functions += other.total_functions;
        self.fully_typed_functions += other.fully_typed_functions;
        self.partially_typed_functions += other.partially_typed_functions;
        self.untyped_functions += other.untyped_functions;
        self.total_parameters += other.total_parameters;
        self.annotated_parameters += other.annotated_parameters;
        self.total_returns += other.total_returns;
        self.annotated_returns += other.annotated_returns;
    }

    fn ratio(numerator: usize, denominator: usize) -> f64 {
        if denominator == 0 {
            100.0
        } else {
            (numerator as f64 / denominator as f64) * 100.0
        }
    }

    fn to_output(&self) -> CoverageStatsOutput {
        let annotated_slots = self.annotated_parameters + self.annotated_returns;
        let total_slots = self.total_parameters + self.total_returns;
        CoverageStatsOutput {
            total_functions: self.total_functions,
            fully_typed_functions: self.fully_typed_functions,
            partially_typed_functions: self.partially_typed_functions,
            untyped_functions: self.untyped_functions,
            total_parameters: self.total_parameters,
            annotated_parameters: self.annotated_parameters,
            total_returns: self.total_returns,
            annotated_returns: self.annotated_returns,
            slot_coverage_percent: Self::ratio(annotated_slots, total_slots),
            fully_typed_function_percent: Self::ratio(
                self.fully_typed_functions,
                self.total_functions,
            ),
        }
    }
}

/// Generate reports from pyrefly type checking results.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct ReportArgs {
    /// Which files to check.
    #[command(flatten)]
    files: FilesArgs,

    /// Configuration override options
    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// Directory where an HTML coverage report should be written (like mypy's --html-report).
    #[arg(long = "html-report", value_name = "DIR")]
    html_report: Option<PathBuf>,
}

impl ReportArgs {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        let ReportArgs {
            files,
            config_override,
            html_report,
        } = self;

        config_override.validate()?;
        let (files_to_check, config_finder) = files.resolve(config_override)?;
        Self::run_inner(files_to_check, config_finder, html_report)
    }

    /// Helper to extract all parameters from Parameters struct
    fn extract_parameters(params: &Parameters) -> Vec<&ruff_python_ast::Parameter> {
        let mut all_params = Vec::new();
        all_params.extend(params.posonlyargs.iter().map(|p| &p.parameter));
        all_params.extend(params.args.iter().map(|p| &p.parameter));
        if let Some(vararg) = &params.vararg {
            all_params.push(vararg);
        }
        all_params.extend(params.kwonlyargs.iter().map(|p| &p.parameter));
        if let Some(kwarg) = &params.kwarg {
            all_params.push(kwarg);
        }
        all_params
    }

    /// Helper to convert byte offset to line and column position
    fn offset_to_position(
        module: &pyrefly_python::module::Module,
        offset: ruff_text_size::TextSize,
    ) -> Position {
        let location = module.lined_buffer().line_index().source_location(
            offset,
            module.lined_buffer().contents(),
            ruff_source_file::PositionEncoding::Utf8,
        );
        Position {
            line: location.line.get(),
            column: location.character_offset.get(),
        }
    }

    /// Helper to convert a text range to a Location
    fn range_to_location(
        module: &pyrefly_python::module::Module,
        range: ruff_text_size::TextRange,
    ) -> Location {
        Location {
            start: Self::offset_to_position(module, range.start()),
            end: Self::offset_to_position(module, range.end()),
        }
    }

    /// Helper to parse suppression comments from source code
    fn parse_suppressions(module: &pyrefly_python::module::Module) -> Vec<Suppression> {
        let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[([^\]]*)\]").unwrap();
        let source = module.lined_buffer().contents();
        let lines: Vec<&str> = source.lines().collect();
        let mut suppressions = Vec::new();

        for (line_idx, line) in lines.iter().enumerate() {
            if let Some(caps) = regex.captures(line) {
                let codes: Vec<String> = caps
                    .get(1)
                    .map(|m| {
                        m.as_str()
                            .split(',')
                            .map(|s| s.trim().to_owned())
                            .filter(|s| !s.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();

                // Find the position of the comment in the line
                if let Some(comment_start) = line.find('#') {
                    let line_number = line_idx + 1; // 1-indexed
                    let start_col = comment_start + 1; // 1-indexed column
                    let end_col = line.len();

                    suppressions.push(Suppression {
                        kind: "ignore".to_owned(),
                        codes,
                        location: Location {
                            start: Position {
                                line: line_number,
                                column: start_col,
                            },
                            end: Position {
                                line: line_number,
                                column: end_col,
                            },
                        },
                    });
                }
            }
        }

        suppressions
    }

    fn parse_functions(
        module: &pyrefly_python::module::Module,
        bindings: Bindings,
    ) -> (Vec<Function>, CoverageCounters) {
        let mut functions = Vec::new();
        let mut coverage = CoverageCounters::default();
        for idx in bindings.keys::<Key>() {
            if let Key::Definition(id) = bindings.idx_to_key(idx)
                && let Binding::Function(x, _pred, _class_meta) = bindings.get(idx)
            {
                let fun = bindings.get(bindings.get(*x).undecorated_idx);
                let func_name = module.code_at(id.range());
                let location = Self::range_to_location(module, fun.def.range);
                let is_method = fun.class_key.is_some();

                // Get return annotation from ReturnTypeKind
                let return_annotation = {
                    let return_key = Key::ReturnType(*id);
                    let return_idx = bindings.key_to_idx(&return_key);
                    if let Binding::ReturnType(ret) = bindings.get(return_idx) {
                        match &ret.kind {
                            ReturnTypeKind::ShouldValidateAnnotation { range, .. } => {
                                Some(module.code_at(*range).to_owned())
                            }
                            ReturnTypeKind::ShouldTrustAnnotation { .. } => {
                                // For trusted annotations, get from AST
                                fun.def
                                    .returns
                                    .as_ref()
                                    .map(|ann| module.code_at(ann.range()).to_owned())
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                };

                // Get parameters
                let mut parameters = Vec::new();
                let all_params = Self::extract_parameters(&fun.def.parameters);
                let mut counted_parameters = 0usize;
                let mut annotated_parameters = 0usize;
                let mut skip_next_method_parameter = is_method;

                for param in all_params {
                    let param_name = module.code_at(param.name.range());
                    let mut should_skip = false;
                    if skip_next_method_parameter {
                        skip_next_method_parameter = false;
                        should_skip = matches!(param_name, "self" | "cls" | "mcs");
                    }

                    let param_annotation = param
                        .annotation
                        .as_ref()
                        .map(|ann| module.code_at(ann.range()).to_owned());

                    if !should_skip {
                        counted_parameters += 1;
                        if param_annotation.is_some() {
                            annotated_parameters += 1;
                        }
                    }

                    parameters.push(Parameter {
                        name: param_name.to_owned(),
                        annotation: param_annotation,
                        location: Self::range_to_location(module, param.range),
                    });
                }
                coverage.record_function(
                    counted_parameters,
                    annotated_parameters,
                    return_annotation.is_some(),
                );
                functions.push(Function {
                    name: func_name.to_owned(),
                    return_annotation,
                    parameters,
                    location,
                    is_method,
                });
            }
        }
        (functions, coverage)
    }

    fn run_inner(
        files_to_check: Box<dyn Includes>,
        config_finder: pyrefly_config::finder::ConfigFinder,
        html_report: Option<PathBuf>,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let state = State::new(config_finder);
        let holder = Forgetter::new(state, false);
        let handles = Handles::new(expanded_file_list);
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
            true,
        );

        let transaction = forgetter.as_mut();
        let (handles, _, sourcedb_errors) = handles.all(holder.as_ref().config_finder());

        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            return Err(anyhow::anyhow!("Failed to query sourcedb."));
        }

        let mut report: HashMap<String, FileReport> = HashMap::new();
        let mut total_coverage = CoverageCounters::default();

        for handle in handles {
            transaction.run(&[handle.dupe()], Require::Everything);

            if let Some(bindings) = transaction.get_bindings(&handle)
                && let Some(module) = transaction.get_module_info(&handle)
            {
                let line_count = module.lined_buffer().line_index().line_count();
                let (functions, coverage) = Self::parse_functions(&module, bindings);
                let suppressions = Self::parse_suppressions(&module);
                total_coverage.absorb(&coverage);

                report.insert(
                    handle.path().as_path().display().to_string(),
                    FileReport {
                        line_count,
                        functions,
                        suppressions,
                        coverage: coverage.to_output(),
                    },
                );
            }
        }

        // Output JSON
        let output = CoverageReport {
            files: report,
            summary: total_coverage.to_output(),
        };

        if let Some(dir) = html_report {
            Self::write_html_report(&dir, &output)?;
        }

        let json = serde_json::to_string_pretty(&output)?;
        println!("{}", json);

        Ok(CommandExitStatus::Success)
    }

    fn write_html_report(out_dir: &Path, report: &CoverageReport) -> anyhow::Result<()> {
        fs::create_dir_all(out_dir)?;
        let document = Self::build_html_document(report);
        let mut file = File::create(out_dir.join("index.html"))?;
        file.write_all(document.as_bytes())?;
        Ok(())
    }

    fn build_html_document(report: &CoverageReport) -> String {
        let mut html = String::new();
        let summary = &report.summary;

        let mut files: Vec<(&String, &FileReport)> = report.files.iter().collect();
        files.sort_by(|a, b| a.0.cmp(b.0));

        let _ = write!(
            html,
            r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Pyrefly Type Annotation Coverage</title>
  <style>
    :root {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111;
      background-color: #fafafa;
    }}
    body {{
      margin: 2rem;
      line-height: 1.5;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 0.5rem;
      text-align: left;
    }}
    th {{
      background: #f0f0f0;
    }}
    .status-good {{
      color: #1a7f37;
      font-weight: 600;
    }}
    .status-warn {{
      color: #b88700;
      font-weight: 600;
    }}
    .status-bad {{
      color: #c93c37;
      font-weight: 600;
    }}
    details {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 6px;
      margin-bottom: 1rem;
      padding: 0.5rem 1rem;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    code {{
      background: #f3f3f3;
      border-radius: 4px;
      padding: 0 0.25rem;
    }}
  </style>
</head>
<body>
  <h1>Pyrefly Type Annotation Coverage</h1>
"#
        );

        let _ = write!(
            html,
            r#"<section>
  <h2>Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Total functions</td><td>{}</td></tr>
      <tr><td>Fully typed functions</td><td>{} ({})</td></tr>
      <tr><td>Partially typed functions</td><td>{}</td></tr>
      <tr><td>Untyped functions</td><td>{}</td></tr>
      <tr><td>Annotated parameters</td><td>{} / {}</td></tr>
      <tr><td>Annotated returns</td><td>{} / {}</td></tr>
      <tr><td>Slot coverage</td><td>{}</td></tr>
      <tr><td>Fully typed coverage</td><td>{}</td></tr>
    </tbody>
  </table>
</section>
"#,
            summary.total_functions,
            summary.fully_typed_functions,
            Self::format_percent(summary.fully_typed_function_percent),
            summary.partially_typed_functions,
            summary.untyped_functions,
            summary.annotated_parameters,
            summary.total_parameters,
            summary.annotated_returns,
            summary.total_returns,
            Self::format_percent(summary.slot_coverage_percent),
            Self::format_percent(summary.fully_typed_function_percent)
        );

        let _ = write!(
            html,
            r#"<section>
  <h2>Files</h2>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Functions</th>
        <th>Fully typed</th>
        <th>Partially typed</th>
        <th>Untyped</th>
        <th>Slot coverage</th>
        <th>Fully typed %</th>
      </tr>
    </thead>
    <tbody>
"#
        );

        for (path, file_report) in &files {
            let coverage = &file_report.coverage;
            let _ = write!(
                html,
                r#"<tr>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
</tr>
"#,
                Self::html_escape(path),
                coverage.total_functions,
                coverage.fully_typed_functions,
                coverage.partially_typed_functions,
                coverage.untyped_functions,
                Self::format_percent(coverage.slot_coverage_percent),
                Self::format_percent(coverage.fully_typed_function_percent)
            );
        }

        html.push_str(
            "    </tbody>\n  </table>\n</section>\n<section>\n  <h2>Per-file Details</h2>\n",
        );

        for (path, file_report) in files {
            let coverage = &file_report.coverage;
            let _ = write!(
                html,
                r#"  <details>
    <summary>{} &mdash; {} slot coverage</summary>
    <div>
      <p><strong>Functions:</strong> {} | <strong>Fully typed:</strong> {} | <strong>Partially typed:</strong> {} | <strong>Untyped:</strong> {}</p>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Signature</th>
            <th>Return</th>
            <th>Status</th>
            <th>Location</th>
          </tr>
        </thead>
        <tbody>
"#,
                Self::html_escape(path),
                Self::format_percent(coverage.slot_coverage_percent),
                coverage.total_functions,
                coverage.fully_typed_functions,
                coverage.partially_typed_functions,
                coverage.untyped_functions
            );

            if file_report.functions.is_empty() {
                html.push_str(
                    r#"<tr><td colspan="5"><em>No functions found in this file.</em></td></tr>"#,
                );
            } else {
                for function in &file_report.functions {
                    let (status, _, _, _) = Self::classify_function(function);
                    let status_class = match status {
                        FunctionCoverageStatus::FullyTyped => "status-good",
                        FunctionCoverageStatus::PartiallyTyped => "status-warn",
                        FunctionCoverageStatus::Untyped => "status-bad",
                    };
                    let status_label = match status {
                        FunctionCoverageStatus::FullyTyped => "Fully typed",
                        FunctionCoverageStatus::PartiallyTyped => "Partially typed",
                        FunctionCoverageStatus::Untyped => "Untyped",
                    };
                    let signature = Self::format_parameter_list(function);
                    let location = Self::format_location(&function.location);
                    let return_annotation = function
                        .return_annotation
                        .as_deref()
                        .map(Self::html_escape)
                        .unwrap_or_else(|| "&mdash;".to_owned());

                    let _ = write!(
                        html,
                        r#"<tr>
  <td><code>{}</code></td>
  <td>{}</td>
  <td>{}</td>
  <td class="{}">{}</td>
  <td>{}</td>
</tr>
"#,
                        Self::html_escape(&function.name),
                        signature,
                        return_annotation,
                        status_class,
                        status_label,
                        Self::html_escape(&location)
                    );
                }
            }

            html.push_str("        </tbody>\n      </table>\n    </div>\n  </details>\n");
        }

        html.push_str("</section>\n</body>\n</html>\n");
        html
    }

    fn html_escape(text: &str) -> String {
        let mut escaped = String::with_capacity(text.len());
        for ch in text.chars() {
            match ch {
                '&' => escaped.push_str("&amp;"),
                '<' => escaped.push_str("&lt;"),
                '>' => escaped.push_str("&gt;"),
                '"' => escaped.push_str("&quot;"),
                '\'' => escaped.push_str("&#39;"),
                _ => escaped.push(ch),
            }
        }
        escaped
    }

    fn format_percent(value: f64) -> String {
        format!("{value:.2}%")
    }

    fn classify_function(function: &Function) -> (FunctionCoverageStatus, usize, usize, bool) {
        let (total_parameters, annotated_parameters, has_return_annotation) =
            Self::function_annotation_stats(function);
        let missing_parameters = total_parameters.saturating_sub(annotated_parameters);
        let status = if missing_parameters == 0 && has_return_annotation {
            FunctionCoverageStatus::FullyTyped
        } else if annotated_parameters == 0 && !has_return_annotation {
            FunctionCoverageStatus::Untyped
        } else {
            FunctionCoverageStatus::PartiallyTyped
        };
        (
            status,
            total_parameters,
            annotated_parameters,
            has_return_annotation,
        )
    }

    fn function_annotation_stats(function: &Function) -> (usize, usize, bool) {
        let mut total_parameters = 0usize;
        let mut annotated_parameters = 0usize;
        let mut skip_method_parameter = function.is_method;

        for param in &function.parameters {
            let mut should_skip = false;
            if skip_method_parameter {
                skip_method_parameter = false;
                should_skip = matches!(param.name.as_str(), "self" | "cls" | "mcs");
            }
            if should_skip {
                continue;
            }
            total_parameters += 1;
            if param.annotation.is_some() {
                annotated_parameters += 1;
            }
        }

        (
            total_parameters,
            annotated_parameters,
            function.return_annotation.is_some(),
        )
    }

    fn format_parameter_list(function: &Function) -> String {
        if function.parameters.is_empty() {
            return "&mdash;".to_owned();
        }
        let mut rendered = String::new();
        for (idx, param) in function.parameters.iter().enumerate() {
            if idx > 0 {
                rendered.push_str(", ");
            }
            let mut part = param.name.clone();
            if let Some(annotation) = &param.annotation {
                part.push_str(": ");
                part.push_str(annotation);
            }
            rendered.push_str(&Self::html_escape(&part));
        }
        rendered
    }

    fn format_location(location: &Location) -> String {
        format!(
            "{}:{}-{}:{}",
            location.start.line, location.start.column, location.end.line, location.end.column
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected} got {actual}"
        );
    }

    #[test]
    fn coverage_counters_classify_functions() {
        let mut counters = CoverageCounters::default();
        counters.record_function(2, 2, true);
        counters.record_function(2, 1, true);
        counters.record_function(0, 0, false);

        let output = counters.to_output();
        assert_eq!(output.total_functions, 3);
        assert_eq!(output.fully_typed_functions, 1);
        assert_eq!(output.partially_typed_functions, 1);
        assert_eq!(output.untyped_functions, 1);
        approx(output.slot_coverage_percent, (5.0 / 7.0) * 100.0);
        approx(output.fully_typed_function_percent, (1.0 / 3.0) * 100.0);
    }

    #[test]
    fn coverage_counters_zero_state_defaults_to_full() {
        let counters = CoverageCounters::default();
        let output = counters.to_output();
        assert_eq!(output.total_functions, 0);
        approx(output.slot_coverage_percent, 100.0);
        approx(output.fully_typed_function_percent, 100.0);
    }

    #[test]
    fn function_annotation_stats_skip_method_receiver() {
        let function = Function {
            name: "MyClass.method".to_owned(),
            return_annotation: Some("None".to_owned()),
            parameters: vec![
                Parameter {
                    name: "self".to_owned(),
                    annotation: None,
                    location: Location {
                        start: Position { line: 1, column: 0 },
                        end: Position { line: 1, column: 4 },
                    },
                },
                Parameter {
                    name: "value".to_owned(),
                    annotation: Some("int".to_owned()),
                    location: Location {
                        start: Position { line: 1, column: 6 },
                        end: Position {
                            line: 1,
                            column: 16,
                        },
                    },
                },
            ],
            location: Location {
                start: Position { line: 1, column: 0 },
                end: Position { line: 3, column: 0 },
            },
            is_method: true,
        };

        let (total, annotated, has_return) = ReportArgs::function_annotation_stats(&function);
        assert_eq!(total, 1);
        assert_eq!(annotated, 1);
        assert!(has_return);
    }

    #[test]
    fn html_report_contains_function_details() {
        let function = Function {
            name: "foo".to_owned(),
            return_annotation: Some("str".to_owned()),
            parameters: vec![
                Parameter {
                    name: "self".to_owned(),
                    annotation: None,
                    location: Location {
                        start: Position { line: 1, column: 0 },
                        end: Position { line: 1, column: 4 },
                    },
                },
                Parameter {
                    name: "value".to_owned(),
                    annotation: Some("list[int]".to_owned()),
                    location: Location {
                        start: Position { line: 1, column: 6 },
                        end: Position {
                            line: 1,
                            column: 24,
                        },
                    },
                },
            ],
            location: Location {
                start: Position { line: 1, column: 0 },
                end: Position { line: 2, column: 0 },
            },
            is_method: true,
        };

        let coverage = CoverageStatsOutput {
            total_functions: 1,
            fully_typed_functions: 1,
            partially_typed_functions: 0,
            untyped_functions: 0,
            total_parameters: 1,
            annotated_parameters: 1,
            total_returns: 1,
            annotated_returns: 1,
            slot_coverage_percent: 100.0,
            fully_typed_function_percent: 100.0,
        };

        let mut files = HashMap::new();
        files.insert(
            "pkg/foo.py".to_owned(),
            FileReport {
                line_count: 10,
                functions: vec![function],
                suppressions: vec![],
                coverage: coverage.clone(),
            },
        );

        let report = CoverageReport {
            files,
            summary: coverage,
        };

        let html = ReportArgs::build_html_document(&report);
        assert!(html.contains("Pyrefly Type Annotation Coverage"));
        assert!(html.contains("pkg/foo.py"));
        assert!(html.contains("Fully typed"));
        assert!(html.contains("list[int]"));
        assert!(html.contains("100.00%"));
    }
}
