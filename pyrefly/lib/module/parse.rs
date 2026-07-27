/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_python::sys_info::PythonVersion;
use ruff_python_ast::ModModule;
use ruff_python_ast::token::Tokens;

use crate::config::error_kind::ErrorKind;
use crate::cython;
use crate::error::collector::ErrorCollector;
use crate::module::module_info::ModuleInfo;

pub fn module_parse(
    module_info: &ModuleInfo,
    contents: &str,
    version: PythonVersion,
    errors: &ErrorCollector,
    keep_tokens: bool,
) -> (ModModule, Option<Tokens>) {
    if cython::is_cython_module(module_info) {
        let syntax_error_ranges = cython::syntax_error_ranges(contents);
        for range in &syntax_error_ranges {
            errors
                .error_builder(
                    *range,
                    ErrorKind::ParseError,
                    "Cython parse error".to_owned(),
                )
                .emit();
        }
        if !syntax_error_ranges.is_empty() {
            let empty = Ast::parse_with_version("", version, module_info.source_type())
                .0
                .into_syntax();
            return (empty, None);
        }
        let lowered = cython::lower_to_python(contents);
        let mut module =
            Ast::parse_with_version(&lowered.source, version, module_info.source_type())
                .0
                .into_syntax();
        let mut prelude =
            Ast::parse_with_version(&lowered.prelude, version, module_info.source_type())
                .0
                .into_syntax();
        prelude.body.append(&mut module.body);
        module.body = prelude.body;
        return (module, None);
    }

    let (parsed, parse_errors, unsupported_syntax_errors) =
        Ast::parse_with_version(contents, version, module_info.source_type());
    for err in parse_errors {
        errors
            .error_builder(
                err.location,
                ErrorKind::ParseError,
                format!("Parse error: {}", err.error),
            )
            .emit();
    }
    for err in unsupported_syntax_errors {
        errors
            .error_builder(err.range, ErrorKind::InvalidSyntax, format!("{err}"))
            .emit();
    }

    let tokens = if keep_tokens {
        Some(parsed.tokens().clone())
    } else {
        None
    };

    (parsed.into_syntax(), tokens)
}
