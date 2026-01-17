/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::path::Path;

pub fn is_cython_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("pyx") | Some("pxd") | Some("pxi")
    )
}

pub fn sanitize_for_parse(contents: &str) -> Cow<'_, str> {
    let mut out = String::with_capacity(contents.len());
    let mut pending_decorator = false;

    for chunk in contents.split_inclusive('\n') {
        let (line, newline) = split_line(chunk);
        let trimmed = line.trim_start_matches([' ', '\t']);

        if trimmed.is_empty() || trimmed.starts_with('#') {
            out.push_str(line);
            out.push_str(newline);
            pending_decorator = false;
            continue;
        }

        if trimmed.starts_with('@') {
            out.push_str(line);
            out.push_str(newline);
            pending_decorator = true;
            continue;
        }

        if let Some(sanitized) = sanitize_cython_line(line, pending_decorator) {
            out.push_str(&sanitized);
        } else {
            out.push_str(line);
        }
        out.push_str(newline);
        pending_decorator = false;
    }

    Cow::Owned(out)
}

fn split_line(line: &str) -> (&str, &str) {
    if let Some(stripped) = line.strip_suffix('\n') {
        if let Some(without_cr) = stripped.strip_suffix('\r') {
            return (without_cr, "\r\n");
        }
        return (stripped, "\n");
    }
    (line, "")
}

fn sanitize_cython_line(line: &str, pending_decorator: bool) -> Option<String> {
    let trimmed = line.trim_start_matches([' ', '\t']);
    let indent_len = line.len() - trimmed.len();
    let indent = &line[..indent_len];
    let available = line.len() - indent_len;

    let trimmed_no_comment = trimmed.split('#').next().unwrap_or("").trim_end();
    if trimmed_no_comment.is_empty() {
        return None;
    }

    let is_cython_stmt = is_cython_statement(trimmed_no_comment);
    if !is_cython_stmt {
        return None;
    }

    let block = trimmed_no_comment.ends_with(':');
    let mut replacement =
        if starts_kw(trimmed_no_comment, "cdef") || starts_kw(trimmed_no_comment, "cpdef") {
            replacement_for_cdef(trimmed_no_comment, block)
        } else {
            replacement_for_other(block)
        };

    if pending_decorator && !is_def_or_class(&replacement) {
        replacement = replacement_for_decorator(block);
    }

    let replacement = fit_replacement(replacement, available, block, pending_decorator);
    Some(pad_replacement(indent, available, &replacement))
}

fn is_cython_statement(trimmed: &str) -> bool {
    starts_kw(trimmed, "cdef")
        || starts_kw(trimmed, "cpdef")
        || starts_kw(trimmed, "ctypedef")
        || starts_kw(trimmed, "cimport")
        || starts_kw(trimmed, "include")
        || starts_kw(trimmed, "DEF")
        || starts_kw(trimmed, "IF")
        || starts_kw(trimmed, "ELIF")
        || starts_kw(trimmed, "ELSE")
        || is_from_cimport(trimmed)
}

fn starts_kw(s: &str, kw: &str) -> bool {
    if !s.starts_with(kw) {
        return false;
    }
    match s[kw.len()..].chars().next() {
        None => true,
        Some(c) => c.is_whitespace() || c == ':' || c == '(',
    }
}

fn is_from_cimport(s: &str) -> bool {
    let mut parts = s.split_whitespace();
    if parts.next() != Some("from") {
        return false;
    }
    parts.any(|part| part == "cimport")
}

fn replacement_for_cdef(trimmed: &str, block: bool) -> String {
    if (trimmed.starts_with("cdef class") || trimmed.starts_with("cpdef class"))
        && let Some(name) = extract_name_after_keyword(trimmed, "class")
    {
        if block {
            return format!("class {name}:");
        }
        return format!("class {name}: pass");
    }

    if trimmed.contains('(')
        && let Some(name) = extract_name_before_paren(trimmed)
    {
        if block {
            return format!("def {name}():");
        }
        return format!("def {name}(): pass");
    }

    replacement_for_other(block)
}

fn replacement_for_other(block: bool) -> String {
    if block {
        "if True:".to_owned()
    } else {
        "pass".to_owned()
    }
}

fn replacement_for_decorator(block: bool) -> String {
    if block {
        "def _():".to_owned()
    } else {
        "def _(): pass".to_owned()
    }
}

fn is_def_or_class(s: &str) -> bool {
    s.starts_with("def ") || s.starts_with("class ")
}

fn fit_replacement(
    preferred: String,
    available: usize,
    block: bool,
    needs_def_or_class: bool,
) -> String {
    if preferred.len() <= available {
        return preferred;
    }

    if needs_def_or_class {
        let inline_def = "def _(): pass";
        let block_def = "def _():";
        let inline_class = "class _: pass";
        let block_class = "class _:";
        if !block && inline_def.len() <= available {
            return inline_def.to_owned();
        }
        if block_def.len() <= available {
            return block_def.to_owned();
        }
        if !block && inline_class.len() <= available {
            return inline_class.to_owned();
        }
        if block_class.len() <= available {
            return block_class.to_owned();
        }
    }

    if block && "if True:".len() <= available {
        return "if True:".to_owned();
    }
    if "pass".len() <= available {
        return "pass".to_owned();
    }
    String::new()
}

fn pad_replacement(indent: &str, available: usize, replacement: &str) -> String {
    let mut out = String::with_capacity(indent.len() + available);
    out.push_str(indent);
    out.push_str(replacement);
    if replacement.len() < available {
        out.push_str(&" ".repeat(available - replacement.len()));
    }
    out
}

fn extract_name_after_keyword<'a>(line: &'a str, keyword: &str) -> Option<&'a str> {
    let idx = line.find(keyword)?;
    let mut iter = line[idx + keyword.len()..].split_whitespace();
    let token = iter.next()?;
    extract_ident_prefix(token)
}

fn extract_name_before_paren<'a>(line: &'a str) -> Option<&'a str> {
    let before = line.split('(').next()?;
    let token = before.split_whitespace().last()?;
    extract_ident_prefix(token)
}

fn extract_ident_prefix(token: &str) -> Option<&str> {
    let bytes = token.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut end = 0;
    for (idx, &b) in bytes.iter().enumerate() {
        let is_valid = if idx == 0 {
            b == b'_' || b.is_ascii_lowercase() || b.is_ascii_uppercase()
        } else {
            b == b'_' || b.is_ascii_lowercase() || b.is_ascii_uppercase() || b.is_ascii_digit()
        };
        if !is_valid {
            break;
        }
        end = idx + 1;
    }
    if end == 0 { None } else { Some(&token[..end]) }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::ast::Ast;
    use crate::module::Module;
    use crate::module_name::ModuleName;
    use crate::module_path::ModulePath;

    #[test]
    fn test_parse_cython_module() {
        let contents = r#"
@decorator
cdef int add(int a, int b):
    return a + b

cdef class Foo:
    cdef int x
    cpdef int bar(self, int y):
        return y

cdef extern from "math.h":
    cdef double sin(double)

cimport numpy as cnp
ctypedef unsigned long size_t
DEF FLAG = 1
IF FLAG:
    cdef int y
"#;
        let module = Module::new(
            ModuleName::unknown(),
            ModulePath::memory(PathBuf::from("test.pyx")),
            Arc::new(contents.to_owned()),
        );
        let (_ast, parse_errors, unsupported) = Ast::parse_module(&module);
        assert!(
            parse_errors.is_empty(),
            "Expected no parse errors, got: {parse_errors:?}"
        );
        assert!(
            unsupported.is_empty(),
            "Expected no unsupported syntax errors, got: {unsupported:?}"
        );
    }
}
