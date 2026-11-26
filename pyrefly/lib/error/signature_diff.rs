/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Range;

use annotate_snippets::AnnotationKind;
use annotate_snippets::Group;
use annotate_snippets::Level as SnippetLevel;
use annotate_snippets::Renderer as SnippetRenderer;
use annotate_snippets::Snippet as SnippetBlock;

use crate::solver::solver::ParamsSubsetError;

/// Parse byte ranges for parameters and return type from a signature string
/// produced by `TypeDisplayContext` with `SignatureHelp` mode.
///
/// Expects format like `def foo(self: T, a: int) -> None: ...`.
/// Returns `(params_range, return_type_range)` as byte ranges, or `None`
/// if the format doesn't match. Callers should fall back gracefully on `None`
/// since the display format could change or contain edge cases we don't handle.
fn signature_parts(sig: &str) -> Option<(Range<usize>, Range<usize>)> {
    let open = sig.find('(')?;

    // Find closing ')' respecting nested parentheses
    let mut balance = 0;
    let mut close = None;
    for (i, c) in sig[open..].char_indices() {
        match c {
            '(' => balance += 1,
            ')' => {
                balance -= 1;
                if balance == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;

    let params = (open + 1)..close;

    // Find " -> " after close
    let arrow_search_start = close;
    let arrow = sig[arrow_search_start..].find(" -> ")? + arrow_search_start;

    let ret_start = arrow + " -> ".len();
    let ret_end = if let Some(pos) = sig[ret_start..].rfind(": ...") {
        ret_start + pos
    } else if let Some(pos) = sig[ret_start..].rfind(':') {
        ret_start + pos
    } else {
        sig.len()
    };
    Some((params, ret_start..ret_end))
}

fn shift(range: Range<usize>, offset: usize) -> Range<usize> {
    (range.start + offset)..(range.end + offset)
}

/// Find the UTF-8-safe byte ranges where two strings differ, using longest
/// common prefix and suffix. Returns `None` if the strings are equal.
///
/// The returned ranges highlight the "differing middle" of each string.
/// When one string is a strict prefix/suffix of the other, a minimal
/// single-character range is returned to ensure there's always something to
/// annotate.
fn diff_ranges(expected: &str, found: &str) -> Option<(Range<usize>, Range<usize>)> {
    if expected == found {
        return None;
    }

    let expected_char_len = expected.chars().count();
    let found_char_len = found.chars().count();

    // Count matching characters from the front.
    let lcp = expected
        .chars()
        .zip(found.chars())
        .take_while(|(a, b)| a == b)
        .count();

    // Count matching characters from the end, not overlapping with the prefix.
    let max_suffix = std::cmp::min(expected_char_len - lcp, found_char_len - lcp);
    let lcs = expected
        .chars()
        .rev()
        .zip(found.chars().rev())
        .take(max_suffix)
        .take_while(|(a, b)| a == b)
        .count();

    let expected_end = expected_char_len - lcs;
    let found_end = found_char_len - lcs;

    // Collect byte offsets for each character index in a single pass per string.
    // Index `i` gives the byte offset of the `i`-th character; the final entry
    // is the total byte length, so lookups for `n == char_count` work too.
    let byte_offsets = |s: &str| -> Vec<usize> {
        s.char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(s.len()))
            .collect()
    };
    let expected_offsets = byte_offsets(expected);
    let found_offsets = byte_offsets(found);

    let expected_span = if expected_end > lcp {
        expected_offsets[lcp]..expected_offsets[expected_end]
    } else {
        // The expected params are a prefix of the found params (or vice versa).
        // Point at the first character after the shared prefix, which in the
        // full source corresponds to the closing `)` or `,` — indicating where
        // parameters are missing or extra. Clamp to the string length to avoid
        // producing an out-of-bounds range when the entire string is a prefix
        // (e.g., for Callable types whose return type ends at the string boundary).
        let next = (lcp + 1).min(expected_char_len);
        expected_offsets[lcp]..expected_offsets[next]
    };
    let found_span = if found_end > lcp {
        found_offsets[lcp]..found_offsets[found_end]
    } else {
        let next = (lcp + 1).min(found_char_len);
        found_offsets[lcp]..found_offsets[next]
    };
    Some((expected_span, found_span))
}

/// A signature as displayed in a signature diff.
pub struct DisplayedSignature<'a> {
    pub text: &'a str,
    /// The display of each parameter, in the order the parameter list displays them. The
    /// parameter indices of a `ParamsSubsetError` point into this list.
    pub params: &'a [String],
}

/// Find the byte range of parameter `index` within the parameters of `sig`, which occupy
/// `params_range` of its text. The displayed parameters appear in order, separated only by
/// commas and the `/` and `*` markers, so searching for each one after the end of the previous
/// one finds its occurrence in the parameter list.
fn param_range(
    sig: &DisplayedSignature,
    params_range: &Range<usize>,
    index: usize,
) -> Option<Range<usize>> {
    let mut start = params_range.start;
    for param in sig.params.get(..=index)? {
        start += sig.text[start..params_range.end].find(param.as_str())?;
        start += param.len();
    }
    Some((start - sig.params[index].len())..start)
}

/// The parts of two signatures that fail the assignability check.
pub enum SignatureFailure<'a> {
    /// The parameters fail as `params` describes, and the return types also fail if
    /// `return_type` is set.
    Params {
        params: &'a ParamsSubsetError,
        return_type: bool,
    },
    /// The parameters are compatible, but the return types are not.
    Return,
}

type Spans = Vec<(Range<usize>, &'static str)>;

fn join_spans(a: Option<(Spans, Spans)>, b: Option<(Spans, Spans)>) -> Option<(Spans, Spans)> {
    match (a, b) {
        (Some((mut exp_spans, mut found_spans)), Some((exp_more, found_more))) => {
            exp_spans.extend(exp_more);
            found_spans.extend(found_more);
            Some((exp_spans, found_spans))
        }
        (a, b) => a.or(b),
    }
}

/// The byte ranges of the expected and found signatures to annotate, with their labels.
fn mismatch_spans(
    expected: &DisplayedSignature,
    found: &DisplayedSignature,
    failure: Option<&SignatureFailure>,
) -> Option<(Spans, Spans)> {
    let (expected_params, expected_ret) = signature_parts(expected.text)?;
    let (found_params, found_ret) = signature_parts(found.text)?;
    let diff = |expected_range: Range<usize>, found_range: Range<usize>, label| {
        let (exp_span, found_span) = diff_ranges(
            &expected.text[expected_range.clone()],
            &found.text[found_range.clone()],
        )?;
        Some((
            vec![(shift(exp_span, expected_range.start), label)],
            vec![(shift(found_span, found_range.start), label)],
        ))
    };
    let params_diff = || diff(expected_params.clone(), found_params.clone(), "parameters");
    let return_diff = || diff(expected_ret.clone(), found_ret.clone(), "return type");
    match failure {
        None => join_spans(params_diff(), return_diff()),
        Some(SignatureFailure::Return) => return_diff(),
        Some(SignatureFailure::Params {
            params:
                ParamsSubsetError {
                    got_index,
                    want_index,
                    ..
                },
            return_type,
        }) => {
            // `None` if the parameter cannot be located, `Some(None)` if there is no parameter.
            let locate = |sig, params_range, index: Option<usize>| match index {
                Some(index) => param_range(sig, params_range, index).map(Some),
                None => Some(None),
            };
            let params_spans = match locate(expected, &expected_params, *want_index).zip(locate(
                found,
                &found_params,
                *got_index,
            )) {
                // Narrow the annotation to the differing part of the two parameters.
                Some((Some(exp_range), Some(found_range))) => {
                    diff(exp_range.clone(), found_range.clone(), "parameter").or(Some((
                        vec![(exp_range, "parameter")],
                        vec![(found_range, "parameter")],
                    )))
                }
                Some((Some(exp_range), None)) => Some((vec![(exp_range, "parameter")], Vec::new())),
                Some((None, Some(found_range))) => {
                    Some((Vec::new(), vec![(found_range, "parameter")]))
                }
                // The failure is not attributed to individual parameters, or they cannot be
                // located in the displayed signatures.
                Some((None, None)) | None => params_diff(),
            };
            join_spans(params_spans, return_type.then(return_diff).flatten())
        }
    }
}

/// Render a visual diff between two function signatures, highlighting the parts that make the
/// found signature incompatible with the expected one.
///
/// When `failure` says which part of the signatures failed the assignability check, only that
/// part is highlighted. Otherwise, or when the failing parameters cannot be located in the
/// displayed signatures, the textual differences between the signatures are highlighted.
///
/// Uses `annotate_snippets` to produce caret-underline annotations, then
/// strips the line-number gutter from the rendered output. Returns `None` if
/// signature parsing fails or if there are no differences to annotate.
///
/// The gutter-stripping logic depends on the current output format of
/// `SnippetRenderer::plain()`. If that format changes upstream, this
/// post-processing may need to be updated.
pub fn render_signature_diff(
    expected: &DisplayedSignature,
    found: &DisplayedSignature,
    failure: Option<&SignatureFailure>,
) -> Option<Vec<String>> {
    let (expected_spans, found_spans) = mismatch_spans(expected, found, failure)?;

    let expected_prefix = "expected: ";
    let found_prefix = "found:    ";
    let expected_line_text = format!("{expected_prefix}{}", expected.text);
    let found_line_text = format!("{found_prefix}{}", found.text);

    // We strip line numbers from the output, so the starting line number doesn't matter
    // for the final text, but it's used by the renderer.
    let line_start = 1;
    let mut source = expected_line_text.clone();
    source.push('\n');
    source.push_str(&found_line_text);
    let found_offset = expected_line_text.len() + 1;

    let annotations = expected_spans
        .into_iter()
        .map(|(span, label)| (shift(span, expected_prefix.len()), label))
        .chain(
            found_spans
                .into_iter()
                .map(|(span, label)| (shift(span, found_offset + found_prefix.len()), label)),
        )
        .map(|(span, label)| AnnotationKind::Primary.span(span).label(label));

    // Without `fold(false)`, a line without annotations would be omitted.
    let mut snippet = SnippetBlock::source(&source)
        .line_start(line_start)
        .fold(false);
    for ann in annotations {
        snippet = snippet.annotation(ann);
    }
    let message = [Group::with_level(SnippetLevel::ERROR.no_name()).element(snippet)];
    let rendered = SnippetRenderer::plain().render(&message).to_string();
    let mut lines: Vec<String> = Vec::new();
    lines.push("Signature mismatch:".to_owned());
    for line in rendered.lines() {
        if let Some(idx) = line.find('|') {
            let (left, right) = line.split_at(idx);
            if left.trim().is_empty() || left.trim().chars().all(|c| c.is_ascii_digit()) {
                let mut trimmed = right.trim_start_matches('|');
                if trimmed.starts_with(' ') {
                    trimmed = &trimmed[1..];
                }
                if trimmed.is_empty() {
                    continue;
                }
                lines.push(trimmed.to_owned());
                continue;
            }
        }
        if !line.trim().is_empty() {
            lines.push(line.to_owned());
        }
    }
    Some(lines)
}

#[cfg(test)]
mod tests {
    use super::DisplayedSignature;
    use crate::test::util::TestEnv;

    /// A signature whose parameters are not located, for tests without a callable subset failure.
    fn signature(text: &str) -> DisplayedSignature<'_> {
        DisplayedSignature { text, params: &[] }
    }

    /// Run a single-module type check and return the error messages.
    fn error_messages(code: &str) -> Vec<String> {
        let (state, handle) = TestEnv::one("main", code).to_state();
        state
            .transaction()
            .get_errors(&[handle("main")])
            .collect_errors()
            .ordinary
            .iter()
            .map(|e| e.msg().to_string())
            .collect()
    }

    /// Integration test verifying the full error message produced by the
    /// override checker includes the signature diff annotation.
    #[test]
    fn test_override_signature_diff_full_message() {
        let messages = error_messages(
            r#"
from abc import ABC

class A(ABC):
    def foo(self, a: int, b: int, c: int):
        raise NotImplementedError()

class B(A):
    def foo(self):
        x = 1
        print(x)
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `() -> None`, which is not assignable to `(a: int, b: int, c: int) -> Unknown`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, a: int, b: int, c: int) -> Unknown: ...
                             ^^^^^^ parameter
  found:    def foo(self: B) -> None: ..."#;
        assert_eq!(messages[0], expected);
    }

    /// Override has too many arguments (inverse of the basic test).
    #[test]
    fn test_signature_diff_too_many_args() {
        let messages = error_messages(
            r#"
class A:
    def foo(self) -> None:
        pass

class B(A):
    def foo(self, x: int, y: str) -> None:
        pass
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: int, y: str) -> None`, which is not assignable to `() -> None`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B) -> None: ...
  found:    def foo(self: B, x: int, y: str) -> None: ...
                             ^^^^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Simple single argument type mismatch.
    #[test]
    fn test_signature_diff_single_param_mismatch() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> None:
        pass

class B(A):
    def foo(self, x: str) -> None:
        pass
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: str) -> None`, which is not assignable to `(x: int) -> None`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> None: ...
                                ^^^ parameter
  found:    def foo(self: B, x: str) -> None: ...
                                ^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Simple return type only mismatch (parameters are identical).
    #[test]
    fn test_signature_diff_return_type_only() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> int:
        return x

class B(A):
    def foo(self, x: int) -> str:
        return ""
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: int) -> str`, which is not assignable to `(x: int) -> int`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> int: ...
                                        ^^^ return type
  found:    def foo(self: B, x: int) -> str: ...
                                        ^^^ return type"#;
        assert_eq!(messages[0], expected);
    }

    /// A narrower return type is a valid override, so only the parameter is annotated.
    /// https://github.com/pyrefly-org/pyrefly/issues/5684
    #[test]
    fn test_signature_diff_compatible_return_type() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: object) -> object: ...

class B(A):
    def foo(self, x: int) -> int: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: int) -> int`, which is not assignable to `(x: object) -> object`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: object) -> object: ...
                                ^^^^^ parameter
  found:    def foo(self: B, x: int) -> int: ...
                                ^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Only `b` is annotated: widening `a` and `c` is a valid override.
    #[test]
    fn test_signature_diff_one_of_several_params() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, a: int, b: int, c: int) -> None: ...

class B(A):
    def foo(self, a: object, b: str, c: object) -> None: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(a: object, b: str, c: object) -> None`, which is not assignable to `(a: int, b: int, c: int) -> None`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, a: int, b: int, c: int) -> None: ...
                                        ^^^ parameter
  found:    def foo(self: B, a: object, b: str, c: object) -> None: ...
                                           ^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// The override adds a required keyword-only parameter that callers of `A.foo` do not pass.
    #[test]
    fn test_signature_diff_extra_required_keyword() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> None: ...

class B(A):
    def foo(self, x: int, *, y: str) -> None: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: int, *, y: str) -> None`, which is not assignable to `(x: int) -> None`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> None: ...
  found:    def foo(self: B, x: int, *, y: str) -> None: ...
                                        ^^^^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// The override drops a keyword-only parameter that callers of `A.foo` may pass.
    #[test]
    fn test_signature_diff_missing_keyword() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, *, x: int, y: int) -> None: ...

class B(A):
    def foo(self, *, x: int) -> None: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(*, x: int) -> None`, which is not assignable to `(*, x: int, y: int) -> None`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, *, x: int, y: int) -> None: ...
                                        ^^^^^^ parameter
  found:    def foo(self: B, *, x: int) -> None: ..."#;
        assert_eq!(messages[0], expected);
    }

    /// A generic method, whose displayed signature lists its type parameters before the
    /// parameters.
    #[test]
    fn test_signature_diff_generic_method() {
        let messages = error_messages(
            r#"
class A:
    def foo[T](self, x: T, y: int) -> T: ...

class B(A):
    def foo[T](self, x: T, y: str) -> T: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `[T](x: T, y: str) -> T`, which is not assignable to `[T](x: T, y: int) -> T`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: T, y: int) -> T: ...
                                      ^^^ parameter
  found:    def foo(self: B, x: T, y: str) -> T: ...
                                      ^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// A classmethod, whose `cls` receiver is bound before the comparison.
    #[test]
    fn test_signature_diff_classmethod() {
        let messages = error_messages(
            r#"
class A:
    @classmethod
    def make(cls, x: int, y: int) -> None: ...

class B(A):
    @classmethod
    def make(cls, x: int, y: str) -> None: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.make` overrides parent class `A` in an inconsistent manner
  `B.make` has type `(x: int, y: str) -> None`, which is not assignable to `(x: int, y: int) -> None`, the type of `A.make`
  Signature mismatch:
  expected: def make(cls: type[B], x: int, y: int) -> None: ...
                                              ^^^ parameter
  found:    def make(cls: type[B], x: int, y: str) -> None: ...
                                              ^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// A staticmethod, which has no receiver.
    #[test]
    fn test_signature_diff_staticmethod() {
        let messages = error_messages(
            r#"
class A:
    @staticmethod
    def make(x: int, y: int) -> object: ...

class B(A):
    @staticmethod
    def make(x: int, y: str) -> int: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.make` overrides parent class `A` in an inconsistent manner
  `B.make` has type `(x: int, y: str) -> int`, which is not assignable to `(x: int, y: int) -> object`, the type of `A.make`
  Signature mismatch:
  expected: def make(x: int, y: int) -> object: ...
                                ^^^ parameter
  found:    def make(x: int, y: str) -> int: ...
                                ^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Overloaded parent method: signature diff should not appear since
    /// overloads have multiple signatures.
    #[test]
    fn test_signature_diff_overloads() {
        let messages = error_messages(
            r#"
from typing import overload

class A:
    @overload
    def foo(self, x: int) -> int: ...
    @overload
    def foo(self, x: str) -> str: ...
    def foo(self, x):
        return x

class B(A):
    def foo(self, x: float) -> float:
        return x
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        // Overloads have multiple signatures, so no signature diff is shown.
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: float) -> float`, which is not assignable to `Overload[
  (x: int) -> int
  (x: str) -> str
]`, the type of `A.foo`"#;
        assert_eq!(messages[0], expected);
    }

    /// Both the parameter and the return types are incompatible, so both are annotated.
    #[test]
    fn test_signature_diff_param_and_return() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> int: ...

class B(A):
    def foo(self, x: str) -> str: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: str) -> str`, which is not assignable to `(x: int) -> int`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> int: ...
                                ^^^     ^^^ return type
                                |
                                parameter
  found:    def foo(self: B, x: str) -> str: ...
                                ^^^     ^^^ return type
                                |
                                parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Both a parameter and the return type are incompatible, but the return types mention a
    /// type parameter, so only the parameter is annotated.
    #[test]
    fn test_signature_diff_generic_return() {
        let messages = error_messages(
            r#"
class A:
    def foo[T](self, x: int, y: T) -> T: ...

class B(A):
    def foo[T](self, x: str, y: T) -> list[T]: ...
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `[T](x: str, y: T) -> list[T]`, which is not assignable to `[T](x: int, y: T) -> T`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int, y: T) -> T: ...
                                ^^^ parameter
  found:    def foo(self: B, x: str, y: T) -> list[T]: ...
                                ^^^ parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Inferred (unannotated) return types: verifies signature diff works
    /// when return types have no explicit annotation.
    #[test]
    fn test_signature_diff_inferred_return() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int):
        return x

class B(A):
    def foo(self, x: str):
        return x
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(x: str) -> str`, which is not assignable to `(x: int) -> int`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> int: ...
                                ^^^     ^^^ return type
                                |
                                parameter
  found:    def foo(self: B, x: str) -> str: ...
                                ^^^     ^^^ return type
                                |
                                parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Lambda override: when the arity differs, the override falls through
    /// to the general BadOverride path, showing the signature diff instead
    /// of a misleading parameter name mismatch.
    #[test]
    fn test_signature_diff_lambda() {
        let messages = error_messages(
            r#"
class A:
    def foo(self, x: int) -> int:
        return x

class B(A):
    foo = lambda self: None
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.foo` overrides parent class `A` in an inconsistent manner
  `B.foo` has type `(self: Unknown) -> None`, which is not assignable to `(x: int) -> int`, the type of `A.foo`
  Signature mismatch:
  expected: def foo(self: B, x: int) -> int: ...
                             ^^^^^^     ^^^ return type
                             |
                             parameter
  found:    (self: Unknown) -> None
             ^^^^^^^^^^^^^     ^^^^ return type
             |
             parameter"#;
        assert_eq!(messages[0], expected);
    }

    /// Named function override: verifies signature diff normalizes the
    /// function name to the attribute name for readability.
    #[test]
    fn test_signature_diff_named_function() {
        let messages = error_messages(
            r#"
def helper(self) -> None:
    pass

class A:
    def method(self, x: int) -> int:
        return x

class B(A):
    method = helper
"#,
        );
        assert_eq!(messages.len(), 1, "Expected one error, got {messages:?}");
        let expected = r#"Class member `B.method` overrides parent class `A` in an inconsistent manner
  `B.method` has type `() -> None`, which is not assignable to `(x: int) -> int`, the type of `A.method`
  Signature mismatch:
  expected: def method(self: B, x: int) -> int: ...
                                ^^^^^^     ^^^ return type
                                |
                                parameter
  found:    def method(self: Unknown) -> None: ...
                                         ^^^^ return type"#;
        assert_eq!(messages[0], expected);
    }

    /// Callable-style signatures (no `def`, no `: ...` suffix) where the
    /// shorter return type is a prefix of the longer one. Previously panicked
    /// because `diff_ranges` produced a span past the end of the source string.
    #[test]
    fn test_render_signature_diff_callable_prefix_return_type() {
        use super::render_signature_diff;
        let expected = "() -> EdgeFoo | EdgeBar";
        let found = "() -> Edge";
        // Should not panic. The diff should highlight the return type difference.
        let result = render_signature_diff(&signature(expected), &signature(found), None);
        assert!(
            result.is_some(),
            "Expected a signature diff for differing return types"
        );
    }

    #[test]
    fn test_render_signature_diff_unicode_literal_return_type() {
        use super::render_signature_diff;

        let expected = "def _money_desc(cls: type[PensionAsset]) -> Literal['累计可领(元)']: ...";
        let found = "def _money_desc(cls: type[PensionAsset]) -> Literal['90岁累计可领(元)']: ...";
        let result = render_signature_diff(&signature(expected), &signature(found), None);
        let lines = result.expect("Expected a signature diff for differing Unicode return types");
        let joined = lines.join("\n");
        assert!(
            joined.contains("return type"),
            "Expected return type annotation in diff, got:\n{joined}"
        );
        assert!(
            !joined.contains("parameters"),
            "Expected no parameter annotation (params are identical), got:\n{joined}"
        );

        let expected = "def _money_desc(cls: type[PensionAsset]) -> Literal['累计可领']: ...";
        let found = "def _money_desc(cls: type[PensionAsset]) -> Literal['累计可累']: ...";
        let result = render_signature_diff(&signature(expected), &signature(found), None);
        let lines = result.expect("Expected a signature diff for differing Unicode return types");
        let joined = lines.join("\n");
        assert!(
            joined.contains("return type"),
            "Expected return type annotation in diff, got:\n{joined}"
        );
        assert!(
            !joined.contains("parameters"),
            "Expected no parameter annotation (params are identical), got:\n{joined}"
        );

        let expected = "def _money_desc(cls: type[PensionAsset]) -> Literal['累计可领']: ...";
        let found = "def _money_desc(cls: type[PensionAsset]) -> Literal['领计可领']: ...";
        let result = render_signature_diff(&signature(expected), &signature(found), None);
        let lines = result.expect("Expected a signature diff for differing Unicode return types");
        let joined = lines.join("\n");
        assert!(
            joined.contains("return type"),
            "Expected return type annotation in diff, got:\n{joined}"
        );
        assert!(
            !joined.contains("parameters"),
            "Expected no parameter annotation (params are identical), got:\n{joined}"
        );
    }
}
