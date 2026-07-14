/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Given a file, record which ignore statements are in it.
//!
//! Given `# type: ignore` we should ignore errors on that line.
//! Originally specified in <https://peps.python.org/pep-0484/>.
//!
//! You can also use the name of the linter, e.g. `# pyright: ignore`,
//! `# pyrefly: ignore`.
//!
//! You can specify a specific error code, e.g. `# type: ignore[invalid-type]`.
//! Note that Pyright will only honor such codes after `# pyright: ignore[code]`.
//!
//! You can also use `# mypy: ignore-errors`, `# pyrefly: ignore-errors`
//! or `# type: ignore` at the beginning of a file to suppress all errors.
//! `# pyrefly: ignore-errors[invalid-type]` suppresses only the listed error
//! codes across the file rather than all errors.
//!
//! For Pyre compatibility we also allow `# pyre-ignore` and `# pyre-fixme`
//! as equivalents to `pyre: ignore`, and `# pyre-ignore-all-errors` as
//! an equivalent to `type: ignore` on its own line.
//!
//! We are permissive with whitespace, allowing `#type:ignore[code]` and
//! `#  type:  ignore  [  code  ]`, but do not allow a space before the colon.

use clap::ValueEnum;
use dupe::Dupe;
use enum_iterator::Sequence;
use pyrefly_util::lined_buffer::LineNumber;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use starlark_map::smallset;

/// Finds the byte offset of the first '#' character that starts a comment, tracking
/// whether we're inside a multi-line triple-quoted string.
///
/// All interesting characters (`#`, `'`, `"`, `\`) are ASCII, so we operate
/// on bytes directly — UTF-8 guarantees these never appear inside multi-byte
/// sequences.
///
/// `in_triple_quote` should be `Some('"')` or `Some('\'')` if the line begins
/// inside an open triple-quoted string from a previous line, or `None` otherwise.
///
/// Returns `(comment_start, new_triple_quote_state)`.
pub fn find_comment_start(
    line: &str,
    in_triple_quote: Option<char>,
) -> (Option<usize>, Option<char>) {
    let mut state = CommentState(
        in_triple_quote
            .map(|quote| {
                vec![Context::String(StringContext {
                    quote: quote as u8,
                    triple: true,
                    interpolated: false,
                })]
            })
            .unwrap_or_default(),
    );
    let comment_start = find_comment_start_stateful(line, &mut state);
    let triple_quote = state.0.iter().rev().find_map(|context| match context {
        Context::String(StringContext {
            quote,
            triple: true,
            ..
        }) => Some(*quote as char),
        _ => None,
    });
    (comment_start, triple_quote)
}

#[derive(Clone, Copy)]
struct StringContext {
    quote: u8,
    triple: bool,
    interpolated: bool,
}

#[derive(Clone, Copy)]
enum Context {
    String(StringContext),
    Interpolation { nesting: u32 },
    FormatSpec,
}

#[derive(Default)]
struct CommentState(Vec<Context>);

/// Whether the quote at `quote` has an f-string or t-string prefix.
fn has_interpolated_prefix(bytes: &[u8], quote: usize) -> bool {
    let has_boundary = |start: usize| {
        start == 0
            || !(bytes[start - 1].is_ascii_alphanumeric()
                || bytes[start - 1] == b'_'
                || !bytes[start - 1].is_ascii())
    };
    (quote >= 2
        && has_boundary(quote - 2)
        && matches!(
            (
                bytes[quote - 2].to_ascii_lowercase(),
                bytes[quote - 1].to_ascii_lowercase()
            ),
            (b'f' | b't', b'r') | (b'r', b'f' | b't')
        ))
        || (quote >= 1
            && has_boundary(quote - 1)
            && matches!(bytes[quote - 1].to_ascii_lowercase(), b'f' | b't'))
}

/// Full string-aware comment finder. The context stack is only allocated for lines
/// containing strings, while the caller retains it to handle multiline f-string
/// interpolations and nested strings.
fn find_comment_start_stateful(line: &str, state: &mut CommentState) -> Option<usize> {
    let bytes = line.as_bytes();

    // Preserve the fast path for the overwhelmingly common case of ordinary code.
    if state.0.is_empty() {
        match bytes
            .iter()
            .position(|&b| b == b'#' || b == b'\'' || b == b'"' || b == b'\\')
        {
            None => return None,
            Some(pos) if bytes[pos] == b'#' => return Some(pos),
            _ => {}
        }
    }

    let mut i = 0;
    while i < bytes.len() {
        match state.0.last().copied() {
            None => match bytes[i] {
                b'"' | b'\'' => {
                    let quote = bytes[i];
                    let triple =
                        bytes.get(i + 1) == Some(&quote) && bytes.get(i + 2) == Some(&quote);
                    state.0.push(Context::String(StringContext {
                        quote,
                        triple,
                        interpolated: has_interpolated_prefix(bytes, i),
                    }));
                    i += if triple { 3 } else { 1 };
                }
                b'#' => return Some(i),
                _ => i += 1,
            },
            Some(Context::String(string)) => {
                if bytes[i] == b'\\' {
                    i += 2;
                } else if bytes[i] == string.quote
                    && (!string.triple
                        || bytes.get(i + 1) == Some(&string.quote)
                            && bytes.get(i + 2) == Some(&string.quote))
                {
                    state.0.pop();
                    i += if string.triple { 3 } else { 1 };
                } else if string.interpolated && bytes[i] == b'{' {
                    if bytes.get(i + 1) == Some(&b'{') {
                        i += 2;
                    } else {
                        state.0.push(Context::Interpolation { nesting: 0 });
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
            Some(Context::FormatSpec) => match bytes[i] {
                b'{' if bytes.get(i + 1) != Some(&b'{') => {
                    state.0.push(Context::Interpolation { nesting: 0 });
                    i += 1;
                }
                b'}' => {
                    state.0.pop();
                    i += 1;
                }
                b'{' => i += 2,
                _ => i += 1,
            },
            Some(Context::Interpolation { nesting }) => match bytes[i] {
                b'"' | b'\'' => {
                    let quote = bytes[i];
                    let triple =
                        bytes.get(i + 1) == Some(&quote) && bytes.get(i + 2) == Some(&quote);
                    state.0.push(Context::String(StringContext {
                        quote,
                        triple,
                        interpolated: has_interpolated_prefix(bytes, i),
                    }));
                    i += if triple { 3 } else { 1 };
                }
                b'#' => {
                    // Suppressions inside an f-string interpolation were historically
                    // ignored. Stop at the Python comment without treating it as one.
                    return None;
                }
                b'(' | b'[' | b'{' => {
                    let last = state.0.len() - 1;
                    state.0[last] = Context::Interpolation {
                        nesting: nesting + 1,
                    };
                    i += 1;
                }
                b')' | b']' => {
                    let last = state.0.len() - 1;
                    state.0[last] = Context::Interpolation {
                        nesting: nesting.saturating_sub(1),
                    };
                    i += 1;
                }
                b'}' => {
                    if nesting == 0 {
                        state.0.pop();
                    } else {
                        let last = state.0.len() - 1;
                        state.0[last] = Context::Interpolation {
                            nesting: nesting - 1,
                        };
                    }
                    i += 1;
                }
                b':' if nesting == 0 => {
                    let last = state.0.len() - 1;
                    state.0[last] = Context::FormatSpec;
                    i += 1;
                }
                _ => i += 1,
            },
        }
    }

    // Ordinary unterminated single-quoted strings do not continue onto the next
    // physical line. An interpolation can continue when it contains multiline code.
    while matches!(
        state.0.last(),
        Some(Context::String(StringContext { triple: false, .. }))
    ) {
        state.0.pop();
    }
    None
}

/// Finds the byte offset of the first '#' character that starts a comment.
/// Returns None if no comment is found or if all '#' are inside strings.
/// Handles escape sequences, single/double quotes, and triple-quoted strings.
///
/// This is string-aware parsing that avoids treating '#' inside strings as comments.
/// For example: `x = "hello # world"  # real comment` correctly identifies the second '#'.
pub fn find_comment_start_in_line(line: &str) -> Option<usize> {
    find_comment_start(line, None).0
}

/// The name of the tool that is being suppressed.
/// Note that the variant names and docstrings are displayed in `pyrefly check --help`.
#[derive(PartialEq, Debug, Clone, Hash, Eq, Dupe, Copy, Sequence)]
#[derive(Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Tool {
    /// Enables `# type: ignore`
    Type,
    /// Enables `# pyrefly: ignore` and `# pyrefly: ignore-errors`
    Pyrefly,
    /// Enables `# pyright: ignore`
    Pyright,
    /// Enables `# mypy: ignore-errors`
    Mypy,
    /// Enables `# ty: ignore`
    Ty,
    /// Enables `# pyre: ignore`, `# pyre-ignore`, `# pyre-fixme`, and `# pyre-ignore-all-errors`
    Pyre,
    /// Enables `# zuban: ignore`
    Zuban,
}

impl Tool {
    /// The maximum length of any tool.
    const MAX_LEN: usize = 7;

    fn from_comment(x: &str) -> Option<Self> {
        match x {
            "type" => Some(Tool::Type),
            "pyrefly" => Some(Tool::Pyrefly),
            "pyre" => Some(Tool::Pyre),
            "pyright" => Some(Tool::Pyright),
            "mypy" => Some(Tool::Mypy),
            "ty" => Some(Tool::Ty),
            "zuban" => Some(Tool::Zuban),
            _ => None,
        }
    }

    pub fn default_enabled() -> SmallSet<Self> {
        smallset! { Self::Type, Self::Pyrefly }
    }

    pub fn all() -> SmallSet<Self> {
        enum_iterator::all::<Self>().collect()
    }
}

/// A simple lexer that deals with the rules around whitespace.
/// As it consumes the string, it will move forward.
struct Lexer<'a>(&'a str);

impl<'a> Lexer<'a> {
    /// The string starts with the given string, return `true` if so.
    fn starts_with(&mut self, x: &str) -> bool {
        match self.0.strip_prefix(x) {
            Some(x) => {
                self.0 = x;
                true
            }
            None => false,
        }
    }

    /// The string starts with `tool:`, return the tool if it does.
    fn starts_with_tool(&mut self) -> Option<Tool> {
        let p = self
            .0
            .as_bytes()
            .iter()
            .take(Tool::MAX_LEN + 1)
            .position(|&c| c == b':')?;
        let tool = Tool::from_comment(&self.0[..p])?;
        self.0 = &self.0[p + 1..];
        Some(tool)
    }

    /// Trim whitespace from the start of the string.
    /// Return `true` if the string was changed.
    fn trim_start(&mut self) -> bool {
        let before = self.0;
        self.0 = self.0.trim_start();
        self.0.len() != before.len()
    }

    /// Return `true` if the string is empty or only whitespace.
    fn blank(&mut self) -> bool {
        self.0.trim_start().is_empty()
    }

    /// Return `true` if the string is at the start of a word boundary.
    /// That means the next char is not something that continues an identifier.
    fn word_boundary(&mut self) -> bool {
        self.0
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric() && c != '-' && c != '_')
    }

    /// Finish and return the rest of the string.
    fn rest(self) -> &'a str {
        self.0
    }
}

#[derive(PartialEq, Debug, Clone, Hash, Eq)]
pub struct Suppression {
    tool: Tool,
    /// The permissible error kinds, use empty Vec to mean any are allowed
    kind: Vec<String>,
    /// The line number where the suppression comment is located.
    /// This may differ from the line the suppression applies to
    /// (e.g., when the comment is on the line above).
    comment_line: LineNumber,
    /// Byte offset within `comment_line` of the `#` that starts the comment.
    comment_offset: usize,
}

impl Suppression {
    /// A blanket suppression for `tool` that matches every error code.
    fn blanket(tool: Tool, comment_line: LineNumber, comment_offset: usize) -> Self {
        Self {
            tool,
            kind: Vec::new(),
            comment_line,
            comment_offset,
        }
    }

    /// Returns the line number where the suppression comment is located.
    pub fn comment_line(&self) -> LineNumber {
        self.comment_line
    }

    /// Returns the byte offset of the comment's `#` within `comment_line`.
    pub fn comment_offset(&self) -> usize {
        self.comment_offset
    }

    /// Returns the error codes that this suppression applies to.
    /// An empty slice means the suppression applies to all error codes.
    pub fn error_codes(&self) -> &[String] {
        &self.kind
    }

    /// Returns the tool that this suppression is for.
    pub fn tool(&self) -> Tool {
        self.tool
    }
}

/// Record the position of lines affected by `# type: ignore[valid-type]` suppressions.
/// For now we don't record the content of the ignore, but we could.
#[derive(Debug, Clone, Default)]
pub struct Ignore {
    /// The line number here represents the line that the suppression applies to,
    /// not the line of the suppression comment.
    ignores: SmallMap<LineNumber, Vec<Suppression>>,
}

impl Ignore {
    pub fn new(code: &str) -> Self {
        Self {
            ignores: Self::parse_ignores(code),
        }
    }

    fn parse_ignores(code: &str) -> SmallMap<LineNumber, Vec<Suppression>> {
        let mut ignores: SmallMap<LineNumber, Vec<Suppression>> = SmallMap::new();
        // If we see a comment on a non-code line, apply it to the next non-comment line.
        let mut pending = Vec::new();
        let mut line = LineNumber::default();
        let mut comment_state = CommentState::default();
        for (idx, line_str) in code.lines().enumerate() {
            let comment_start = find_comment_start_stateful(line_str, &mut comment_state);
            let is_comment_only_line = comment_start
                .is_some_and(|comment_start| line_str[..comment_start].trim_start().is_empty());
            line = LineNumber::from_zero_indexed(idx as u32);
            if !pending.is_empty() && (line_str.is_empty() || !is_comment_only_line) {
                ignores.entry(line).or_default().append(&mut pending);
            }
            let Some(comment_start) = comment_start else {
                continue;
            };
            // We know `#` is at `comment_start`, so the first split is an empty string
            for x in line_str[comment_start..].split('#').skip(1) {
                if let Some(supp) = Self::parse_ignore_comment(x, line, comment_start) {
                    if is_comment_only_line {
                        pending.push(supp);
                    } else {
                        ignores.entry(line).or_default().push(supp);
                    }
                }
            }
        }
        if !pending.is_empty() {
            ignores
                .entry(line.increment())
                .or_default()
                .append(&mut pending);
        }
        ignores
    }

    /// Given the content of a comment, parse it as a suppression.
    /// `comment_line` and `comment_offset` locate the `#` starting the comment.
    fn parse_ignore_comment(
        l: &str,
        comment_line: LineNumber,
        comment_offset: usize,
    ) -> Option<Suppression> {
        let mut lex = Lexer(l);
        lex.trim_start();

        let mut tool = None;
        if let Some(t) = lex.starts_with_tool() {
            lex.trim_start();
            if lex.starts_with("ignore") {
                tool = Some(t);
            }
        } else if lex.starts_with("pyre-ignore") || lex.starts_with("pyre-fixme") {
            tool = Some(Tool::Pyre);
        }
        let tool = tool?;

        // We have seen `type: ignore` or `pyre-ignore`. Now look for `[code]` or the end.
        let gap = lex.trim_start();
        if lex.starts_with("[") {
            let rest = lex.rest();
            let inside = rest.split_once(']').map_or(rest, |x| x.0);
            return Some(Suppression {
                tool,
                kind: parse_error_codes(inside),
                comment_line,
                comment_offset,
            });
        } else if gap || lex.word_boundary() {
            return Some(Suppression::blanket(tool, comment_line, comment_offset));
        }
        None
    }

    pub fn is_ignored(
        &self,
        start_line: LineNumber,
        kind: &str,
        enabled_ignores: &SmallSet<Tool>,
    ) -> bool {
        if let Some(suppressions) = self.ignores.get(&start_line)
            && suppressions.iter().any(|supp| {
                enabled_ignores.contains(&supp.tool)
                    && match supp.tool {
                        // We only check the subkind if they do `# pyrefly: ignore`
                        Tool::Pyrefly => {
                            supp.kind.is_empty() || supp.kind.iter().any(|x| x == kind)
                        }
                        _ => true,
                    }
            })
        {
            return true;
        }
        false
    }

    /// Similar to `is_ignored`, but it only returns true if the error is ignored
    /// by a suppression that targets a specific line.
    pub fn is_ignored_by_suppression_line(
        &self,
        suppression_line: LineNumber,
        start_line: LineNumber,
        end_line: LineNumber,
        kind: &str,
        enabled_ignores: &SmallSet<Tool>,
    ) -> bool {
        // If the error does not overlap the range, skip the more expensive check
        if start_line > suppression_line || end_line < suppression_line {
            return false;
        }
        let Some(suppressions) = self.ignores.get(&suppression_line) else {
            return false;
        };
        if suppressions.iter().any(|supp| {
            enabled_ignores.contains(&supp.tool)
                && match supp.tool {
                    // We only check the subkind if they do `# pyrefly: ignore`
                    Tool::Pyrefly => supp.kind.is_empty() || supp.kind.iter().any(|x| x == kind),
                    _ => true,
                }
        }) {
            return true;
        }
        false
    }

    // gets either just pyrefly ignores or pyrefly and type: ignore comments
    pub fn get_pyrefly_ignores(&self, all: bool) -> SmallSet<LineNumber> {
        let ignore_iter = self.ignores.iter();
        let filtered_ignores: Box<dyn Iterator<Item = (&LineNumber, &Vec<Suppression>)>> = if all {
            Box::new(ignore_iter.filter(|ignore| {
                ignore
                    .1
                    .iter()
                    .any(|s| s.tool == Tool::Pyrefly || s.tool == Tool::Type)
            }))
        } else {
            Box::new(ignore_iter.filter(|ignore| ignore.1.iter().any(|s| s.tool == Tool::Pyrefly)))
        };
        filtered_ignores.map(|(line, _)| *line).collect()
    }

    /// Returns an iterator over all suppressions in the file.
    /// Each item is a (line_number, suppressions) pair where line_number is where the suppression applies.
    pub fn iter(&self) -> impl Iterator<Item = (&LineNumber, &Vec<Suppression>)> {
        self.ignores.iter()
    }

    /// Gets the suppressions for a specific line.
    pub fn get(&self, line: &LineNumber) -> Option<&Vec<Suppression>> {
        self.ignores.get(line)
    }

    /// Returns true if there are no suppressions.
    pub fn is_empty(&self) -> bool {
        self.ignores.is_empty()
    }
}

/// Returns true if `line` falls inside one of the sorted multiline string ranges.
fn is_in_multiline_string(
    multiline_string_ranges: &[(LineNumber, LineNumber)],
    line: LineNumber,
) -> bool {
    let idx = multiline_string_ranges.partition_point(|(start, _)| *start <= line);
    idx > 0 && {
        let (start, end) = multiline_string_ranges[idx - 1];
        line >= start && line <= end
    }
}

/// Parse top-level `ignore-errors` / `ignore-all-errors` / `type: ignore` directives.
///
/// Scans the beginning of the file for comment-only lines (including blank lines
/// and lines inside multiline strings like docstrings). Returns the file-level
/// suppressions found; Pyrefly entries may carry specific error codes
/// (`# pyrefly: ignore-errors[code]`), while other tools are blanket-only.
///
/// After a docstring, only `ignore-errors` directives are recognized — bare
/// `# type: ignore` is not, since it could plausibly be meant as a per-line
/// suppression for code that follows.
pub fn parse_ignore_all(
    code: &str,
    multiline_string_ranges: &[(LineNumber, LineNumber)],
) -> Vec<Suppression> {
    let mut res = Vec::new();
    let mut prev_ignore = None;
    let mut seen_docstring = false;

    for (idx, raw_line) in code.lines().enumerate() {
        let line = LineNumber::from_zero_indexed(idx as u32);
        let trimmed = raw_line.trim();

        // Lines inside a multiline string (e.g. a module docstring) are not
        // code — skip them but record that we've passed through a docstring.
        if is_in_multiline_string(multiline_string_ranges, line) {
            seen_docstring = true;
            continue;
        }

        // Lines that open/close a triple-quoted string are also part of the
        // preamble — skip them.
        if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
            seen_docstring = true;
            continue;
        }

        // Stop at the first non-empty, non-comment line (i.e. actual code).
        // A pending `# type: ignore` followed directly by code is a per-line
        // suppression, not an ignore-all directive, so we discard it.
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            break;
        }

        if let Some((tool, prev_line, prev_offset)) = prev_ignore {
            // The previous `# type: ignore` was followed by another comment or
            // blank line, so it is a whole-file suppression.
            res.push(Suppression::blanket(tool, prev_line, prev_offset));
            prev_ignore = None;
        }

        let mut lex = Lexer(trimmed);
        if !lex.starts_with("#") {
            continue;
        }
        // `trimmed` starts with `#`, so its offset is the line's leading whitespace.
        let comment_offset = raw_line.len() - raw_line.trim_start().len();
        lex.trim_start();
        if lex.starts_with("pyre-ignore-all-errors") {
            res.push(Suppression::blanket(Tool::Pyre, line, comment_offset));
        } else if let Some(tool) = lex.starts_with_tool() {
            lex.trim_start();
            if lex.starts_with("ignore-errors") {
                lex.trim_start();
                // Parse an optional `[code, ...]` list, sharing `parse_error_codes`
                // with the line-level parser. A file-level directive must close its
                // bracket and have nothing after it, so malformed lines are rejected.
                let kind = if lex.starts_with("[") {
                    lex.0.split_once(']').and_then(|(inside, after)| {
                        // Drop empty entries so `[]`/trailing commas act as a blanket
                        // ignore rather than a directive that matches nothing.
                        Lexer(after).blank().then(|| {
                            parse_error_codes(inside)
                                .into_iter()
                                .filter(|code| !code.is_empty())
                                .collect()
                        })
                    })
                } else if lex.blank() {
                    Some(Vec::new())
                } else {
                    None
                };
                // Only Pyrefly honors specific codes; other tools are blanket-only.
                if let Some(kind) = kind
                    && (tool == Tool::Pyrefly || kind.is_empty())
                {
                    res.push(Suppression {
                        tool,
                        kind,
                        comment_line: line,
                        comment_offset,
                    });
                }
            } else if !seen_docstring && lex.starts_with("ignore") && lex.blank() {
                // After a docstring, bare `# type: ignore` is not recognized
                // as an ignore-all directive.
                prev_ignore = Some((tool, line, comment_offset));
            }
        }
    }
    res
}

/// Split the comma-separated error codes inside a `[...]` suppression into trimmed names.
fn parse_error_codes(inside: &str) -> Vec<String> {
    inside.split(',').map(|x| x.trim().to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use pyrefly_util::prelude::SliceExt;

    use super::*;

    #[test]
    fn test_parse_ignores() {
        fn f(x: &str, expect: &[(Tool, u32)]) {
            assert_eq!(
                &Ignore::parse_ignores(x)
                    .into_iter()
                    .flat_map(|(line, xs)| xs.map(|x| (x.tool, line.get())))
                    .collect::<Vec<_>>(),
                expect,
                "{x:?}"
            );
        }

        f("stuff # type: ignore # and then stuff", &[(Tool::Type, 1)]);
        f("more # stuff # type: ignore", &[(Tool::Type, 1)]);
        f(" pyrefly: ignore", &[]);
        f("normal line", &[]);
        f(
            "code # pyright: ignore\n# pyre-fixme\nmore code",
            &[(Tool::Pyright, 1), (Tool::Pyre, 3)],
        );
        f(
            "# type: ignore\n# pyright: ignore\n# bad\n\ncode",
            &[(Tool::Type, 4), (Tool::Pyright, 4)],
        );

        // Ignore `# pyrefly: ignore` inside a string but not before/after
        f("x = 1 + '# pyrefly: ignore'", &[]);
        f("x = ''  # pyrefly: ignore", &[(Tool::Pyrefly, 1)]);
        f("x = '''# pyrefly: ignore'''", &[]);
        f(
            r#"
x = """
x = 1  # pyrefly: ignore
"""
        "#,
            &[],
        );
        f(
            r#"
import textwrap
textwrap.dedent("""\
x = 1  # pyrefly: ignore
""")
        "#,
            &[],
        );
        f(
            r#"
x = """  # pyrefly: ignore
"""
        "#,
            &[],
        );
        f(
            r#"
x = """
# pyrefly: ignore"""
        "#,
            &[],
        );
        f(
            r#"
x = """
"""  # pyrefly: ignore
        "#,
            &[(Tool::Pyrefly, 3)],
        );
        f(
            r#"_ = f'start{"""message
"""}end'
y: int = "hello"  # pyrefly: ignore[bad-assignment]"#,
            &[(Tool::Pyrefly, 3)],
        );
        f(
            r#"_ = f'start{"""message
"""}# pyrefly: ignore'
y: int = "hello"  # pyrefly: ignore[bad-assignment]"#,
            &[(Tool::Pyrefly, 3)],
        );
        f("x = ''''''  # pyrefly: ignore", &[(Tool::Pyrefly, 1)]);
    }

    #[test]
    fn test_suppression_comment_offset() {
        fn f(x: &str, expect: &[(u32, usize)]) {
            assert_eq!(
                &Ignore::parse_ignores(x)
                    .into_iter()
                    .flat_map(|(_, xs)| xs.map(|x| (x.comment_line.get(), x.comment_offset)))
                    .collect::<Vec<_>>(),
                expect,
                "{x:?}"
            );
        }

        f("x = 1  # type: ignore", &[(1, 7)]);
        // Not the `#` inside the string literal
        f(r##"x: str = "#hash"  # type: ignore"##, &[(1, 18)]);
        // Line starts inside a triple-quoted string that closes mid-line
        f("x = \"\"\"\n#fake\"\"\" # type: ignore", &[(2, 9)]);
        // A comment above code keeps its own line and offset
        f("  # type: ignore\nx = 1", &[(1, 2)]);
    }

    #[test]
    fn test_parse_ignore_comment() {
        fn f(x: &str, tool: Option<Tool>, kind: &[&str]) {
            let dummy_line = LineNumber::default();
            assert_eq!(
                Ignore::parse_ignore_comment(x, dummy_line, 0),
                tool.map(|tool| Suppression {
                    tool,
                    kind: kind.map(|x| (*x).to_owned()),
                    comment_line: dummy_line,
                    comment_offset: 0,
                }),
                "{x:?}"
            );
        }

        f("ignore: pyrefly", None, &[]);
        f("pyrefly: ignore", Some(Tool::Pyrefly), &[]);
        f(
            "pyrefly: ignore[bad-return]",
            Some(Tool::Pyrefly),
            &["bad-return"],
        );
        f("pyrefly: ignore[]", Some(Tool::Pyrefly), &[""]);
        f("pyrefly: ignore[bad-]", Some(Tool::Pyrefly), &["bad-"]);

        // Check spacing
        f(" type: ignore ", Some(Tool::Type), &[]);
        f("type:ignore", Some(Tool::Type), &[]);
        f("type :ignore", None, &[]);

        // Check extras
        // Mypy rejects that, Pyright accepts it
        f("type: ignore because it is wrong", Some(Tool::Type), &[]);
        f("type: ignore_none", None, &[]);
        f("type: ignore1", None, &[]);
        f("type: ignore?", Some(Tool::Type), &[]);

        f("pyright: ignore", Some(Tool::Pyright), &[]);
        f(
            "pyright: ignore[something]",
            Some(Tool::Pyright),
            &["something"],
        );

        f("pyre-ignore", Some(Tool::Pyre), &[]);
        f("pyre-ignore[7]", Some(Tool::Pyre), &["7"]);
        f("pyre-fixme[7]", Some(Tool::Pyre), &["7"]);
        f(
            "pyre-fixme[61]: `x` may not be initialized here.",
            Some(Tool::Pyre),
            &["61"],
        );
        f("pyre-fixme: core type error", Some(Tool::Pyre), &[]);

        f("zuban: ignore", Some(Tool::Zuban), &[]);
        f(
            "zuban: ignore[something]",
            Some(Tool::Zuban),
            &["something"],
        );

        // For a malformed comment, at least do something with it (works well incrementally)
        f("type: ignore[hello", Some(Tool::Type), &["hello"]);
    }

    #[test]
    fn test_find_comment_start_in_line() {
        // Test basic comment finding
        assert_eq!(find_comment_start_in_line("x = 1  # comment"), Some(7));
        assert_eq!(find_comment_start_in_line("no comment here"), None);

        // Test string-aware parsing
        assert_eq!(
            find_comment_start_in_line(r#"x = "hello # world"  # real"#),
            Some(21)
        );
        assert_eq!(
            find_comment_start_in_line(r#"x = 'hello # world'  # real"#),
            Some(21)
        );

        // Test escaped quotes
        assert_eq!(
            find_comment_start_in_line(r#"x = "she said \"hi\" # not" # real"#),
            Some(28)
        );

        // Test multiple hashes
        assert_eq!(find_comment_start_in_line("# first # second"), Some(0));

        // A '#' in an f-string format specifier is not a comment.
        let format_spec = r##"x = f"{1:#x}"  # real"##;
        assert_eq!(
            find_comment_start_in_line(format_spec),
            format_spec.rfind('#')
        );
    }

    #[test]
    fn test_parse_ignore_all() {
        fn f(x: &str, ignores: &[(Tool, u32, &[&str])]) {
            assert_eq!(
                parse_ignore_all(x, &[]),
                ignores
                    .iter()
                    .map(|x| Suppression {
                        tool: x.0,
                        kind: x.2.iter().map(|x| (*x).to_owned()).collect(),
                        comment_line: LineNumber::new(x.1).unwrap(),
                        comment_offset: 0,
                    })
                    .collect::<Vec<_>>(),
                "{x:?}"
            );
        }

        f(
            "# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 1, &[])],
        );
        f(
            "# pyrefly: ignore-errors[bad-assignment]\nx = 5",
            &[(Tool::Pyrefly, 1, &["bad-assignment"])],
        );
        f(
            "# pyrefly: ignore-errors [ bad-assignment, bad-return ]\nx = 5",
            &[(Tool::Pyrefly, 1, &["bad-assignment", "bad-return"])],
        );
        // Empty brackets and trailing commas drop empty entries, acting as a blanket ignore.
        f(
            "# pyrefly: ignore-errors[]\nx = 5",
            &[(Tool::Pyrefly, 1, &[])],
        );
        f(
            "# pyrefly: ignore-errors[bad-assignment,]\nx = 5",
            &[(Tool::Pyrefly, 1, &["bad-assignment"])],
        );
        // A missing closing bracket is malformed and rejected, not silently accepted.
        f("# pyrefly: ignore-errors[bad-assignment\nx = 5", &[]);
        f(
            "# comment\n# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 2, &[])],
        );
        f(
            "#comment\n  # indent\n# pyrefly: ignore-errors\nx = 5",
            &[(Tool::Pyrefly, 3, &[])],
        );
        f("x = 5\n# pyrefly: ignore-errors", &[]);
        // Directives are only recognized in the preamble; once real code (including an
        // import) appears the scan stops, so a later typed directive is inert — whether
        // it trails code, trails an import, or is sandwiched between code lines.
        f("x = 5\n# pyrefly: ignore-errors[bad-assignment]", &[]);
        f(
            "import os\n# pyrefly: ignore-errors[bad-assignment]\nx = 5",
            &[],
        );
        f(
            "x = 5\n# pyrefly: ignore-errors[bad-assignment]\ny = 6",
            &[],
        );
        f("# type: ignore\n\nx = 5", &[(Tool::Type, 1, &[])]);
        f(
            "# comment\n# type: ignore\n# comment\nx = 5",
            &[(Tool::Type, 2, &[])],
        );
        f("# type: ignore\nx = 5", &[]);
        f("# pyre-ignore-all-errors\nx = 5", &[(Tool::Pyre, 1, &[])]);
        f(
            "# mypy: ignore-errors\n#pyrefly:ignore-errors",
            &[(Tool::Mypy, 1, &[]), (Tool::Pyrefly, 2, &[])],
        );
        f("# mypy: ignore-errors[bad-assignment]\nx = 5", &[]);

        // Anything else on the line (other than space) makes it invalid
        f("# pyrefly: ignore-errors because I want to\nx = 5", &[]);
        f("# pyrefly: ignore-errors # because I want to\nx = 5", &[]);
        f(
            "# pyrefly: ignore-errors[bad-assignment] # because I want to\nx = 5",
            &[],
        );
        f(
            "# pyrefly: ignore-errors \nx = 5",
            &[(Tool::Pyrefly, 1, &[])],
        );
    }

    #[test]
    fn test_parse_ignore_all_with_docstring() {
        fn f(x: &str, ranges: &[(LineNumber, LineNumber)], ignores: &[(Tool, u32, &[&str])]) {
            assert_eq!(
                parse_ignore_all(x, ranges),
                ignores
                    .iter()
                    .map(|x| Suppression {
                        tool: x.0,
                        kind: x.2.iter().map(|x| (*x).to_owned()).collect(),
                        comment_line: LineNumber::new(x.1).unwrap(),
                        comment_offset: 0,
                    })
                    .collect::<Vec<_>>(),
                "{x:?}"
            );
        }

        // ignore-errors after a docstring should work
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\n# pyrefly: ignore-errors\nx = 5",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[(Tool::Pyrefly, 4, &[])],
        );

        // typed ignore-errors[code] after a docstring should also work
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\n# pyrefly: ignore-errors[bad-assignment]\nx = 5",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[(Tool::Pyrefly, 4, &["bad-assignment"])],
        );

        // bare `# type: ignore` after docstring should NOT be recognized
        f(
            "\"\"\"\nmodule docstring\n\"\"\"\n# type: ignore\n\nx = 5",
            &[(
                LineNumber::from_zero_indexed(0),
                LineNumber::from_zero_indexed(2),
            )],
            &[],
        );

        // ignore-errors before a docstring should still work
        f(
            "# pyrefly: ignore-errors\n\"\"\"\nmodule docstring\n\"\"\"\nx = 5",
            &[(
                LineNumber::from_zero_indexed(1),
                LineNumber::from_zero_indexed(3),
            )],
            &[(Tool::Pyrefly, 1, &[])],
        );
    }
}
