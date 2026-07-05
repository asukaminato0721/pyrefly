/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::Position;
use pyrefly_config::pyproject::PyProjectDependency;
use pyrefly_config::pyproject::dependency_at_offset;
use pyrefly_util::lined_buffer::LinedBuffer;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Deserialize;

use crate::lsp::wasm::hover::HoverResult;

const PACKAGE_QUERY_TIMEOUT: Duration = Duration::from_secs(3);
// The selected interpreter provides both environment metadata and HTTPS without a new Rust client.
const PACKAGE_QUERY: &str = r#"
import importlib.metadata
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime

name = sys.argv[1]
result = {}
try:
    result["installed_version"] = importlib.metadata.version(name)
except importlib.metadata.PackageNotFoundError:
    pass
try:
    request = urllib.request.Request(
        "https://pypi.org/pypi/" + urllib.parse.quote(name, safe="") + "/json",
        headers={"User-Agent": "pyrefly-lsp"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        payload = json.load(response)
    info = payload.get("info") or {}
    result["summary"] = info.get("summary")
    result["version"] = info.get("version")
    urls = info.get("project_urls") or {}
    result["homepage"] = (
        urls.get("Homepage")
        or urls.get("Source")
        or info.get("home_page")
        or info.get("package_url")
    )
    releases = (payload.get("releases") or {}).get(info.get("version"), [])
    uploads = [entry.get("upload_time_iso_8601") for entry in releases]
    uploads = [upload for upload in uploads if upload]
    if uploads:
        result["published_at"] = int(
            datetime.fromisoformat(max(uploads).replace("Z", "+00:00")).timestamp()
        )
except Exception:
    pass
print(json.dumps(result))
"#;

#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct PackageMetadata {
    summary: Option<String>,
    version: Option<String>,
    published_at: Option<u64>,
    homepage: Option<String>,
    installed_version: Option<String>,
}

/// Query local and package-index metadata with the selected Python environment.
pub(super) fn package_metadata(interpreter: &Path, name: &str) -> PackageMetadata {
    let mut child = match Command::new(interpreter)
        .args(["-c", PACKAGE_QUERY, name])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return PackageMetadata::default(),
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < PACKAGE_QUERY_TIMEOUT => {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return PackageMetadata::default();
            }
        }
    }
    let mut output = String::new();
    if child
        .stdout
        .take()
        .is_none_or(|mut stdout| stdout.read_to_string(&mut output).is_err())
    {
        return PackageMetadata::default();
    }
    serde_json::from_str(&output).unwrap_or_default()
}

/// Find the dependency under an LSP position.
pub(super) fn dependency(source: &str, position: Position) -> Option<PyProjectDependency> {
    let buffer = LinedBuffer::new(Arc::new(source.to_owned()));
    let offset = buffer.from_lsp_position(position, None).to_usize();
    dependency_at_offset(source, offset)
}

/// Build a standard LSP hover for a dependency and its available metadata.
pub(super) fn hover(
    source: &str,
    dependency: &PyProjectDependency,
    metadata: &PackageMetadata,
) -> Option<HoverResult> {
    let buffer = LinedBuffer::new(Arc::new(source.to_owned()));
    let start = TextSize::new(u32::try_from(dependency.range.start).ok()?);
    let end = TextSize::new(u32::try_from(dependency.range.end).ok()?);
    let mut paragraphs = Vec::new();
    if let Some(summary) = metadata.summary.as_deref().map(str::trim)
        && !summary.is_empty()
    {
        paragraphs.push(escape_markdown(summary));
    }
    if let Some(version) = &metadata.version {
        let published = metadata
            .published_at
            .and_then(relative_time)
            .map(|age| format!(" published {age}"))
            .unwrap_or_default();
        paragraphs.push(format!(
            "Latest version: {}{published}",
            escape_markdown(version)
        ));
    }
    if let Some(version) = &metadata.installed_version {
        paragraphs.push(format!("Installed version: {}", escape_markdown(version)));
    } else if let Some(version) = &dependency.configured_version {
        paragraphs.push(format!("Configured version: {}", escape_markdown(version)));
    }
    if let Some(homepage) = metadata.homepage.as_deref().filter(|url| {
        (url.starts_with("https://") || url.starts_with("http://"))
            && !url
                .chars()
                .any(|character| character.is_whitespace() || matches!(character, '<' | '>'))
    }) {
        paragraphs.push(format!("<{homepage}>"));
    }
    if paragraphs.is_empty() {
        return None;
    }
    Some(HoverResult {
        hover: Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: paragraphs.join("\n\n"),
            }),
            range: Some(buffer.to_lsp_range(TextRange::new(start, end), None)),
        },
        can_increase_verbosity: false,
    })
}

fn relative_time(timestamp: u64) -> Option<String> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    let seconds = now.saturating_sub(timestamp).max(1);
    for (unit, size) in [
        ("year", 365 * 24 * 60 * 60),
        ("month", 30 * 24 * 60 * 60),
        ("week", 7 * 24 * 60 * 60),
        ("day", 24 * 60 * 60),
        ("hour", 60 * 60),
        ("minute", 60),
    ] {
        let amount = seconds / size;
        if amount >= 1 {
            return Some(format!(
                "{amount} {unit}{} ago",
                if amount == 1 { "" } else { "s" }
            ));
        }
    }
    Some(format!("{seconds} seconds ago"))
}

fn escape_markdown(text: &str) -> String {
    text.chars().fold(String::new(), |mut escaped, character| {
        if matches!(
            character,
            '\\' | '`' | '*' | '_' | '{' | '}' | '[' | ']' | '<' | '>' | '#' | '|' | '!'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
        escaped
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_dependency_hover() {
        let source = "[project]\ndependencies = [\"requests==2.32.0\"]\n";
        let result = hover(
            source,
            &dependency(source, Position::new(1, 18)).unwrap(),
            &PackageMetadata {
                summary: Some("HTTP *client*".to_owned()),
                version: Some("2.32.4".to_owned()),
                homepage: Some("https://example.com".to_owned()),
                installed_version: Some("2.32.3".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();
        let HoverContents::Markup(contents) = result.hover.contents else {
            panic!("expected markdown hover")
        };
        assert_eq!(
            contents.value,
            "HTTP \\*client\\*\n\nLatest version: 2.32.4\n\nInstalled version: 2.32.3\n\n<https://example.com>"
        );
        assert_eq!(
            result.hover.range,
            Some(lsp_types::Range::new(
                Position::new(1, 17),
                Position::new(1, 25),
            ))
        );
    }

    #[test]
    fn configured_version_is_fallback() {
        let source = "[project]\ndependencies = [\"requests==2.32.0\"]\n";
        let result = hover(
            source,
            &dependency(source, Position::new(1, 18)).unwrap(),
            &PackageMetadata::default(),
        )
        .unwrap();
        let HoverContents::Markup(contents) = result.hover.contents else {
            panic!("expected markdown hover")
        };
        assert_eq!(contents.value, "Configured version: 2.32.0");
    }
}
