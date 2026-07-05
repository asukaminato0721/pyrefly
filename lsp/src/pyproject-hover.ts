/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as child_process from 'child_process';
import * as https from 'https';
import * as vscode from 'vscode';
import {PythonEnvironment} from './python-environment';

type Dependency = {
  name: string;
  configuredVersion?: string;
  range: vscode.Range;
};

type PypiMetadata = {
  summary?: string;
  version?: string;
  publishedAt?: Date;
  homepage?: string;
};

const pypiMetadataCache = new Map<string, Promise<PypiMetadata | undefined>>();
const installedVersionCache = new Map<string, Promise<string | undefined>>();

export function registerPyprojectHoverProvider(
  context: vscode.ExtensionContext,
  pythonEnv: PythonEnvironment,
) {
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(
      [
        {scheme: 'file', language: 'toml', pattern: '**/pyproject.toml'},
        {scheme: 'untitled', language: 'toml'},
      ],
      {
        async provideHover(document, position, token) {
          if (
            document.uri.scheme === 'file' &&
            document.uri.fsPath.split(/[\\/]/).pop() !== 'pyproject.toml'
          ) {
            return undefined;
          }
          const dependency = findPyprojectDependency(document, position);
          if (!dependency) {
            return undefined;
          }
          const [metadata, installedVersion] = await Promise.all([
            fetchPypiMetadata(dependency.name),
            resolveInstalledVersion(pythonEnv, document.uri, dependency.name),
          ]);
          if (token.isCancellationRequested) {
            return undefined;
          }
          const markdown = formatHover(dependency, metadata, installedVersion);
          return markdown ? new vscode.Hover(markdown, dependency.range) : undefined;
        },
      },
    ),
  );
}

export function findPyprojectDependency(
  document: vscode.TextDocument,
  position: vscode.Position,
): Dependency | undefined {
  const line = document.lineAt(position.line).text;
  const section = findSection(document, position.line);
  const quoted = dependencyFromQuotedString(line, position);
  if (quoted) {
    const arrayKey = findEnclosingArrayKey(document, position.line);
    if (
      (section === 'project' && arrayKey === 'dependencies') ||
      section === 'project.optional-dependencies' ||
      section === 'dependency-groups'
    ) {
      return quoted;
    }
  }
  if (
    section === 'tool.poetry.dependencies' ||
    (section?.startsWith('tool.poetry.group.') &&
      section.endsWith('.dependencies'))
  ) {
    return dependencyFromPoetryEntry(line, position);
  }
  return undefined;
}

function findSection(document: vscode.TextDocument, line: number): string | undefined {
  for (let i = line; i >= 0; i--) {
    const match = document.lineAt(i).text.match(/^\s*\[([^\]]+)\]\s*(?:#.*)?$/);
    if (match) {
      return match[1].trim();
    }
  }
  return undefined;
}

function findEnclosingArrayKey(
  document: vscode.TextDocument,
  line: number,
): string | undefined {
  for (let i = line; i >= 0; i--) {
    const text = document.lineAt(i).text;
    if (/^\s*\[/.test(text)) {
      return undefined;
    }
    const match = text.match(/^\s*([A-Za-z0-9_.-]+)\s*=\s*\[/);
    if (match) {
      return match[1];
    }
    if (i < line && text.includes(']')) {
      return undefined;
    }
  }
  return undefined;
}

function dependencyFromQuotedString(
  line: string,
  position: vscode.Position,
): Dependency | undefined {
  for (const span of quotedSpans(line)) {
    if (position.character < span.start || position.character > span.end) {
      continue;
    }
    const text = line.slice(span.start + 1, span.end);
    const parsed = parseDependencySpec(text);
    if (!parsed) {
      return undefined;
    }
    return {
      ...parsed,
      range: new vscode.Range(
        position.line,
        span.start + 1,
        position.line,
        span.start + 1 + parsed.name.length,
      ),
    };
  }
  return undefined;
}

function quotedSpans(line: string): {start: number; end: number}[] {
  const spans = [];
  let quote: string | undefined;
  let start = -1;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (quote) {
      if (char === quote && line[i - 1] !== '\\') {
        spans.push({start, end: i});
        quote = undefined;
      }
    } else if (char === '"' || char === "'") {
      quote = char;
      start = i;
    }
  }
  return spans;
}

function dependencyFromPoetryEntry(
  line: string,
  position: vscode.Position,
): Dependency | undefined {
  const match = line.match(/^\s*("[^"]+"|'[^']+'|[A-Za-z0-9_.-]+)\s*=\s*(.+)$/);
  if (!match) {
    return undefined;
  }
  const rawName = match[1];
  const keyStart = line.indexOf(rawName);
  const keyEnd = keyStart + rawName.length;
  const name = rawName.replace(/^["']|["']$/g, '');
  if (name === 'python') {
    return undefined;
  }
  const value = match[2].trim();
  const range = new vscode.Range(position.line, keyStart, position.line, keyEnd);
  if (position.character < keyStart || position.character > line.length) {
    return undefined;
  }
  return {
    name,
    configuredVersion: poetryVersion(value),
    range,
  };
}

function parseDependencySpec(
  spec: string,
): {name: string; configuredVersion?: string} | undefined {
  const match = spec
    .trim()
    .match(/^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\s*\[[^\]]+\])?\s*(.*)$/);
  if (!match || match[1] === 'python') {
    return undefined;
  }
  return {
    name: match[1],
    configuredVersion: exactVersion(match[2]),
  };
}

function exactVersion(spec: string): string | undefined {
  return spec.match(/^\s*==\s*([^,;\s]+)/)?.[1];
}

function poetryVersion(value: string): string | undefined {
  const quoted = value.match(/^["']([^"']+)["']/)?.[1];
  if (quoted) {
    return quoted;
  }
  return value.match(/version\s*=\s*["']([^"']+)["']/)?.[1];
}

async function fetchPypiMetadata(name: string): Promise<PypiMetadata | undefined> {
  const normalized = normalizePackageName(name);
  let cached = pypiMetadataCache.get(normalized);
  if (!cached) {
    cached = fetchPypiMetadataUncached(name);
    pypiMetadataCache.set(normalized, cached);
  }
  return cached;
}

function fetchPypiMetadataUncached(
  name: string,
): Promise<PypiMetadata | undefined> {
  return new Promise(resolve => {
    const request = https.get(
      {
        hostname: 'pypi.org',
        path: `/pypi/${encodeURIComponent(name)}/json`,
        headers: {'User-Agent': 'pyrefly-vscode'},
        timeout: 3000,
      },
      response => {
        if (response.statusCode !== 200) {
          response.resume();
          resolve(undefined);
          return;
        }
        let body = '';
        response.setEncoding('utf8');
        response.on('data', chunk => {
          body += chunk;
        });
        response.on('end', () => {
          try {
            const json = JSON.parse(body);
            const version = json.info?.version;
            const releases = version ? json.releases?.[version] : undefined;
            const publishedAt = Array.isArray(releases)
              ? releases
                  .map(release => release.upload_time_iso_8601)
                  .filter(Boolean)
                  .sort()
                  .pop()
              : undefined;
            resolve({
              summary: json.info?.summary,
              version,
              publishedAt: publishedAt ? new Date(publishedAt) : undefined,
              homepage:
                json.info?.project_urls?.Homepage ??
                json.info?.project_urls?.Source ??
                json.info?.home_page ??
                json.info?.package_url,
            });
          } catch {
            resolve(undefined);
          }
        });
      },
    );
    request.on('timeout', () => {
      request.destroy();
      resolve(undefined);
    });
    request.on('error', () => resolve(undefined));
  });
}

async function resolveInstalledVersion(
  pythonEnv: PythonEnvironment,
  uri: vscode.Uri,
  name: string,
): Promise<string | undefined> {
  const interpreter = await pythonEnv.getInterpreterPath(uri);
  if (!interpreter) {
    return undefined;
  }
  const key = `${interpreter}\0${normalizePackageName(name)}`;
  let cached = installedVersionCache.get(key);
  if (!cached) {
    cached = queryInstalledVersion(interpreter, name);
    installedVersionCache.set(key, cached);
  }
  return cached;
}

function queryInstalledVersion(
  interpreter: string,
  name: string,
): Promise<string | undefined> {
  const script = [
    'import importlib.metadata',
    'import sys',
    'try:',
    '    print(importlib.metadata.version(sys.argv[1]))',
    'except importlib.metadata.PackageNotFoundError:',
    '    pass',
  ].join('\n');
  return new Promise(resolve => {
    child_process.execFile(
      interpreter,
      ['-c', script, name],
      {timeout: 2000},
      (error, stdout) => {
        if (error) {
          resolve(undefined);
          return;
        }
        const version = stdout.trim();
        resolve(version === '' ? undefined : version);
      },
    );
  });
}

function formatHover(
  dependency: Dependency,
  metadata?: PypiMetadata,
  installedVersion?: string,
): vscode.MarkdownString | undefined {
  if (!metadata && !installedVersion && !dependency.configuredVersion) {
    return undefined;
  }
  const markdown = new vscode.MarkdownString(undefined, true);
  const summary = metadata?.summary?.trim();
  if (summary) {
    markdown.appendText(summary);
  }
  if (metadata?.version) {
    appendParagraph(
      markdown,
      `Latest version: ${metadata.version}${publishedSuffix(metadata.publishedAt)}`,
    );
  }
  if (installedVersion) {
    appendParagraph(markdown, `Installed version: ${installedVersion}`);
  } else if (dependency.configuredVersion) {
    appendParagraph(markdown, `Configured version: ${dependency.configuredVersion}`);
  }
  if (metadata?.homepage) {
    appendParagraph(markdown, metadata.homepage);
  }
  return markdown;
}

function appendParagraph(markdown: vscode.MarkdownString, text: string) {
  if (markdown.value.length > 0) {
    markdown.appendMarkdown('\n\n');
  }
  markdown.appendText(text);
}

function publishedSuffix(date?: Date): string {
  if (!date || Number.isNaN(date.getTime())) {
    return '';
  }
  return ` published ${relativeTime(date)}`;
}

function relativeTime(date: Date): string {
  const seconds = Math.max(1, Math.floor((Date.now() - date.getTime()) / 1000));
  const units: [string, number][] = [
    ['year', 365 * 24 * 60 * 60],
    ['month', 30 * 24 * 60 * 60],
    ['week', 7 * 24 * 60 * 60],
    ['day', 24 * 60 * 60],
    ['hour', 60 * 60],
    ['minute', 60],
  ];
  for (const [unit, size] of units) {
    const amount = Math.floor(seconds / size);
    if (amount >= 1) {
      return `${amount} ${unit}${amount === 1 ? '' : 's'} ago`;
    }
  }
  return `${seconds} seconds ago`;
}

function normalizePackageName(name: string): string {
  return name.toLowerCase().replace(/[-_.]+/g, '-');
}
