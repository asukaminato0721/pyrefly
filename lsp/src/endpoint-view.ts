/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {LanguageClient} from 'vscode-languageclient/node';

type EndpointRange = {
  start: {line: number; character: number};
  end: {line: number; character: number};
};

type EndpointLocation = {
  uri: string;
  range: EndpointRange;
};

type EndpointItem = {
  path: string;
  methods: string[];
  location: EndpointLocation;
};

class EndpointTreeItem extends vscode.TreeItem {
  constructor(public readonly endpoint: EndpointItem) {
    super(endpoint.path, vscode.TreeItemCollapsibleState.None);
    if (endpoint.methods.length > 0) {
      this.description = endpoint.methods.join(', ');
    }
    this.contextValue = 'pyreflyEndpoint';
    this.command = {
      command: 'pyrefly.openEndpoint',
      title: 'Open Endpoint',
      arguments: [endpoint.location],
    };
  }
}

export class EndpointViewProvider
  implements vscode.TreeDataProvider<EndpointTreeItem>
{
  private readonly onDidChangeTreeDataEmitter = new vscode.EventEmitter<
    EndpointTreeItem | undefined
  >();
  readonly onDidChangeTreeData = this.onDidChangeTreeDataEmitter.event;

  private endpoints: EndpointItem[] = [];
  private loadedOnce = false;

  constructor(
    private readonly client: LanguageClient,
    private readonly outputChannel: vscode.OutputChannel,
  ) {}

  refresh(): void {
    this.loadEndpoints()
      .then(() => {
        this.onDidChangeTreeDataEmitter.fire(undefined);
      })
      .catch(err => {
        this.outputChannel.appendLine(
          `[Pyrefly] Failed to refresh endpoints: ${String(err)}`,
        );
      });
  }

  getTreeItem(element: EndpointTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(): Promise<EndpointTreeItem[]> {
    if (!this.loadedOnce) {
      await this.loadEndpoints();
    }
    return this.endpoints.map(endpoint => new EndpointTreeItem(endpoint));
  }

  private async loadEndpoints(): Promise<void> {
    this.loadedOnce = true;
    try {
      const result = await this.client.sendRequest<EndpointItem[]>(
        'pyrefly/endpoints',
        {},
      );
      if (Array.isArray(result)) {
        this.endpoints = result.sort((a, b) =>
          a.path.localeCompare(b.path),
        );
      } else {
        this.endpoints = [];
      }
    } catch (err) {
      this.endpoints = [];
      this.outputChannel.appendLine(
        `[Pyrefly] Failed to load endpoints: ${String(err)}`,
      );
    }
  }
}

export async function openEndpointLocation(
  location: EndpointLocation,
): Promise<void> {
  const uri = vscode.Uri.parse(location.uri);
  const range = new vscode.Range(
    location.range.start.line,
    location.range.start.character,
    location.range.end.line,
    location.range.end.character,
  );
  const doc = await vscode.workspace.openTextDocument(uri);
  await vscode.window.showTextDocument(doc, {
    selection: range,
    preserveFocus: false,
  });
}
