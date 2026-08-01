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

type TypeArgs = {text?: string} | undefined;

async function defaultType(args: TypeArgs): Promise<void> {
  await vscode.commands.executeCommand('default:type', args);
}

export function registerTripleQuoteAutoClose(
  context: vscode.ExtensionContext,
  getClient: () => LanguageClient,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('type', async (args: TypeArgs) => {
      const editor = vscode.window.activeTextEditor;
      const quote = args?.text;
      if (
        editor === undefined ||
        editor.document.languageId !== 'python' ||
        (quote !== '"' && quote !== "'") ||
        vscode.workspace
          .getConfiguration('editor', editor.document.uri)
          .get('autoClosingQuotes') === 'never'
      ) {
        await defaultType(args);
        return;
      }

      const document = editor.document;
      if (
        editor.selections.every(
          selection =>
            selection.isEmpty &&
            document.getText(
              new vscode.Range(
                selection.active,
                selection.active.translate(0, 1),
              ),
            ) === quote,
        )
      ) {
        await vscode.commands.executeCommand('cursorRight');
        return;
      }

      if (
        editor.selections.some(
          selection =>
            !selection.isEmpty ||
            selection.active.character < 2 ||
            document.getText(
              new vscode.Range(
                selection.active.translate(0, -2),
                selection.active,
              ),
            ) !== quote.repeat(2),
        )
      ) {
        await defaultType(args);
        return;
      }

      const positions = editor.selections.map(selection => selection.active);
      const client = getClient();
      const version = document.version;
      try {
        let shouldClose: boolean[] = [];
        for (let attempt = 0; attempt < 2; attempt++) {
          await new Promise(resolve => setTimeout(resolve, 0));
          try {
            shouldClose = await Promise.all(
              positions.map(position =>
                client.sendRequest<boolean>(
                  'pyrefly/textDocument/autoCloseTripleQuote',
                  {
                    textDocument:
                      client.code2ProtocolConverter.asTextDocumentIdentifier(
                        document,
                      ),
                    position:
                      client.code2ProtocolConverter.asPosition(position),
                    quote,
                  },
                ),
              ),
            );
            break;
          } catch (error) {
            if (attempt === 1) {
              throw error;
            }
          }
        }
        if (
          shouldClose.every(Boolean) &&
          vscode.window.activeTextEditor === editor &&
          document.version === version
        ) {
          await defaultType(args);
          return;
        }
      } catch {
        // Typing must continue normally if the language server is unavailable.
      }
      if (
        vscode.window.activeTextEditor !== editor ||
        document.version !== version
      ) {
        await defaultType(args);
        return;
      }
      await editor.insertSnippet(new vscode.SnippetString(quote), positions);
    }),
  );
}
