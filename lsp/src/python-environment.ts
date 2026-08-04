/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {PythonExtension} from '@vscode/python-extension';
import {PythonEnvironments} from '@vscode/python-environments';

const DISMISSED_KEY = 'pyrefly.dismissedPythonExtensionWarning';

interface InterpreterProvider {
  getPath(uri?: vscode.Uri): Promise<string | undefined>;
  onDidChange(callback: () => void): vscode.Disposable;
}

export function selectInterpreterPath(
  managerId: string | undefined,
  preferredPath: string | undefined,
  legacyPath: string | undefined,
): string | undefined {
  return managerId === 'ms-python.python:system' &&
    legacyPath !== undefined &&
    /(^|[\\/])\.pixi[\\/]envs[\\/]/.test(legacyPath)
    ? legacyPath
    : preferredPath;
}

export class PythonEnvironment {
  private provider: Promise<InterpreterProvider | undefined>;
  private listeners: (() => void)[] = [];
  private listenerDisposables: vscode.Disposable[] = [];
  private context: vscode.ExtensionContext;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.provider = this.tryResolveProvider().then(provider => {
      if (!provider) {
        this.showInstallWarning();
      }
      return provider;
    });
    this.watchExtensionChanges();
  }

  private async tryResolveProvider(): Promise<InterpreterProvider | undefined> {
    const [pythonEnvironments, pythonExtension] = await Promise.all([
      PythonEnvironments.api().catch(() => undefined),
      PythonExtension.api().catch(() => undefined),
    ]);

    if (pythonEnvironments) {
      const preferred = pythonEnvironments;
      const legacy = pythonExtension;
      return {
        async getPath(uri?: vscode.Uri) {
          const env = await preferred.getEnvironment(uri);
          const path = env?.execInfo?.run?.executable;
          if (
            env?.envId.managerId === 'ms-python.python:system' &&
            legacy
          ) {
            // The legacy Python extension detects Pixi without a separate
            // environment-manager extension.
            const legacyPath = await legacy.environments.getActiveEnvironmentPath(
              uri,
            );
            return selectInterpreterPath(
              env.envId.managerId,
              path,
              legacyPath.path,
            );
          }
          return path;
        },
        onDidChange(callback: () => void) {
          const disposables = [
            preferred.onDidChangeEnvironment(() => callback()),
          ];
          if (legacy) {
            disposables.push(
              legacy.environments.onDidChangeActiveEnvironmentPath(callback),
            );
          }
          return vscode.Disposable.from(...disposables);
        },
      };
    }

    if (pythonExtension) {
      const legacy = pythonExtension;
      return {
        async getPath(uri?: vscode.Uri) {
          const envPath =
            await legacy.environments.getActiveEnvironmentPath(uri);
          return envPath.path.length > 0 ? envPath.path : undefined;
        },
        onDidChange(callback: () => void) {
          return legacy.environments.onDidChangeActiveEnvironmentPath(callback);
        },
      };
    }

    return undefined;
  }

  private showInstallWarning() {
    if (this.context.globalState.get(DISMISSED_KEY)) {
      return;
    }
    const install = 'Install';
    const dismiss = "Don't Show Again";
    vscode.window
      .showInformationMessage(
        'Install the Python extension (ms-python.python) for improved experience with Pyrefly, including automatic Python environment detection.',
        install,
        dismiss,
      )
      .then(selection => {
        if (selection === install) {
          vscode.commands.executeCommand(
            'workbench.extensions.installExtension',
            'ms-python.python',
          );
        } else if (selection === dismiss) {
          this.context.globalState.update(DISMISSED_KEY, true);
        }
      });
  }

  private watchExtensionChanges() {
    let hadPyEnvs = !!vscode.extensions.getExtension(
      'ms-python.vscode-python-envs',
    );
    let hadMsPython = !!vscode.extensions.getExtension('ms-python.python');

    this.context.subscriptions.push(
      vscode.extensions.onDidChange(() => {
        const hasPyEnvs = !!vscode.extensions.getExtension(
          'ms-python.vscode-python-envs',
        );
        const hasMsPython = !!vscode.extensions.getExtension(
          'ms-python.python',
        );

        if (hasPyEnvs === hadPyEnvs && hasMsPython === hadMsPython) {
          return;
        }
        hadPyEnvs = hasPyEnvs;
        hadMsPython = hasMsPython;

        this.tryResolveProvider()
          .then(provider => {
            for (const d of this.listenerDisposables) {
              d.dispose();
            }
            this.listenerDisposables = [];

            this.provider = Promise.resolve(provider);

            if (provider) {
              for (const listener of this.listeners) {
                listener();
                this.listenerDisposables.push(
                  provider.onDidChange(listener),
                );
              }
            }
          })
          .catch(() => {});
      }),
    );
  }

  async getInterpreterPath(uri?: vscode.Uri): Promise<string | undefined> {
    const provider = await this.provider;
    if (!provider) {
      return undefined;
    }
    return provider.getPath(uri);
  }

  async onDidChangeInterpreter(
    callback: () => void,
  ): Promise<vscode.Disposable> {
    this.listeners.push(callback);
    const provider = await this.provider;
    if (provider) {
      this.listenerDisposables.push(provider.onDidChange(callback));
    }
    return new vscode.Disposable(() => {
      const idx = this.listeners.indexOf(callback);
      if (idx >= 0) {
        this.listeners.splice(idx, 1);
      }
    });
  }
}
