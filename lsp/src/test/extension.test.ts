/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import {findPyprojectDependency} from '../pyproject-hover';

suite('Extension Test Suite', () => {
	const extension: vscode.Extension<unknown> | undefined = vscode.extensions.getExtension('meta.pyrefly');

	test('Test activation', async function () {
		// On macos-13, we've noticed successful test activation take up to 3500ms.
		this.timeout(10000);
		await extension?.activate();
		assert.ok(true);
	});

	test('Finds PEP 621 dependencies', async function () {
		const document = await vscode.workspace.openTextDocument({
			language: 'toml',
			content: [
				'[project]',
				'dependencies = [',
				'  "requests==2.32.0",',
				']',
			].join('\n'),
		});

		const dependency = findPyprojectDependency(
			document,
			new vscode.Position(2, 5),
		);

		assert.strictEqual(dependency?.name, 'requests');
		assert.strictEqual(dependency?.configuredVersion, '2.32.0');
	});

	test('Finds optional dependency groups', async function () {
		const document = await vscode.workspace.openTextDocument({
			language: 'toml',
			content: [
				'[project.optional-dependencies]',
				'dev = [',
				'  "pytest>=8",',
				']',
			].join('\n'),
		});

		const dependency = findPyprojectDependency(
			document,
			new vscode.Position(2, 5),
		);

		assert.strictEqual(dependency?.name, 'pytest');
		assert.strictEqual(dependency?.configuredVersion, undefined);
	});

	test('Ignores non-dependency strings', async function () {
		const document = await vscode.workspace.openTextDocument({
			language: 'toml',
			content: ['[project]', 'description = "not a dependency"'].join('\n'),
		});

		assert.strictEqual(
			findPyprojectDependency(document, new vscode.Position(1, 16)),
			undefined,
		);
	});
});
