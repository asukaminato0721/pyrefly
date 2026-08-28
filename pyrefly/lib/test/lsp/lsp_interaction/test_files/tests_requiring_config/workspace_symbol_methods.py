# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class WorkspaceSymbolMethodHost:
    def __init__(self) -> None:
        self._workspace_symbol_private_member = 1

    def _workspace_symbol_private_method(self) -> None:
        return

    def workspace_symbol_method_deterministic_name(self) -> None:
        return
