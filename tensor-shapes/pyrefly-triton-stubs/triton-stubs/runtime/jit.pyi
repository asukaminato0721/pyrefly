# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any, overload

class JITFunction[**P, R]:
    fn: Callable[P, R]
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    # Launch arguments can be host tensors that Triton converts to device pointers.
    def __getitem__(
        self, grid: tuple[int, ...] | Callable[[dict[str, Any]], tuple[int, ...]]
    ) -> Callable[..., Any]: ...

class _JitDecorator:
    def __call__[**P, R](self, fn: Callable[P, R]) -> JITFunction[P, R]: ...

@overload
def jit[**P, R](fn: Callable[P, R], **kwargs: Any) -> JITFunction[P, R]: ...
@overload
def jit(**kwargs: Any) -> _JitDecorator: ...
