# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    type Float32 = tl.dtype[Literal["fp32"]]
    type FloatPointer = tl.tensor[[], tl.pointer_type[Float32]]


@triton.jit
def vector_add(x: FloatPointer, y: FloatPointer, out: FloatPointer) -> None:
    offsets = tl.arange(0, 128)
    x_values = tl.load(x + offsets)
    y_values = tl.load(y + offsets)
    values = x_values + y_values
    if TYPE_CHECKING:
        assert_type(offsets, tl.tensor[[128], tl.dtype[Literal["int32"]]])
        assert_type(values, tl.tensor[[128], Float32])
    tl.store(out + offsets, values)


@triton.jit
def invalid_load() -> None:
    offsets = tl.arange(0, 128)
    tl.load(offsets)  # E: not assignable to parameter `pointer`
