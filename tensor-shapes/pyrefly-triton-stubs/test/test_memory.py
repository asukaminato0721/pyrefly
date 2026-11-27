# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, Literal

import triton.language as tl
from shape_extensions import IntVar

type Float32 = tl.dtype[Literal["fp32"]]
type Int32 = tl.dtype[Literal["int32"]]


def test_load(pointer: tl.tensor[[], tl.pointer_type[Float32]]) -> None:
    assert_type(tl.load(pointer), tl.tensor[[], Float32])
    offsets = tl.arange(0, 128)
    pointers = pointer + offsets
    assert_type(pointers, tl.tensor[[128], tl.pointer_type[Float32]])
    assert_type(offsets + pointer, tl.tensor[[128], tl.pointer_type[Float32]])
    assert_type(pointers.dtype.element_ty, Float32)
    mask = offsets < 64
    assert_type(mask, tl.tensor[[128], tl.dtype[Literal["int1"]]])
    assert_type(tl.load(pointers, mask=mask, other=0), tl.tensor[[128], Float32])


def test_pointer_broadcast(pointer: tl.tensor[[16, 1], tl.pointer_type[Int32]]) -> None:
    offsets = tl.arange(0, 32)[None, :]
    pointers = pointer + offsets
    assert_type(tl.load(pointers), tl.tensor[[16, 32], Int32])


def test_symbolic_memory[N: IntVar](
    pointer: tl.tensor[[N], tl.pointer_type[Float32]],
) -> None:
    assert_type(tl.load(pointer), tl.tensor[[N], Float32])
    assert_type(tl.store(pointer, 1), tl.tensor[[], tl.dtype[Literal["void"]]])


def test_store_casts_and_broadcasts(
    pointer: tl.tensor[[16, 32], tl.pointer_type[Float32]],
) -> None:
    integers = tl.arange(0, 32)
    assert_type(tl.store(pointer, integers), tl.tensor[[], tl.dtype[Literal["void"]]])
    tl.store(pointer, 1)
    tl.store(pointer, tl.full((1, 32), 1, tl.int32))


def test_memory_rejections(pointer: tl.tensor[[128], tl.pointer_type[Float32]]) -> None:
    values = tl.full((128,), 0, tl.float32)
    tl.load(values)  # E: not assignable to parameter `pointer`
    tl.store(values, 0)  # E: not assignable to parameter `pointer`
    tl.store(pointer, "bad")  # E: not assignable to parameter `value`
    tl.load(pointer, mask=tl.arange(0, 128))  # E: not assignable to parameter `mask`
    pointer + pointer  # E: not supported
    pointer + values  # E: not supported


def test_memory_broadcast_rejections(
    pointer: tl.tensor[[128], tl.pointer_type[Float32]],
) -> None:
    mask = tl.full((64,), True, tl.int1)
    values = tl.full((64,), 1, tl.float32)
    tl.load(pointer, mask=mask)  # E: cannot broadcast to the pointer shape
    tl.load(pointer, other=values)  # E: cannot broadcast to the pointer shape
    tl.store(pointer, values)  # E: cannot broadcast to the pointer shape
    tl.store(pointer, 1, mask=mask)  # E: cannot broadcast to the pointer shape
