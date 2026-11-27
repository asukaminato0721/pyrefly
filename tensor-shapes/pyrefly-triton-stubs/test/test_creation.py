# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, assert_type, Literal

import triton
import triton.language as tl
from shape_extensions import Int, IntTuple, IntVar


def test_arange() -> None:
    assert_type(tl.arange(0, 128), tl.tensor[[128], tl.dtype[Literal["int32"]]])
    assert_type(tl.arange(16, 32), tl.tensor[[16], tl.dtype[Literal["int32"]]])
    assert_type(tl.arange(0, 128).shape, tl.tuple[tl.constexpr])


def test_symbolic_arange[N: IntVar](end: Int[N]) -> None:
    assert_type(tl.arange(0, end), tl.tensor[[N], tl.dtype[Literal["int32"]]])


def test_gradual_arange(end: int) -> None:
    assert_type(tl.arange(0, end), tl.tensor[[int], tl.dtype[Literal["int32"]]])


def test_full_and_cast() -> None:
    values = tl.full((16, 32), 0, tl.float32)
    assert_type(values, tl.tensor[[16, 32], tl.dtype[Literal["fp32"]]])
    assert_type(values.to(tl.int32), tl.tensor[[16, 32], tl.dtype[Literal["int32"]]])
    assert_type(tl.full((16,), 0), tl.tensor[[16], Any])


def test_gradual_shape(shape: tuple[int, ...]) -> None:
    assert_type(
        tl.full(shape, 0, tl.float32), tl.tensor[IntTuple, tl.dtype[Literal["fp32"]]]
    )


def test_invalid_arange() -> None:
    tl.arange(16, 8)  # E: arange end must be greater than start
    tl.arange(0, "bad")  # E: While checking parameter `end`


@triton.jit
def test_constexpr_arange(block: tl.constexpr) -> None:
    assert_type(tl.arange(0, block), tl.tensor[[int], tl.dtype[Literal["int32"]]])
