# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, assert_type, Literal

import triton.language as tl


def test_broadcast_add() -> None:
    row = tl.full((1, 32), 1.0, tl.float32)
    column = tl.full((16, 1), 2.0, tl.float32)
    assert_type(row + column, tl.tensor[[16, 32], tl.dtype[Literal["fp32"]]])
    assert_type(column + row, tl.tensor[[16, 32], tl.dtype[Literal["fp32"]]])


def test_newaxis() -> None:
    offsets = tl.arange(0, 128)
    assert_type(offsets[:, None], tl.tensor[[128, 1], tl.dtype[Literal["int32"]]])
    assert_type(offsets[None, :], tl.tensor[[1, 128], tl.dtype[Literal["int32"]]])


def test_scalar_add() -> None:
    offsets = tl.arange(0, 128)
    values = tl.full((128,), 1, tl.float32)
    assert_type(offsets + 1, tl.tensor[[128], tl.dtype[Literal["int32"]]])
    assert_type(values + 1.0, tl.tensor[[128], tl.dtype[Literal["fp32"]]])
    assert_type(offsets + 1.0, tl.tensor[[128], Any])


def test_mixed_dtype_is_gradual() -> None:
    integers = tl.arange(0, 128)
    floats = tl.full((128,), 1.0, tl.float32)
    assert_type(integers + floats, tl.tensor[[128], Any])


def test_broadcast_rejection() -> None:
    a = tl.full((16, 32), 0, tl.float32)
    b = tl.full((16, 64), 0, tl.float32)
    a + b  # E: broadcast
