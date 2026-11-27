#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-check the stub rules with Triton's CPU interpreter and NumPy memory."""

from __future__ import annotations

import unittest

import numpy as np
import triton.language as tl
from triton.language.semantic import IncompatibleTypeErrorImpl, TritonSemantic
from triton.runtime.interpreter import InterpreterBuilder, TensorHandle


class TritonRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        """Use Triton's real semantic rules without a GPU driver."""
        self.semantic = TritonSemantic(InterpreterBuilder())

    def assert_tensor(self, value, shape, dtype) -> None:
        """Check the runtime shape and dtype against the static expectations."""
        self.assertEqual(tuple(dim.value for dim in value.shape), shape)
        self.assertEqual(value.dtype, dtype)

    def pointer(self, array):
        """Keep the NumPy array alive at the call site while using its address."""
        pointer_type = tl.pointer_type(tl.float32)
        handle = TensorHandle(np.array([array.ctypes.data], np.uint64), pointer_type)
        return tl.tensor(handle, pointer_type)

    def test_arange(self) -> None:
        self.assert_tensor(tl.arange(0, 128, _semantic=self.semantic), (128,), tl.int32)
        self.assert_tensor(tl.arange(16, 32, _semantic=self.semantic), (16,), tl.int32)

    def test_full_and_cast(self) -> None:
        value = tl.full((16, 32), 0, tl.float32, _semantic=self.semantic)
        self.assert_tensor(value, (16, 32), tl.float32)
        self.assert_tensor(
            value.to(tl.int32, _semantic=self.semantic), (16, 32), tl.int32
        )

    def test_broadcast_add(self) -> None:
        row = tl.full((1, 32), 1.0, tl.float32, _semantic=self.semantic)
        column = tl.full((16, 1), 2.0, tl.float32, _semantic=self.semantic)
        result = row.__add__(column, _semantic=self.semantic)
        self.assert_tensor(result, (16, 32), tl.float32)
        np.testing.assert_array_equal(result.handle.data, np.full((16, 32), 3.0))

    def test_newaxis(self) -> None:
        value = tl.arange(0, 128, _semantic=self.semantic)
        self.assert_tensor(
            value.__getitem__((slice(None), None), _semantic=self.semantic),
            (128, 1),
            tl.int32,
        )
        self.assert_tensor(
            value.__getitem__((None, slice(None)), _semantic=self.semantic),
            (1, 128),
            tl.int32,
        )

    def test_scalar_and_mixed_dtype_add(self) -> None:
        integers = tl.arange(0, 128, _semantic=self.semantic)
        floats = tl.full((128,), 1.0, tl.float32, _semantic=self.semantic)
        self.assert_tensor(
            integers.__add__(1, _semantic=self.semantic), (128,), tl.int32
        )
        self.assert_tensor(
            floats.__add__(1.0, _semantic=self.semantic), (128,), tl.float32
        )
        self.assert_tensor(
            integers.__add__(1.0, _semantic=self.semantic), (128,), tl.float32
        )
        self.assert_tensor(
            integers.__add__(floats, _semantic=self.semantic), (128,), tl.float32
        )

    def test_pointer_load_and_mask(self) -> None:
        array = np.arange(128, dtype=np.float32)
        pointer = self.pointer(array)
        self.assert_tensor(tl.load(pointer, _semantic=self.semantic), (), tl.float32)
        offsets = tl.arange(0, 128, _semantic=self.semantic)
        pointers = pointer.__add__(offsets, _semantic=self.semantic)
        mask = offsets.__lt__(64, _semantic=self.semantic)
        self.assert_tensor(pointers, (128,), tl.pointer_type(tl.float32))
        self.assert_tensor(mask, (128,), tl.int1)
        loaded = tl.load(pointers, mask=mask, other=0, _semantic=self.semantic)
        self.assert_tensor(loaded, (128,), tl.float32)
        np.testing.assert_array_equal(loaded.handle.data[:64], array[:64])
        np.testing.assert_array_equal(loaded.handle.data[64:], np.zeros(64))

    def test_store_casts_and_broadcasts(self) -> None:
        array = np.zeros((16, 32), dtype=np.float32)
        offsets = tl.arange(0, 512, _semantic=self.semantic)
        offsets = tl.reshape(offsets, (16, 32), _semantic=self.semantic)
        pointers = self.pointer(array).__add__(offsets, _semantic=self.semantic)
        integers = tl.arange(0, 32, _semantic=self.semantic)
        self.assert_tensor(
            tl.store(pointers, integers, _semantic=self.semantic), (), tl.void
        )
        np.testing.assert_array_equal(array, np.broadcast_to(np.arange(32), (16, 32)))

    def test_invalid_arange(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            tl.arange(16, 8, _semantic=self.semantic)

    def test_invalid_broadcast(self) -> None:
        left = tl.full((16, 32), 0, tl.float32, _semantic=self.semantic)
        right = tl.full((16, 64), 0, tl.float32, _semantic=self.semantic)
        with self.assertRaises((TypeError, ValueError)):
            left.__add__(right, _semantic=self.semantic)

    def test_invalid_memory_operations(self) -> None:
        values = tl.full((128,), 0, tl.float32, _semantic=self.semantic)
        offsets = tl.arange(0, 128, _semantic=self.semantic)
        array = np.zeros(128, dtype=np.float32)
        pointer = self.pointer(array).__add__(offsets, _semantic=self.semantic)
        operations = (
            lambda: tl.load(values, _semantic=self.semantic),
            lambda: tl.store(values, 0, _semantic=self.semantic),
            lambda: tl.store(pointer, "bad", _semantic=self.semantic),
            lambda: tl.load(pointer, mask=offsets, _semantic=self.semantic),
            lambda: pointer.__add__(pointer, _semantic=self.semantic),
            lambda: pointer.__add__(values, _semantic=self.semantic),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(
                    (TypeError, ValueError, IncompatibleTypeErrorImpl)
                ):
                    operation()

    def test_memory_broadcast_rejections(self) -> None:
        array = np.zeros(128, dtype=np.float32)
        offsets = tl.arange(0, 128, _semantic=self.semantic)
        pointer = self.pointer(array).__add__(offsets, _semantic=self.semantic)
        mask = tl.full((64,), True, tl.int1, _semantic=self.semantic)
        values = tl.full((64,), 1, tl.float32, _semantic=self.semantic)
        operations = (
            lambda: tl.load(pointer, mask=mask, _semantic=self.semantic),
            lambda: tl.load(pointer, other=values, _semantic=self.semantic),
            lambda: tl.store(pointer, values, _semantic=self.semantic),
            lambda: tl.store(pointer, 1, mask=mask, _semantic=self.semantic),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises((TypeError, ValueError)):
                    operation()


if __name__ == "__main__":
    unittest.main()
