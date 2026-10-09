# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for https://github.com/facebook/pyrefly/issues/5161."""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import nn, Tensor


def test_missing_constants() -> None:
    assert_type(torch.channels_last, torch.memory_format)
    assert_type(torch.uint8, torch.dtype)
    tensor = torch.zeros((2, 3, 4, 5)).to(memory_format=torch.channels_last)
    assert_shape(tensor.shape, (2, 3, 4, 5))
    assert tensor.is_contiguous(memory_format=torch.channels_last)
    assert torch.tensor([1, 2], dtype=torch.uint8).dtype == torch.uint8


def test_float_tensor() -> None:
    tensor: torch.FloatTensor = torch.FloatTensor([[1, 2], [3, 4]])
    assert_type(tensor, Tensor)
    assert_shape(torch.FloatTensor([[1, 2], [3, 4]]).shape, (2, 2))
    assert tensor.dtype == torch.float32


def test_unique() -> None:
    tensor = torch.tensor([[1, 1, 2], [1, 1, 2]])
    assert_type(torch.unique(tensor), Tensor)
    assert_shape(torch.unique(tensor).shape, IntTuple, runtime=(2,))
    assert torch.unique(tensor).tolist() == [1, 2]
    values, inverse = torch.unique(tensor, return_inverse=True)
    assert_type(values, Tensor)
    assert_type(inverse, Tensor)
    assert torch.equal(values[inverse], tensor)
    values, counts = torch.unique(tensor, return_counts=True)
    assert_type(counts, Tensor)
    assert counts.tolist() == [4, 2]
    values, inverse, counts = torch.unique(tensor, True, True, True, 0)
    assert_type(values, Tensor)
    assert_type(inverse, Tensor)
    assert_type(counts, Tensor)
    assert tuple(values.shape) == (1, 3)
    assert inverse.tolist() == [0, 0]
    assert counts.tolist() == [2]
    assert_type(torch.unique(tensor, True, False, True), tuple[Tensor, Tensor])
    assert_type(
        torch.unique(tensor, return_inverse=True, dim=-1), tuple[Tensor, Tensor]
    )


def test_module_parameters() -> None:
    model = nn.Linear(2, 2)
    for name, parameter in model.named_parameters():
        assert_type(name, str)
        assert_type(parameter, nn.Parameter)
        assert isinstance(parameter, nn.Parameter)
    for parameter in model.parameters():
        assert_type(parameter, nn.Parameter)
        assert isinstance(parameter, nn.Parameter)
    parameters = list(model.parameters())
    assert_shape(parameters[0].shape, IntTuple, runtime=(2, 2))
    assert_shape(parameters[1].shape, IntTuple, runtime=(2,))


def test_scalar_comparisons() -> None:
    tensor = torch.zeros(3)
    assert_type(tensor != 0.0, Tensor[[3]])
    assert_type(tensor == 0.0, Tensor[[3]])
    assert_shape((tensor != 0.0).float().shape, (3,))
    assert_shape(torch.all(tensor == 0.0).shape, ())


def test_adaptive_pool_scalar_size() -> None:
    tensor = torch.ones((2, 3, 4, 5))
    assert_shape(nn.AdaptiveAvgPool2d(1)(tensor).shape, (2, 3, 1, 1))
    assert_shape(nn.AdaptiveAvgPool2d(2)(tensor).shape, (2, 3, 2, 2))
    assert_shape(nn.AdaptiveAvgPool2d((1, 2))(tensor).shape, (2, 3, 1, 2))


def test_randint_without_low() -> None:
    assert_shape(torch.randint(10, (1,)).shape, (1,))
    assert_shape(torch.randint(high=10, size=(2, 3)).shape, (2, 3))
    assert_shape(torch.randint(2, 10, (3,)).shape, (3,))
    assert_shape(torch.randint(low=2, high=10, size=(3,)).shape, (3,))


if TYPE_CHECKING:

    def check_tensor_narrowing(value: float | Tensor, index: Tensor) -> None:
        if torch.is_tensor(value):
            assert_type(value, Tensor)
            value[index]
        else:
            assert_type(value, float)

    def check_float_tensor_annotation(value: torch.FloatTensor | None) -> None:
        if value is not None:
            assert_type(value, Tensor)
            value[:, 1:]

    def check_unique_dynamic_flags(value: Tensor, flag: bool) -> None:
        assert_type(
            torch.unique(value, return_inverse=flag, return_counts=flag),
            Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
        )
