# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, shape_transform

@shape_transform
def arange_shape(start: Int, end: Int) -> IntTuple:
    if dsl.is_concrete_int(start) and dsl.is_concrete_int(end):
        if end < start or end == start:
            return dsl.Invalid("arange end must be greater than start")
    return dsl.IntTuple((end - start,))

@shape_transform
def broadcast_into(target: IntTuple, source: IntTuple) -> IntTuple:
    if len(target) < len(source):
        return dsl.Invalid("memory operand cannot broadcast to the pointer shape")
    offset = len(target) - len(source)
    if any(
        source[index] != 1 and source[index] != target[offset + index]
        for index in range(len(source))
    ):
        return dsl.Invalid("memory operand cannot broadcast to the pointer shape")
    return target

@shape_transform
def store_shape(validated: IntTuple) -> IntTuple:
    return validated[:0]
