# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import Any, Literal, overload

from shape_extensions import broadcast, Index, index_shape, Int, IntTuple
from triton.language._shapes import arange_shape, broadcast_into, store_shape

class dtype[Name: str = str]:
    def __init__(self, name: Name) -> None: ...
    @property
    def name(self) -> Name: ...

class pointer_type[Element: dtype = dtype](dtype[str]):
    def __init__(
        self, element_ty: Element, address_space: int = 1, const: bool = False
    ) -> None: ...
    @property
    def element_ty(self) -> Element: ...

type _Numeric = dtype[Literal["int1", "int32", "fp32"]]
type _Integer = dtype[Literal["int1", "int32"]]
type _Int32 = dtype[Literal["int32"]]
type _Float32 = dtype[Literal["fp32"]]
type _Bool = dtype[Literal["int1"]]
type _Void = dtype[Literal["void"]]
_DType = dtype

int1: _Bool
int32: _Int32
float32: _Float32
void: _Void

class tensor[Shape: IntTuple = IntTuple, DType: dtype = Any]:
    @property
    def shape(self) -> tuple[constexpr]: ...
    @property
    def dtype(self) -> DType: ...
    def __getitem__[I: Index](
        self, index: I
    ) -> tensor[index_shape(Shape, I), DType]: ...
    def to[NewDType: _DType](
        self,
        dtype: NewDType,
        fp_downcast_rounding: str | None = None,
        bitcast: bool = False,
    ) -> tensor[Shape, NewDType]: ...
    @overload
    def __add__[Other: IntTuple, Element: _DType](
        self: tensor[Shape, pointer_type[Element]], other: tensor[Other, _Integer]
    ) -> tensor[broadcast(Shape, Other), pointer_type[Element]]: ...
    @overload
    def __add__[Other: IntTuple, Element: _DType](
        self: tensor[Shape, _Integer], other: tensor[Other, pointer_type[Element]]
    ) -> tensor[broadcast(Shape, Other), pointer_type[Element]]: ...
    @overload
    def __add__(
        self: tensor[Shape, pointer_type], other: int
    ) -> tensor[Shape, DType]: ...
    @overload
    def __add__[Other: IntTuple](
        self: tensor[Shape, _Float32], other: tensor[Other, _Float32]
    ) -> tensor[broadcast(Shape, Other), _Float32]: ...
    @overload
    def __add__[Other: IntTuple](
        self: tensor[Shape, _Int32], other: tensor[Other, _Int32]
    ) -> tensor[broadcast(Shape, Other), _Int32]: ...
    @overload
    def __add__[Other: IntTuple](
        self: tensor[Shape, _Bool], other: tensor[Other, _Bool]
    ) -> tensor[broadcast(Shape, Other), _Bool]: ...
    @overload
    def __add__[Other: IntTuple](
        self: tensor[Shape, _Numeric], other: tensor[Other, _Numeric]
    ) -> tensor[broadcast(Shape, Other)]: ...
    @overload
    def __add__(
        self: tensor[Shape, _Float32], other: float
    ) -> tensor[Shape, _Float32]: ...
    @overload
    def __add__(self: tensor[Shape, _Int32], other: int) -> tensor[Shape, _Int32]: ...
    @overload
    def __add__(self: tensor[Shape, _Numeric], other: float) -> tensor[Shape]: ...
    __radd__ = __add__
    @overload
    def __lt__[Other: IntTuple](
        self: tensor[Shape, _Numeric], other: tensor[Other, _Numeric]
    ) -> tensor[broadcast(Shape, Other), _Bool]: ...
    @overload
    def __lt__(self: tensor[Shape, _Numeric], other: float) -> tensor[Shape, _Bool]: ...

class constexpr:
    value: Any
    def __init__(self, value: Any) -> None: ...

class tuple[Element]:
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Element]: ...
    def __getitem__(self, index: int) -> Element: ...

def arange[Start: Int, End: Int](
    start: Start, end: End, _semantic: Any = None
) -> tensor[arange_shape(Start, End), _Int32]: ...
@overload
def full[Shape: IntTuple, DType: dtype](
    shape: Shape,
    value: int | float,
    dtype: DType,
    _semantic: Any = None,
) -> tensor[Shape, DType]: ...
@overload
def full[Shape: IntTuple](
    shape: Shape,
    value: int | float,
    dtype: None = None,
    _semantic: Any = None,
) -> tensor[Shape]: ...
def load[Shape: IntTuple, Element: dtype, Mask: IntTuple = [], Other: IntTuple = []](
    pointer: tensor[Shape, pointer_type[Element]],
    mask: tensor[Mask, _Bool] | bool | None = None,
    other: tensor[Other, _Numeric] | float | None = None,
    *,
    cache_modifier: str = "",
    eviction_policy: str = "",
    volatile: bool = False,
    _semantic: Any = None,
) -> tensor[broadcast_into(broadcast_into(Shape, Mask), Other), Element]: ...
def store[Shape: IntTuple, Element: dtype, Value: IntTuple = [], Mask: IntTuple = []](
    pointer: tensor[Shape, pointer_type[Element]],
    value: tensor[Value, _Numeric] | float,
    mask: tensor[Mask, _Bool] | bool | None = None,
    *,
    cache_modifier: str = "",
    eviction_policy: str = "",
    _semantic: Any = None,
) -> tensor[store_shape(broadcast_into(broadcast_into(Shape, Value), Mask)), _Void]: ...
