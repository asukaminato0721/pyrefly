/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_struct_unpack,
    r#"
import struct
from collections.abc import Iterator
from typing import Any, assert_type

assert_type(struct.unpack(">bhl", b""), tuple[int, int, int])
assert_type(struct.unpack("2s?xfd", b""), tuple[bytes, bool, float, float])
assert_type(struct.unpack("0s 2c", b""), tuple[bytes, bytes, bytes])
assert_type(struct.unpack(b"i", b""), tuple[int])
assert_type(struct.unpack("", b""), tuple[()])
assert_type(struct.unpack_from("?F", b""), tuple[bool, complex])
assert_type(struct.iter_unpack("i2s", b""), Iterator[tuple[int, bytes]])

fmt: str = "i"
assert_type(struct.unpack(fmt, b""), tuple[Any, ...])
assert_type(struct.unpack("not a format", b""), tuple[Any, ...])
"#,
);

testcase!(
    test_struct_pack,
    r#"
import struct

class Indexable:
    def __index__(self) -> int:
        return 0

class Floatable:
    def __float__(self) -> float:
        return 0.0

struct.pack(">bhl", 1, 2, 3)
struct.pack("2s?xfd", b"x", object(), 1, 2.0)
struct.pack("0s2c", bytearray(), b"x", b"y")
struct.pack_into("i", bytearray(), 0, 1)
struct.pack("if", Indexable(), Floatable())

struct.pack("ii", 1)  # E: `struct.pack` expects 2 values, got 1
struct.pack("i", 1, 2)  # E: `struct.pack` expects 1 value, got 2
struct.pack_into("i", bytearray(), 0)  # E: `struct.pack_into` expects 1 value, got 0
struct.pack("i", "not an int")  # E: Argument `Literal['not an int']` is not assignable to parameter with type `int` in function `_struct.pack`
struct.pack("s", "not bytes")  # E: Argument `Literal['not bytes']` is not assignable to parameter with type `bytearray | bytes` in function `_struct.pack`
struct.pack("c", bytearray(b"x"))  # E: Argument `bytearray` is not assignable to parameter with type `bytes` in function `_struct.pack`
"#,
);

testcase!(
    test_struct_function_aliases,
    r#"
from struct import pack as encode, unpack as decode
from typing import assert_type

assert_type(decode("i?", b""), tuple[int, bool])
encode("i", "bad")  # E: Argument `Literal['bad']` is not assignable to parameter with type `int` in function `_struct.pack`
"#,
);
