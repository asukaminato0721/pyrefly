/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_singledispatchmethod,
    r#"
from functools import singledispatchmethod
from typing import assert_type

class Formatter:
    @singledispatchmethod
    def format(self, value: object, suffix: str = "") -> str:
        return str(value) + suffix

    @format.register
    def integer(self, value: int, suffix: str = "") -> int:
        return value

    @format.register(bytes)
    def binary(self, value: object, suffix: str = "") -> bytes:
        return b""

f = Formatter()
assert_type(f.format(1), int)
assert_type(f.format(b""), bytes)
assert_type(f.format("text"), str)
assert_type(f.integer(1), int)
f.format(1, 2)  # E: Argument `Literal[2]` is not assignable to parameter `suffix` with type `str`
f.format("text", 1)  # E: Argument `Literal[1]` is not assignable to parameter `suffix` with type `str`
f.integer("text")  # E: Argument `Literal['text']` is not assignable to parameter `value` with type `int`
assert_type(Formatter.format(f, 1), str)
method = f.format
assert_type(method(1), int)
"#,
);

testcase!(
    test_singledispatchmethod_runtime_mro,
    r#"
from functools import singledispatchmethod
from typing import assert_type

class Left: pass
class Right: pass
class Both(Left, Right): pass

class Dispatch:
    @singledispatchmethod
    def method(self, value: object) -> None:
        return None

    @method.register
    def right(self, value: Right) -> bytes:
        return b""

    @method.register
    def left(self, value: Left) -> str:
        return ""

    @method.register
    def floating(self, value: float) -> float:
        return value

dispatch = Dispatch()
assert_type(dispatch.method(Both()), str)
assert_type(dispatch.method(1), None)
assert_type(dispatch.method(1.0), float)
"#,
);

testcase!(
    test_singledispatchmethod_union_and_specificity,
    r#"
from functools import singledispatchmethod
from typing import Any, assert_type

class Dispatch:
    @singledispatchmethod
    def method(self, value: object) -> bytes:
        return b""

    @method.register
    def boolean(self, value: bool) -> str:
        return ""

    @method.register
    def integer(self, value: int) -> int:
        return value

    @method.register
    def text(self, value: str | float) -> float:
        return 0.0

def check(value: int | bytes, unknown: Any) -> None:
    dispatch = Dispatch()
    assert_type(dispatch.method(True), str)
    assert_type(dispatch.method(1), int)
    assert_type(dispatch.method("text"), float)
    assert_type(dispatch.method(1.0), float)
    assert_type(dispatch.method(value), int | bytes)
    assert_type(dispatch.method(unknown), Any)
"#,
);

testcase!(
    test_singledispatchmethod_staticmethod_and_generics,
    r#"
from functools import singledispatchmethod
from typing import Self, assert_type

class Dispatch:
    @singledispatchmethod
    @staticmethod
    def method[T](value: T) -> T:
        return value

    @method.register
    @staticmethod
    def integer[T](value: int, result: T) -> T:
        return result

assert_type(Dispatch.method("text"), str)
assert_type(Dispatch().method(1, "text"), str)

class Builder:
    @singledispatchmethod
    def build(self, value: object) -> object:
        return value

    @build.register
    def integer(self, value: int) -> Self:
        return self

class Child(Builder): pass
assert_type(Child().build(1), Child)

class Box[T]:
    def __init__(self, value: T) -> None:
        self.value = value

    @singledispatchmethod
    def get(self, key: object) -> None:
        return None

    @get.register
    def integer(self, key: int) -> T:
        return self.value

assert_type(Box("text").get(1), str)
"#,
);

testcase!(
    test_singledispatchmethod_imported_registry,
    {
        let mut env = TestEnv::new();
        env.add(
            "base",
            r#"
from functools import singledispatchmethod

class Base:
    @singledispatchmethod
    def method(self, value: object) -> str:
        return ""

    @method.register
    def integer(self, value: int) -> int:
        return value
"#,
        );
        env
    },
    r#"
from base import Base
from typing import assert_type

class Child(Base):
    @Base.method.register(bytes)
    def binary(self, value: bytes) -> bytes:
        return value

assert_type(Child().method(1), int)
assert_type(Base().method(b""), bytes)
assert_type(Child().method(b""), bytes)
"#,
);

testcase!(
    test_singledispatchmethod_shared_registry,
    r#"
from functools import singledispatchmethod
from typing import assert_type

class Base:
    @singledispatchmethod
    def method(self, value: object) -> str:
        return "fallback"

class Child(Base):
    @Base.method.register
    def integer(self, value: int) -> int:
        return value

class Sibling(Base):
    pass

class Independent(Base):
    @singledispatchmethod
    def method(self, value: object) -> str:
        return "separate registry"

class Later(Base):
    @Base.method.register(bytes)
    def binary(self, value: bytes) -> bytes:
        return value

assert_type(Base().method(1), int)
assert_type(Child().method(1), int)
assert_type(Sibling().method(1), int)
assert_type(Sibling().method("text"), str)
assert_type(Independent().method(1), str)
assert_type(Base().method(b""), bytes)
assert_type(Sibling().method(b""), bytes)
assert_type(Independent().method(b""), str)
"#,
);

testcase!(
    test_singledispatchmethod_classmethod,
    r#"
from functools import singledispatchmethod
from typing import assert_type

class Factory:
    @singledispatchmethod
    @classmethod
    def make(cls, value: object) -> str:
        return "fallback"

    @make.register
    @classmethod
    def integer(cls, value: int) -> int:
        return value

assert_type(Factory.make(1), int)
assert_type(Factory().make(1), int)
assert_type(Factory.make("text"), str)
"#,
);
