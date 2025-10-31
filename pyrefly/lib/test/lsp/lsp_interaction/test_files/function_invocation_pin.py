from __future__ import annotations

from typing import TypeVar


T = TypeVar("T")


def foo(a: T, b: T) -> T:
    return b


def bar(x: int | None, y: int | None) -> int | None:
    if x is None:
        # Regression test for https://github.com/facebook/pyrefly/issues/105
        return foo(x, y)
    if y is None:
        return foo(x, y)
    return y

