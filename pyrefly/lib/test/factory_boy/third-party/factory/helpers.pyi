from typing import Any, Callable

from factory.declarations import PostGeneration

def post_generation(fun: Callable[..., Any]) -> PostGeneration: ...
