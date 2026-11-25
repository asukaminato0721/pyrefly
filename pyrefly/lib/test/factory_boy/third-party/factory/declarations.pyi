from typing import Any, Callable

class LazyAttribute:
    def __init__(self, function: Callable[..., Any]) -> None: ...

class PostGeneration:
    def __init__(self, function: Callable[..., Any]) -> None: ...
