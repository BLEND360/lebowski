from typing import Any, Callable


class postfork:
    def __init__(self, f: Callable[[], None]) -> None:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...
