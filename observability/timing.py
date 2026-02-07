"""Timing utilities for performance measurement."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from .metrics import get_metrics_client

F = TypeVar("F", bound=Callable[..., Any])


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, tags: dict[str, str] | None = None, emit_metric: bool = True):
        self.name = name
        self.tags = tags or {}
        self.emit_metric = emit_metric
        self.start_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self) -> "TimingContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if self.emit_metric:
            get_metrics_client().timing(self.name, self.elapsed_ms, self.tags)


@contextmanager
def timed(
    name: str, tags: dict[str, str] | None = None, emit_metric: bool = True
) -> Generator[TimingContext, None, None]:
    """Context manager for timing code blocks.

    Usage:
        with timed("my_operation") as t:
            do_work()
        print(f"Took {t.elapsed_ms:.2f}ms")
    """
    ctx = TimingContext(name, tags, emit_metric)
    with ctx:
        yield ctx


def timed_function(
    name: str | None = None, tags: dict[str, str] | None = None
) -> Callable[[F], F]:
    """Decorator for timing function execution.

    Usage:
        @timed_function("my_function")
        def my_function():
            ...
    """

    def decorator(func: F) -> F:
        metric_name = name or f"function.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed(metric_name, tags):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
