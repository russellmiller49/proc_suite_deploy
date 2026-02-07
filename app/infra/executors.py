"""Executor helpers for running CPU-bound work off the event loop."""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, TypeVar

from fastapi import FastAPI

R = TypeVar("R")


async def run_cpu(app: FastAPI, fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    """Run a blocking function in the app's CPU executor."""
    loop = asyncio.get_running_loop()
    executor = getattr(app.state, "cpu_executor", None)
    bound = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(executor, bound)


__all__ = ["run_cpu"]
