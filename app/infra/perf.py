"""Lightweight performance helpers."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

logger = logging.getLogger(__name__)


@dataclass
class Timing:
    name: str
    started_at: float
    elapsed_ms: float = 0.0


@contextmanager
def timed(name: str, *, extra: dict[str, Any] | None = None) -> Generator[Timing, None, None]:
    """Time a block and log duration at DEBUG.

    Do not include raw note text in `extra`.
    """
    timing = Timing(name=name, started_at=time.perf_counter())
    try:
        yield timing
    finally:
        timing.elapsed_ms = (time.perf_counter() - timing.started_at) * 1000.0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s elapsed_ms=%.2f", name, timing.elapsed_ms, extra=extra or {})


__all__ = ["Timing", "timed"]
