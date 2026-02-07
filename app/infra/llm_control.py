"""Concurrency limiting + retry utilities for outbound LLM calls."""

from __future__ import annotations

import hashlib
import random
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Mapping

from app.infra.settings import get_infra_settings


@lru_cache(maxsize=1)
def get_llm_semaphore() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(value=get_infra_settings().llm_concurrency)


@contextmanager
def llm_slot() -> None:
    """Global concurrency gate for LLM requests (thread-safe)."""
    sem = get_llm_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def make_llm_cache_key(*, model: str, prompt: str, prompt_version: str) -> str:
    payload = f"{model}\n{prompt_version}\n{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return None


def backoff_seconds(attempt: int, *, base: float = 0.75, cap: float = 10.0) -> float:
    """Exponential backoff with jitter."""
    exp = base * (2**max(0, int(attempt)))
    jitter = random.uniform(0.0, base)
    return min(cap, exp + jitter)


def within_deadline(deadline: float) -> bool:
    return time.monotonic() < deadline


__all__ = [
    "backoff_seconds",
    "get_llm_semaphore",
    "llm_slot",
    "make_llm_cache_key",
    "parse_retry_after_seconds",
    "within_deadline",
]
