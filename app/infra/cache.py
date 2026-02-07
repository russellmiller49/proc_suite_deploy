"""Simple caching primitives with PHI-safe keys.

This module is intentionally lightweight and thread-safe. Callers should ensure
cache keys never include raw note text (hash keys instead).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Entry:
    value: Any
    expires_at: float | None


class MemoryCache:
    def __init__(self, *, max_size: int = 1024) -> None:
        self._max_size = max(1, int(max_size))
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, _Entry]" = OrderedDict()

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.expires_at is not None and entry.expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return entry.value

    def set(self, key: str, value: Any, *, ttl_s: float | None = None) -> None:
        expires_at = None
        if ttl_s is not None:
            ttl_s = float(ttl_s)
            if ttl_s > 0:
                expires_at = time.time() + ttl_s

        with self._lock:
            if key in self._items:
                self._items.pop(key, None)
            self._items[key] = _Entry(value=value, expires_at=expires_at)
            self._items.move_to_end(key)
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class RedisCache:
    """Optional Redis cache (requires `redis` Python package)."""

    def __init__(self, redis_url: str) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("redis package not installed; cannot use RedisCache") from exc

        self._client = redis.Redis.from_url(redis_url)

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return raw

    def set(self, key: str, value: Any, *, ttl_s: float | None = None) -> None:
        ttl_seconds = int(ttl_s) if ttl_s is not None and ttl_s > 0 else None
        payload: bytes
        try:
            payload = json.dumps(value).encode("utf-8")
        except TypeError:
            payload = str(value).encode("utf-8")
        if ttl_seconds is not None:
            self._client.setex(key, ttl_seconds, payload)
        else:
            self._client.set(key, payload)


_llm_memory_cache = MemoryCache(max_size=1024)
_ml_memory_cache = MemoryCache(max_size=2048)


def get_llm_memory_cache() -> MemoryCache:
    return _llm_memory_cache


def get_ml_memory_cache() -> MemoryCache:
    return _ml_memory_cache


__all__ = [
    "MemoryCache",
    "RedisCache",
    "get_llm_memory_cache",
    "get_ml_memory_cache",
]
