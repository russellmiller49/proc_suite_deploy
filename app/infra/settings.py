"""Infrastructure/runtime settings (env-driven).

This module centralizes operational flags used by the API runtime so that
performance behavior can be toggled quickly via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(*names: str, default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _get_float(*names: str, default: float) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return default


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return None


@dataclass(frozen=True)
class InfraSettings:
    """Runtime toggles for API startup and performance controls."""

    skip_warmup: bool
    background_warmup: bool
    wait_for_ready_s: float

    cpu_workers: int

    llm_concurrency: int
    llm_timeout_s: float

    enable_redis_cache: bool
    enable_llm_cache: bool
    enable_ml_cache: bool

    redis_url: str | None

    @staticmethod
    def from_env() -> "InfraSettings":
        skip_warmup = _truthy(_env_first("SKIP_WARMUP", "PROCSUITE_SKIP_WARMUP"))

        background_warmup_raw = _env_first("BACKGROUND_WARMUP", "PROCSUITE_BACKGROUND_WARMUP")
        background_warmup = True if background_warmup_raw is None else _truthy(background_warmup_raw)

        wait_for_ready_s = _get_float("WAIT_FOR_READY_S", "PROCSUITE_WAIT_FOR_READY_S", default=0.0)

        cpu_workers = max(1, _get_int("CPU_WORKERS", "PROCSUITE_CPU_WORKERS", default=1))

        llm_concurrency = max(1, _get_int("LLM_CONCURRENCY", "PROCSUITE_LLM_CONCURRENCY", default=2))
        llm_timeout_s = _get_float("LLM_TIMEOUT_S", "PROCSUITE_LLM_TIMEOUT_S", default=120.0)

        enable_redis_cache = _truthy(_env_first("ENABLE_REDIS_CACHE", "PROCSUITE_ENABLE_REDIS_CACHE"))
        enable_llm_cache = _truthy(_env_first("ENABLE_LLM_CACHE", "PROCSUITE_ENABLE_LLM_CACHE"))
        enable_ml_cache = _truthy(_env_first("ENABLE_ML_CACHE", "PROCSUITE_ENABLE_ML_CACHE"))

        redis_url = _env_first("REDIS_URL", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_URL")

        return InfraSettings(
            skip_warmup=skip_warmup,
            background_warmup=background_warmup,
            wait_for_ready_s=wait_for_ready_s,
            cpu_workers=cpu_workers,
            llm_concurrency=llm_concurrency,
            llm_timeout_s=llm_timeout_s,
            enable_redis_cache=enable_redis_cache,
            enable_llm_cache=enable_llm_cache,
            enable_ml_cache=enable_ml_cache,
            redis_url=redis_url,
        )


@lru_cache(maxsize=1)
def get_infra_settings() -> InfraSettings:
    return InfraSettings.from_env()


__all__ = ["InfraSettings", "get_infra_settings"]
