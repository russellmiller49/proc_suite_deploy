"""Async HTTP utilities for LLM providers.

This module is provider-agnostic and can be used to wrap outbound calls with:
- an asyncio semaphore for concurrency limiting
- exponential backoff with jitter on 429 / transient 5xx
- a hard time budget (`LLM_TIMEOUT_S`)
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from app.infra.llm_control import backoff_seconds, parse_retry_after_seconds
from app.infra.settings import get_infra_settings


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    retry_on_statuses: tuple[int, ...] = (429,)


async def post_json_with_retries(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str] | None,
    json_body: Any,
    sem: asyncio.Semaphore,
    policy: RetryPolicy | None = None,
) -> httpx.Response:
    policy = policy or RetryPolicy()
    settings = get_infra_settings()
    deadline = time.monotonic() + float(settings.llm_timeout_s)

    last_exc: Exception | None = None
    attempt = 0
    while True:
        attempt += 1
        try:
            async with sem:
                resp = await client.post(url, headers=dict(headers or {}), json=json_body)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            status_code = None
            retry_after = None
        else:
            status_code = resp.status_code
            if status_code < 400:
                return resp

            retry_after = parse_retry_after_seconds(resp.headers)
            should_retry = status_code in policy.retry_on_statuses or 500 <= status_code <= 599
            if not should_retry:
                return resp

        if attempt >= policy.max_retries or time.monotonic() >= deadline:
            if last_exc is not None:
                raise last_exc
            return resp  # type: ignore[has-type]

        sleep_s = retry_after if retry_after is not None else backoff_seconds(attempt - 1)
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            if last_exc is not None:
                raise last_exc
            return resp  # type: ignore[has-type]

        # Add small jitter even if Retry-After is provided.
        jitter = random.uniform(0.0, 0.25)
        await asyncio.sleep(min(sleep_s + jitter, remaining))


__all__ = ["RetryPolicy", "post_json_with_retries"]
