"""Readiness gating for heavy endpoints.

This module is intentionally dependency-light to avoid import cycles with
`app.api.fastapi_app`.
"""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request

from app.infra.settings import get_infra_settings

_DEFAULT_RETRY_AFTER_S = 10


async def require_ready(request: Request) -> None:
    """Fail fast (503) if heavy resources are still warming up."""
    if bool(getattr(request.app.state, "model_ready", False)):
        return

    model_error = getattr(request.app.state, "model_error", None)
    if model_error:
        raise HTTPException(status_code=503, detail=f"Warmup failed: {model_error}")

    wait_s = float(get_infra_settings().wait_for_ready_s)
    if wait_s > 0:
        try:
            await asyncio.wait_for(request.app.state.ready_event.wait(), timeout=wait_s)
        except TimeoutError:
            pass

        if bool(getattr(request.app.state, "model_ready", False)):
            return

        model_error = getattr(request.app.state, "model_error", None)
        if model_error:
            raise HTTPException(status_code=503, detail=f"Warmup failed: {model_error}")

    raise HTTPException(
        status_code=503,
        detail="Service warming up",
        headers={"Retry-After": str(_DEFAULT_RETRY_AFTER_S)},
    )


__all__ = ["require_ready"]
