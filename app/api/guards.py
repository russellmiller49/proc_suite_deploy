from __future__ import annotations

import os

from fastapi import HTTPException, status


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def enforce_legacy_endpoints_allowed() -> None:
    if not _env_flag("PROCSUITE_ALLOW_LEGACY_ENDPOINTS", "0"):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Use /api/v1/process",
        )


def enforce_request_mode_override_allowed(mode_value: str | None) -> None:
    if not mode_value:
        return
    if not _env_flag("PROCSUITE_ALLOW_REQUEST_MODE_OVERRIDE", "0"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Request mode overrides are disabled. "
                "Set PROCSUITE_ALLOW_REQUEST_MODE_OVERRIDE=1 to enable."
            ),
        )


__all__ = ["enforce_legacy_endpoints_allowed", "enforce_request_mode_override_allowed"]
