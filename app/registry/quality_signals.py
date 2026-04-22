from __future__ import annotations

import json
from typing import Any


QUALITY_SIGNAL_PREFIX = "QUALITY_SIGNAL:"


def make_quality_signal_warning(
    signal: str,
    *,
    field: str | None = None,
    action: str | None = None,
    detail: str | None = None,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Encode a structured quality-pass signal into a legacy warning string."""
    payload: dict[str, Any] = {"signal": str(signal).strip()}
    if field:
        payload["field"] = field
    if action:
        payload["action"] = action
    if detail:
        payload["detail"] = detail
    if source:
        payload["source"] = source
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key and value not in (None, "", [], {}):
                payload[str(key)] = value
    return QUALITY_SIGNAL_PREFIX + json.dumps(payload, separators=(",", ":"), sort_keys=True)


def parse_quality_signal_warning(warning: Any) -> dict[str, Any] | None:
    """Parse a structured quality-pass signal from a warning string."""
    if not isinstance(warning, str) or not warning.startswith(QUALITY_SIGNAL_PREFIX):
        return None
    raw = warning[len(QUALITY_SIGNAL_PREFIX) :].strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    signal = str(payload.get("signal") or "").strip()
    if not signal:
        return None
    payload["signal"] = signal
    return payload


__all__ = [
    "QUALITY_SIGNAL_PREFIX",
    "make_quality_signal_warning",
    "parse_quality_signal_warning",
]
