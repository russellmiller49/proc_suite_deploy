"""PHI-safe logging helpers."""

from __future__ import annotations

import hashlib


def safe_log_text(text: str | None) -> str:
    """Return a PHI-safe representation of an arbitrary text blob.

    This function intentionally does NOT return raw text. It returns only a short
    hash and length so logs can correlate repeated inputs without storing PHI.
    """
    normalized = (text or "").strip()
    if not normalized:
        return "<empty>"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"<sha256={digest} len={len(normalized)}>"


__all__ = ["safe_log_text"]
