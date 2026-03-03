"""Safe text helpers for canonical case snapshots."""

from __future__ import annotations

import re


_ISO_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"
)
_US_NUMERIC_DATE_RE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:\d{2}|\d{4})\b"
)
_EU_NUMERIC_DATE_RE = re.compile(
    r"\b(?:0?[1-9]|[12]\d|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:\d{2}|\d{4})\b"
)
_MONTH_NAME_RE = re.compile(
    r"\b(?:"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\s+\d{1,2}(?:\s*,?\s*(?:19|20)\d{2})?\b",
    flags=re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


def strip_date_like_strings(text: str) -> str:
    value = text or ""
    for regex in (_ISO_DATE_RE, _US_NUMERIC_DATE_RE, _EU_NUMERIC_DATE_RE, _MONTH_NAME_RE):
        value = regex.sub("[DATE]", value)
    return value


def compact_text(text: str, *, max_chars: int = 180) -> str:
    clean = _WS_RE.sub(" ", strip_date_like_strings(text or "")).strip()
    if not clean:
        return ""
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


__all__ = ["compact_text", "strip_date_like_strings"]
