from __future__ import annotations

import re

# Absolute date patterns (PHI leak guardrails). Intentionally conservative:
# - prefers clear YYYY-MM-DD / MM/DD/YYYY shapes
# - does not attempt to match relative expressions like "POD #2"
_ISO_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/]"
    r"(?:0?[1-9]|[12]\d|3[01])\b"
)
_US_NUMERIC_DATE_RE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/]"
    r"(?:\d{2}|\d{4})\b"
)
_MONTH_NAME_RE = re.compile(
    r"\b(?:"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\s+\d{1,2}(?:\s*,?\s*(?:19|20)\d{2})?\b",
    flags=re.IGNORECASE,
)

_DOC_T_OFFSET_RE = re.compile(r"\bT\s*([+-]\d+)\b")
_SYSTEM_HEADER_RE = re.compile(r"\[SYSTEM:[^\]]*\]")
_SYSTEM_HEADER_LINE_RE = re.compile(r"^\s*\[SYSTEM:[^\]]*\]\s*(?:\r?\n)+")


def strip_bracket_tokens(text: str) -> str:
    """Compatibility helper: remove bracketed tokens like `[DATE: ...]`."""

    return re.sub(r"\[[A-Z_ ]{2,32}:[^\]]*\]", " ", text or "")


def count_date_like_strings(text: str) -> int:
    """Return the number of absolute date-like matches in *text*.

    Never returns the matched substrings to avoid echoing PHI.
    """

    clean = text or ""
    if not clean.strip():
        return 0

    count = 0
    for regex in (_ISO_DATE_RE, _US_NUMERIC_DATE_RE, _MONTH_NAME_RE):
        count += len(list(regex.finditer(clean)))
    return count


def extract_doc_t_offset_days(text: str) -> int | None:
    """Extract a bundle-relative doc offset from text.

    Convention: the client prepends a header containing `T+N` / `T-N` tokens.
    This parser only accepts signed offsets (requires + or -).
    """

    if not text:
        return None

    head = text[:400]
    system = _SYSTEM_HEADER_RE.search(head)
    if system:
        match = _DOC_T_OFFSET_RE.search(system.group(0))
    else:
        match = _DOC_T_OFFSET_RE.search(head)
    if not match:
        return None

    raw = match.group(1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def strip_system_header(text: str) -> str:
    """Remove a leading `[SYSTEM: ...]` header line from a bundle document."""

    if not text:
        return ""
    return _SYSTEM_HEADER_LINE_RE.sub("", text, count=1)


__all__ = [
    "count_date_like_strings",
    "extract_doc_t_offset_days",
    "strip_bracket_tokens",
    "strip_system_header",
]
