"""Common imaging parsing helpers (deterministic)."""

from __future__ import annotations

import re
from typing import Any

from app.registry.aggregation.sanitize import compact_text


_SIZE3_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(cm|mm)?",
    flags=re.IGNORECASE,
)
_SIZE2_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(cm|mm)",
    flags=re.IGNORECASE,
)
_SINGLE_SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(cm|mm)\b", flags=re.IGNORECASE)

_SUV_MAX_RE = re.compile(r"SUV\s*(?:max(?:imum)?)?\s*[:=]?\s*(\d+(?:\.\d+)?)", flags=re.IGNORECASE)
_SUV_DELAYED_RE = re.compile(
    r"(?:increases?\s+to|delayed\s+SUV\s*(?:max(?:imum)?)?\s*[:=]?)\s*(\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

_RESPONSE_INCREASE_RE = re.compile(
    r"\b(?:increased|increase|enlarged|enlarging|new\s+lesion|new\s+node)\b",
    re.IGNORECASE,
)
_RESPONSE_DECREASE_RE = re.compile(r"\b(?:decreased|decrease|smaller|resolution|resolved|regression)\b", re.IGNORECASE)
_RESPONSE_STABLE_RE = re.compile(r"\b(?:stable|unchanged|similar|no\s+new\s+lesions?)\b", re.IGNORECASE)

_COMPARATIVE_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnew\b", re.IGNORECASE), "New"),
    (re.compile(r"\bresolved|resolution\b", re.IGNORECASE), "Resolved"),
    (re.compile(r"\bincreased|enlarged|greater\b", re.IGNORECASE), "Increased"),
    (re.compile(r"\bdecreased|smaller|reduced\b", re.IGNORECASE), "Decreased"),
    (re.compile(r"\bstable|unchanged|similar\b", re.IGNORECASE), "Stable"),
]

_IMPRESSION_RE = re.compile(r"\bimpression\b\s*[:\-]\s*(.+)$", flags=re.IGNORECASE | re.MULTILINE)


def _to_mm(value: str, unit: str | None) -> int:
    num = float(value)
    if (unit or "").lower() == "cm":
        num *= 10.0
    return int(round(num))


def parse_sizes_mm(text: str) -> tuple[int | None, int | None, int | None]:
    value = text or ""
    match3 = _SIZE3_RE.search(value)
    if match3:
        unit = match3.group(4) or "mm"
        return (
            _to_mm(match3.group(1), unit),
            _to_mm(match3.group(2), unit),
            _to_mm(match3.group(3), unit),
        )

    match2 = _SIZE2_RE.search(value)
    if match2:
        unit = match2.group(3)
        return (
            _to_mm(match2.group(1), unit),
            _to_mm(match2.group(2), unit),
            None,
        )

    single = _SINGLE_SIZE_RE.search(value)
    if single:
        mm = _to_mm(single.group(1), single.group(2))
        return (mm, None, None)

    return (None, None, None)


def parse_suv_values(text: str) -> tuple[float | None, float | None]:
    value = text or ""
    suv = None
    delayed = None

    m1 = _SUV_MAX_RE.search(value)
    if m1:
        try:
            suv = float(m1.group(1))
        except ValueError:
            suv = None

    m2 = _SUV_DELAYED_RE.search(value)
    if m2:
        try:
            delayed = float(m2.group(1))
        except ValueError:
            delayed = None

    return suv, delayed


def classify_comparative_change(text: str) -> str | None:
    value = text or ""
    for pattern, label in _COMPARATIVE_MAP:
        if pattern.search(value):
            return label
    return None


def classify_response(text: str) -> str:
    value = text or ""
    has_increase = bool(_RESPONSE_INCREASE_RE.search(value))
    has_decrease = bool(_RESPONSE_DECREASE_RE.search(value))
    has_stable = bool(_RESPONSE_STABLE_RE.search(value))

    if has_increase and has_decrease:
        return "Mixed"
    if has_increase:
        return "Progression"
    if has_decrease:
        return "Response"
    if has_stable:
        return "Stable"
    return "Indeterminate"


def extract_overall_impression(text: str) -> str | None:
    value = text or ""
    match = _IMPRESSION_RE.search(value)
    if match:
        return compact_text(match.group(1), max_chars=220)

    # Fallback: first informative line.
    for line in value.splitlines():
        clean = line.strip()
        if len(clean) < 8:
            continue
        if clean.lower().startswith(("findings", "comparison")):
            continue
        return compact_text(clean, max_chars=220)
    return None


def build_snapshot(
    *,
    relative_day_offset: int | None,
    modality: str,
    subtype: str | None,
    text: str,
    qa_flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "relative_day_offset": int(relative_day_offset or 0),
        "modality": modality,
        "subtype": subtype,
        "response": classify_response(text),
        "overall_impression_text": extract_overall_impression(text),
        "qa_flags": sorted(set(qa_flags or [])),
    }


__all__ = [
    "build_snapshot",
    "classify_comparative_change",
    "classify_response",
    "extract_overall_impression",
    "parse_sizes_mm",
    "parse_suv_values",
]
