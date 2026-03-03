"""Normalization utilities for imaging extraction."""

from __future__ import annotations

import re


_LOBE_PATTERNS: list[tuple[str, str]] = [
    (r"\bright\s+upper\s+lobe\b|\bRUL\b", "RUL"),
    (r"\bright\s+middle\s+lobe\b|\bRML\b", "RML"),
    (r"\bright\s+lower\s+lobe\b|\bRLL\b", "RLL"),
    (r"\bleft\s+upper\s+lobe\b|\bLUL\b", "LUL"),
    (r"\bleft\s+lower\s+lobe\b|\bLLL\b", "LLL"),
]

_SEGMENT_HINTS = [
    "posterior basal segment",
    "apicoposterior segment",
    "superior segment",
    "anterior segment",
    "posterior segment",
    "medial segment",
    "lateral segment",
]


def normalize_lobe(text: str) -> str | None:
    value = text or ""
    for pattern, label in _LOBE_PATTERNS:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return label
    return None


def extract_laterality(text: str) -> str | None:
    value = text or ""
    if re.search(r"\b(?:right|rt)\b", value, flags=re.IGNORECASE):
        return "R"
    if re.search(r"\b(?:left|lt)\b", value, flags=re.IGNORECASE):
        return "L"
    return None


def normalize_segment(text: str) -> str | None:
    value = (text or "").lower()
    for phrase in _SEGMENT_HINTS:
        if phrase in value:
            return phrase

    match = re.search(r"\bsegment\s+([a-z0-9\- ]{1,24})\b", value)
    if match:
        return match.group(1).strip()

    rb = re.search(r"\b[RL]B\d{1,2}\b", text or "", flags=re.IGNORECASE)
    if rb:
        return rb.group(0).upper()
    return None


def build_target_key(*, lobe: str | None, laterality: str | None, segment: str | None) -> str:
    if lobe and segment:
        safe_segment = re.sub(r"\s+", "_", segment.strip().lower())
        return f"{lobe}:{safe_segment}"
    if lobe:
        return lobe
    if laterality and segment:
        safe_segment = re.sub(r"\s+", "_", segment.strip().lower())
        return f"{laterality}:{safe_segment}"
    if laterality:
        return f"{laterality}:unknown"
    return "target_unknown"


__all__ = [
    "build_target_key",
    "extract_laterality",
    "normalize_lobe",
    "normalize_segment",
]
