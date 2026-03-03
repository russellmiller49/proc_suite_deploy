"""Normalization helpers for pathology case-event extraction."""

from __future__ import annotations

import re


_STATION_TOKEN_RE = re.compile(r"\b(?:station\s*)?((?:[1247]|10|11)\s*[RrLl]?\s*[A-Za-z]{0,2})\b")
_LOBE_MAP = {
    "rul": "RUL",
    "right upper lobe": "RUL",
    "rml": "RML",
    "right middle lobe": "RML",
    "rll": "RLL",
    "right lower lobe": "RLL",
    "lul": "LUL",
    "left upper lobe": "LUL",
    "lll": "LLL",
    "left lower lobe": "LLL",
}


def normalize_station(value: str | None) -> str | None:
    if not value:
        return None
    norm = re.sub(r"[^0-9A-Za-z]", "", value).upper().strip()
    return norm or None


def find_station_tokens(text: str) -> list[str]:
    found: list[str] = []
    for match in _STATION_TOKEN_RE.finditer(text or ""):
        station = normalize_station(match.group(1))
        if station and station not in found:
            found.append(station)
    return found


def extract_laterality(text: str) -> str | None:
    value = (text or "").lower()
    if re.search(r"\b(?:right|rt|r)\b", value):
        return "R"
    if re.search(r"\b(?:left|lt|l)\b", value):
        return "L"
    return None


def normalize_lobe(text: str) -> str | None:
    value = (text or "").lower()
    for raw, normalized in _LOBE_MAP.items():
        if raw in value:
            return normalized
    return None


def build_target_key(*, laterality: str | None, lobe: str | None, segment: str | None = None) -> str | None:
    if lobe:
        if segment:
            seg = re.sub(r"\s+", "_", segment.strip().lower())
            return f"{lobe}:{seg}"
        return lobe
    if laterality and segment:
        seg = re.sub(r"\s+", "_", segment.strip().lower())
        return f"{laterality}:{seg}"
    return None


__all__ = [
    "build_target_key",
    "extract_laterality",
    "find_station_tokens",
    "normalize_lobe",
    "normalize_station",
]
