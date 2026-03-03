"""Deterministic pathology event extraction for case aggregation."""

from __future__ import annotations

import re
from typing import Any

from app.registry.aggregation.pathology.normalize import (
    build_target_key,
    extract_laterality,
    find_station_tokens,
    normalize_lobe,
)
from app.registry.aggregation.sanitize import compact_text


_SPECIMEN_BLOCK_RE = re.compile(r"(?ims)^\s*[A-Z]\.\s*(.+?)(?=^\s*[A-Z]\.\s*|\Z)")
_SECTION_HEADING_RE = re.compile(r"(?im)^[ \t]*[A-Z][A-Z0-9 /()\-:]{2,}[ \t]*$")
_FINAL_DIAGNOSIS_HEADING_RE = re.compile(r"(?im)^[ \t]*FINAL\s+DIAGNOSIS\b.*$")
_LINE_SPLIT_RE = re.compile(r"[\r\n]+")

_NEGATIVE_RE = re.compile(
    r"(?:negative\s+for\s+malignan|no\s+evidence\s+of\s+malignan|no\s+malignan\w*\s+(?:identified|seen))",
    re.IGNORECASE,
)
_POSITIVE_RE = re.compile(
    r"\b(?:diagnostic\s+category\s*:\s*malignant|positive\s+for\s+malignan|malignan\w*|adenocarcinoma|carcinoma)\b",
    re.IGNORECASE,
)
_NON_DIAGNOSTIC_RE = re.compile(r"\b(?:non[-\s]?diagnostic|insufficient|limited\s+by)\b", re.IGNORECASE)
_ATYPICAL_RE = re.compile(r"\batypical\b", re.IGNORECASE)
_SUSPICIOUS_RE = re.compile(r"\bsuspicious\b", re.IGNORECASE)

_PERIPHERAL_RE = re.compile(
    r"\b(?:lung\s+nodule|pulmonary\s+nodule|lesion|mass)\b[^\n\.]{0,120}\b(?:right|left|rt|lt|RUL|RML|RLL|LUL|LLL)\b",
    re.IGNORECASE,
)


def _split_blocks(text: str) -> list[str]:
    blocks = [m.group(1).strip() for m in _SPECIMEN_BLOCK_RE.finditer(text or "") if m.group(1).strip()]
    if blocks:
        return blocks
    stripped = (text or "").strip()
    return [stripped] if stripped else []


def _extract_section_after_heading(text: str, heading_re: re.Pattern[str]) -> str | None:
    value = text or ""
    heading_match = heading_re.search(value)
    if not heading_match:
        return None

    tail = value[heading_match.end() :]
    next_heading = _SECTION_HEADING_RE.search(tail)
    section = tail[: next_heading.start()] if next_heading else tail
    clean = section.strip()
    return clean or None


def _split_prefer_final_diagnosis_blocks(text: str) -> tuple[list[str], bool]:
    final_section = _extract_section_after_heading(text, _FINAL_DIAGNOSIS_HEADING_RE)
    if final_section:
        blocks = _split_blocks(final_section)
        if blocks:
            return blocks, True
    return _split_blocks(text), False


def _station_tokens_for_block(block: str, *, header_only: bool) -> list[str]:
    lines = [line.strip() for line in _LINE_SPLIT_RE.split(block or "") if line.strip()]
    if not lines:
        return []
    header_tokens = find_station_tokens(lines[0])
    if header_tokens:
        return header_tokens
    if header_only:
        return []
    return find_station_tokens(block)


def _classify_path_result(block: str) -> str | None:
    value = block or ""
    if _NEGATIVE_RE.search(value):
        return "Negative"
    if _NON_DIAGNOSTIC_RE.search(value):
        return "Non-diagnostic"
    if _ATYPICAL_RE.search(value):
        return "Atypical"
    if _SUSPICIOUS_RE.search(value):
        return "Suspicious"

    positive = _POSITIVE_RE.search(value)
    if not positive:
        return None
    window_start = max(0, positive.start() - 20)
    window = value[window_start : positive.start()].lower()
    if "negative" in window:
        return None
    return "Positive"


def _extract_diagnosis_text(block: str, path_result: str | None) -> str | None:
    value = block or ""
    if re.search(r"granulomatous\s+lymphadenitis", value, re.IGNORECASE):
        return "Granulomatous lymphadenitis"

    adeno_match = re.search(r"adenocarcinoma[^\n\.]{0,140}", value, re.IGNORECASE)
    if adeno_match:
        phrase = compact_text(adeno_match.group(0), max_chars=120)
        words = phrase.split()
        if len(words) <= 15:
            return phrase
        return " ".join(words[:15])

    if path_result:
        for line in _LINE_SPLIT_RE.split(value):
            clean = line.strip()
            if not clean:
                continue
            if path_result.lower() in clean.lower() or "diagnosis" in clean.lower() or "malignan" in clean.lower():
                return compact_text(clean, max_chars=120)

    return compact_text(value, max_chars=120) or None


def _extract_peripheral_update(block: str, path_result: str | None) -> dict[str, Any] | None:
    if not _PERIPHERAL_RE.search(block or ""):
        return None

    laterality = extract_laterality(block)
    lobe = normalize_lobe(block)
    target_key = build_target_key(laterality=laterality, lobe=lobe)

    return {
        "target_key": target_key,
        "laterality": laterality,
        "lobe": lobe,
        "segment": None,
        "path_result": path_result,
        "path_diagnosis_text": _extract_diagnosis_text(block, path_result),
    }


def extract_pathology_event(text: str) -> dict[str, Any]:
    """Extract station/peripheral pathology updates from scrubbed text."""

    node_updates: list[dict[str, Any]] = []
    peripheral_updates: list[dict[str, Any]] = []
    qa_flags: list[str] = []

    blocks, using_final_diagnosis_blocks = _split_prefer_final_diagnosis_blocks(text)
    for block in blocks:
        path_result = _classify_path_result(block)
        diagnosis = _extract_diagnosis_text(block, path_result)

        stations = _station_tokens_for_block(block, header_only=using_final_diagnosis_blocks)
        if using_final_diagnosis_blocks and not stations:
            qa_flags.append("station_not_found_in_final_diagnosis_block")

        for station in stations:
            node_updates.append(
                {
                    "station": station,
                    "path_result": path_result,
                    "path_diagnosis_text": diagnosis,
                }
            )

        peripheral = _extract_peripheral_update(block, path_result)
        if peripheral:
            if not peripheral.get("target_key"):
                qa_flags.append("unmatched_peripheral_target")
            peripheral_updates.append(peripheral)

    # Dedupe by station (latest block wins).
    dedup_nodes: dict[str, dict[str, Any]] = {}
    for update in node_updates:
        station = str(update.get("station") or "").strip().upper()
        if station:
            dedup_nodes[station] = update

    return {
        "node_updates": list(dedup_nodes.values()),
        "peripheral_updates": peripheral_updates,
        "qa_flags": sorted(set(qa_flags)),
    }


__all__ = ["extract_pathology_event"]
