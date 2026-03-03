"""Deterministic CT event extractor."""

from __future__ import annotations

import re
from typing import Any

from app.registry.aggregation.imaging.normalize import (
    build_target_key,
    extract_laterality,
    normalize_lobe,
    normalize_segment,
)
from app.registry.aggregation.imaging.parse_common import (
    build_snapshot,
    classify_comparative_change,
    parse_sizes_mm,
)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\n])\s+")

_NEGATED_NODULE_RE = re.compile(r"\bno\s+(?:suspicious\s+)?(?:pulmonary\s+)?(?:nodule|lesion|mass)\b", re.IGNORECASE)
_TARGET_CUE_RE = re.compile(r"\b(?:nodule|lesion|mass)\b", re.IGNORECASE)
_DENSITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpart[-\s]?solid\b", re.IGNORECASE), "Part-solid"),
    (re.compile(r"\bground\s+glass|\bGGO\b", re.IGNORECASE), "GGO"),
    (re.compile(r"\bcavitary\b", re.IGNORECASE), "Cavitary"),
    (re.compile(r"\bsolid\b", re.IGNORECASE), "Solid"),
]

_MEDIASTINAL_NODE_RE = re.compile(
    r"\b(?:mediastinal|hilar)\b[^\n\.]{0,100}\b(\d+(?:\.\d+)?)\s*(cm|mm)\b",
    re.IGNORECASE,
)


def _density(sentence: str) -> str | None:
    for pattern, label in _DENSITY_PATTERNS:
        if pattern.search(sentence):
            return label
    return None


def extract_ct_event(
    text: str,
    *,
    relative_day_offset: int | None,
    event_subtype: str | None,
) -> dict[str, Any]:
    qa_flags: list[str] = []
    clean = text or ""

    if re.search(r"\bno\s+mediastinal\s+lymphadenopathy\b", clean, re.IGNORECASE):
        qa_flags.append("no_mediastinal_adenopathy")

    peripheral_targets: list[dict[str, Any]] = []
    mediastinal_targets: list[dict[str, Any]] = []

    for sentence in _SENTENCE_SPLIT_RE.split(clean):
        sample = sentence.strip()
        if not sample:
            continue

        if _NEGATED_NODULE_RE.search(sample):
            continue
        if _TARGET_CUE_RE.search(sample):
            lobe = normalize_lobe(sample)
            laterality = extract_laterality(sample)
            segment = normalize_segment(sample)
            long_mm, short_mm, cc_mm = parse_sizes_mm(sample)
            target_key = build_target_key(lobe=lobe, laterality=laterality, segment=segment)
            peripheral_targets.append(
                {
                    "target_key": target_key,
                    "laterality": laterality,
                    "lobe": lobe,
                    "segment": segment,
                    "size_mm_long": long_mm,
                    "size_mm_short": short_mm,
                    "size_mm_cc": cc_mm,
                    "density": _density(sample),
                    "comparative_change": classify_comparative_change(sample),
                }
            )
            continue

        mediastinal_match = _MEDIASTINAL_NODE_RE.search(sample)
        if mediastinal_match:
            size = float(mediastinal_match.group(1))
            if mediastinal_match.group(2).lower() == "cm":
                size *= 10.0
            mediastinal_targets.append(
                {
                    "station": None,
                    "location_text": "mediastinal" if "mediastinal" in sample.lower() else "hilar",
                    "short_axis_mm": int(round(size)),
                    "comparative_change": classify_comparative_change(sample),
                }
            )

    snapshot = build_snapshot(
        relative_day_offset=relative_day_offset,
        modality="ct",
        subtype=event_subtype,
        text=clean,
        qa_flags=qa_flags,
    )

    return {
        "imaging_snapshot": snapshot,
        "targets_update": {
            "peripheral_targets": peripheral_targets,
            "mediastinal_targets": mediastinal_targets,
        },
        "qa_flags": sorted(set(qa_flags)),
    }


__all__ = ["extract_ct_event"]
