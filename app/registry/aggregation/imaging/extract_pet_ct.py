"""Deterministic PET/CT event extractor."""

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
    parse_suv_values,
)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\n])\s+")
_TARGET_CUE_RE = re.compile(r"\b(?:nodule|lesion|mass)\b", re.IGNORECASE)
_NODE_CUE_RE = re.compile(r"\b(?:hilar|mediastinal|thoracic\s+lymph\s+node)\b", re.IGNORECASE)
_NO_NODES_RE = re.compile(
    r"\bno\s+(?:suspicious|hypermetabolic)\s+(?:thoracic\s+)?lymph\s+nodes\b|\bno\s+hypermetabolic\s+.*thoracic\s+lymph\s+nodes\b",
    re.IGNORECASE,
)


def extract_pet_ct_event(
    text: str,
    *,
    relative_day_offset: int | None,
    event_subtype: str | None,
) -> dict[str, Any]:
    qa_flags: list[str] = []
    clean = text or ""

    peripheral_targets: list[dict[str, Any]] = []
    mediastinal_targets: list[dict[str, Any]] = []

    if _NO_NODES_RE.search(clean):
        qa_flags.append("no_hypermetabolic_thoracic_nodes")

    for sentence in _SENTENCE_SPLIT_RE.split(clean):
        sample = sentence.strip()
        if not sample:
            continue

        suvmax, delayed_suv = parse_suv_values(sample)

        if _TARGET_CUE_RE.search(sample):
            lobe = normalize_lobe(sample)
            laterality = extract_laterality(sample)
            segment = normalize_segment(sample)
            long_mm, short_mm, cc_mm = parse_sizes_mm(sample)

            peripheral_targets.append(
                {
                    "target_key": build_target_key(lobe=lobe, laterality=laterality, segment=segment),
                    "laterality": laterality,
                    "lobe": lobe,
                    "segment": segment,
                    "size_mm_long": long_mm,
                    "size_mm_short": short_mm,
                    "size_mm_cc": cc_mm,
                    "pet_avid": True if suvmax is not None else None,
                    "pet_suvmax": suvmax,
                    "pet_delayed_suvmax": delayed_suv,
                    "comparative_change": classify_comparative_change(sample),
                }
            )
            continue

        if _NODE_CUE_RE.search(sample):
            location = None
            if re.search(r"\bleft\s+hilar\b", sample, re.IGNORECASE):
                location = "left hilar"
            elif re.search(r"\bright\s+hilar\b", sample, re.IGNORECASE):
                location = "right hilar"
            elif re.search(r"\bhilar\b", sample, re.IGNORECASE):
                location = "hilar"
            elif re.search(r"\bmediastinal\b", sample, re.IGNORECASE):
                location = "mediastinal"

            mediastinal_targets.append(
                {
                    "station": None,
                    "location_text": location,
                    "pet_avid": True if suvmax is not None else None,
                    "pet_suvmax": suvmax,
                    "pet_delayed_suvmax": delayed_suv,
                    "comparative_change": classify_comparative_change(sample),
                }
            )

    snapshot = build_snapshot(
        relative_day_offset=relative_day_offset,
        modality="pet_ct",
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


__all__ = ["extract_pet_ct_event"]
