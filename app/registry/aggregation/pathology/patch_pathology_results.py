"""Patch pathology_results fields for case aggregation.

This module applies extracted pathology summary fields into `registry_json["pathology_results"]`
using conservative "fill-missing-only" semantics while honoring manual JSON-pointer locks.
"""

from __future__ import annotations

from typing import Any

from app.registry.aggregation.locks import (
    assign_if_unlocked,
    ensure_dict,
    is_pointer_locked,
    pointer_join,
)


_SCALAR_FIELDS = [
    "final_diagnosis",
    "final_staging",
    "histology",
    "pdl1_tps_percent",
    "pdl1_tps_text",
    "microbiology_results",
    "pathology_result_date",
]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return not bool(value)
    return False


def patch_pathology_results(
    registry_json: dict[str, Any],
    *,
    extracted: dict[str, Any],
    manual_overrides: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    changed = False
    qa_flags = list(extracted.get("qa_flags") or [])

    update = extracted.get("pathology_results_update")
    if not isinstance(update, dict) or not update:
        return False, sorted(set(qa_flags))

    pathology_pointer = pointer_join("pathology_results")
    if is_pointer_locked(manual_overrides, pathology_pointer):
        qa_flags.append("pathology_results_locked")
        return False, sorted(set(qa_flags))

    pathology = ensure_dict(registry_json, "pathology_results")

    for field in _SCALAR_FIELDS:
        value = update.get(field)
        if value is None:
            continue
        if not _is_missing(pathology.get(field)):
            continue
        changed |= assign_if_unlocked(
            pathology,
            key=field,
            value=value,
            pointer=pointer_join("pathology_results", field),
            manual_overrides=manual_overrides,
        )

    markers = update.get("molecular_markers")
    if isinstance(markers, dict) and markers:
        markers_pointer = pointer_join("pathology_results", "molecular_markers")
        if is_pointer_locked(manual_overrides, markers_pointer):
            qa_flags.append("molecular_markers_locked")
        else:
            dest = ensure_dict(pathology, "molecular_markers")
            for marker, value in markers.items():
                key = str(marker or "").strip()
                if not key or value is None:
                    continue
                if not _is_missing(dest.get(key)):
                    continue
                changed |= assign_if_unlocked(
                    dest,
                    key=key,
                    value=value,
                    pointer=pointer_join("pathology_results", "molecular_markers", key),
                    manual_overrides=manual_overrides,
                )

    return changed, sorted(set(qa_flags))


__all__ = ["patch_pathology_results"]

