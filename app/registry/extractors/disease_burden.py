"""Deterministic disease burden extraction (Text → Numeric anchors).

This module is the canonical import surface for deterministic disease-burden
extractors. Implementation currently lives in
`app.registry.processing.disease_burden` to keep compatibility with older
imports while the registry pipeline is refactored.
"""

from __future__ import annotations

from app.registry.processing.disease_burden import (
    ExtractedLesionAxes,
    ExtractedNumeric,
    apply_disease_burden_overrides,
    extract_unambiguous_lesion_size_mm,
    extract_unambiguous_suv_max,
    extract_unambiguous_target_lesion_axes_mm,
    extract_unambiguous_target_lesion_size_and_axes_mm,
)

__all__ = [
    "ExtractedLesionAxes",
    "ExtractedNumeric",
    "apply_disease_burden_overrides",
    "extract_unambiguous_lesion_size_mm",
    "extract_unambiguous_suv_max",
    "extract_unambiguous_target_lesion_axes_mm",
    "extract_unambiguous_target_lesion_size_and_axes_mm",
]

