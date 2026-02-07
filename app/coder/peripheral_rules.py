"""Mapping from peripheral lesion evidence to CPT candidates."""

from __future__ import annotations

from typing import Iterable, Sequence

from app.coder.types import CodeCandidate, PeripheralLesionEvidence


def _normalize_actions(actions: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for action in actions:
        if not action:
            continue
        normalized.add(action.strip().lower())
    return normalized


def peripheral_lesions_to_candidates(
    lesions: Sequence[PeripheralLesionEvidence],
) -> list[CodeCandidate]:
    """Convert peripheral lesion evidence into CPT code candidates."""

    if not lesions:
        return []

    want_31628 = False
    want_31626 = False
    want_31624 = False
    want_31627 = False
    want_31654 = False

    for lesion in lesions:
        actions = _normalize_actions(lesion.actions)

        if {"cryobiopsy", "tblb", "lung_biopsy"} & actions:
            want_31628 = True

        if "fiducial" in actions:
            want_31626 = True

        if "bal" in actions or "lavage" in actions:
            want_31624 = True

        if lesion.navigation:
            want_31627 = True

        if lesion.radial_ebus:
            want_31654 = True

    candidates: list[CodeCandidate] = []

    if want_31628:
        candidates.append(
            CodeCandidate(code="31628", confidence=0.9, reason="peripheral:cryobiopsy", evidence=None)
        )
    if want_31626:
        candidates.append(
            CodeCandidate(code="31626", confidence=0.85, reason="peripheral:fiducial", evidence=None)
        )
    if want_31624:
        candidates.append(
            CodeCandidate(code="31624", confidence=0.8, reason="peripheral:bal", evidence=None)
        )
    if want_31627:
        candidates.append(
            CodeCandidate(code="31627", confidence=0.8, reason="peripheral:navigation", evidence=None)
        )
    if want_31654:
        candidates.append(
            CodeCandidate(code="+31654", confidence=0.8, reason="peripheral:radial_ebus", evidence=None)
        )

    return candidates


__all__ = ["peripheral_lesions_to_candidates"]
