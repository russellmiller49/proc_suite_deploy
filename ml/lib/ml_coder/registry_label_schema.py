"""Canonical label schema for the registry multi-label procedure classifier.

This module is the single import point for:
- the canonical procedure label IDs and their ordering
- human-friendly titles for UI (Prodigy choice options)
- stable encounter_id generation for batch/annotation workflows

The underlying label IDs are aligned to the IP Registry schema via
`app.registry.v2_booleans.PROCEDURE_BOOLEAN_FIELDS`.
"""

from __future__ import annotations

import hashlib

from app.registry.v2_booleans import PROCEDURE_BOOLEAN_FIELDS

REGISTRY_LABELS: list[str] = list(PROCEDURE_BOOLEAN_FIELDS)

# Labels that remain part of the registry schema but are not currently predicted
# by the ML head due to ultra-low support (e.g., n=1).
#
# This list is intentionally conservative; training code can optionally mark
# additional labels as dormant based on a minimum-positive threshold.
DORMANT_LABELS: list[str] = [
    "pleural_biopsy",
]

ACTIVE_LABELS: list[str] = [l for l in REGISTRY_LABELS if l not in set(DORMANT_LABELS)]

REGISTRY_LABEL_TITLES: dict[str, str] = {
    "bal": "BAL",
    "blvr": "BLVR",
    "ipc": "IPC",
    "linear_ebus": "Linear EBUS",
    "radial_ebus": "Radial EBUS",
    "tbna_conventional": "TBNA (Conventional)",
    "transbronchial_biopsy": "Transbronchial Biopsy",
    "transbronchial_cryobiopsy": "Transbronchial Cryobiopsy",
    "whole_lung_lavage": "Whole Lung Lavage",
    "rigid_bronchoscopy": "Rigid Bronchoscopy",
    "chest_tube": "Chest Tube",
    "medical_thoracoscopy": "Medical Thoracoscopy",
    "tumor_debulking_non_thermal": "Tumor Debulking (Non-Thermal)",
    "percutaneous_tracheostomy": "Percutaneous Tracheostomy",
    "peg_insertion": "PEG Insertion",
}


def prodigy_options() -> list[dict[str, str]]:
    """Return Prodigy `choice` options for registry labels."""
    return [{"id": label, "text": title_for_label(label)} for label in REGISTRY_LABELS]


def title_for_label(label: str) -> str:
    """Return a human-friendly title for a label ID."""
    title = REGISTRY_LABEL_TITLES.get(label)
    if title:
        return title
    return label.replace("_", " ").title()


def compute_encounter_id(note_text: str) -> str:
    """Compute a stable encounter ID from note text.

    Used for:
    - Prodigy manifests (avoid re-sampling)
    - merging human labels into training data (dedup)
    - encounter-level split grouping
    """
    normalized = (note_text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def validate_schema() -> None:
    """Validate the canonical label schema invariants."""
    expected = len(PROCEDURE_BOOLEAN_FIELDS)
    if len(REGISTRY_LABELS) != expected:
        raise AssertionError(f"Expected {expected} registry labels, got {len(REGISTRY_LABELS)}")
    if len(set(REGISTRY_LABELS)) != len(REGISTRY_LABELS):
        raise AssertionError("Duplicate label IDs detected in REGISTRY_LABELS")
    if not all(isinstance(x, str) and x.strip() for x in REGISTRY_LABELS):
        raise AssertionError("All REGISTRY_LABELS must be non-empty strings")
    unknown_dormant = set(DORMANT_LABELS) - set(REGISTRY_LABELS)
    if unknown_dormant:
        raise AssertionError(f"DORMANT_LABELS contains unknown labels: {sorted(unknown_dormant)}")

    # Titles must exist for every label (either explicit or derived).
    titles = {label: title_for_label(label) for label in REGISTRY_LABELS}
    if set(titles.keys()) != set(REGISTRY_LABELS):
        raise AssertionError("REGISTRY_LABEL_TITLES mismatch with REGISTRY_LABELS")
    if not all(isinstance(v, str) and v.strip() for v in titles.values()):
        raise AssertionError("All label titles must be non-empty strings")
