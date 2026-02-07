"""Helpers for resolving the registry ML label schema.

The canonical set/order of registry procedure labels lives in
`app.registry.v2_booleans.PROCEDURE_BOOLEAN_FIELDS` (aligned to the IP Registry
schema). This module provides a small helper to prefer the on-disk label list
(`data/ml_training/registry_label_fields.json`) when present while still falling
back to the canonical list if the file is missing or malformed.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.registry.v2_booleans import PROCEDURE_BOOLEAN_FIELDS

DEFAULT_REGISTRY_LABEL_FIELDS_PATH = Path("data/ml_training/registry_label_fields.json")


def canonical_registry_procedure_labels() -> list[str]:
    """Return the canonical registry procedure label schema."""
    return list(PROCEDURE_BOOLEAN_FIELDS)


def load_registry_procedure_labels(
    label_fields_path: str | Path | None = None,
) -> list[str]:
    """Load registry procedure labels, preferring a JSON file when available.

    Guardrails:
    - Avoid hardcoding label IDs outside the canonical mapping module.
    - If the JSON file is missing or invalid, fall back to the canonical list.

    Args:
        label_fields_path: Optional path to a JSON string-list of labels.

    Returns:
        List of canonical label IDs in canonical order.
    """
    canonical = canonical_registry_procedure_labels()

    path = Path(label_fields_path) if label_fields_path else DEFAULT_REGISTRY_LABEL_FIELDS_PATH
    if not path.exists():
        return canonical

    try:
        data = json.loads(path.read_text())
    except Exception:
        return canonical

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return canonical

    # If the file contains at least the canonical set, keep canonical order to
    # avoid drifting due to sorted/inferred column ordering elsewhere.
    file_set = set(data)
    if all(label in file_set for label in canonical):
        return canonical

    # File is missing some labels (likely filtered/incomplete). Use canonical to
    # keep UI/training schema stable.
    return canonical
