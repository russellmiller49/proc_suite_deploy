"""Structurer-first extraction wiring (optional; behind feature flag).

This module is reserved for a future implementation that converts a raw note
directly into a RegistryRecord via an agents-based structurer.
"""

from __future__ import annotations

from typing import Any

from app.registry.schema import RegistryRecord


def structure_note_to_registry_record(
    note_text: str,
    *,
    note_id: str | None = None,
) -> tuple[RegistryRecord, dict[str, Any]]:
    raise NotImplementedError(
        "REGISTRY_EXTRACTION_ENGINE=agents_structurer is not implemented yet; "
        "falling back to deterministic RegistryEngine extraction"
    )


__all__ = ["structure_note_to_registry_record"]

