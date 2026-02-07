"""Compatibility re-export for the V3 extraction (event-log) schema.

The extraction engine uses `app.registry.schema.ip_v3_extraction`.
This module is kept to avoid breaking older imports.
"""

from __future__ import annotations

from app.registry.schema.ip_v3_extraction import (  # noqa: F401
    EvidenceSpan,
    IPRegistryV3,
    LesionDetails,
    Outcomes,
    ProcedureEvent,
    ProcedureTarget,
)

__all__ = [
    "EvidenceSpan",
    "ProcedureTarget",
    "LesionDetails",
    "Outcomes",
    "ProcedureEvent",
    "IPRegistryV3",
]

