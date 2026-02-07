"""Shared EBUS node-event primitives.

These types are used in multiple schema layers (V2 dynamic registry + V3 proc schemas)
to represent station-level interactions while keeping a single authoritative definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NodeActionType = Literal[
    "inspected_only",  # Visual/Ultrasound only (NO needle)
    "needle_aspiration",  # TBNA / FNA
    "core_biopsy",  # FNB / Core needle
    "forceps_biopsy",  # Mini-forceps / intranodal forceps
]

NodeOutcomeType = Literal[
    "benign",
    "malignant",
    "suspicious",
    "nondiagnostic",
    "deferred_to_final_path",
    "unknown",
]


class NodeInteraction(BaseModel):
    """Represents an interaction with an EBUS lymph node station.

    Distinguishes inspection-only from actual sampling.
    """

    model_config = ConfigDict(extra="ignore")

    station: str
    action: NodeActionType
    outcome: NodeOutcomeType | None = None
    passes: int | None = None
    pass_count: int | None = Field(
        default=None,
        description="Alias for passes (kept for backward compatibility with older field naming).",
    )
    elastography_pattern: str | None = None
    rose_result: str | None = Field(
        default=None,
        description="ROSE / onsite pathology result (free text or normalized category).",
    )
    evidence_quote: str

    @model_validator(mode="after")
    def _sync_pass_fields(self) -> "NodeInteraction":
        if self.passes is None and self.pass_count is not None:
            object.__setattr__(self, "passes", self.pass_count)
        elif self.pass_count is None and self.passes is not None:
            object.__setattr__(self, "pass_count", self.passes)
        return self


__all__ = ["NodeActionType", "NodeOutcomeType", "NodeInteraction"]
