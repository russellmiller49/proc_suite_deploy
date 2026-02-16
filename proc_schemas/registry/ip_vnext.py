"""IP Registry schema vNext (final; API/UI-facing).

This schema adds anchored evidence spans + an explicit evidence status.
It is additive and not yet wired into production extraction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatus(str, Enum):
    anchored = "ANCHORED"
    inferred_unanchored = "INFERRED_UNANCHORED"
    inferred_noquote = "INFERRED_NOQUOTE"


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: int
    end: int


class EvidenceFinal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    quote: str | None = Field(default=None, description="Verbatim quote from the redacted input.")
    evidence_span: EvidenceSpan | None = Field(
        default=None,
        description="Anchored [start,end] offsets into the redacted input, when available.",
    )
    evidence_status: EvidenceStatus = Field(
        default=EvidenceStatus.inferred_noquote,
        description="Whether the evidence quote was successfully anchored to a span.",
    )
    confidence: float | None = Field(
        default=None,
        description="Optional numeric confidence (0-1) for the evidence anchoring/result.",
    )


class ProcedureTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anatomy_type: str | None = None
    lobe: str | None = None
    segment: str | None = None
    station: str | None = None


class LesionDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lesion_type: str | None = None
    size_mm: float | None = None
    long_axis_mm: float | None = None
    short_axis_mm: float | None = None
    craniocaudal_mm: float | None = None
    morphology: str | None = None
    suv_max: float | None = None
    location: str | None = None
    size_text: str | None = None


class Outcomes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    airway_lumen_pre: str | None = None
    airway_lumen_post: str | None = None
    pre_obstruction_pct: int | None = None
    post_obstruction_pct: int | None = None
    pre_diameter_mm: float | None = None
    post_diameter_mm: float | None = None
    symptoms: str | None = None
    pleural: str | None = None
    complications: str | None = None


class ProcedureEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str
    type: str
    method: str | None = None
    target: ProcedureTarget = Field(default_factory=ProcedureTarget)
    lesion: LesionDetails = Field(default_factory=LesionDetails)
    devices: list[str] = Field(default_factory=list)
    specimens: list[str] = Field(default_factory=list)
    outcomes: Outcomes = Field(default_factory=Outcomes)
    evidence: EvidenceFinal | None = None

    measurements: Any | None = None
    findings: Any | None = None
    stent_size: str | None = None
    stent_material_or_brand: str | None = None
    catheter_size_fr: float | None = None


class IPRegistryVNext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    note_id: str
    source_filename: str
    schema_version: Literal["vnext"] = "vnext"
    established_tracheostomy_route: bool = False
    procedures: list[ProcedureEvent] = Field(default_factory=list)


__all__ = [
    "EvidenceFinal",
    "EvidenceSpan",
    "EvidenceStatus",
    "IPRegistryVNext",
    "LesionDetails",
    "Outcomes",
    "ProcedureEvent",
    "ProcedureTarget",
]

