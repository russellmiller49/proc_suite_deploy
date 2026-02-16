"""IP Registry schema vNext (draft; LLM-facing).

This schema is intended to be used with constrained decoding / structured outputs.
It is additive and not yet wired into production extraction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceConfidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class EvidenceQuoteDraft(BaseModel):
    """Draft evidence quote with minimal surrounding context."""

    model_config = ConfigDict(extra="ignore")

    rationale: str = Field(
        ...,
        description="Short internal/debug rationale for why this quote supports the extracted value.",
    )
    prefix_3_words: str = Field(
        ...,
        description="Exactly three words immediately before the quote in the redacted input (best-effort).",
    )
    exact_quote: str = Field(
        ...,
        description="Verbatim quote copied from the redacted input (must be a substring).",
    )
    suffix_3_words: str = Field(
        ...,
        description="Exactly three words immediately after the quote in the redacted input (best-effort).",
    )
    confidence: EvidenceConfidence | None = Field(
        default=None,
        description="Optional coarse confidence bucket for this quote/value pair.",
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


class ProcedureEventDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str
    type: str
    method: str | None = None
    target: ProcedureTarget = Field(default_factory=ProcedureTarget)
    lesion: LesionDetails = Field(default_factory=LesionDetails)
    devices: list[str] = Field(default_factory=list)
    specimens: list[str] = Field(default_factory=list)
    outcomes: Outcomes = Field(default_factory=Outcomes)
    evidence: EvidenceQuoteDraft | None = None

    measurements: Any | None = None
    findings: Any | None = None
    stent_size: str | None = None
    stent_material_or_brand: str | None = None
    catheter_size_fr: float | None = None


class IPRegistryVNextDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    note_id: str
    source_filename: str
    schema_version: Literal["vnext_draft"] = "vnext_draft"
    established_tracheostomy_route: bool = False
    procedures: list[ProcedureEventDraft] = Field(default_factory=list)


__all__ = [
    "EvidenceConfidence",
    "EvidenceQuoteDraft",
    "IPRegistryVNextDraft",
    "LesionDetails",
    "Outcomes",
    "ProcedureEventDraft",
    "ProcedureTarget",
]

