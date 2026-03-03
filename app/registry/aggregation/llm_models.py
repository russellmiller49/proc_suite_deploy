"""Pydantic models for optional LLM fallback in append-event aggregation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PathologyResultsLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_diagnosis: str | None = Field(default=None)
    final_staging: str | None = Field(default=None)
    microbiology_results: str | None = Field(default=None)
    pathology_result_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD when explicitly stated.")


class ClinicalFollowupLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hospital_admission: bool | None = Field(default=None)
    icu_admission: bool | None = Field(default=None)
    deceased: bool | None = Field(default=None)
    disease_status: Literal["Progression", "Stable", "Response", "Mixed", "Indeterminate"] | None = Field(default=None)


class ImagingSnapshotLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: Literal["Progression", "Stable", "Response", "Mixed", "Indeterminate"] | None = Field(default=None)
    overall_impression_text: str | None = Field(default=None)


__all__ = ["ClinicalFollowupLLM", "ImagingSnapshotLLM", "PathologyResultsLLM"]

