"""Reporter-only schema overrides for interactive, partially-populated bundles.

The registry/extraction pipeline can often identify *that* a procedure was
performed without all documentation details (counts, segments, tests, etc.).

For the interactive Reporter Builder, we need those procedures to still be
represented in a ProcedureBundle so validation can prompt the user for missing
details, and rendering can produce a draft without crashing.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class AirwayDilationPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    airway_segment: str | None = None
    technique: str | None = None
    dilation_sizes_mm: List[int] = Field(default_factory=list)
    post_dilation_diameter_mm: int | None = None
    notes: str | None = None


class TransbronchialNeedleAspirationPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lung_segment: str | None = None
    needle_tools: str | None = None
    samples_collected: int | None = None
    tests: List[str] = Field(default_factory=list)


class BALPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lung_segment: str | None = None
    instilled_volume_cc: int | None = None
    returned_volume_cc: int | None = None
    tests: List[str] = Field(default_factory=list)


class BronchialBrushingPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lung_segment: str | None = None
    samples_collected: int | None = None
    brush_tool: str | None = None
    tests: List[str] = Field(default_factory=list)


class BronchialWashingPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    airway_segment: str | None = None
    instilled_volume_ml: int | None = None
    returned_volume_ml: int | None = None
    tests: List[str] = Field(default_factory=list)


class TransbronchialCryobiopsyPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lung_segment: str | None = None
    num_samples: int | None = None
    sample_size_mm: float | None = None
    cryoprobe_size_mm: float | None = None
    freeze_seconds: int | None = None
    thaw_seconds: int | None = None
    blocker_type: str | None = None
    blocker_volume_ml: float | None = None
    blocker_location: str | None = None
    tests: List[str] = Field(default_factory=list)
    radial_vessel_check: bool | None = None
    notes: str | None = None


class PeripheralAblationPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modality: str | None = None
    target: str | None = None
    power_w: int | None = None
    duration_min: float | None = None
    max_temp_c: int | None = None
    notes: str | None = None


class EndobronchialCatheterPlacementPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    catheter_size_fr: int | None = None
    target_airway: str | None = None
    obstruction_pct: int | None = None
    fluoro_used: bool | None = None
    dummy_check: bool | None = None
    notes: str | None = None


class MicrodebriderDebridementPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    airway_segment: str | None = None
    notes: str | None = None


class EndobronchialTumorDestructionPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modality: str | None = None
    airway_segment: str | None = None
    notes: str | None = None


class AirwayStentPlacementPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stent_type: str | None = None
    diameter_mm: int | None = None
    length_mm: int | None = None
    airway_segment: str | None = None
    notes: str | None = None


class MedicalThoracoscopyPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    side: str | None = None
    findings: str | None = None
    interventions: List[str] = Field(default_factory=list)
    specimens: List[str] = Field(default_factory=list)
    notes: str | None = None


class RigidBronchoscopyPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    size_or_model: str | None = None
    hf_jv: bool | None = None
    interventions: List[str] = Field(default_factory=list)
    flexible_scope_used: bool | None = None
    estimated_blood_loss_ml: int | None = None
    specimens: List[str] | None = None
    post_procedure_plan: str | None = None
    dilation_sizes_mm: List[int] = Field(default_factory=list)
    post_dilation_diameter_mm: int | None = None

    # Reporter-only enrichments
    target_airway: str | None = None
    pre_obstruction_pct: int | None = None
    post_obstruction_pct: int | None = None
    findings: str | None = None
