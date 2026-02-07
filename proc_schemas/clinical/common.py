from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny


class PatientInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    age: int | None = None
    sex: str | None = None
    patient_id: str | None = None
    mrn: str | None = None


class EncounterInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    date: str | None = None
    encounter_id: str | None = None
    location: str | None = None
    referred_physician: str | None = None
    attending: str | None = None
    assistant: str | None = None


class SedationInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str | None = None
    description: str | None = None


class AnesthesiaInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str | None = None
    description: str | None = None


class PreAnesthesiaAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    anticoagulant_use: str | None = None
    prophylactic_antibiotics: bool | None = None
    asa_status: str
    anesthesia_plan: str
    sedation_history: str | None = None
    time_out_confirmed: bool | None = None


class OperativeShellInputs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    indication_text: str | None = None
    preop_diagnosis_text: str | None = None
    postop_diagnosis_text: str | None = None
    procedures_summary: str | None = None
    cpt_summary: str | None = None
    estimated_blood_loss: str | None = None
    complications_text: str | None = None
    specimens_text: str | None = None
    impression_plan: str | None = None


class ProcedureInput(BaseModel):
    """Input for a single procedure in the bundle.

    IMPORTANT: Extraction rules for data fields:
    1. NEVER guess or hallucinate values not present in source text
    2. If a value is not explicitly stated, set it to null (not a default)
    3. Add missing fields to bundle.acknowledged_omissions[proc_id]
    4. sequence should reflect chronological order from source text
    """

    model_config = ConfigDict(extra="ignore")
    proc_type: str
    schema_id: str
    proc_id: str | None = None
    data: SerializeAsAny[dict[str, Any] | BaseModel]
    cpt_candidates: List[str | int] = Field(default_factory=list)
    # Chronological sequence from source text (1, 2, 3...)
    sequence: int | None = Field(
        default=None,
        description="Chronological order from source dictation (1-based). Do not reorder by CPT or type.",
    )


class ProcedureBundle(BaseModel):
    """Bundle containing all procedure data for report generation.

    The `procedures` list contains core procedure data rendered via the macro system.
    The `addons` list contains slugs for supplementary addon templates (rare events,
    complications, transitional statements) rendered as a secondary snippet library.
    """

    model_config = ConfigDict(extra="ignore")
    patient: PatientInfo
    encounter: EncounterInfo
    procedures: List[ProcedureInput]
    sedation: SedationInfo | None = None
    anesthesia: AnesthesiaInfo | None = None
    pre_anesthesia: PreAnesthesiaAssessment | dict[str, Any] | None = None
    indication_text: str | None = None
    preop_diagnosis_text: str | None = None
    postop_diagnosis_text: str | None = None
    impression_plan: str | None = None
    estimated_blood_loss: str | None = None
    complications_text: str | None = None
    specimens_text: str | None = None
    free_text_hint: str | None = None
    acknowledged_omissions: dict[str, list[str]] = Field(default_factory=dict)
    # Addon slugs for supplementary templates (rare events, complications, etc.)
    addons: list[str] = Field(
        default_factory=list,
        description="List of addon template slugs to render as supplementary content",
    )


class ProcedurePatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    proc_id: str
    updates: dict[str, Any] = Field(default_factory=dict)
    acknowledge_missing: list[str] = Field(default_factory=list)


class BundlePatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    procedures: list[ProcedurePatch]


__all__ = [
    "PatientInfo",
    "EncounterInfo",
    "SedationInfo",
    "AnesthesiaInfo",
    "PreAnesthesiaAssessment",
    "OperativeShellInputs",
    "ProcedureInput",
    "ProcedureBundle",
    "ProcedurePatch",
    "BundlePatch",
]
