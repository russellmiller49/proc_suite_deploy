from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Thoracentesis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    effusion_size: str
    effusion_echogenicity: str
    loculations: str | None = None
    ultrasound_findings: str | None = None
    anesthesia_lidocaine_1_pct_ml: int | None = None
    intercostal_space: str
    entry_location: str
    volume_removed_ml: int
    fluid_appearance: str
    specimen_tests: List[str]
    cxr_ordered: bool | None = None


class ThoracentesisDetailed(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    ultrasound_feasible: bool | None = None
    anesthesia_lidocaine_ml: int | None = None
    intercostal_space: str
    entry_location: str
    volume_removed_ml: int
    fluid_appearance: str
    drainage_device: str | None = None
    suction_cmh2o: str | None = None
    specimen_tests: List[str] | None = None
    cxr_ordered: bool | None = None
    sutured: bool | None = None
    effusion_volume: str | None = None
    effusion_echogenicity: str | None = None
    loculations: str | None = None
    diaphragm_motion: str | None = None
    lung_sliding_pre: str | None = None
    lung_sliding_post: str | None = None
    lung_consolidation: str | None = None
    pleura_description: str | None = None
    pleural_guidance: str | None = None


class ThoracentesisManometry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    guidance: str | None = None
    opening_pressure_cmh2o: float | None = None
    pressure_readings: List[str] | None = None
    stopping_criteria: str | None = None
    total_removed_ml: int
    post_procedure_imaging: str | None = None
    effusion_size: str | None = None
    effusion_echogenicity: str | None = None
    loculations: str | None = None
    diaphragm_motion: str | None = None
    lung_sliding_pre: str | None = None
    lung_sliding_post: str | None = None
    lung_consolidation: str | None = None
    pleura_description: str | None = None


class ChestTube(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    intercostal_space: str
    entry_line: str
    guidance: str | None = None
    fluid_removed_ml: int | None = None
    fluid_appearance: str | None = None
    specimen_tests: List[str] | None = None
    cxr_ordered: bool | None = None
    effusion_volume: str | None = None
    effusion_echogenicity: str | None = None
    loculations: str | None = None
    diaphragm_motion: str | None = None
    lung_sliding_pre: str | None = None
    lung_sliding_post: str | None = None
    lung_consolidation: str | None = None
    pleura_description: str | None = None


class TunneledPleuralCatheterInsert(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str | None = None
    intercostal_space: str | None = None
    entry_location: str | None = None
    tunnel_length_cm: int | None = None
    exit_site: str | None = None
    anesthesia_lidocaine_ml: int | None = None
    fluid_removed_ml: int | None = None
    fluid_appearance: str | None = None
    pleural_pressures: dict[str, float | int] | None = None
    drainage_device: str | None = None
    suction: str | None = None
    specimen_tests: List[str] | None = None
    cxr_ordered: bool | None = None
    pleural_guidance: str | None = None
    effusion_volume: str | None = None
    effusion_echogenicity: str | None = None
    loculations: str | None = None
    diaphragm_motion: str | None = None
    lung_sliding_pre: str | None = None
    lung_sliding_post: str | None = None
    lung_consolidation: str | None = None
    pleura_description: str | None = None


class TunneledPleuralCatheterRemove(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    insertion_date: str | None = None
    reason: str | None = None
    site_assessment: str | None = None
    anesthesia_lidocaine_ml: int | None = None
    sutured: bool | None = None
    complications: str | None = None
    antibiotics: str | None = None


class PigtailCatheter(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    intercostal_space: str
    entry_location: str
    size_fr: str
    anesthesia_lidocaine_ml: int | None = None
    fluid_removed_ml: int | None = None
    fluid_appearance: str | None = None
    specimen_tests: List[str] | None = None
    cxr_ordered: bool | None = None


class TransthoracicNeedleBiopsy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    needle_gauge: str
    samples_collected: int
    imaging_modality: str | None = None
    cxr_ordered: bool | None = None


class Paracentesis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    site_description: str | None = None
    volume_removed_ml: int
    fluid_character: str | None = None
    tests: List[str] | None = None
    imaging_guidance: str | None = None


class PEGPlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    incision_location: str | None = None
    endoscope_time_seconds: int | None = None
    wire_route: str | None = None
    bumper_depth_cm: float | None = None
    tube_size_fr: int | None = None
    procedural_time_min: int | None = None
    complications: str | None = None


class PEGExchange(BaseModel):
    model_config = ConfigDict(extra="ignore")
    new_tube_size_fr: int | None = None
    bumper_depth_cm: float | None = None
    complications: str | None = None


class PleurxInstructions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    followup_timeframe: str | None = None
    contact_info: str | None = None


class ChestTubeDischargeInstructions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    drainage_plan: str | None = None
    infection_signs: str | None = None
    followup_timeframe: str | None = None


class PEGDischargeInstructions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    feeding_plan: str | None = None
    medication_plan: str | None = None
    wound_care: str | None = None
    contact_info: str | None = None


__all__ = [
    "ChestTube",
    "ChestTubeDischargeInstructions",
    "Paracentesis",
    "PEGDischargeInstructions",
    "PEGExchange",
    "PEGPlacement",
    "PigtailCatheter",
    "PleurxInstructions",
    "Thoracentesis",
    "ThoracentesisDetailed",
    "ThoracentesisManometry",
    "TransthoracicNeedleBiopsy",
    "TunneledPleuralCatheterInsert",
    "TunneledPleuralCatheterRemove",
]
