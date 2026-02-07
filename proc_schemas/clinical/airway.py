from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class EMNBronchoscopy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    navigation_system: str
    target_lung_segment: str
    lesion_size_cm: float | None = None
    tool_to_target_distance_cm: float | None = None
    navigation_catheter: str | None = None
    registration_method: str | None = None
    adjunct_imaging: List[str] | None = None
    notes: str | None = None


class FiducialMarkerPlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airway_location: str
    marker_details: str | None = None
    confirmation_method: str | None = None


class RadialEBUSSurvey(BaseModel):
    model_config = ConfigDict(extra="ignore")
    location: str
    rebus_features: str | None = None
    notes: str | None = None


class RoboticIonBronchoscopy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    navigation_plan_source: str | None = None
    vent_mode: str
    vent_rr: int
    vent_tv_ml: int
    vent_peep_cm_h2o: float
    vent_fio2_pct: int
    vent_flow_rate: str | None = None
    vent_pmean_cm_h2o: float | None = None
    cbct_performed: bool | None = None
    radial_pattern: str | None = None
    notes: str | None = None


class IonRegistrationComplete(BaseModel):
    model_config = ConfigDict(extra="ignore")
    method: str | None = None
    airway_landmarks: List[str] | None = None
    fiducial_error_mm: float | None = None
    alignment_quality: str | None = None
    notes: str | None = None


class RoboticNavigation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    platform: str | None = None
    lesion_location: str | None = None
    registration_method: str | None = None
    registration_error_mm: float | None = None
    notes: str | None = None


class IonRegistrationPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")
    indication: str
    scope_of_registration: str | None = None
    registered_landmarks: List[str] | None = None
    registration_start_time: str | None = None
    registration_complete_time: str | None = None
    navigation_start_time: str | None = None
    time_to_primary_nodule_min: float | None = None
    navigation_time_min: float | None = None
    divergence_pct: float | None = None
    rebus_pattern: str | None = None
    tool_in_lesion_confirmation: str | None = None
    rose_adequacy: str | None = None
    diagnostic_yield_pct: float | None = None
    followup_plan: str | None = None


class IonRegistrationDrift(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cause: str | None = None
    findings: str | None = None
    mitigation: str | None = None
    post_correction_alignment: str | None = None
    proceeded_strategy: str | None = None


class CBCTFusion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ventilation_settings: str | None = None
    translation_mm: str | None = None
    rotation_degrees: str | None = None
    overlay_result: str | None = None
    confirmatory_spin_result: str | None = None
    notes: str | None = None


class ToolInLesionConfirmation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    confirmation_method: str
    margin_mm: float | None = None
    rebus_pattern: str | None = None
    lesion_size_mm: float | None = None
    fluoro_angle_deg: str | None = None
    projection: str | None = None
    screenshots_saved: bool | None = None
    notes: str | None = None


class RoboticMonarchBronchoscopy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    radial_pattern: str | None = None
    cbct_used: bool | None = None
    vent_mode: str | None = None
    vent_rr: int | None = None
    vent_tv_ml: int | None = None
    vent_peep_cm_h2o: float | None = None
    vent_fio2_pct: int | None = None
    vent_flow_rate: str | None = None
    vent_pmean_cm_h2o: float | None = None
    notes: str | None = None


class RadialEBUSSampling(BaseModel):
    model_config = ConfigDict(extra="ignore")
    guide_sheath_diameter: str | None = None
    ultrasound_pattern: str | None = None
    lesion_size_mm: float | None = None
    sampling_tools: List[str] = Field(default_factory=list)
    passes_per_tool: str | None = None
    fluoro_used: bool | None = None
    rose_result: str | None = None
    specimens: List[str] | None = None
    cxr_ordered: bool | None = None
    notes: str | None = None


class CBCTAugmentedBronchoscopy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ventilation_settings: str | None = None
    adjustment_description: str | None = None
    final_position: str | None = None
    radiation_parameters: str | None = None
    notes: str | None = None


class DyeMarkerPlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    guidance_method: str
    needle_gauge: str
    distance_from_pleura_cm: float | None = None
    dye_type: str
    dye_concentration: str | None = None
    volume_ml: float
    diffusion_observed: str | None = None
    notes: str | None = None


class EBUSStationSample(BaseModel):
    model_config = ConfigDict(extra="ignore")
    station_name: str
    size_mm: float | None = None
    passes: int | None = None
    echo_features: str | None = None
    biopsy_tools: List[str] = Field(default_factory=list)
    rose_result: str | None = None
    comments: str | None = None


class EBUSTBNA(BaseModel):
    model_config = ConfigDict(extra="ignore")
    needle_gauge: str | None = None
    stations: List[EBUSStationSample]
    elastography_used: bool | None = None
    elastography_pattern: str | None = None
    rose_available: bool | None = None
    overall_rose_diagnosis: str | None = None


class EBUSIntranodalForcepsBiopsy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    station_name: str
    size_mm: int | None = None
    ultrasound_features: str | None = None
    needle_gauge: str
    core_samples: int
    rose_result: str | None = None
    specimen_medium: str | None = None


class EBUS19GFNB(BaseModel):
    model_config = ConfigDict(extra="ignore")
    station_name: str
    passes: int
    rose_result: str | None = None
    elastography_pattern: str | None = None
    findings: str | None = None


class ValvePlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    valve_type: str
    valve_size: str | None = None
    lobe: str
    segment: str | None = None


class BLVRValvePlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    balloon_occlusion_performed: bool | None = None
    chartis_used: bool | None = None
    collateral_ventilation_absent: bool | None = None
    lobes_treated: List[str]
    valves: List[ValvePlacement]
    air_leak_reduction: str | None = None
    notes: str | None = None


class BLVRValveRemovalExchange(BaseModel):
    model_config = ConfigDict(extra="ignore")
    indication: str
    device_brand: str | None = None
    locations: List[str] = Field(default_factory=list)
    valves_removed: int
    valves_exchanged: int | None = None
    replacement_sizes: str | None = None
    mucosa_status: str | None = None
    tolerance_notes: str | None = None


class BLVRPostProcedureProtocol(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cxr_schedule: List[str] | None = None
    monitoring_plan: str | None = None
    steroids_plan: str | None = None
    antibiotics_plan: str | None = None
    ambulation_plan: str | None = None
    discharge_plan: str | None = None


class BLVRDischargeInstructions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    activity_restrictions: str | None = None
    monitoring_plan: str | None = None
    follow_up_plan: str | None = None
    contact_info: str | None = None


class TransbronchialCryobiopsy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lung_segment: str
    num_samples: int
    cryoprobe_size_mm: float | None = None
    freeze_seconds: int | None = None
    thaw_seconds: int | None = None
    blocker_type: str | None = None
    blocker_volume_ml: float | None = None
    blocker_location: str | None = None
    tests: List[str] | None = None
    radial_vessel_check: bool | None = None
    notes: str | None = None


class EndobronchialCryoablation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    site: str
    cryoprobe_size_mm: float | None = None
    freeze_seconds: int | None = None
    thaw_seconds: int | None = None
    cycles: int | None = None
    pattern: str | None = None
    post_patency: str | None = None
    notes: str | None = None


class CryoExtractionMucus(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airway_segment: str
    probe_size_mm: float | None = None
    freeze_seconds: int | None = None
    num_casts: int | None = None
    ventilation_result: str | None = None
    notes: str | None = None


class BPFLocalizationOcclusion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    culprit_segment: str
    balloon_type: str | None = None
    balloon_size_mm: int | None = None
    leak_reduction: str | None = None
    methylene_blue_used: bool | None = None
    contrast_used: bool | None = None
    instillation_findings: str | None = None
    notes: str | None = None


class BPFValvePlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    etiology: str | None = None
    culprit_location: str
    valve_type: str | None = None
    valve_size: str | None = None
    valves_placed: int | None = None
    leak_reduction: str | None = None
    additional_valves: str | None = None
    post_plan: str | None = None


class BPFSealantApplication(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sealant_type: str
    volume_ml: float | None = None
    dwell_minutes: int | None = None
    leak_reduction: str | None = None
    applications: int | None = None
    notes: str | None = None


class EndobronchialHemostasis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airway_segment: str
    iced_saline_ml: int | None = None
    epinephrine_concentration: str | None = None
    epinephrine_volume_ml: float | None = None
    tranexamic_acid_dose: str | None = None
    topical_thrombin_dose: str | None = None
    balloon_type: str | None = None
    balloon_location: str | None = None
    balloon_duration_sec: int | None = None
    balloon_cycles: int | None = None
    hemostasis_result: str | None = None
    escalation_plan: str | None = None
    tolerance: str | None = None


class EndobronchialBlockerPlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    blocker_type: str
    size: str | None = None
    side: str
    location: str
    inflation_volume_ml: float | None = None
    secured_method: str | None = None
    indication: str | None = None
    tolerance: str | None = None


class PhotodynamicTherapyLight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    agent: str
    administration_time: str | None = None
    lesion_site: str
    wavelength_nm: int | None = None
    fluence_j_cm2: float | None = None
    duration_minutes: int | None = None
    notes: str | None = None


class PhotodynamicTherapyDebridement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    site: str
    debridement_tool: str | None = None
    pre_patency_pct: int | None = None
    post_patency_pct: int | None = None
    bleeding: bool | None = None
    notes: str | None = None


class ForeignBodyRemoval(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airway_segment: str
    tools_used: List[str]
    passes: int | None = None
    removed_intact: bool | None = None
    mucosal_trauma: str | None = None
    bleeding: str | None = None
    hemostasis_method: str | None = None
    cxr_ordered: bool | None = None


class AwakeFiberopticIntubation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lidocaine_concentration: str | None = None
    lidocaine_volume_ml: int | None = None
    sedative: str | None = None
    ett_size: str
    route: str
    depth_cm: float | None = None
    tolerated: bool | None = None


class DoubleLumenTubePlacement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    size_fr: int
    alignment: str | None = None
    adjustments: str | None = None
    tolerated: bool | None = None


class AirwayStentSurveillance(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stent_type: str
    location: str
    findings: List[str] | None = None
    interventions: List[str] | None = None
    final_patency_pct: int | None = None


class WholeLungLavage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    side: str
    dlt_size_fr: int | None = None
    position: str | None = None
    total_volume_l: float | None = None
    max_volume_l: float | None = None
    aliquot_volume_l: float | None = None
    dwell_time_min: int | None = None
    num_cycles: int | None = None
    notes: str | None = None


class EUSB(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stations_sampled: List[str]
    needle_gauge: str | None = None
    passes: int | None = None
    rose_result: str | None = None
    complications: str | None = None


class BronchialWashing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airway_segment: str
    instilled_volume_ml: int
    returned_volume_ml: int
    tests: List[str]


class BronchialBrushing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lung_segment: str
    samples_collected: int
    brush_tool: str | None = None
    tests: List[str]


class BronchoalveolarLavageAlt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lung_segment: str
    instilled_volume_cc: int
    returned_volume_cc: int
    tests: List[str]


class EndobronchialBiopsy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airway_segment: str
    samples_collected: int
    tests: List[str]
    hemostasis_method: str | None = None
    lesion_removed: bool | None = None


class TransbronchialLungBiopsy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lung_segment: str
    samples_collected: int
    forceps_tools: str
    tests: List[str]


class TransbronchialNeedleAspiration(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lung_segment: str
    needle_tools: str
    samples_collected: int
    tests: List[str]


class TransbronchialBiopsyBasic(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lobe: str
    segment: str | None = None
    guidance: str
    tool: str
    number_of_biopsies: int
    specimen_tests: List[str] | None = None
    complications: str | None = None
    notes: str | None = None


class TherapeuticAspiration(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airway_segment: str
    aspirate_type: str


class RigidBronchoscopy(BaseModel):
    model_config = ConfigDict(extra="ignore")
    size_or_model: str | None = None
    hf_jv: bool | None = None
    interventions: List[str]
    flexible_scope_used: bool | None = None
    estimated_blood_loss_ml: int | None = None
    specimens: List[str] | None = None
    post_procedure_plan: str | None = None


class BAL(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lung_segment: str
    instilled_volume_cc: int
    returned_volume_cc: int
    tests: List[str]


class BronchoscopyShell(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sedation_type: str | None = None
    airway_route: str | None = None
    airway_overview: str | None = None
    right_lung_overview: str | None = None
    left_lung_overview: str | None = None
    mucosa_overview: str | None = None
    secretions_overview: str | None = None
    summary: str | None = None


__all__ = [
    "BAL",
    "AwakeFiberopticIntubation",
    "AirwayStentSurveillance",
    "BLVRDischargeInstructions",
    "BLVRPostProcedureProtocol",
    "BLVRValvePlacement",
    "BLVRValveRemovalExchange",
    "BPFLocalizationOcclusion",
    "BPFSealantApplication",
    "BPFValvePlacement",
    "BronchialBrushing",
    "BronchialWashing",
    "BronchoalveolarLavageAlt",
    "BronchoscopyShell",
    "CBCTAugmentedBronchoscopy",
    "CBCTFusion",
    "CryoExtractionMucus",
    "DyeMarkerPlacement",
    "EBUS19GFNB",
    "EBUSIntranodalForcepsBiopsy",
    "EBUSStationSample",
    "EBUSTBNA",
    "EMNBronchoscopy",
    "EndobronchialBiopsy",
    "EndobronchialBlockerPlacement",
    "EndobronchialCryoablation",
    "EndobronchialHemostasis",
    "EUSB",
    "FiducialMarkerPlacement",
    "ForeignBodyRemoval",
    "IonRegistrationComplete",
    "IonRegistrationDrift",
    "IonRegistrationPartial",
    "RoboticNavigation",
    "DoubleLumenTubePlacement",
    "PhotodynamicTherapyDebridement",
    "PhotodynamicTherapyLight",
    "RadialEBUSSampling",
    "RadialEBUSSurvey",
    "RigidBronchoscopy",
    "RoboticIonBronchoscopy",
    "RoboticMonarchBronchoscopy",
    "TherapeuticAspiration",
    "ToolInLesionConfirmation",
    "TransbronchialCryobiopsy",
    "TransbronchialLungBiopsy",
    "TransbronchialNeedleAspiration",
    "TransbronchialBiopsyBasic",
    "ValvePlacement",
    "WholeLungLavage",
]
