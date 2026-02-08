"""IP Registry Schema v3 (registry entry schema).

This is the next-generation schema with enhanced features:
- Structured event timeline
- Enhanced complication modeling
- Better laterality tracking
- Procedure outcome flags

Note: this is **not** the LLM-facing V3 extraction event-log schema used by the
extraction engine. That extraction schema lives at `app.registry.schema.ip_v3_extraction`.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, List, Optional, Literal

from pydantic import BaseModel, Field

from proc_schemas.shared.ebus_events import NodeInteraction


# =============================================================================
# EBUS Node Interaction (Granular)
# =============================================================================


class LinearEBUS(BaseModel):
    """Linear EBUS with granular per-node events."""

    performed: bool = Field(False, description="Was Linear EBUS performed?")
    node_events: List[NodeInteraction] = Field(
        default_factory=list,
        description="List of all lymph node interactions, both sampled and inspected.",
    )
    needle_gauge: Optional[str] = Field(None, description="Size of needle used (e.g., '22G', '19G').")

    @property
    def stations_sampled(self) -> List[str]:
        """Derived property for CPT logic (e.g. 31653 requires count >= 3)."""
        return [
            event.station
            for event in self.node_events
            if event.action in ("needle_aspiration", "core_biopsy", "forceps_biopsy")
        ]

    @property
    def stations_inspected_only(self) -> List[str]:
        """Derived property for reporting."""
        return [event.station for event in self.node_events if event.action == "inspected_only"]


class PatientInfo(BaseModel):
    """Patient demographic information."""

    patient_id: str = ""
    mrn: str = ""
    age: Optional[int] = None
    sex: Optional[Literal["M", "F", "O"]] = None
    bmi: Optional[float] = None
    smoking_status: Optional[str] = None
    ecog_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="ECOG performance status score (0-4) when explicitly documented.",
    )
    ecog_text: Optional[str] = Field(
        default=None,
        description="Raw ECOG/Zubrod performance status text when not a single integer (e.g., '0-1').",
    )


class ProcedureInfo(BaseModel):
    """Procedure metadata."""

    procedure_id: str = ""
    procedure_date: Optional[date] = None
    procedure_type: str = ""
    indication: str = ""
    urgency: Literal["routine", "urgent", "emergent"] = "routine"
    operator: str = ""
    facility: str = ""


class Sedation(BaseModel):
    """Sedation details with timing."""

    type: Literal["none", "moderate", "deep", "general"] = "moderate"
    agents: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    provider: str = ""
    independent_observer: bool = False


class AnatomicLocation(BaseModel):
    """Detailed anatomic location with laterality."""

    name: str  # "4R", "RUL", "trachea"
    laterality: Optional[Literal["left", "right", "bilateral", "midline"]] = None
    segment: Optional[str] = None
    subsegment: Optional[str] = None
    lymph_node_station: Optional[str] = None


class BiopsySite(BaseModel):
    """Enhanced biopsy site with more detail."""

    location: AnatomicLocation
    technique: str = ""  # "EBUS-TBNA", "TBLB", "forceps", "cryobiopsy"
    passes: Optional[int] = None
    specimens_obtained: Optional[int] = None
    rose_result: Optional[str] = None
    lymphocytes_present: Optional[bool] = Field(
        default=None,
        description="Lymphocytes/lymphoid tissue present when explicitly documented (ROSE/pathology). None when not assessable.",
    )
    pathology_result: Optional[str] = None
    adequacy: Optional[Literal["adequate", "inadequate", "pending"]] = None


class StentPlacement(BaseModel):
    """Enhanced stent placement details."""

    location: AnatomicLocation
    action_type: Optional[Literal["placement", "removal", "revision", "assessment_only"]] = Field(
        default=None,
        description="High-level stent action classification.",
    )
    type: str = ""  # "silicone", "metal", "hybrid"
    subtype: str = ""  # "Y-stent", "straight", "tracheobronchial"
    size: str = ""  # "14x40mm"
    manufacturer: str = ""
    deployment_successful: bool = True
    complications: List[str] = Field(default_factory=list)


class ProcedureEvent(BaseModel):
    """A timestamped event during the procedure."""

    timestamp: Optional[datetime] = None
    event_type: str  # "start", "biopsy", "stent", "complication", "end"
    description: str
    location: Optional[AnatomicLocation] = None
    outcome: Optional[str] = None


class Finding(BaseModel):
    """A finding from the procedure."""

    category: str  # "anatomic", "pathologic", "incidental"
    description: str
    severity: Optional[str] = None
    location: Optional[AnatomicLocation] = None
    action_taken: Optional[str] = None


class Complication(BaseModel):
    """Enhanced complication with timing and causality."""

    type: str  # "bleeding", "pneumothorax", "hypoxia"
    severity: Literal["mild", "moderate", "severe"] = "mild"
    nashville_bleeding_grade: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="Nashville bleeding grade (0-4) when applicable (typically for bleeding events).",
    )
    onset: Literal["immediate", "delayed", "post-procedure"] = "immediate"
    onset_time: Optional[datetime] = None
    related_to: Optional[str] = None  # Which procedure step caused it
    intervention: Optional[str] = None
    resolved: bool = True
    resolution_time: Optional[datetime] = None


class ProcedureOutcome(BaseModel):
    """Summary of procedure outcome."""

    class ProcedureSuccessStatus(str, Enum):
        SUCCESS = "success"
        PARTIAL_SUCCESS = "partial_success"
        FAILED = "failed"
        ABORTED = "aborted"
        UNKNOWN = "unknown"

    completed: bool = True
    aborted: bool = False
    abort_reason: Optional[str] = None
    # New (2026-02): richer outcome status + reasons (kept additive for backward compatibility).
    procedure_success_status: ProcedureSuccessStatus = ProcedureSuccessStatus.UNKNOWN
    aborted_reason: Optional[str] = None
    complication_intervention: Optional[str] = None
    complication_duration: Optional[str] = None
    diagnostic_yield: Optional[str] = None
    therapeutic_success: Optional[bool] = None
    follow_up_planned: bool = False
    follow_up_notes: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        # Keep abort_reason and aborted_reason in sync for compatibility.
        if self.aborted_reason is None and self.abort_reason:
            self.aborted_reason = self.abort_reason
        elif self.abort_reason is None and self.aborted_reason:
            self.abort_reason = self.aborted_reason


class BalloonOcclusion(BaseModel):
    """Balloon occlusion / endobronchial blocker workflow details (e.g., air leak localization)."""

    performed: bool = False
    occlusion_location: Optional[str] = None
    air_leak_result: Optional[str] = None


class LesionMorphology(str, Enum):
    SPICULATED = "Spiculated"
    GROUND_GLASS = "Ground Glass"
    SOLID = "Solid"
    PART_SOLID = "Part-solid"
    CAVITARY = "Cavitary"
    CALCIFIED = "Calcified"


class LesionCharacteristics(BaseModel):
    """Structured target lesion characteristics (axes, morphology, location)."""

    size_long_axis_mm: float | None = Field(
        default=None,
        ge=0,
        description="Largest (long-axis) lesion dimension in mm when documented.",
    )
    size_short_axis_mm: float | None = Field(
        default=None,
        ge=0,
        description="Smaller (short-axis) lesion dimension in mm when documented.",
    )
    morphology: List[LesionMorphology] = Field(
        default_factory=list,
        description="Lesion morphology descriptors (e.g., spiculated, ground glass, solid).",
    )
    location_text: str | None = Field(
        default=None,
        description="Free-text anatomic lesion location (e.g., 'RLL posterior segment').",
    )
    bronchus_sign: Optional[Literal["Positive", "Negative", "Not assessed"]] = Field(
        default=None,
        description="CT bronchus sign for peripheral lesions when documented.",
    )
    distance_from_pleura_mm: float | None = Field(
        default=None,
        ge=0,
        description="Distance to pleura in mm when explicitly documented (0 if abutting).",
    )
    air_bronchogram_present: Optional[bool] = Field(
        default=None,
        description="Air bronchogram documented on CT when explicitly stated.",
    )
    pet_suv_max: float | None = Field(
        default=None,
        ge=0,
        description="Maximum PET SUV when explicitly documented.",
    )


class CentralAirwayObstruction(BaseModel):
    """Central airway obstruction structured summary (pre/post)."""

    class ObstructionType(str, Enum):
        INTRINSIC = "Intrinsic"
        EXTRINSIC = "Extrinsic"
        MIXED = "Mixed"

    obstruction_type: ObstructionType | None = Field(
        default=None,
        description="Intrinsic (endoluminal) vs extrinsic (compressive) vs mixed obstruction type.",
    )
    obstruction_percent_pre: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Pre-intervention obstruction percent (0-100) when documented.",
    )
    obstruction_percent_post: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Post-intervention obstruction percent (0-100) when documented.",
    )
    classification: str | None = Field(
        default=None,
        description="Optional severity classification (e.g., 'Myer-Cotton Grade').",
    )


class ClinicalContextV3(BaseModel):
    """Structured clinical context fields for V3."""

    lesion_characteristics: LesionCharacteristics | None = None
    central_airway_obstruction: CentralAirwayObstruction | None = None


class IPRegistryV3(BaseModel):
    """IP Registry Schema v3 - next-generation schema.

    New in v3:
    - Structured event timeline
    - Enhanced complication modeling with timing and causality
    - Better laterality tracking via AnatomicLocation
    - Procedure outcome flags
    - BMI and smoking status
    - Operator and facility info
    """

    # Metadata
    schema_version: Literal["v3"] = "v3"
    registry_id: str = "ip_registry"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Patient and procedure
    patient: PatientInfo = Field(default_factory=PatientInfo)
    procedure: ProcedureInfo = Field(default_factory=ProcedureInfo)

    # Sedation
    sedation: Sedation = Field(default_factory=Sedation)

    clinical_context: ClinicalContextV3 = Field(default_factory=ClinicalContextV3)

    established_tracheostomy_route: bool = Field(
        False,
        description="True when bronchoscopy/tracheoscopy is performed through an established tracheostomy route.",
    )

    # Event timeline
    events: List[ProcedureEvent] = Field(default_factory=list)

    # EBUS-TBNA
    ebus_performed: bool = False
    ebus_stations: List[BiopsySite] = Field(default_factory=list)
    ebus_station_count: int = 0

    # Transbronchial biopsy
    tblb_performed: bool = False
    tblb_sites: List[BiopsySite] = Field(default_factory=list)
    tblb_technique: Optional[Literal["forceps", "cryobiopsy", "both"]] = None

    # Navigation
    navigation_performed: bool = False
    navigation_system: str = ""
    navigation_target_reached: Optional[bool] = None

    # Radial EBUS
    radial_ebus_performed: bool = False
    radial_ebus_probe_position: Optional[
        Literal["Concentric", "Eccentric", "Adjacent", "Not visualized"]
    ] = Field(
        default=None,
        description="Radial EBUS view/probe position classification when documented.",
    )
    radial_ebus_findings: List[str] = Field(default_factory=list)

    # BAL
    bal_performed: bool = False
    bal_sites: List[AnatomicLocation] = Field(default_factory=list)
    bal_volume_ml: Optional[int] = None
    bal_return_ml: Optional[int] = None

    # Therapeutic procedures
    dilation_performed: bool = False
    dilation_sites: List[AnatomicLocation] = Field(default_factory=list)
    dilation_technique: str = ""

    stent_placed: bool = False
    stents: List[StentPlacement] = Field(default_factory=list)

    ablation_performed: bool = False
    ablation_technique: str = ""
    ablation_sites: List[AnatomicLocation] = Field(default_factory=list)

    blvr_performed: bool = False
    blvr_valves: int = 0
    blvr_target_lobe: str = ""
    blvr_chartis_performed: bool = False
    blvr_cv_result: Optional[str] = None

    # Balloon occlusion / endobronchial blocker workflows (non-Chartis).
    balloon_occlusion: BalloonOcclusion = Field(default_factory=BalloonOcclusion)

    # Findings
    findings: List[Finding] = Field(default_factory=list)

    # Complications
    complications: List[Complication] = Field(default_factory=list)
    any_complications: bool = False

    # Outcome
    outcome: ProcedureOutcome = Field(default_factory=ProcedureOutcome)

    # Disposition
    disposition: str = ""  # "home", "observation", "admit"
    length_of_stay_hours: Optional[int] = None

    # Free text
    impression: str = ""
    recommendations: str = ""

    model_config = {"frozen": False}
