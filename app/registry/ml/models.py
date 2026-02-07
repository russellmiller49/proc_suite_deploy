"""Pydantic models for atomic clinical action extraction.

These models represent the structured output of clinical action extraction,
designed to be the source of truth for deterministic CPT code derivation.

Design principles:
1. Atomic actions: Each model captures a single, verifiable clinical action
2. Evidence-backed: All extractions include source text spans
3. Confidence-scored: Extraction confidence for audit and review
4. CPT-derivable: Fields map directly to CPT coding logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from pydantic import BaseModel, Field

from app.common.spans import Span


# =============================================================================
# Base Types
# =============================================================================


@dataclass(slots=True)
class ActionResult:
    """Single field extraction result with evidence and confidence.

    This mirrors the SlotResult pattern used by deterministic extractors,
    ensuring compatibility with the existing extraction infrastructure.

    Attributes:
        value: Extracted value (type depends on field)
        evidence: Source text spans supporting the extraction
        confidence: Confidence score (0.0-1.0)
        source: Extraction method ("deterministic", "ml", "llm", "inferred")
    """

    value: Any
    evidence: list[Span]
    confidence: float
    source: Literal["deterministic", "ml", "llm", "inferred"] = "deterministic"


# =============================================================================
# Procedure-Specific Action Models
# =============================================================================


class EBUSActions(BaseModel):
    """EBUS (Endobronchial Ultrasound) clinical actions.

    CPT derivation:
    - 31652: EBUS-TBNA, 1-2 stations (performed=True, len(stations) in [1,2])
    - 31653: EBUS-TBNA, 3+ stations (performed=True, len(stations) >= 3)
    """

    performed: bool = False
    stations: list[str] = Field(
        default_factory=list,
        description="Lymph node stations sampled (e.g., ['4R', '7', '11L'])",
    )
    rose_available: bool | None = None
    rose_result: str | None = None
    intranodal_forceps: bool = False

    @property
    def station_count(self) -> int:
        """Number of unique stations sampled."""
        return len(set(self.stations))

    @property
    def cpt_station_bucket(self) -> Literal["0", "1-2", "3+"] | None:
        """Station count bucket for CPT coding."""
        if not self.performed:
            return None
        count = self.station_count
        if count == 0:
            return "0"
        if count <= 2:
            return "1-2"
        return "3+"


class BiopsyActions(BaseModel):
    """Biopsy-related clinical actions.

    CPT derivation:
    - 31625: Transbronchial biopsy (transbronchial.performed=True)
    - 31628: Transbronchial needle aspiration (tbna_conventional.performed=True)
    - 31629: Endobronchial biopsy (endobronchial.performed=True)
    """

    transbronchial_performed: bool = False
    transbronchial_sites: list[str] = Field(
        default_factory=list,
        description="Biopsy sites (e.g., ['RUL', 'LLL'])",
    )
    transbronchial_count: int | None = None
    transbronchial_tool: str | None = None

    cryobiopsy_performed: bool = False
    cryobiopsy_sites: list[str] = Field(default_factory=list)
    cryobiopsy_probe_size: str | None = None

    endobronchial_performed: bool = False
    endobronchial_sites: list[str] = Field(default_factory=list)

    tbna_conventional_performed: bool = False
    tbna_sites: list[str] = Field(default_factory=list)


class NavigationActions(BaseModel):
    """Navigational bronchoscopy clinical actions.

    CPT derivation:
    - 31627: Computer-assisted navigation (add-on code)
    """

    performed: bool = False
    platform: str | None = Field(
        default=None,
        description="Navigation system (e.g., 'superDimension', 'Ion', 'Monarch')",
    )
    is_robotic: bool = False
    radial_ebus_used: bool = False
    targets: list[str] = Field(
        default_factory=list,
        description="Navigation targets (e.g., ['RUL nodule', 'LLL mass'])",
    )
    cone_beam_ct_used: bool = False


class BALActions(BaseModel):
    """Bronchoalveolar lavage clinical actions.

    CPT derivation:
    - 31624: Bronchoscopy with BAL
    """

    performed: bool = False
    sites: list[str] = Field(
        default_factory=list,
        description="BAL sites (e.g., ['RML', 'Lingula'])",
    )
    volume_ml: int | None = None
    return_volume_ml: int | None = None


class BrushingsActions(BaseModel):
    """Bronchial brushings clinical actions.

    CPT derivation:
    - 31623: Bronchoscopy with brushing/protected specimen
    """

    performed: bool = False
    sites: list[str] = Field(default_factory=list)
    protected_brush: bool = False


class BronchialWashActions(BaseModel):
    """Bronchial wash clinical actions.

    CPT derivation:
    - 31622: Diagnostic bronchoscopy with wash (bundled)
    """

    performed: bool = False
    sites: list[str] = Field(default_factory=list)


class PleuralActions(BaseModel):
    """Pleural procedure clinical actions.

    CPT derivation:
    - 32554-32557: Thoracentesis codes
    - 32550: PleurX insertion
    - 32601/32606: Thoracoscopy codes
    """

    thoracentesis_performed: bool = False
    thoracentesis_diagnostic: bool = False
    thoracentesis_therapeutic: bool = False
    thoracentesis_volume_ml: int | None = None

    chest_tube_performed: bool = False
    chest_tube_type: str | None = None

    ipc_performed: bool = False
    ipc_action: Literal["insertion", "removal", "exchange", None] = None

    thoracoscopy_performed: bool = False
    thoracoscopy_type: Literal["medical", "surgical", None] = None

    pleurodesis_performed: bool = False
    pleurodesis_agent: str | None = None

    pleural_biopsy_performed: bool = False


class CAOActions(BaseModel):
    """Central airway obstruction therapeutic actions.

    CPT derivation:
    - 31641: Bronchoscopy with destruction of lesion
    - 31638: Balloon bronchoplasty
    """

    performed: bool = False
    location: str | None = None
    modalities: list[str] = Field(
        default_factory=list,
        description="Treatment modalities (e.g., ['APC', 'cryotherapy', 'mechanical'])",
    )
    pre_stenosis_pct: int | None = None
    post_stenosis_pct: int | None = None

    dilation_performed: bool = False
    dilation_type: str | None = None

    thermal_ablation_performed: bool = False
    cryotherapy_performed: bool = False
    mechanical_debridement_performed: bool = False


class StentActions(BaseModel):
    """Airway stent clinical actions.

    CPT derivation:
    - 31636: Bronchial stent placement
    - 31631: Tracheal stent placement
    """

    performed: bool = False
    action: Literal["insertion", "removal", "exchange", "repositioning", None] = None
    location: str | None = None
    stent_type: str | None = None
    stent_size: str | None = None


class BLVRActions(BaseModel):
    """Bronchoscopic lung volume reduction clinical actions.

    CPT derivation:
    - 31647: Valve removal
    - 31651: Valve insertion (each valve)
    """

    performed: bool = False
    target_lobe: str | None = None
    valve_count: int | None = None
    valve_type: str | None = None
    chartis_performed: bool = False
    chartis_result: str | None = None


class TherapeuticActions(BaseModel):
    """Other therapeutic bronchoscopy actions."""

    aspiration_performed: bool = False
    foreign_body_removal_performed: bool = False
    foreign_body_type: str | None = None

    whole_lung_lavage_performed: bool = False
    wll_side: Literal["left", "right", None] = None

    bronchial_thermoplasty_performed: bool = False
    thermoplasty_lobe: str | None = None


class ComplicationActions(BaseModel):
    """Procedure complications."""

    any_complication: bool = False
    complications: list[str] = Field(default_factory=list)
    bleeding: bool = False
    bleeding_severity: Literal["minimal", "moderate", "severe", None] = None
    pneumothorax: bool = False
    hypoxia: bool = False
    other: str | None = None


class SedationActions(BaseModel):
    """Sedation and anesthesia details."""

    sedation_type: Literal["MAC", "moderate", "general", "topical", None] = None
    airway_type: Literal["flexible", "rigid", "both", None] = None


# =============================================================================
# Aggregate Clinical Actions Model
# =============================================================================


class ClinicalActions(BaseModel):
    """Complete structured representation of clinical actions from a procedure note.

    This is the primary output of the ActionPredictor and serves as the
    source of truth for deterministic CPT code derivation.

    All fields are extracted with evidence and confidence scores, enabling:
    1. Deterministic CPT coding based on structured data
    2. Audit trails via evidence spans
    3. Confidence-based review flagging
    """

    # Core procedure actions
    ebus: EBUSActions = Field(default_factory=EBUSActions)
    biopsy: BiopsyActions = Field(default_factory=BiopsyActions)
    navigation: NavigationActions = Field(default_factory=NavigationActions)
    bal: BALActions = Field(default_factory=BALActions)
    brushings: BrushingsActions = Field(default_factory=BrushingsActions)
    bronchial_wash: BronchialWashActions = Field(default_factory=BronchialWashActions)

    # Pleural procedures
    pleural: PleuralActions = Field(default_factory=PleuralActions)

    # Therapeutic interventions
    cao: CAOActions = Field(default_factory=CAOActions)
    stent: StentActions = Field(default_factory=StentActions)
    blvr: BLVRActions = Field(default_factory=BLVRActions)
    therapeutic: TherapeuticActions = Field(default_factory=TherapeuticActions)

    # Context
    complications: ComplicationActions = Field(default_factory=ComplicationActions)
    sedation: SedationActions = Field(default_factory=SedationActions)

    # Metadata
    diagnostic_bronchoscopy: bool = False
    rigid_bronchoscopy: bool = False

    def get_performed_procedures(self) -> list[str]:
        """Return list of performed procedure names."""
        performed = []

        if self.diagnostic_bronchoscopy:
            performed.append("diagnostic_bronchoscopy")
        if self.rigid_bronchoscopy:
            performed.append("rigid_bronchoscopy")
        if self.ebus.performed:
            performed.append("linear_ebus")
        if self.navigation.performed:
            performed.append("navigational_bronchoscopy")
        if self.navigation.radial_ebus_used:
            performed.append("radial_ebus")
        if self.biopsy.transbronchial_performed:
            performed.append("transbronchial_biopsy")
        if self.biopsy.cryobiopsy_performed:
            performed.append("transbronchial_cryobiopsy")
        if self.biopsy.endobronchial_performed:
            performed.append("endobronchial_biopsy")
        if self.biopsy.tbna_conventional_performed:
            performed.append("tbna_conventional")
        if self.bal.performed:
            performed.append("bal")
        if self.brushings.performed:
            performed.append("brushings")
        if self.bronchial_wash.performed:
            performed.append("bronchial_wash")
        if self.pleural.thoracentesis_performed:
            performed.append("thoracentesis")
        if self.pleural.chest_tube_performed:
            performed.append("chest_tube")
        if self.pleural.ipc_performed:
            performed.append("ipc")
        if self.pleural.thoracoscopy_performed:
            performed.append("medical_thoracoscopy")
        if self.pleural.pleurodesis_performed:
            performed.append("pleurodesis")
        if self.pleural.pleural_biopsy_performed:
            performed.append("pleural_biopsy")
        if self.cao.thermal_ablation_performed:
            performed.append("thermal_ablation")
        if self.cao.cryotherapy_performed:
            performed.append("cryotherapy")
        if self.cao.dilation_performed:
            performed.append("airway_dilation")
        if self.stent.performed:
            performed.append("airway_stent")
        if self.blvr.performed:
            performed.append("blvr")
        if self.therapeutic.aspiration_performed:
            performed.append("therapeutic_aspiration")
        if self.therapeutic.foreign_body_removal_performed:
            performed.append("foreign_body_removal")
        if self.therapeutic.whole_lung_lavage_performed:
            performed.append("whole_lung_lavage")
        if self.therapeutic.bronchial_thermoplasty_performed:
            performed.append("bronchial_thermoplasty")

        return performed


# =============================================================================
# Prediction Result Container
# =============================================================================


class PredictionResult(BaseModel):
    """Complete result from ActionPredictor.predict().

    Contains:
    - actions: Structured ClinicalActions (primary output)
    - field_extractions: Per-field ActionResult with evidence
    - metadata: Extraction statistics and configuration
    """

    actions: ClinicalActions
    field_extractions: dict[str, ActionResult] = Field(default_factory=dict)
    confidence_overall: float = Field(
        default=0.0, description="Overall confidence (min of all field confidences)"
    )
    extraction_method: Literal["deterministic", "hybrid", "llm"] = "hybrid"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}  # For ActionResult dataclass

    def get_low_confidence_fields(self, threshold: float = 0.7) -> list[str]:
        """Return fields with confidence below threshold."""
        return [
            field
            for field, result in self.field_extractions.items()
            if result.confidence < threshold
        ]

    def needs_review(self, threshold: float = 0.7) -> bool:
        """Check if extraction needs human review."""
        return len(self.get_low_confidence_fields(threshold)) > 0
