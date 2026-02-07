"""
Pydantic Models for the Procedure Suite ML Advisor Integration

These models are designed to integrate with the existing proc_autocode and
proc_report schema patterns. They follow the established conventions:
- ConfigDict for model configuration
- Field with descriptions and examples
- Literal types for constrained values
- Optional with None defaults
- Consistent naming conventions

Usage:
    from app.proc_ml_advisor.schemas import (
        MLAdvisorInput,
        MLAdvisorSuggestion,
        HybridCodingResult,
        CodingTrace,
    )
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMS
# =============================================================================

class AdvisorBackend(str, Enum):
    """Available ML advisor backends."""
    STUB = "stub"
    GEMINI = "gemini"
    # Future extensions:
    # OPENAI = "openai"
    # LOCAL = "local"


class CodingPolicy(str, Enum):
    """Policy for combining rule and advisor codes."""
    RULES_ONLY = "rules_only"  # v1: Rules are authoritative
    ADVISOR_AUGMENTS = "advisor_augments"  # Future: Advisor can add codes
    HUMAN_REVIEW = "human_review"  # Future: Flag for human decision


class CodeType(str, Enum):
    """Type of medical code."""
    CPT = "CPT"
    HCPCS = "HCPCS"
    ICD10_CM = "ICD10-CM"
    ICD10_PCS = "ICD10-PCS"


class ConfidenceLevel(str, Enum):
    """Qualitative confidence levels for coding decisions."""
    HIGH = "high"  # >0.9
    MEDIUM = "medium"  # 0.7-0.9
    LOW = "low"  # 0.5-0.7
    UNCERTAIN = "uncertain"  # <0.5


class ProcedureCategory(str, Enum):
    """Procedure categories for interventional pulmonology."""
    BRONCHOSCOPY = "bronchoscopy"
    EBUS = "ebus"
    NAVIGATION = "navigation"
    PLEURAL = "pleural"
    ABLATION = "ablation"
    VALVE = "valve"
    STENT = "stent"
    AIRWAY = "airway"
    OTHER = "other"


# =============================================================================
# CODE-LEVEL MODELS
# =============================================================================

class CodeWithConfidence(BaseModel):
    """
    A single CPT/HCPCS code with confidence score and metadata.

    Matches the pattern used in proc_autocode for code results.
    """
    model_config = ConfigDict(
        frozen=True,
        use_attribute_docstrings=True,
        json_schema_extra={
            "example": {
                "code": "31653",
                "code_type": "CPT",
                "confidence": 0.95,
                "confidence_level": "high",
                "description": "EBUS-guided TBNA, 3+ stations",
                "is_addon": False,
            }
        },
    )

    code: str = Field(
        ...,
        description="CPT or HCPCS code (e.g., '31653', 'C9751')",
        min_length=4,
        max_length=10,
    )
    code_type: CodeType = Field(
        default=CodeType.CPT,
        description="Type of code (CPT, HCPCS, etc.)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    confidence_level: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Qualitative confidence level",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable code description",
    )
    is_addon: bool = Field(
        default=False,
        description="True if this is an add-on code (prefixed with +)",
    )
    primary_code: Optional[str] = Field(
        default=None,
        description="For add-on codes, the primary code it should be billed with",
    )

    @property
    def display_code(self) -> str:
        """Return code with + prefix for add-ons."""
        return f"+{self.code}" if self.is_addon else self.code


class CodeModifier(BaseModel):
    """
    A modifier applied to a CPT code.

    Follows CPT modifier conventions for interventional pulmonology.
    """
    model_config = ConfigDict(frozen=True)

    modifier: str = Field(
        ...,
        description="Modifier code (e.g., '-50', '-51', '-59')",
        pattern=r"^-?\d{2}$",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for applying this modifier",
    )
    applies_to: Optional[str] = Field(
        default=None,
        description="Code this modifier applies to",
    )


class NCCIWarning(BaseModel):
    """
    NCCI edit warning or bundling concern.

    Based on the NCCI rules in ip_golden_knowledge_v2_2.json.
    """
    model_config = ConfigDict(frozen=True)

    warning_id: str = Field(
        ...,
        description="Unique identifier for this warning type",
    )
    codes_involved: list[str] = Field(
        default_factory=list,
        description="Codes involved in this bundling issue",
    )
    message: str = Field(
        ...,
        description="Human-readable warning message",
    )
    severity: Literal["info", "warning", "error"] = Field(
        default="warning",
        description="Severity of the bundling concern",
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Suggested resolution or modifier to use",
    )
    citation: Optional[str] = Field(
        default=None,
        description="Reference to NCCI policy or CPT guideline",
    )


# =============================================================================
# STRUCTURED REPORT MODELS (INPUT)
# =============================================================================

class SamplingStation(BaseModel):
    """
    EBUS/TBNA sampling station details.

    Uses IASLC nomenclature for lymph node stations.
    """
    model_config = ConfigDict(frozen=True)

    station: str = Field(
        ...,
        description="IASLC station (e.g., '4R', '7', '11L')",
        pattern=r"^(2[RL]?|4[RL]|5|6|7|8|9|10[RL]|11[RL]|12[RL]|13|14)$",
    )
    needle_gauge: Optional[int] = Field(
        default=None,
        ge=18,
        le=25,
        description="Needle gauge used (typically 21 or 22)",
    )
    passes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of needle passes",
    )
    rose_result: Optional[str] = Field(
        default=None,
        description="ROSE cytology result if available",
    )
    adequate: Optional[bool] = Field(
        default=None,
        description="Whether sample was adequate",
    )


class PleuralProcedureDetails(BaseModel):
    """
    Details for pleural procedures (thoracentesis, IPC, pleuroscopy).

    Matches the thoracentesis_core and ipc_core templates.
    """
    model_config = ConfigDict(frozen=True)

    laterality: Literal["left", "right", "bilateral"] = Field(
        ...,
        description="Side of the procedure",
    )
    volume_ml: Optional[int] = Field(
        default=None,
        ge=0,
        description="Volume removed in mL",
    )
    fluid_appearance: Optional[str] = Field(
        default=None,
        description="Color and clarity of fluid",
    )
    imaging_guidance: bool = Field(
        default=True,
        description="Whether imaging guidance was used",
    )
    imaging_modality: Optional[Literal["ultrasound", "fluoroscopy", "CT", "none"]] = Field(
        default="ultrasound",
        description="Type of imaging guidance",
    )
    permanent_image: bool = Field(
        default=False,
        description="Whether permanent image was archived",
    )
    catheter_french: Optional[int] = Field(
        default=None,
        ge=6,
        le=32,
        description="Catheter size in French (for drainage/IPC)",
    )
    tunneled: Optional[bool] = Field(
        default=None,
        description="Whether catheter is tunneled (for IPC)",
    )


class BronchoscopyProcedureDetails(BaseModel):
    """
    Details for bronchoscopy procedures.

    Supports both diagnostic and therapeutic bronchoscopy.
    """
    model_config = ConfigDict(frozen=True)

    scope_type: Literal["flexible", "rigid", "combined"] = Field(
        default="flexible",
        description="Type of bronchoscope used",
    )
    navigation_used: bool = Field(
        default=False,
        description="Whether navigation system was used",
    )
    navigation_system: Optional[str] = Field(
        default=None,
        description="Navigation system (superDimension, Ion, Monarch, etc.)",
    )
    ebus_performed: bool = Field(
        default=False,
        description="Whether EBUS was performed",
    )
    ebus_type: Optional[Literal["linear", "radial"]] = Field(
        default=None,
        description="Type of EBUS probe",
    )
    stations_sampled: list[SamplingStation] = Field(
        default_factory=list,
        description="Lymph node stations sampled (for EBUS-TBNA)",
    )
    bal_performed: bool = Field(
        default=False,
        description="Whether BAL was performed",
    )
    biopsy_sites: list[str] = Field(
        default_factory=list,
        description="Sites where biopsies were taken",
    )
    therapeutic_procedures: list[str] = Field(
        default_factory=list,
        description="Therapeutic procedures performed (dilation, stent, etc.)",
    )


class SedationDetails(BaseModel):
    """
    Moderate sedation details for time-based coding.

    Used for 99152/99153 coding.
    """
    model_config = ConfigDict(frozen=True)

    sedation_provided: bool = Field(
        default=False,
        description="Whether proceduralist provided sedation",
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Sedation start time (HH:MM)",
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Sedation end time (HH:MM)",
    )
    total_minutes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total sedation time in minutes",
    )
    independent_observer: bool = Field(
        default=False,
        description="Whether independent trained observer was present",
    )


class StructuredProcedureReport(BaseModel):
    """
    Structured procedure report - the primary input to the autocode pipeline.

    This model represents the output of the reporter module that feeds into
    the coder. It follows the BaseProcedure pattern from proc_report.
    """
    model_config = ConfigDict(
        validate_assignment=False,
        use_attribute_docstrings=True,
        json_schema_extra={
            "example": {
                "report_id": "rpt-12345",
                "procedure_category": "ebus",
                "procedure_types": ["EBUS-TBNA"],
                "bronchoscopy": {
                    "scope_type": "flexible",
                    "ebus_performed": True,
                    "ebus_type": "linear",
                    "stations_sampled": [
                        {"station": "4R", "passes": 3},
                        {"station": "7", "passes": 2},
                        {"station": "11L", "passes": 2},
                    ],
                },
            }
        },
    )

    # Identifiers
    report_id: Optional[str] = Field(
        default=None,
        description="Unique report identifier",
    )
    encounter_id: Optional[str] = Field(
        default=None,
        description="Associated encounter/visit ID",
    )

    # Classification
    procedure_category: ProcedureCategory = Field(
        ...,
        description="Primary procedure category",
    )
    procedure_types: list[str] = Field(
        default_factory=list,
        description="Specific procedures performed",
    )

    # Raw text (for ML advisor)
    raw_text: Optional[str] = Field(
        default=None,
        description="Original procedure report text",
    )

    # Structured components (procedure-specific)
    bronchoscopy: Optional[BronchoscopyProcedureDetails] = Field(
        default=None,
        description="Bronchoscopy-specific details",
    )
    pleural: Optional[PleuralProcedureDetails] = Field(
        default=None,
        description="Pleural procedure-specific details",
    )
    sedation: Optional[SedationDetails] = Field(
        default=None,
        description="Sedation details for time-based coding",
    )

    # Metadata
    procedure_date: Optional[datetime] = Field(
        default=None,
        description="Date of procedure",
    )
    facility_type: Optional[Literal["facility", "non-facility", "asc"]] = Field(
        default="facility",
        description="Place of service for billing",
    )

    @property
    def station_count(self) -> int:
        """Count of EBUS sampling stations (for 31652 vs 31653)."""
        if self.bronchoscopy and self.bronchoscopy.stations_sampled:
            return len(self.bronchoscopy.stations_sampled)
        return 0


# =============================================================================
# ML ADVISOR MODELS
# =============================================================================

class MLAdvisorInput(BaseModel):
    """
    Input to the ML advisor.

    Contains all context needed for the model to suggest codes.
    This is constructed from the StructuredProcedureReport and rule engine output.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "trc-abc123",
                "report_text": "Bronchoscopy with EBUS-TBNA sampling of stations 4R, 7, and 11L...",
                "structured_report": {},
                "autocode_codes": ["31622", "31653"],
            }
        },
    )

    # Identifiers
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this advisor call",
    )
    report_id: Optional[str] = Field(
        default=None,
        description="Report identifier if available",
    )

    # Input content
    report_text: str = Field(
        default="",
        description="Raw procedure report text",
    )
    structured_report: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured report data as dict",
    )

    # Context from rule engine
    autocode_codes: list[str] = Field(
        default_factory=list,
        description="Codes already assigned by rule engine",
    )

    # Additional context
    procedure_category: Optional[ProcedureCategory] = Field(
        default=None,
        description="Procedure category if known",
    )
    facility_type: Optional[str] = Field(
        default=None,
        description="Place of service",
    )


class MLAdvisorSuggestion(BaseModel):
    """
    Output from the ML advisor.

    Contains suggested codes with confidence scores and explanations.
    The advisor ONLY suggests - it does not make final coding decisions.
    """
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "candidate_codes": ["31622", "31653", "31625"],
                "code_confidence": {"31622": 0.95, "31653": 0.92, "31625": 0.75},
                "explanation": "Consider adding 31625 for BAL if documented",
                "additions": ["31625"],
                "removals": [],
            }
        },
    )

    # Suggested codes
    candidate_codes: list[str] = Field(
        default_factory=list,
        description="All codes the advisor thinks apply",
    )
    code_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence score per code (0.0 to 1.0)",
    )

    # Explanation
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of suggestions",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings or concerns from the advisor",
    )

    # Comparison to rule codes
    additions: list[str] = Field(
        default_factory=list,
        description="Codes advisor suggests that rules didn't include",
    )
    removals: list[str] = Field(
        default_factory=list,
        description="Rule codes the advisor thinks may be wrong",
    )

    # Raw output (for debugging)
    raw_model_output: Optional[dict[str, Any]] = Field(
        default=None,
        description="Raw output from the LLM for debugging",
    )

    # Metadata
    model_name: Optional[str] = Field(
        default=None,
        description="Model used for suggestions",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="API call latency in milliseconds",
    )
    tokens_used: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total tokens used in the request",
    )

    @property
    def disagreements(self) -> list[str]:
        """All codes where advisor and rules differ."""
        return self.additions + self.removals

    @property
    def has_suggestions(self) -> bool:
        """True if advisor has meaningful suggestions."""
        return bool(self.candidate_codes) and self.model_name != "stub"


# =============================================================================
# HYBRID RESULT MODELS (OUTPUT)
# =============================================================================

class RuleEngineResult(BaseModel):
    """
    Output from the deterministic rule engine.

    Represents the authoritative coding decision in v1.
    """
    model_config = ConfigDict(frozen=True)

    codes: list[CodeWithConfidence] = Field(
        default_factory=list,
        description="Assigned CPT/HCPCS codes with confidence",
    )
    modifiers: list[CodeModifier] = Field(
        default_factory=list,
        description="Modifiers to apply",
    )
    ncci_warnings: list[NCCIWarning] = Field(
        default_factory=list,
        description="NCCI bundling warnings",
    )
    mer_applied: bool = Field(
        default=False,
        description="Whether Multiple Endoscopy Rule was applied",
    )
    rationales: dict[str, str] = Field(
        default_factory=dict,
        description="Code-level rationales from rule engine",
    )

    @property
    def code_list(self) -> list[str]:
        """Simple list of code strings."""
        return [c.code for c in self.codes]

    @property
    def confidence_dict(self) -> dict[str, float]:
        """Code to confidence mapping."""
        return {c.code: c.confidence for c in self.codes}


class HybridCodingResult(BaseModel):
    """
    Combined result from rule engine and ML advisor.

    This is the primary output model for the /code_with_advisor endpoint.
    In v1, final_codes always equals rule_codes.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "final_codes": ["31622", "31653"],
                "rule_codes": ["31622", "31653"],
                "rule_confidence": {"31622": 0.95, "31653": 0.90},
                "advisor_candidate_codes": ["31622", "31653", "31625"],
                "disagreements": ["31625"],
                "policy": "rules_only",
            }
        },
    )

    # Final output (rule engine in v1)
    final_codes: list[str] = Field(
        ...,
        description="Authoritative final codes (rules win in v1)",
    )

    # Rule engine results
    rule_codes: list[str] = Field(
        default_factory=list,
        description="Codes from deterministic rule engine",
    )
    rule_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores from rule engine",
    )
    rule_rationales: dict[str, str] = Field(
        default_factory=dict,
        description="Rationales from rule engine",
    )
    rule_modifiers: list[CodeModifier] = Field(
        default_factory=list,
        description="Modifiers from rule engine",
    )
    ncci_warnings: list[NCCIWarning] = Field(
        default_factory=list,
        description="NCCI warnings from rule engine",
    )
    mer_applied: bool = Field(
        default=False,
        description="Whether MER was applied",
    )

    # Advisor results
    advisor_candidate_codes: list[str] = Field(
        default_factory=list,
        description="Codes suggested by ML advisor",
    )
    advisor_code_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence per advisor code",
    )
    advisor_explanation: Optional[str] = Field(
        default=None,
        description="Human-readable advisor explanation",
    )

    # Comparison
    disagreements: list[str] = Field(
        default_factory=list,
        description="Codes where advisor and rules differ",
    )
    advisor_additions: list[str] = Field(
        default_factory=list,
        description="Codes advisor suggests but rules don't have",
    )
    advisor_removals: list[str] = Field(
        default_factory=list,
        description="Rule codes advisor thinks are wrong",
    )

    # Metadata
    policy: CodingPolicy = Field(
        default=CodingPolicy.RULES_ONLY,
        description="Policy used to combine results",
    )
    advisor_model: Optional[str] = Field(
        default=None,
        description="ML model used for suggestions",
    )
    advisor_latency_ms: Optional[float] = Field(
        default=None,
        description="Advisor API latency in ms",
    )

    @property
    def has_disagreements(self) -> bool:
        """True if advisor and rules disagree."""
        return bool(self.disagreements)


# =============================================================================
# CODING TRACE MODEL (LOGGING)
# =============================================================================

class CodingTrace(BaseModel):
    """
    Immutable record of a single autocode pipeline run.

    Used for building training datasets and evaluation.
    Logged to JSONL file after each coding run.
    """
    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "trace_id": "trc-abc123",
                "timestamp": "2025-11-29T10:30:00Z",
                "source": "api.code_with_advisor",
                "autocode_codes": ["31622", "31653"],
                "advisor_candidate_codes": ["31622", "31653", "31625"],
            }
        },
    )

    # Identifiers
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trace identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this trace was created",
    )
    report_id: Optional[str] = Field(
        default=None,
        description="Associated report ID if available",
    )

    # Inputs
    report_text: str = Field(
        default="",
        description="Raw procedure report text",
    )
    structured_report: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured report data",
    )
    procedure_category: Optional[str] = Field(
        default=None,
        description="Procedure category",
    )

    # Rule engine outputs
    autocode_codes: list[str] = Field(
        default_factory=list,
        description="Codes from rule engine",
    )
    autocode_modifiers: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Modifiers per code",
    )
    autocode_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence per code",
    )
    autocode_rationales: dict[str, str] = Field(
        default_factory=dict,
        description="Rationale per code",
    )
    ncci_warnings: list[str] = Field(
        default_factory=list,
        description="NCCI warning messages",
    )
    mer_applied: bool = Field(
        default=False,
        description="Whether MER was applied",
    )

    # Advisor outputs
    advisor_candidate_codes: list[str] = Field(
        default_factory=list,
        description="Codes from ML advisor",
    )
    advisor_code_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Advisor confidence per code",
    )
    advisor_explanation: Optional[str] = Field(
        default=None,
        description="Advisor explanation",
    )
    advisor_disagreements: list[str] = Field(
        default_factory=list,
        description="Where advisor and rules differ",
    )
    advisor_model: Optional[str] = Field(
        default=None,
        description="Model used for advisor",
    )
    advisor_latency_ms: Optional[float] = Field(
        default=None,
        description="Advisor latency in ms",
    )

    # Final output (for human review tracking)
    final_codes: Optional[list[str]] = Field(
        default=None,
        description="Human-reviewed final codes (if available)",
    )
    human_override: bool = Field(
        default=False,
        description="Whether human modified the codes",
    )

    # Cross-module linking (for integrated feedback loop)
    reporter_trace_id: Optional[str] = Field(
        default=None,
        description="Link to associated ReporterTrace for error attribution",
    )
    extraction_gaps: list[str] = Field(
        default_factory=list,
        description="Fields that were empty/uncertain from reporter extraction",
    )
    coding_limited_by_extraction: bool = Field(
        default=False,
        description="Whether coding was limited by extraction gaps",
    )

    # Provenance
    source: str = Field(
        default="unknown",
        description="Source of this coding run (e.g., 'api.code_with_advisor')",
    )
    pipeline_version: Optional[str] = Field(
        default=None,
        description="Version of the coding pipeline",
    )


# =============================================================================
# REPORTER TRACE MODEL (Phase 2)
# =============================================================================

class ReporterTrace(BaseModel):
    """
    Trace for reporter module extractions.

    Used to track extraction quality and link to downstream coding traces
    for error attribution. Part of the integrated ML feedback loop.

    The reporter transforms free-text procedure notes into structured reports.
    Tracking extraction quality helps identify:
    - Missing field extraction patterns
    - Low confidence extractions that need review
    - Extraction gaps that cause downstream coding errors
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "rpt-001",
                "input_text": "EBUS performed with sampling of 4R, 7, and 11L...",
                "extracted_fields": {"stations": ["4R", "7", "11L"], "bal_performed": True},
                "extraction_confidence": {"stations": 0.95, "bal_performed": 0.88},
                "field_completeness": 0.85,
            }
        },
    )

    # Identifiers
    trace_id: str = Field(
        default_factory=lambda: f"rpt-{uuid.uuid4().hex[:8]}",
        description="Unique trace identifier for reporter",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this trace was created",
    )
    report_id: Optional[str] = Field(
        default=None,
        description="Associated report ID if available",
    )

    # Input
    input_text: str = Field(
        default="",
        description="Original free-text procedure note",
    )
    input_source: Literal["free_text", "extractor_hints", "ehr_import", "qa_sandbox"] = Field(
        default="free_text",
        description="Source of the input text",
    )
    procedure_type_hint: Optional[str] = Field(
        default=None,
        description="Procedure type hint provided by user",
    )

    # Extraction output
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields extracted from the free text",
    )
    extraction_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence score per extracted field (0.0 to 1.0)",
    )
    extraction_model: Optional[str] = Field(
        default=None,
        description="Model used for extraction (e.g., 'gemini-1.5-pro')",
    )
    extraction_prompt_version: Optional[str] = Field(
        default=None,
        description="Version of the extraction prompt template",
    )

    # Quality metrics
    field_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of required fields successfully extracted",
    )
    missing_required_fields: list[str] = Field(
        default_factory=list,
        description="Required fields that could not be extracted",
    )
    low_confidence_fields: list[str] = Field(
        default_factory=list,
        description="Fields extracted with low confidence (<0.7)",
    )

    # Downstream impact tracking (populated by coder)
    coding_gaps_due_to_extraction: list[str] = Field(
        default_factory=list,
        description="Codes that couldn't be assigned due to extraction gaps",
    )
    linked_coding_trace_id: Optional[str] = Field(
        default=None,
        description="ID of the CodingTrace that used this extraction",
    )

    # Ground truth (from human review in QA)
    corrected_fields: Optional[dict[str, Any]] = Field(
        default=None,
        description="Human-corrected field values",
    )
    human_reviewed: bool = Field(
        default=False,
        description="Whether a human has reviewed this extraction",
    )

    # Provenance
    source: str = Field(
        default="unknown",
        description="Source of this extraction run",
    )
    pipeline_version: Optional[str] = Field(
        default=None,
        description="Version of the reporter pipeline",
    )


# =============================================================================
# REGISTRY TRACE MODEL (Phase 3)
# =============================================================================

class RegistryTrace(BaseModel):
    """
    Trace for registry export operations.

    Used to track data quality for registry submissions and link back
    to upstream reporter/coder traces for error attribution.

    The registry module transforms structured reports and codes into
    registry-ready bundles for external submission.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "reg-001",
                "report_id": "rpt-12345",
                "target_registry": "aabip",
                "assigned_codes": ["31622", "31653"],
                "field_completeness": 0.95,
                "validation_passed": True,
            }
        },
    )

    # Identifiers
    trace_id: str = Field(
        default_factory=lambda: f"reg-{uuid.uuid4().hex[:8]}",
        description="Unique trace identifier for registry",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this trace was created",
    )
    report_id: str = Field(
        ...,
        description="Associated report ID",
    )

    # Input
    structured_report: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured report from reporter module",
    )
    assigned_codes: list[str] = Field(
        default_factory=list,
        description="CPT/HCPCS codes from coder module",
    )
    target_registry: Literal["aabip", "sts", "aquire", "internal", "custom"] = Field(
        default="internal",
        description="Target registry for export",
    )

    # Output
    registry_bundle: dict[str, Any] = Field(
        default_factory=dict,
        description="Final registry-formatted data bundle",
    )
    export_format: Literal["json", "csv", "hl7", "fhir", "custom"] = Field(
        default="json",
        description="Export format used",
    )

    # Validation
    validation_passed: bool = Field(
        default=True,
        description="Whether pre-submission validation passed",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Validation errors (hard failures)",
    )
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings (soft issues)",
    )
    field_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of registry fields successfully populated",
    )
    missing_registry_fields: list[str] = Field(
        default_factory=list,
        description="Registry fields that couldn't be populated",
    )

    # Submission tracking
    submission_status: Optional[Literal["pending", "submitted", "accepted", "rejected"]] = Field(
        default=None,
        description="Status of registry submission",
    )
    rejection_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons for registry rejection (if rejected)",
    )
    submission_id: Optional[str] = Field(
        default=None,
        description="External submission/confirmation ID",
    )

    # Cross-module linking
    reporter_trace_id: Optional[str] = Field(
        default=None,
        description="Link to associated ReporterTrace",
    )
    coding_trace_id: Optional[str] = Field(
        default=None,
        description="Link to associated CodingTrace",
    )

    # Provenance
    source: str = Field(
        default="unknown",
        description="Source of this registry export",
    )
    pipeline_version: Optional[str] = Field(
        default=None,
        description="Version of the registry pipeline",
    )


# =============================================================================
# UNIFIED TRACE MODEL (Phase 4 - Integrated Feedback Loop)
# =============================================================================

class UnifiedTrace(BaseModel):
    """
    Unified trace linking all three modules for error attribution.

    This model connects reporter, coder, and registry traces for
    end-to-end quality tracking and improvement routing.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "unified_trace_id": "unified-001",
                "reporter_trace_id": "rpt-001",
                "coding_trace_id": "code-001",
                "registry_trace_id": "reg-001",
                "error_attribution": "reporter",
                "root_cause": "Failed to extract EBUS stations from text",
            }
        },
    )

    # Identifiers
    unified_trace_id: str = Field(
        default_factory=lambda: f"unified-{uuid.uuid4().hex[:8]}",
        description="Unique unified trace identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this unified trace was created",
    )

    # Module trace links
    reporter_trace_id: Optional[str] = Field(
        default=None,
        description="Link to ReporterTrace",
    )
    coding_trace_id: Optional[str] = Field(
        default=None,
        description="Link to CodingTrace",
    )
    registry_trace_id: Optional[str] = Field(
        default=None,
        description="Link to RegistryTrace",
    )

    # Error attribution
    has_errors: bool = Field(
        default=False,
        description="Whether any errors were detected",
    )
    error_attribution: Optional[Literal["reporter", "coder", "registry", "unknown"]] = Field(
        default=None,
        description="Which module is responsible for the error",
    )
    root_cause: Optional[str] = Field(
        default=None,
        description="Description of the root cause",
    )
    improvement_recommendation: Optional[str] = Field(
        default=None,
        description="Recommended improvement action",
    )

    # Quality scores
    overall_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Combined quality score across all modules",
    )
    reporter_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reporter extraction quality score",
    )
    coder_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Coder accuracy score",
    )
    registry_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Registry completeness score",
    )

    # Human review
    human_reviewed: bool = Field(
        default=False,
        description="Whether a human has reviewed this case",
    )
    human_feedback: Optional[str] = Field(
        default=None,
        description="Free-text feedback from human reviewer",
    )
    human_corrections: dict[str, Any] = Field(
        default_factory=dict,
        description="Human-made corrections by module",
    )


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class CodeRequest(BaseModel):
    """
    Request model for coding endpoints.

    Accepts either structured report or raw text.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "report_text": "Bronchoscopy with EBUS-TBNA...",
                "procedure_category": "ebus",
                "facility_type": "facility",
            }
        },
    )

    # Either structured report or raw text
    structured_report: Optional[StructuredProcedureReport] = Field(
        default=None,
        description="Structured procedure report",
    )
    report_text: Optional[str] = Field(
        default=None,
        description="Raw procedure report text",
    )

    # Optional context
    procedure_category: Optional[ProcedureCategory] = Field(
        default=None,
        description="Procedure category if known",
    )
    facility_type: Optional[Literal["facility", "non-facility", "asc"]] = Field(
        default="facility",
        description="Place of service",
    )

    # Advisor control
    include_advisor: bool = Field(
        default=True,
        description="Whether to include ML advisor suggestions",
    )


class CodeResponse(BaseModel):
    """
    Response model for coding endpoints.

    Includes both rule engine and advisor results.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "final_codes": ["31622", "31653"],
                "codes": [
                    {"code": "31622", "confidence": 0.95, "description": "Diagnostic bronchoscopy"},
                    {"code": "31653", "confidence": 0.92, "description": "EBUS-TBNA, 3+ stations"},
                ],
                "modifiers": ["-51"],
                "advisor_suggestions": {"31625": 0.75},
            }
        },
    )

    # Final codes (rule engine in v1)
    final_codes: list[str] = Field(
        ...,
        description="Final CPT/HCPCS codes",
    )

    # Detailed code information
    codes: list[CodeWithConfidence] = Field(
        default_factory=list,
        description="Codes with full details",
    )
    modifiers: list[str] = Field(
        default_factory=list,
        description="Applicable modifiers",
    )

    # Warnings
    ncci_warnings: list[str] = Field(
        default_factory=list,
        description="NCCI bundling warnings",
    )
    documentation_gaps: list[str] = Field(
        default_factory=list,
        description="Missing documentation for billing",
    )

    # Advisor (optional)
    advisor_suggestions: Optional[dict[str, float]] = Field(
        default=None,
        description="Advisor code suggestions with confidence",
    )
    advisor_explanation: Optional[str] = Field(
        default=None,
        description="Advisor explanation",
    )
    disagreements: list[str] = Field(
        default_factory=list,
        description="Where advisor and rules differ",
    )

    # Metadata
    mer_applied: bool = Field(
        default=False,
        description="Whether MER was applied",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Trace ID for debugging",
    )


# =============================================================================
# EVALUATION MODELS
# =============================================================================

class EvaluationMetrics(BaseModel):
    """
    Metrics from evaluating coding traces.

    Used by the evaluation harness.
    """
    model_config = ConfigDict(frozen=True)

    # Dataset statistics
    total_traces: int = Field(default=0)
    traces_with_advisor: int = Field(default=0)
    traces_with_final: int = Field(default=0)

    # Agreement metrics
    full_agreement: int = Field(default=0)
    advisor_suggested_extras: int = Field(default=0)
    advisor_suggested_removals: int = Field(default=0)

    # Code coverage
    unique_rule_codes: int = Field(default=0)
    unique_advisor_codes: int = Field(default=0)

    # Accuracy metrics (vs human review)
    rule_precision: Optional[float] = Field(default=None)
    rule_recall: Optional[float] = Field(default=None)
    advisor_precision: Optional[float] = Field(default=None)
    advisor_recall: Optional[float] = Field(default=None)

    @property
    def agreement_rate(self) -> Optional[float]:
        """Percentage of traces where advisor and rules agree."""
        if self.traces_with_advisor == 0:
            return None
        return self.full_agreement / self.traces_with_advisor
