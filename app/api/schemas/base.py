"""Base Pydantic schemas for the FastAPI integration layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer, model_validator

from app.coder.schema import CoderOutput
from app.common.spans import Span
from app.registry.schema import RegistryRecord
from app.reporting import BundlePatch, MissingFieldIssue, ProcedureBundle, QuestionSpec


class CoderRequest(BaseModel):
    note: str
    allow_weak_sedation_docs: bool = False
    explain: bool = False
    locality: str = "00"
    setting: str = "facility"
    mode: str | None = None
    use_ml_first: bool = Field(
        default=False,
        description=(
            "If True, use ML-first hybrid pipeline (SmartHybridOrchestrator) "
            "with ternary classification (HIGH_CONF/GRAY_ZONE/LOW_CONF). "
            "If False, use legacy rule+LLM union merge."
        ),
    )


class HybridPipelineMetadata(BaseModel):
    """Metadata from the ML-first hybrid pipeline."""

    difficulty: str = Field(
        "", description="ML case difficulty: high_confidence, gray_zone, or low_confidence"
    )
    source: str = Field(
        "", description="Decision source: ml_rules_fastpath or hybrid_llm_fallback"
    )
    llm_used: bool = Field(False, description="Whether LLM was called for this case")
    ml_candidates: list[str] = Field(
        default_factory=list, description="CPT codes suggested by ML model"
    )
    fallback_reason: str | None = Field(
        None, description="Why LLM fallback was triggered (if applicable)"
    )
    rules_error: str | None = Field(
        None, description="Rules validation error message (if any)"
    )


CoderResponse = CoderOutput


class RegistryRequest(BaseModel):
    note: str
    explain: bool = False
    mode: str | None = None


class RegistryResponse(RegistryRecord):
    evidence: dict[str, list[Span]] = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    extraction: dict[str, Any]
    strict: bool = False


class VerifyResponse(BaseModel):
    bundle: ProcedureBundle
    issues: list[MissingFieldIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    inference_notes: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class JsonPatchOperation(BaseModel):
    op: Literal["add", "replace", "remove"]
    path: str
    value: Any | None = None


class RenderRequest(BaseModel):
    bundle: ProcedureBundle
    patch: BundlePatch | list[JsonPatchOperation] | None = None
    embed_metadata: bool = False
    strict: bool = False
    debug: bool = False
    include_debug: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_json_patch_values(cls, data: Any) -> Any:
        """Coerce interactive UI patch values into schema-friendly shapes."""
        if not isinstance(data, dict):
            return data
        patch_payload = data.get("patch")
        if not isinstance(patch_payload, list):
            return data

        normalized_patch: list[Any] = []
        for op in patch_payload:
            if isinstance(op, BaseModel):
                op_data = op.model_dump(exclude_none=False)
            elif isinstance(op, dict):
                op_data = dict(op)
            else:
                normalized_patch.append(op)
                continue

            path = op_data.get("path")
            if not isinstance(path, str):
                normalized_patch.append(op_data)
                continue

            value = op_data.get("value")
            if path.endswith("/echo_features") and isinstance(value, list):
                parts = [str(item).strip() for item in value if str(item).strip()]
                op_data["value"] = ", ".join(parts)
            elif path.endswith("/tests"):
                if isinstance(value, str):
                    normalized = value.replace(";", ",").replace("\n", ",")
                    op_data["value"] = [
                        part.strip() for part in normalized.split(",") if part.strip()
                    ]
                elif isinstance(value, list):
                    op_data["value"] = [str(item).strip() for item in value if str(item).strip()]

            normalized_patch.append(op_data)

        out = dict(data)
        out["patch"] = normalized_patch
        return out


class RenderResponse(BaseModel):
    bundle: ProcedureBundle
    markdown: str | None
    issues: list[MissingFieldIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    inference_notes: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    debug_notes: list[dict[str, Any]] | None = None

    @model_serializer(mode="wrap")
    def _serialize_optional_debug_notes(self, handler):
        data = handler(self)
        if data.get("debug_notes") is None:
            data.pop("debug_notes", None)
        return data


class QuestionsRequest(BaseModel):
    bundle: ProcedureBundle
    strict: bool = False


class QuestionsResponse(BaseModel):
    bundle: ProcedureBundle
    issues: list[MissingFieldIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    inference_notes: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    questions: list[QuestionSpec] = Field(default_factory=list)
    markdown: str | None = None


class SeedFromTextRequest(BaseModel):
    text: str
    already_scrubbed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    strict: bool = False
    debug: bool = False
    include_debug: bool = False


class SeedFromTextResponse(BaseModel):
    bundle: ProcedureBundle
    markdown: str | None
    issues: list[MissingFieldIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    inference_notes: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    questions: list[QuestionSpec] = Field(default_factory=list)
    missing_field_prompts: list[MissingFieldPrompt] = Field(
        default_factory=list,
        description="Suggested missing fields to document (completeness nudges).",
    )
    debug_notes: list[dict[str, Any]] | None = None

    @model_serializer(mode="wrap")
    def _serialize_optional_debug_notes(self, handler):
        data = handler(self)
        if data.get("debug_notes") is None:
            data.pop("debug_notes", None)
        return data


class KnowledgeMeta(BaseModel):
    version: str
    sha256: str


class QARunRequest(BaseModel):
    """Request schema for QA sandbox endpoint."""

    note_text: str
    modules_run: str = "all"  # "reporter", "coder", "registry", or "all"
    procedure_type: str | None = None


class UnifiedProcessRequest(BaseModel):
    """Request schema for unified registry + coder endpoint (extraction-first)."""

    note: str = Field(..., description="The procedure note text to process")
    already_scrubbed: bool = Field(
        False,
        description=(
            "If true, the server will skip PHI scrubbing and treat the note as already "
            "de-identified/scrubbed."
        ),
    )
    locality: str = Field("00", description="Geographic locality for RVU calculations")
    include_financials: bool = Field(True, description="Whether to include RVU/payment info")
    explain: bool = Field(False, description="Include extraction evidence/rationales")
    include_v3_event_log: bool = Field(
        False,
        description=(
            "If true, also run the event-log V3 extractor and include the raw "
            "event payload under `registry_v3_event_log`."
        ),
    )


class CodeSuggestionSummary(BaseModel):
    """Simplified code suggestion for unified response."""

    code: str
    description: str
    confidence: float
    rationale: str = ""
    review_flag: str = "optional"


class MissingFieldPrompt(BaseModel):
    """Suggested missing field to improve note completeness."""

    group: str = Field(
        default="",
        description="UI grouping label (e.g., Global, Navigation, EBUS).",
    )
    path: str = Field(
        ...,
        description=(
            "Dotted path relative to the registry root (supports [*] wildcards for arrays)."
        ),
    )
    label: str = Field(..., description="Short human label for the missing field.")
    severity: Literal["required", "recommended"] = Field(
        default="recommended",
        description="Required = high-priority completeness; Recommended = helpful but optional.",
    )
    message: str = Field(..., description="Actionable guidance for the user.")


class ReviewStatus(str, Enum):
    UNVERIFIED = "unverified"
    PENDING_PHI_REVIEW = "pending_phi_review"
    FINALIZED = "finalized"


class UnifiedProcessResponse(BaseModel):
    """Response schema combining registry extraction and CPT coding."""

    # Registry output
    registry: dict[str, Any] = Field(default_factory=dict, description="Extracted registry fields")
    registry_v3_event_log: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional raw V3 event-log extraction payload (note_id/source_filename/procedures[]). "
            "Present only when `include_v3_event_log=true`."
        ),
    )
    evidence: dict[str, Any] = Field(default_factory=dict, description="Extraction evidence spans")

    missing_field_prompts: list[MissingFieldPrompt] = Field(
        default_factory=list,
        description="Suggested missing fields to document (completeness nudges).",
    )

    # Coder output
    cpt_codes: list[str] = Field(default_factory=list, description="Derived CPT codes")
    suggestions: list[CodeSuggestionSummary] = Field(
        default_factory=list, description="Code suggestions with confidence"
    )

    # Financials (optional)
    total_work_rvu: float | None = None
    estimated_payment: float | None = None
    per_code_billing: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    pipeline_mode: str = "extraction_first"
    coder_difficulty: str = ""
    needs_manual_review: bool = False
    audit_warnings: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = Field(
        default=ReviewStatus.UNVERIFIED,
        description="Review status: unverified, pending_phi_review, or finalized",
    )

    # Versions
    kb_version: str = ""
    policy_version: str = ""
    processing_time_ms: float = 0.0


__all__ = [
    "CoderRequest",
    "CoderResponse",
    "CodeSuggestionSummary",
    "MissingFieldPrompt",
    "HybridPipelineMetadata",
    "KnowledgeMeta",
    "QARunRequest",
    "RegistryRequest",
    "RegistryResponse",
    "RenderRequest",
    "RenderResponse",
    "JsonPatchOperation",
    "QuestionsRequest",
    "QuestionsResponse",
    "SeedFromTextRequest",
    "SeedFromTextResponse",
    "UnifiedProcessRequest",
    "UnifiedProcessResponse",
    "ReviewStatus",
    "VerifyRequest",
    "VerifyResponse",
]
