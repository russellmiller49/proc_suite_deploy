"""QA Sandbox response schemas.

This module defines the structured response types for the /qa/run endpoint,
including ModuleResult envelopes for each module (registry, reporter, coder)
and the composite QARunResponse.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ModuleStatus(str, Enum):
    """Status of a module execution."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


T = TypeVar("T")


class ModuleResult(BaseModel, Generic[T]):
    """Generic envelope for module execution results.

    Attributes:
        status: Execution status (success, error, skipped)
        data: Typed payload on success, None on error/skip
        error_message: Human-readable error description on failure
        error_code: Machine-readable error code on failure
    """

    status: ModuleStatus
    data: T | None = None
    error_message: str | None = None
    error_code: str | None = None


class RegistryData(BaseModel):
    """Structured data from registry extraction.

    Attributes:
        record: The extracted registry record (procedure data)
        evidence: Field-to-spans mapping showing source evidence
    """

    record: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class ReporterData(BaseModel):
    """Structured data from reporter module.

    Attributes:
        markdown: Generated procedure note markdown
        bundle: Structured procedure bundle (when registry data available)
        issues: List of validation issues (missing critical fields)
        warnings: List of warning messages
        procedure_core: Procedure core data (simple reporter fallback)
        indication: Indication text (simple reporter fallback)
        postop: Post-op plan text (simple reporter fallback)
        fallback_used: Whether simple reporter fallback was used
    """

    markdown: str | None = None
    bundle: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Simple reporter fallback fields
    procedure_core: dict[str, Any] | None = None
    indication: dict[str, Any] | None = None
    postop: dict[str, Any] | None = None
    fallback_used: bool = False
    render_mode: Literal["structured", "simple_fallback"] | None = None
    fallback_reason: str | None = None
    reporter_errors: list[str] = Field(default_factory=list)


class CodeEntry(BaseModel):
    """A single code suggestion entry."""

    cpt: str
    description: str | None = None
    confidence: float | None = None
    source: str | None = None
    hybrid_decision: str | None = None
    review_flag: bool = False


class CoderData(BaseModel):
    """Structured data from coder module.

    Attributes:
        codes: List of suggested CPT codes with metadata
        total_work_rvu: Total work RVU value (if calculated)
        estimated_payment: Estimated payment amount (if calculated)
        bundled_codes: Codes that were bundled/excluded by NCCI
        kb_version: Knowledge base version used
        policy_version: Coding policy version used
        model_version: LLM model version (if used)
        processing_time_ms: Processing time in milliseconds
    """

    codes: list[CodeEntry] = Field(default_factory=list)
    total_work_rvu: float | None = None
    estimated_payment: float | None = None
    bundled_codes: list[str] = Field(default_factory=list)
    kb_version: str | None = None
    policy_version: str | None = None
    model_version: str | None = None
    processing_time_ms: int | None = None

    # Allow model_version without protected namespace warnings.
    model_config = ConfigDict(protected_namespaces=())


class QARunResponse(BaseModel):
    """Composite response for QA sandbox endpoint.

    Attributes:
        overall_status: Aggregate status (completed, partial_success, failed)
        registry: Registry module result envelope
        reporter: Reporter module result envelope
        coder: Coder module result envelope
        reporter_version: Reporter engine version
        coder_version: Coder service version
        repo_branch: Git branch name
        repo_commit_sha: Git commit SHA
    """

    # Allow fields like model_version without protected namespace warnings.
    model_config = ConfigDict(protected_namespaces=())

    overall_status: str = "completed"
    registry: ModuleResult[RegistryData] | None = None
    reporter: ModuleResult[ReporterData] | None = None
    coder: ModuleResult[CoderData] | None = None

    # Legacy flat outputs (kept for backward compatibility with UI/API routes)
    registry_output: dict[str, Any] | None = None
    reporter_output: dict[str, Any] | None = None
    coder_output: dict[str, Any] | None = None

    # Model provenance (registry predictor bundle)
    model_backend: str | None = None
    model_version: str | None = None

    reporter_version: str | None = None
    coder_version: str | None = None
    repo_branch: str | None = None
    repo_commit_sha: str | None = None


__all__ = [
    "CodeEntry",
    "CoderData",
    "ModuleResult",
    "ModuleStatus",
    "QARunResponse",
    "RegistryData",
    "ReporterData",
]
