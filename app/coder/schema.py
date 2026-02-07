"""Pydantic data structures for the coder pipeline."""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field

from app.common.spans import Span

__all__ = [
    "DetectedIntent",
    "CodeDecision",
    "BundleDecision",
    "CoderOutput",
    "PerCodeBilling",
    "FinancialSummary",
    "LLMCodeSuggestion",
]


class DetectedIntent(BaseModel):
    """Represents an intermediate intent inferred from the note."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    intent: str
    value: str | None = None
    payload: dict[str, Any] | None = None
    confidence: float | None = None
    evidence: List[Span] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)


class CodeDecision(BaseModel):
    """Finalized CPT decision including bundles and MER metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cpt: str
    description: str
    modifiers: list[str] = Field(default_factory=list)
    rationale: str | list[str] = ""
    evidence: List[Span] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    mer_role: str | None = None  # "primary", "secondary", "add_on"
    mer_allowed: float | None = None
    mer_explanation: str | None = None  # Human-readable explanation for MER
    confidence: float = 0.0
    rule_trace: List[str] = Field(default_factory=list)


class BundleDecision(BaseModel):
    """Explains how bundling/NCCI edits affected the code set."""

    pair: tuple[str, str]
    action: str
    reason: str
    rule: str | None = None


class PerCodeBilling(BaseModel):
    cpt_code: str
    description: str | None = None
    modifiers: list[str] = Field(default_factory=list)

    # Base MPFS values (per-claim if billed alone)
    work_rvu: float = 0.0
    total_facility_rvu: float = 0.0
    total_nonfacility_rvu: float = 0.0
    facility_payment: float = 0.0
    nonfacility_payment: float = 0.0

    # Adjusted values after MER / multiple procedure rules
    allowed_facility_rvu: float = 0.0
    allowed_nonfacility_rvu: float = 0.0
    allowed_facility_payment: float = 0.0
    allowed_nonfacility_payment: float = 0.0

    mer_role: str | None = None  # "primary", "secondary", "add_on"
    mer_allowed: float | None = None
    mer_reduction: float | None = None
    mp_rule: str | None = None   # e.g. "multiple_endoscopy_primary", "multiple_procedure_50pct"


class FinancialSummary(BaseModel):
    conversion_factor: float
    locality: str
    per_code: list[PerCodeBilling]
    total_work_rvu: float
    total_facility_payment: float
    total_nonfacility_payment: float


class LLMCodeSuggestion(BaseModel):
    cpt: str
    description: str | None = None
    rationale: str | None = None


class CoderOutput(BaseModel):
    """Top-level payload returned by the coder pipeline."""

    codes: list[CodeDecision]
    intents: list[DetectedIntent] = Field(default_factory=list)
    mer_summary: dict[str, Any] | None = None
    ncci_actions: list[BundleDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    version: str = "0.1.0"

    # NEW
    financials: FinancialSummary | None = None

    # LLM Advisor
    llm_suggestions: list[LLMCodeSuggestion] = Field(default_factory=list)
    llm_disagreements: list[str] = Field(default_factory=list)

    # Optional strict JSON payload for LLM assistant clients
    llm_assistant_payload: dict[str, Any] | None = None

    # ML-first hybrid pipeline metadata
    hybrid_metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Metadata from ML-first hybrid pipeline (SmartHybridOrchestrator). "
            "Includes difficulty classification, source (fastpath vs fallback), "
            "LLM usage, and ML candidate codes."
        ),
    )

    # Legacy alias for explanation text
    explanation: str | None = None
