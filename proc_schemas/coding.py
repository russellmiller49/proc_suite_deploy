"""Coding lifecycle models.

These models track codes through the suggestion → review → finalization lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from .reasoning import ReasoningFields


class CodeSuggestion(BaseModel):
    """AI-generated code suggestion awaiting review."""

    code: str
    description: str = ""
    source: Literal["rule", "llm", "hybrid", "manual", "extraction_first"] = "hybrid"
    hybrid_decision: Optional[str] = None  # HybridDecision enum value

    rule_confidence: Optional[float] = None
    llm_confidence: Optional[float] = None
    final_confidence: float = 0.0

    reasoning: ReasoningFields = Field(default_factory=ReasoningFields)
    review_flag: Literal["required", "recommended", "optional"] = "optional"

    # Evidence
    trigger_phrases: List[str] = Field(default_factory=list)
    evidence_verified: bool = False

    # Metadata
    suggestion_id: Optional[str] = None
    procedure_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": False, "protected_namespaces": ()}


class ReviewAction(BaseModel):
    """Clinician review decision on a CodeSuggestion."""

    suggestion_id: str = ""  # Links to the CodeSuggestion being reviewed
    action: Literal["accept", "reject", "modify"] = "accept"
    reviewer_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
    modified_code: Optional[str] = None  # If action == "modify"
    modified_description: Optional[str] = None
    final_code: Optional["FinalCode"] = None  # The resulting FinalCode if accepted/modified

    model_config = {"frozen": False, "protected_namespaces": ()}


class FinalCode(BaseModel):
    """Approved code ready for billing/registry."""

    code: str
    description: str = ""
    source: Literal["rule", "llm", "hybrid", "manual", "extraction_first"] = "hybrid"

    reasoning: ReasoningFields = Field(default_factory=ReasoningFields)
    review: Optional[ReviewAction] = None

    # Linkage
    procedure_id: str = ""
    suggestion_id: Optional[str] = None  # Links to original CodeSuggestion

    # Financial
    work_rvu: float = 0.0
    total_facility_rvu: float = 0.0
    facility_payment: float = 0.0
    modifiers: List[str] = Field(default_factory=list)

    finalized_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": False, "protected_namespaces": ()}


class CodingResult(BaseModel):
    """Complete result from the coding pipeline."""

    procedure_id: str
    suggestions: List[CodeSuggestion] = Field(default_factory=list)
    final_codes: List[FinalCode] = Field(default_factory=list)

    # Procedure classification
    procedure_type: str = Field(
        default="unknown",
        description="Procedure type classification (e.g., bronch_diagnostic, bronch_ebus, pleural, blvr)",
    )

    # Warnings and notes
    warnings: List[str] = Field(default_factory=list)
    ncci_notes: List[str] = Field(default_factory=list)
    mer_notes: List[str] = Field(default_factory=list)

    # Provenance
    kb_version: str = ""
    policy_version: str = ""
    model_version: str = ""

    # Timing
    processing_time_ms: float = 0.0
    llm_latency_ms: float = Field(
        default=0.0,
        description="LLM advisor latency in milliseconds (separate from total pipeline)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": False, "protected_namespaces": ()}
