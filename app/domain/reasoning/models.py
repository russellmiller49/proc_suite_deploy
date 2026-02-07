"""Reasoning and provenance models."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceSpan(BaseModel):
    """A span of text that provides evidence for a decision."""

    source_id: str = ""  # "note", "path_report", "registry_form"
    start: int = 0
    end: int = 0
    text: str = ""

    model_config = {"frozen": True}


class ReasoningFields(BaseModel):
    """Full reasoning and provenance for a coding/extraction decision."""

    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0

    # Provenance (filled by application/infra layers)
    model_version: str = ""  # e.g. "gemini-1.5-pro-002"
    kb_version: str = ""  # "ip_coding_billing.v2.8"
    policy_version: str = ""  # "smart_hybrid_v2"
    keyword_map_version: Optional[str] = None
    registry_schema_version: Optional[str] = None
    negation_detector_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Decision trace
    trigger_phrases: List[str] = Field(default_factory=list)
    rule_paths: List[str] = Field(default_factory=list)
    confounders_checked: List[str] = Field(default_factory=list)
    qa_flags: List[str] = Field(default_factory=list)
    mer_notes: str = ""
    ncci_notes: str = ""

    model_config = {"frozen": False, "protected_namespaces": ()}
