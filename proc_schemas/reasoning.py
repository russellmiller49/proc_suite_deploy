"""Reasoning and provenance models for audit trails."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceSpan(BaseModel):
    """A span of text that provides evidence for a decision."""

    source_id: str = ""  # "note", "path_report", "registry_form"
    text: str = ""
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Aliases for compatibility
    @property
    def start(self) -> int:
        return self.start_char or 0

    @property
    def end(self) -> int:
        return self.end_char or 0


class ReasoningFields(BaseModel):
    """Full reasoning and provenance for a coding/extraction decision."""

    # Allow fields like model_version without protected namespace warnings.
    model_config = {"protected_namespaces": ()}

    # Evidence
    trigger_phrases: List[str] = Field(default_factory=list)
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)

    # Rationale
    coding_rationale: str = ""
    bundling_rationale: str = ""
    explanation: str = ""

    # Decision trace
    rule_paths: List[str] = Field(default_factory=list)
    confounders_checked: List[str] = Field(default_factory=list)
    qa_flags: List[str] = Field(default_factory=list)

    # Confidence
    confidence: float = 0.0

    # Compliance notes
    mer_notes: str = ""
    ncci_notes: str = ""

    # Provenance (filled by application/infra layers)
    model_version: str = ""  # e.g. "gemini-1.5-pro-002"
    kb_version: str = ""  # "ip_coding_billing.v2.8"
    policy_version: str = ""  # "smart_hybrid_v2"
    keyword_map_version: Optional[str] = None
    registry_schema_version: Optional[str] = None
    negation_detector_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
