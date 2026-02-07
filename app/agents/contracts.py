"""Agent contracts defining I/O schemas for the 3-agent reporter pipeline.

This module defines the data contracts between:
- ParserAgent: Splits raw text into segments and extracts entities
- SummarizerAgent: Produces section summaries from segments/entities
- StructurerAgent: Maps summaries to registry model and generates codes
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Tuple, Literal


class AgentWarning(BaseModel):
    """A warning from an agent that doesn't prevent output."""

    code: str  # "MISSING_HEADER", "AMBIGUOUS_SECTION"
    message: str
    section: Optional[str] = None


class AgentError(BaseModel):
    """An error from an agent that may prevent successful output."""

    code: str  # "NO_SECTIONS_FOUND", "PARSING_FAILED"
    message: str
    section: Optional[str] = None


class Segment(BaseModel):
    """A segmented portion of the note with optional character spans."""

    id: str = ""
    type: str  # "HISTORY", "PROCEDURE", "FINDINGS", "IMPRESSION", etc.
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    spans: List[Tuple[int, int]] = Field(default_factory=list)


class Entity(BaseModel):
    """An entity extracted from the note, such as a station or stent."""

    label: str
    value: str
    name: str = ""  # For backwards compatibility
    type: str = ""
    offsets: Optional[Tuple[int, int]] = None
    evidence_segment_id: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None


class Trace(BaseModel):
    """Trace metadata capturing what triggered an agent's output."""

    trigger_phrases: List[str] = Field(default_factory=list)
    rule_paths: List[str] = Field(default_factory=list)
    confounders_checked: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    notes: Optional[str] = None


class ParserIn(BaseModel):
    """Input to the Parser agent."""

    note_id: str
    raw_text: str


class ParserOut(BaseModel):
    """Output from the Parser agent."""

    note_id: str = ""
    segments: List[Segment] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    trace: Trace = Field(default_factory=Trace)
    warnings: List[AgentWarning] = Field(default_factory=list)
    errors: List[AgentError] = Field(default_factory=list)
    status: Literal["ok", "degraded", "failed"] = "ok"


class SummarizerIn(BaseModel):
    """Input to the Summarizer agent."""

    parser_out: ParserOut


class SummarizerOut(BaseModel):
    """Output from the Summarizer agent."""

    note_id: str = ""
    summaries: Dict[str, str] = Field(default_factory=dict)
    caveats: List[str] = Field(default_factory=list)
    trace: Trace = Field(default_factory=Trace)
    warnings: List[AgentWarning] = Field(default_factory=list)
    errors: List[AgentError] = Field(default_factory=list)
    status: Literal["ok", "degraded", "failed"] = "ok"


class StructurerIn(BaseModel):
    """Input to the Structurer agent."""

    summarizer_out: SummarizerOut


class StructurerOut(BaseModel):
    """Output from the Structurer agent."""

    note_id: str = ""
    registry: Dict[str, Any] = Field(default_factory=dict)
    codes: Dict[str, Any] = Field(default_factory=dict)
    rationale: Dict[str, Any] = Field(default_factory=dict)
    trace: Trace = Field(default_factory=Trace)
    warnings: List[AgentWarning] = Field(default_factory=list)
    errors: List[AgentError] = Field(default_factory=list)
    status: Literal["ok", "degraded", "failed"] = "ok"


class PipelineResult(BaseModel):
    """Full result from running the 3-agent pipeline."""

    pipeline_status: Literal["ok", "degraded", "failed_parser", "failed_summarizer", "failed_structurer"]
    parser: Optional[ParserOut] = None
    summarizer: Optional[SummarizerOut] = None
    structurer: Optional[StructurerOut] = None
    registry: Optional[Dict[str, Any]] = None
    codes: Optional[Dict[str, Any]] = None
