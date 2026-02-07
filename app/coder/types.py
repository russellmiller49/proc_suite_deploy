"""Internal coding pipeline types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass
class EvidenceSpan:
    """Snippet or offset supporting a code decision.

    Note: Snippets/offsets may contain PHI and should never be logged verbatim.
    """

    start: int | None = None
    end: int | None = None
    snippet: str | None = None
    source: Literal["llm", "rule", "heuristic", "manual"] | None = None


@dataclass
class CodeCandidate:
    """Internal representation of a candidate CPT code."""

    code: str
    confidence: float = 1.0
    reason: str | None = None
    evidence: Sequence[EvidenceSpan] | None = None


@dataclass
class EBUSNodeEvidence:
    """Structured evidence for an EBUS lymph node station."""

    station: str
    action: Literal["Inspection", "Sampling"]
    method: str | None = None


@dataclass
class PeripheralLesionEvidence:
    """Structured evidence describing a peripheral lesion intervention."""

    lobe: str | None
    segment: str | None
    actions: list[str]
    navigation: bool
    radial_ebus: bool
