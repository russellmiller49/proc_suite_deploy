"""Types for Phase 6 guarded registry self-correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SelfCorrectionTrigger:
    target_cpt: str
    ml_prob: float
    ml_bucket: str | None
    reason: str


@dataclass(frozen=True)
class JudgeProposal:
    target_cpt: str
    patch: list[dict]
    evidence_quotes: list[str]
    rationale: str
    model_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SelfCorrectionMetadata:
    trigger: SelfCorrectionTrigger
    applied_paths: list[str]
    evidence_quotes: list[str]
    config_snapshot: dict[str, Any]


__all__ = [
    "SelfCorrectionTrigger",
    "JudgeProposal",
    "ValidationResult",
    "SelfCorrectionMetadata",
]

