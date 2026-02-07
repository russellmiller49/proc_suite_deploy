"""Types for RAW-ML audit comparison reporting.

These dataclasses provide a structured, machine-readable representation of:
- The ML audit set (RAW-ML predictor output)
- Deterministic CPT codes derived from RegistryRecord-only rules
- A comparison report capturing discrepancies (no auto-merge behavior)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditPrediction:
    cpt: str
    prob: float
    bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"cpt": self.cpt, "prob": self.prob, "bucket": self.bucket}


@dataclass(frozen=True)
class AuditConfigSnapshot:
    use_buckets: bool
    top_k: int
    min_prob: float
    self_correct_min_prob: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_buckets": self.use_buckets,
            "top_k": self.top_k,
            "min_prob": self.min_prob,
            "self_correct_min_prob": self.self_correct_min_prob,
        }


@dataclass(frozen=True)
class AuditCompareReport:
    derived_codes: list[str]
    ml_audit_codes: list[AuditPrediction]
    missing_in_derived: list[AuditPrediction]
    missing_in_ml: list[str]
    high_conf_omissions: list[AuditPrediction]
    ml_difficulty: str | None
    config: AuditConfigSnapshot
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "derived_codes": list(self.derived_codes),
            "ml_audit_codes": [p.to_dict() for p in self.ml_audit_codes],
            "missing_in_derived": [p.to_dict() for p in self.missing_in_derived],
            "missing_in_ml": list(self.missing_in_ml),
            "high_conf_omissions": [p.to_dict() for p in self.high_conf_omissions],
            "ml_difficulty": self.ml_difficulty,
            "config": self.config.to_dict(),
            "warnings": list(self.warnings),
        }


__all__ = ["AuditCompareReport", "AuditConfigSnapshot", "AuditPrediction"]

