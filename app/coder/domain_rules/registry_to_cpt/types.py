from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DerivedCode:
    code: str
    rationale: str
    rule_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class RegistryCPTDerivation:
    codes: list[DerivedCode] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

