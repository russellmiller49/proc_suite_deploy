from __future__ import annotations

from dataclasses import dataclass

from proc_schemas.clinical import ProcedureBundle


@dataclass
class NormalizationNote:
    kind: str
    path: str | None
    message: str
    source: str | None = None


@dataclass
class NormalizationResult:
    bundle: ProcedureBundle
    notes: list[NormalizationNote]

