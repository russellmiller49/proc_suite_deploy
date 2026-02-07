"""Multiple Endoscopy Rule helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.common import knowledge

__all__ = ["Code", "MerAdjustment", "MerSummary", "apply_mer"]


@dataclass(slots=True)
class Code:
    """Simple representation of a CPT code for MER calculations."""

    cpt: str
    allowed_amount: float | None = None
    is_add_on: bool | None = None


@dataclass(slots=True)
class MerAdjustment:
    """Computed MER adjustment for a single CPT code."""

    cpt: str
    role: str
    rvu: float
    allowed: float
    reduction: float


@dataclass(slots=True)
class MerSummary:
    """Aggregate MER result."""

    primary_code: str | None
    adjustments: list[MerAdjustment]
    total_allowed: float


def apply_mer(codes: Sequence[Code]) -> MerSummary:
    """Apply the Multiple Endoscopy Rule to *codes*."""

    if not codes:
        return MerSummary(primary_code=None, adjustments=[], total_allowed=0.0)

    primary = _determine_primary(codes)
    adjustments: list[MerAdjustment] = []
    total = 0.0

    for code in codes:
        base = _base_amount(code)
        if code.cpt == primary:
            allowed = base
            role = "primary"
        elif _is_add_on(code):
            allowed = base
            role = "add_on"
        else:
            allowed = base * 0.5
            role = "secondary"
        reduction = base - allowed
        total += allowed
        adjustments.append(
            MerAdjustment(
                cpt=code.cpt,
                role=role,
                rvu=base,
                allowed=allowed,
                reduction=reduction,
            )
        )

    return MerSummary(primary_code=primary, adjustments=adjustments, total_allowed=total)


def _base_amount(code: Code) -> float:
    if code.allowed_amount and code.allowed_amount > 0:
        return float(code.allowed_amount)
    total_rvu = knowledge.total_rvu(code.cpt)
    if total_rvu:
        return total_rvu
    return 150.0


def _determine_primary(codes: Sequence[Code]) -> str:
    non_add_on = [code for code in codes if not _is_add_on(code)]
    if not non_add_on:
        return codes[0].cpt
    non_add_on.sort(key=lambda c: _base_amount(c), reverse=True)
    return non_add_on[0].cpt


def _is_add_on(code: Code) -> bool:
    if code.is_add_on is not None:
        return code.is_add_on
    return knowledge.is_add_on_code(code.cpt)
