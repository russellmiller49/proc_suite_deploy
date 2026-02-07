"""Knowledge Base domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProcedureInfo:
    """Information about a procedure from the knowledge base."""

    code: str
    description: str
    category: str
    work_rvu: float
    facility_pe_rvu: float
    malpractice_rvu: float
    total_facility_rvu: float
    is_addon: bool
    parent_codes: list[str]
    bundled_with: list[str]
    mer_group: str | None
    modifiers: list[str]
    notes: str | None
    raw_data: dict[str, Any]


@dataclass(frozen=True)
class NCCIPair:
    """An NCCI edit pair."""

    primary: str
    secondary: str
    modifier_allowed: bool
    reason: str


@dataclass(frozen=True)
class MERGroup:
    """A Mutually Exclusive Rule group."""

    group_id: str
    codes: list[str]
    description: str
