"""MER (Mutually Exclusive Rules) for procedure coding.

These rules handle cases where certain procedures cannot be coded together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain.knowledge_base.repository import KnowledgeBaseRepository


@dataclass(frozen=True)
class MERConflict:
    """A detected MER conflict."""

    codes: list[str]
    group_id: str
    resolution: str  # Which code was kept
    reason: str


@dataclass(frozen=True)
class MERResult:
    """Result of applying MER rules to a code set."""

    kept_codes: list[str]
    removed_codes: list[str]
    conflicts: list[MERConflict]
    warnings: list[str]


def apply_mer_rules(
    codes: Sequence[str],
    kb_repo: KnowledgeBaseRepository,
) -> MERResult:
    """Apply MER rules to a set of codes.

    When mutually exclusive codes are present, typically the higher-value
    code is kept and others in the same MER group are removed.

    Args:
        codes: List of CPT codes to check.
        kb_repo: Knowledge base repository for MER group lookups.

    Returns:
        MERResult with kept codes, removed codes, and conflict details.
    """
    kept = list(codes)
    removed: list[str] = []
    conflicts: list[MERConflict] = []
    warnings: list[str] = []

    # Group codes by their MER group
    mer_groups: dict[str, list[str]] = {}
    for code in codes:
        group = kb_repo.get_mer_group(code)
        if group:
            mer_groups.setdefault(group, []).append(code)

    # For each MER group with multiple codes, keep only one
    for group_id, group_codes in mer_groups.items():
        if len(group_codes) <= 1:
            continue

        # Sort by RVU value (highest first) to keep the most valuable
        def get_rvu(code: str) -> float:
            info = kb_repo.get_procedure_info(code)
            return info.total_facility_rvu if info else 0.0

        sorted_codes = sorted(group_codes, key=get_rvu, reverse=True)
        kept_code = sorted_codes[0]
        codes_to_remove = sorted_codes[1:]

        for code in codes_to_remove:
            if code in kept:
                kept.remove(code)
                removed.append(code)

        conflict = MERConflict(
            codes=group_codes,
            group_id=group_id,
            resolution=kept_code,
            reason=f"MER group {group_id}: kept {kept_code} (highest RVU)",
        )
        conflicts.append(conflict)
        warnings.append(
            f"MER conflict in group {group_id}: kept {kept_code}, "
            f"removed {', '.join(codes_to_remove)}"
        )

    return MERResult(
        kept_codes=kept,
        removed_codes=removed,
        conflicts=conflicts,
        warnings=warnings,
    )
