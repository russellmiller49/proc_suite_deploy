"""NCCI (National Correct Coding Initiative) edit rules.

These are pure functional rules that determine bundling relationships
between CPT codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain.knowledge_base.repository import KnowledgeBaseRepository


@dataclass(frozen=True)
class NCCIEdit:
    """Result of an NCCI edit check."""

    primary: str
    secondary: str
    action: str  # "deny", "allow_with_modifier", "allow"
    modifier_allowed: bool
    reason: str


@dataclass(frozen=True)
class NCCIResult:
    """Result of applying NCCI edits to a code set."""

    kept_codes: list[str]
    removed_codes: list[str]
    edits_applied: list[NCCIEdit]
    warnings: list[str]


def apply_ncci_edits(
    codes: Sequence[str],
    kb_repo: KnowledgeBaseRepository,
) -> NCCIResult:
    """Apply NCCI bundling edits to a set of codes.

    Args:
        codes: List of CPT codes to check.
        kb_repo: Knowledge base repository for NCCI pair lookups.

    Returns:
        NCCIResult with kept codes, removed codes, and edit details.
    """
    kept = list(codes)
    removed: list[str] = []
    edits: list[NCCIEdit] = []
    warnings: list[str] = []

    # Check each pair of codes
    for primary in codes:
        ncci_pairs = kb_repo.get_ncci_pairs(primary)
        for pair in ncci_pairs:
            if pair.secondary in kept and pair.secondary != primary:
                if pair.modifier_allowed:
                    edit = NCCIEdit(
                        primary=pair.primary,
                        secondary=pair.secondary,
                        action="allow_with_modifier",
                        modifier_allowed=True,
                        reason=pair.reason or "NCCI allows with modifier",
                    )
                    edits.append(edit)
                    warnings.append(
                        f"NCCI edit: {pair.secondary} bundled with {pair.primary}; "
                        "modifier may allow separate payment"
                    )
                else:
                    # Remove the secondary code
                    if pair.secondary in kept:
                        kept.remove(pair.secondary)
                        removed.append(pair.secondary)
                    edit = NCCIEdit(
                        primary=pair.primary,
                        secondary=pair.secondary,
                        action="deny",
                        modifier_allowed=False,
                        reason=pair.reason or "NCCI bundling applies",
                    )
                    edits.append(edit)

    return NCCIResult(
        kept_codes=kept,
        removed_codes=removed,
        edits_applied=edits,
        warnings=warnings,
    )
