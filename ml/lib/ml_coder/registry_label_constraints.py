"""Deterministic label constraints for registry multi-label supervision.

This module is a single source of truth for any label normalization rules used
to reduce contradictory supervision and training/serving skew.

Design:
- Pure-Python and dependency-light (no pandas/torch), so it can be imported by
  data prep, merge scripts, and any label hydration layer.
- Operates on dict-like row objects (e.g., CSV rows, hydrated label dicts).
"""

from __future__ import annotations

import re
from typing import Any, MutableMapping

BAL_FIELD = "bal"
BRONCHIAL_WASH_FIELD = "bronchial_wash"
CRYO_FIELD = "transbronchial_cryobiopsy"
TBB_FIELD = "transbronchial_biopsy"

RIGID_FIELD = "rigid_bronchoscopy"
DEBULKING_FIELD = "tumor_debulking_non_thermal"

_RE_BAL = re.compile(r"\b(bal|bronchoalveolar\s+lavage)\b", flags=re.IGNORECASE)
_RE_BRONCHIAL_WASH = re.compile(r"\bbronchial\s+wash(?:ing)?s?\b", flags=re.IGNORECASE)


def _as_int(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        return 0


def _has_bal_wording(text: str) -> bool:
    return bool(_RE_BAL.search(text or ""))


def _has_bronchial_wash_wording(text: str) -> bool:
    return bool(_RE_BRONCHIAL_WASH.search(text or ""))


def apply_label_constraints(
    row: MutableMapping[str, Any],
    *,
    note_text: str | None = None,
    inplace: bool = True,
) -> dict[str, Any]:
    """Apply deterministic label constraints to a row-like mapping.

    Constraints:
    - BAL vs bronchial_wash:
        Default: if BAL=1, set bronchial_wash=0.
        Conservative exception: if note text contains explicit bronchial wash
        wording and contains no BAL wording, treat it as wash (set BAL=0).
    - transbronchial_cryobiopsy implies transbronchial_biopsy:
        If CRYO=1, force TBB=1.

    Args:
        row: Mapping containing label fields (and optionally note_text).
        note_text: Optional explicit note text; falls back to row["note_text"].
        inplace: If True (default), mutate the input mapping; otherwise operate
                 on a shallow copy.

    Returns:
        The normalized row dict (input mapping if inplace=True).
    """
    out: dict[str, Any] = row if inplace else dict(row)
    text = note_text if note_text is not None else str(out.get("note_text") or "")

    # --- BAL vs bronchial_wash ---
    bal = _as_int(out.get(BAL_FIELD, 0))
    wash = _as_int(out.get(BRONCHIAL_WASH_FIELD, 0))
    if bal == 1 and wash == 1:
        if _has_bronchial_wash_wording(text) and not _has_bal_wording(text):
            # Explicitly wash-only text → treat BAL as a false positive.
            out[BAL_FIELD] = 0
            out[BRONCHIAL_WASH_FIELD] = 1
        else:
            # Default: BAL dominates to avoid contradictory supervision.
            out[BAL_FIELD] = 1
            out[BRONCHIAL_WASH_FIELD] = 0

    # --- Cryo implies TBB ---
    cryo = _as_int(out.get(CRYO_FIELD, 0))
    if cryo == 1:
        out[CRYO_FIELD] = 1
        out[TBB_FIELD] = 1

    # --- Tumor debulking implies rigid bronchoscopy ---
    # In the registry schema, non-thermal tumor debulking is expected to be a
    # rigid CAO intervention in nearly all cases; keeping these aligned reduces
    # contradictory supervision (and matches review guidance).
    debulk = _as_int(out.get(DEBULKING_FIELD, 0))
    rigid = _as_int(out.get(RIGID_FIELD, 0))
    if debulk == 1 and rigid == 0:
        out[DEBULKING_FIELD] = 1
        out[RIGID_FIELD] = 1

    return out


def registry_consistency_flags(row: MutableMapping[str, Any]) -> dict[str, bool]:
    """Lightweight QA flags for known semantic overlaps (no label forcing)."""
    rigid = _as_int(row.get(RIGID_FIELD, 0)) == 1
    debulk = _as_int(row.get(DEBULKING_FIELD, 0)) == 1
    return {
        "rigid_and_debulking": rigid and debulk,
        "rigid_without_debulking": rigid and not debulk,
        "debulking_without_rigid": debulk and not rigid,
    }
