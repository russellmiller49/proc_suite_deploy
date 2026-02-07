"""Helpers for enforcing PHI review gating before coding.

Coding must use scrubbed text from ProcedureData and, when configured,
only allow coding after PHI review is completed.
"""

from __future__ import annotations

import os
import uuid

from sqlalchemy.orm import Session

from app.phi.models import ProcedureData, ProcessingStatus


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def is_phi_review_required() -> bool:
    """Check CODER_REQUIRE_PHI_REVIEW flag."""

    return _env_bool("CODER_REQUIRE_PHI_REVIEW", False)


def load_procedure_for_coding(
    db: Session,
    procedure_id: uuid.UUID,
    require_review: bool | None = None,
) -> ProcedureData | None:
    """Load a ProcedureData for coding, enforcing review when required.

    Returns:
        ProcedureData when available and allowed, or None when review is not
        required and the record is missing (legacy direct-text path).

    Raises:
        PermissionError: when review is required but status is not PHI_REVIEWED.
        ValueError: when the procedure is missing or lacks scrubbed text while
                    review is required.
    """

    if require_review is None:
        require_review = is_phi_review_required()

    proc = db.get(ProcedureData, procedure_id)
    if proc is None:
        if require_review:
            raise ValueError("Procedure not found")
        return None

    if require_review and proc.status != ProcessingStatus.PHI_REVIEWED:
        raise PermissionError("Procedure is not eligible for coding; PHI review required")

    if not proc.scrubbed_text:
        if require_review:
            raise ValueError("Procedure missing scrubbed text for coding")
        # Allow legacy path to fall back to caller-provided text
        return None

    return proc


__all__ = ["load_procedure_for_coding", "is_phi_review_required"]
