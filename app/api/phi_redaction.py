"""Unified PHI redaction at API entry point.

This module provides a helper function for applying PHI redaction once
at the beginning of the extraction pipeline, rather than at multiple
points throughout the codebase.

Usage:
    from app.api.phi_redaction import apply_phi_redaction
    from app.api.phi_dependencies import get_phi_scrubber

    @app.post("/endpoint")
    async def handler(
        req: Request,
        phi_scrubber = Depends(get_phi_scrubber),
    ):
        redaction = apply_phi_redaction(req.note, phi_scrubber)
        # Use redaction.text for all downstream processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.phi.ports import PHIScrubberPort

logger = logging.getLogger(__name__)


@dataclass
class RedactionResult:
    """Result of PHI redaction at pipeline entry.

    Attributes:
        text: The (possibly scrubbed) text to use for downstream processing.
        was_scrubbed: True if Presidio scrubbing was actually applied.
        entity_count: Number of PHI entities detected and redacted.
        warning: Warning message if scrubbing was skipped or failed.
    """

    text: str
    was_scrubbed: bool
    entity_count: int
    warning: str | None


def apply_phi_redaction(
    raw_text: str,
    scrubber: PHIScrubberPort | None,
    *,
    already_scrubbed: bool = False,
) -> RedactionResult:
    """Apply PHI redaction at pipeline entry.

    This function should be called once at the start of each API route handler
    that accepts procedure note text. The returned text should be used for all
    downstream processing (registry extraction, CPT coding, LLM calls, etc.).

    Args:
        raw_text: The input text (may be raw or pre-scrubbed from vault).
        scrubber: PHI scrubber instance, or None if unavailable.
        already_scrubbed: If True, skip redaction (text came from PHI vault).

    Returns:
        RedactionResult with scrubbed text and metadata.

    Behavior:
        - If already_scrubbed=True: Returns text unchanged (from vault).
        - If scrubber is None: Logs warning, returns text unchanged.
        - If scrubber fails: Logs warning, returns text unchanged.
        - Otherwise: Returns scrubbed text with entity count.
    """
    if already_scrubbed:
        logger.debug("PHI redaction skipped: text already scrubbed from vault")
        return RedactionResult(
            text=raw_text,
            was_scrubbed=False,
            entity_count=0,
            warning=None,
        )

    if scrubber is None:
        logger.warning("PHI scrubber not configured; proceeding with raw text")
        return RedactionResult(
            text=raw_text,
            was_scrubbed=False,
            entity_count=0,
            warning="PHI scrubber not configured",
        )

    try:
        result = scrubber.scrub(raw_text)
        entity_count = len(result.entities)
        logger.debug(
            "PHI scrubbed at pipeline entry",
            extra={"entity_count": entity_count},
        )
        return RedactionResult(
            text=result.scrubbed_text,
            was_scrubbed=True,
            entity_count=entity_count,
            warning=None,
        )
    except Exception as e:
        logger.warning(
            "PHI scrubbing failed; proceeding with raw text",
            exc_info=True,
            extra={"error": str(e)},
        )
        return RedactionResult(
            text=raw_text,
            was_scrubbed=False,
            entity_count=0,
            warning=f"PHI scrubbing failed: {e}",
        )


__all__ = ["RedactionResult", "apply_phi_redaction"]
