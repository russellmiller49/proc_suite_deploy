"""Registry completeness prompting helpers.

These helpers generate suggested "missing data" prompts that the UI can use to
nudge users to document research/quality-critical fields in the source note.
"""

from __future__ import annotations

from app.registry.completeness.missing_field_prompts import (
    MissingFieldPrompt,
    generate_missing_field_prompts,
)

__all__ = [
    "MissingFieldPrompt",
    "generate_missing_field_prompts",
]

