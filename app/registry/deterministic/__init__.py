"""Deterministic extractors that provide anchors for LLM + rules.

These helpers are intentionally lightweight and avoid schema-specific coupling:
they return small, serializable "anchors" that can be injected into prompts or
used for post-processing when the LLM output is missing obvious details.
"""

from __future__ import annotations

from app.registry.deterministic.anatomy import (
    extract_deterministic_anatomy,
    extract_volume_anchors,
)

__all__ = [
    "extract_deterministic_anatomy",
    "extract_volume_anchors",
]

