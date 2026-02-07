from __future__ import annotations

from typing import Any, Sequence

from app.common.sectionizer import Section
from app.registry.slots.base import SlotResult


class NoOpLLMExtractor:
    """LLM extractor drop-in that performs no extraction (deterministic-only mode)."""

    slot_name = "llm_disabled"

    def extract(self, _text: str, _sections: Sequence[Section], **_kwargs: Any) -> SlotResult:
        return SlotResult(value={}, evidence=[], confidence=0.0)


__all__ = ["NoOpLLMExtractor"]

