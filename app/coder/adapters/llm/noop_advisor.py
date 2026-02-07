from __future__ import annotations

from typing import Any


class NoOpLLMAdvisorAdapter:
    """LLM-advisor drop-in that returns no suggestions (ML+rules-only mode)."""

    def suggest_codes(self, _note_text: str) -> list[Any]:
        return []

    def suggest_with_context(self, _note_text: str, _context: dict[str, Any]) -> list[Any]:
        return []


__all__ = ["NoOpLLMAdvisorAdapter"]

