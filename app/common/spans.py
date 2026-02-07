"""Span helpers shared across coder and registry app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["Span", "context_window", "dedupe_spans"]


@dataclass(slots=True)
class Span:
    """Represents a text span captured from the raw procedure note."""

    text: str
    start: int
    end: int
    section: str | None = None
    confidence: float | None = None

    def context(self, document: str, window: int = 40) -> str:
        """Return a display-friendly context window around the span."""

        return context_window(document, self.start, self.end, window)


def context_window(text: str, start: int, end: int, window: int = 40) -> str:
    """Return a context window from *text* with *window* characters of padding."""

    start_index = max(0, start - window)
    end_index = min(len(text), end + window)
    snippet = text[start_index:end_index]
    return snippet.strip()


def dedupe_spans(spans: Sequence[Span]) -> list[Span]:
    """Remove duplicate spans (by offsets) while preserving original order."""

    seen: set[tuple[int, int]] = set()
    unique: list[Span] = []
    for span in spans:
        key = (span.start, span.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)
    return unique
