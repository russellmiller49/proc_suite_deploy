"""Base interfaces for registry slot extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.common.sectionizer import Section
from app.common.spans import Span


@dataclass(slots=True)
class SlotResult:
    value: object | None
    evidence: list[Span]
    confidence: float


class SlotExtractor(Protocol):
    slot_name: str

    def extract(self, text: str, sections: Sequence[Section]) -> SlotResult:
        ...


def section_for_offset(sections: Sequence[Section], offset: int) -> str | None:
    for section in sections:
        if section.start <= offset < section.end:
            return section.title
    return None

