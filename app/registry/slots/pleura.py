"""Pleural procedure extractor."""

from __future__ import annotations

import re

from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotResult, section_for_offset

PLEURAL_MAP = {
    "Thoracentesis": re.compile(r"thoracentesis", re.IGNORECASE),
    "Chest Tube": re.compile(r"chest tube", re.IGNORECASE),
    "Pleurodesis": re.compile(r"pleurodesis", re.IGNORECASE),
}


class PleuraExtractor:
    slot_name = "pleural_procedures"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        values: list[str] = []
        evidence: list[Span] = []
        for label, pattern in PLEURAL_MAP.items():
            for match in pattern.finditer(text):
                values.append(label)
                evidence.append(
                    Span(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        section=section_for_offset(sections, match.start()),
                    )
                )
        unique = sorted(dict.fromkeys(values))
        return SlotResult(unique, evidence, 0.7 if unique else 0.0)


__all__ = ["PleuraExtractor"]

