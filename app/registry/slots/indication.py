"""Extract procedure indication from the note."""

from __future__ import annotations

import re

from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotExtractor, SlotResult, section_for_offset

INDICATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Pulmonary Nodule", re.compile(r"(pulmonary|lung) nodule", re.IGNORECASE)),
    ("Mass", re.compile(r"mass|tumor", re.IGNORECASE)),
    ("Lymphadenopathy", re.compile(r"lymphadenopathy|adenopathy", re.IGNORECASE)),
    ("Pleural Effusion", re.compile(r"pleural effusion", re.IGNORECASE)),
    ("Airway Stenosis", re.compile(r"stenosis|stricture|obstruction", re.IGNORECASE)),
]


class IndicationExtractor:
    slot_name = "indication"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        for label, pattern in INDICATION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            span = Span(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                section=section_for_offset(sections, match.start()),
            )
            return SlotResult(label, [span], 0.8)
        return SlotResult(None, [], 0.0)


__all__ = ["IndicationExtractor"]

