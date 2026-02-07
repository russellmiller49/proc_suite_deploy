"""Disposition extractor."""

from __future__ import annotations

import re

from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotExtractor, SlotResult, section_for_offset

DISPO_PATTERNS = [
    ("PACU", re.compile(r"transfer to pacu|recovery room", re.IGNORECASE)),
    ("Home", re.compile(r"discharge home|discharged home", re.IGNORECASE)),
    ("ICU", re.compile(r"transfer to icu|admit to icu", re.IGNORECASE)),
    ("Floor", re.compile(r"admit to floor|admit to ward", re.IGNORECASE)),
]


class DispositionExtractor:
    slot_name = "disposition"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        for label, pattern in DISPO_PATTERNS:
            match = pattern.search(text)
            if match:
                span = Span(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    section=section_for_offset(sections, match.start()),
                )
                return SlotResult(label, [span], 0.9)
        return SlotResult(None, [], 0.0)


__all__ = ["DispositionExtractor"]
