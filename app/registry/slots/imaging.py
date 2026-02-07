"""Imaging archive compliance extractor."""

from __future__ import annotations

import re

from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotResult, section_for_offset

ARCHIVE_PATTERN = re.compile(r"images? (?:were )?(?:archived|saved)", re.IGNORECASE)


class ImagingExtractor:
    slot_name = "imaging_archived"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        match = ARCHIVE_PATTERN.search(text)
        if not match:
            return SlotResult(False, [], 0.0)
        span = Span(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
            section=section_for_offset(sections, match.start()),
        )
        return SlotResult(True, [span], 0.8)


__all__ = ["ImagingExtractor"]

