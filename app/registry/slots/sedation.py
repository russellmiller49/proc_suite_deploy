"""Sedation and anesthesia extractor."""

from __future__ import annotations

import re
from datetime import time

from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotResult, section_for_offset

SEDATION_PATTERN = re.compile(r"moderate sedation|conscious sedation", re.IGNORECASE)
ANESTHESIA_PATTERN = re.compile(r"general anesthesia|mac anesthesia|monitored anesthesia care", re.IGNORECASE)
TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2})")


class SedationExtractor:
    slot_name = "sedation"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        evidence: list[Span] = []
        sedation_type = None
        anesthesia_type = None

        lower = text.lower()
        sed_match = SEDATION_PATTERN.search(text)
        if sed_match and "no moderate sedation" not in lower:
            sedation_type = "Moderate Sedation"
            evidence.append(_span_from_match(text, sed_match, sections))

        anes_match = ANESTHESIA_PATTERN.search(text)
        if anes_match:
            anesthesia_type = "GA" if "general" in anes_match.group(0).lower() else "MAC"
            evidence.append(_span_from_match(text, anes_match, sections))

        times = TIME_PATTERN.findall(text)
        sedation_start = _parse_time(times[0]) if times else None
        sedation_stop = _parse_time(times[1]) if len(times) > 1 else None

        if not (sedation_type or anesthesia_type):
            return SlotResult(None, [], 0.0)

        value = {
            "sedation_type": sedation_type,
            "anesthesia_type": anesthesia_type,
            "sedation_start": sedation_start,
            "sedation_stop": sedation_stop,
        }

        confidence = 0.75
        return SlotResult(value, evidence, confidence)


def _span_from_match(text: str, match: re.Match[str], sections: list[Section]) -> Span:
    return Span(
        text=match.group(0),
        start=match.start(),
        end=match.end(),
        section=section_for_offset(sections, match.start()),
    )


def _parse_time(value: str | None) -> time | None:
    if not value:
        return None
    hour, minute = value.split(":")
    try:
        return time(hour=int(hour), minute=int(minute))
    except ValueError:
        return None


__all__ = ["SedationExtractor"]
