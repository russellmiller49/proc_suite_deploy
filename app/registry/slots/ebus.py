"""EBUS-related slot extraction."""

from __future__ import annotations

import re

from app.coder.dictionary import get_station_pattern_map
from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotResult, section_for_offset

NAVIGATION_PATTERNS = (
    re.compile(r"electromagnetic navigation", re.IGNORECASE),
    re.compile(r"EMN", re.IGNORECASE),
    re.compile(r"navigation bronchoscopy", re.IGNORECASE),
)

RADIAL_PATTERNS = (
    re.compile(r"radial (?:ebus|ultrasound)", re.IGNORECASE),
    re.compile(r"radial probe", re.IGNORECASE),
)


class EbusExtractor:
    slot_name = "ebus"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        station_patterns = get_station_pattern_map()
        evidence: list[Span] = []
        stations: list[str] = []

        for station, patterns in station_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    stations.append(station)
                    evidence.append(
                        Span(
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            section=section_for_offset(sections, match.start()),
                        )
                    )

        navigation = _flag(text, NAVIGATION_PATTERNS, sections, evidence)
        radial = _flag(text, RADIAL_PATTERNS, sections, evidence)

        value = {
            "stations": _preserve_order(stations),
            "navigation": navigation,
            "radial": radial,
        }

        confidence = 0.0
        if value["stations"]:
            confidence = 0.8
        elif navigation or radial:
            confidence = 0.6

        return SlotResult(value, evidence, confidence)


def _flag(text: str, patterns: tuple[re.Pattern[str], ...], sections: list[Section], evidence: list[Span]) -> bool:
    found = False
    for pattern in patterns:
        for match in pattern.finditer(text):
            found = True
            evidence.append(
                Span(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    section=section_for_offset(sections, match.start()),
                )
            )
    return found


def _preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


__all__ = ["EbusExtractor"]
