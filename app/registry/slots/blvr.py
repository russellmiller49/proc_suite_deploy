"""BLVR slot extraction."""

from __future__ import annotations

import re

from app.coder.dictionary import get_lobe_pattern_map
from app.common.sectionizer import Section
from app.common.spans import Span

from ..schema import BLVRData
from .base import SlotResult, section_for_offset

VALVE_PATTERN = re.compile(r"valve", re.IGNORECASE)
VALVE_COUNT_RE = re.compile(r"(\d+)\s+\w*\s*valve", re.IGNORECASE)
CHARTIS_PATTERN = re.compile(r"(RUL|RML|RLL|LUL|LLL)[^\n]*CV\s*([+-])", re.IGNORECASE)
MANUFACTURERS = {
    "Zephyr": re.compile(r"zephyr", re.IGNORECASE),
    "Spiration": re.compile(r"spiration", re.IGNORECASE),
}


class BLVRExtractor:
    slot_name = "blvr"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        lobe_patterns = get_lobe_pattern_map()
        lower = text.lower()
        if "valve" not in lower and "chartis" not in lower:
            return SlotResult(None, [], 0.0)

        lobes: set[str] = set()
        valve_details: dict[str, int] = {}
        evidence: list[Span] = []
        for lobe, patterns in lobe_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    lobes.add(lobe)
                    segment = _expand_sentence(text, match.start())
                    if "valve" in segment.lower():
                        counts = [int(group) for group in VALVE_COUNT_RE.findall(segment)]
                        valve_details[lobe] = valve_details.get(lobe, 0) + (sum(counts) or 1)
                    evidence.append(
                        Span(
                            text=segment.strip(),
                            start=text.find(segment),
                            end=text.find(segment) + len(segment),
                            section=section_for_offset(sections, match.start()),
                        )
                    )

        chartis_map: dict[str, str] = {}
        for match in CHARTIS_PATTERN.finditer(text):
            lobe = match.group(1).upper()
            status = "negative" if match.group(2) == "-" else "positive"
            chartis_map[lobe] = status
            evidence.append(
                Span(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    section=section_for_offset(sections, match.start()),
                )
            )

        manufacturer = None
        for label, pattern in MANUFACTURERS.items():
            m = pattern.search(text)
            if m:
                manufacturer = label
                evidence.append(
                    Span(
                        text=m.group(0),
                        start=m.start(),
                        end=m.end(),
                        section=section_for_offset(sections, m.start()),
                    )
                )
                break

        if not lobes and not valve_details and not chartis_map:
            return SlotResult(None, [], 0.0)

        total_valves = sum(valve_details.values()) if valve_details else None
        value = BLVRData(
            lobes=sorted(lobes) or ["Unknown"],
            valve_count=total_valves,
            valve_details=[{"lobe": lobe, "count": count} for lobe, count in valve_details.items()],
            manufacturer=manufacturer,
            chartis=chartis_map,
        )
        return SlotResult(value, evidence, 0.75)


def _expand_sentence(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index)
    end = text.find("\n", index)
    if start == -1:
        start = 0
    if end == -1:
        end = len(text)
    return text[start:end]


__all__ = ["BLVRExtractor"]
