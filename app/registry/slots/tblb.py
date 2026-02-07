"""Transbronchial biopsy slot extractor."""

from __future__ import annotations

from app.coder.dictionary import get_lobe_pattern_map
from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotResult, section_for_offset


class TBLBExtractor:
    slot_name = "tblb_lobes"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        lobe_patterns = get_lobe_pattern_map()
        lobes: list[str] = []
        spans: list[Span] = []
        for lobe, patterns in lobe_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    segment = _expand_sentence(text, match.start())
                    lower_segment = segment.lower()
                    if (
                        "biopsy" not in lower_segment
                        and "biopsies" not in lower_segment
                        and "tblb" not in lower_segment
                    ):
                        continue
                    lobes.append(lobe)
                    spans.append(
                        Span(
                            text=segment.strip(),
                            start=text.find(segment),
                            end=text.find(segment) + len(segment),
                            section=section_for_offset(sections, match.start()),
                        )
                    )
        unique = sorted(dict.fromkeys(lobes))
        confidence = 0.0 if not unique else 0.8
        return SlotResult(unique, spans, confidence)


def _expand_sentence(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index)
    end = text.find("\n", index)
    if start == -1:
        start = 0
    if end == -1:
        end = len(text)
    return text[start:end]


__all__ = ["TBLBExtractor"]
