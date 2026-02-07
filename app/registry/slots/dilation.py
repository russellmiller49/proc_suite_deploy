"""Airway dilation slot extractor."""

from __future__ import annotations

from app.coder.dictionary import get_site_pattern_map
from app.common import knowledge
from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotResult, section_for_offset


class DilationExtractor:
    slot_name = "dilation_sites"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        site_patterns = get_site_pattern_map()
        if "dilation" not in text.lower() and "dilatation" not in text.lower():
            return SlotResult([], [], 0.0)
        sites: list[str] = []
        spans: list[Span] = []
        for site, patterns in site_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    segment = _expand_sentence(text, match.start())
                    lower_segment = segment.lower()
                    if "dilation" not in lower_segment and "dilatation" not in lower_segment:
                        continue
                    start = text.find(segment)
                    spans.append(
                        Span(
                            text=segment.strip(),
                            start=start,
                            end=start + len(segment),
                            section=section_for_offset(sections, match.start()),
                        )
                    )
                    sites.append(site)
        preferred_lobes = set(knowledge.lobe_aliases().keys())
        ordered_unique = list(dict.fromkeys(sites))
        filtered = [site for site in ordered_unique if site in preferred_lobes]
        unique = filtered or [site for site in ordered_unique if site != "LOBAR"]
        return SlotResult(unique, spans, 0.7 if unique else 0.0)


def _expand_sentence(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index)
    end = text.find("\n", index)
    if start == -1:
        start = 0
    if end == -1:
        end = len(text)
    return text[start:end]


__all__ = ["DilationExtractor"]
