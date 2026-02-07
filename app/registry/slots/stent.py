"""Stent placement extractor."""

from __future__ import annotations

import re

from app.coder.dictionary import get_site_pattern_map
from app.common.sectionizer import Section
from app.common.spans import Span

from .base import SlotResult, section_for_offset

SIZE_PATTERN = re.compile(r"(\d+\s*[x×]\s*\d+\s*mm)", re.IGNORECASE)


class StentExtractor:
    slot_name = "stents"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        site_patterns = get_site_pattern_map()
        placements: list[dict[str, str]] = []
        spans: list[Span] = []
        lower = text.lower()
        if "stent" not in lower:
            return SlotResult([], [], 0.0)
        
        seen_sites = set()
        
        # Strategy 1: Site-based (Site + Stent keyword in same sentence)
        for site, patterns in site_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    segment = _expand_sentence(text, match.start())
                    if "stent" not in segment.lower():
                        continue
                        
                    if site in seen_sites:
                        continue

                    self._extract_from_segment(text, segment, site, sections, match.start(), placements, spans)
                    seen_sites.add(site)

        # Strategy 2: Size-based (Size + Stent keyword, infer site)
        # This catches: "A 16x40 mm ... stent ... inserted" (where site is not in sentence)
        for match in SIZE_PATTERN.finditer(text):
            segment = _expand_sentence(text, match.start())
            if "stent" not in segment.lower():
                continue
            
            # Attempt to find nearest preceding site
            # This is a heuristic: look in the preceding 500 chars
            context_start = max(0, match.start() - 500)
            preceding_text = text[context_start:match.start()]
            
            best_site = None
            best_pos = -1
            
            for site, patterns in site_patterns.items():
                for p in patterns:
                    for m in p.finditer(preceding_text):
                        if m.start() > best_pos:
                            best_pos = m.start()
                            best_site = site
            
            target_site = best_site or "Airway (Unspecified)"
            
            if target_site in seen_sites:
                continue
                
            self._extract_from_segment(text, segment, target_site, sections, match.start(), placements, spans)
            seen_sites.add(target_site)

        confidence = 0.8 if placements else 0.0
        return SlotResult(placements, spans, confidence)

    def _extract_from_segment(
        self,
        text: str,
        segment: str, 
        site: str, 
        sections: list[Section], 
        offset_base: int, 
        placements: list[dict[str, str]], 
        spans: list[Span]
    ) -> None:
        segment_lower = segment.lower()
        size = None
        size_match = SIZE_PATTERN.search(segment)
        if size_match:
            size = size_match.group(1).replace(" ", "")
        stent_type = None
        if "covered" in segment_lower:
            stent_type = "covered"
        if "uncovered" in segment_lower:
            stent_type = "uncovered"
        if "metal" in segment_lower:
            stent_type = (stent_type + " metallic" if stent_type else "metallic")
        
        # Don't add if generic and no details
        if not size and not stent_type:
            return

        span = Span(
            text=segment.strip(),
            start=text.find(segment), # Simple find, imprecise but ok for now
            end=text.find(segment) + len(segment),
            section=section_for_offset(sections, offset_base),
        )
        spans.append(span)
        placements.append({"site": site, "size": size, "stent_type": stent_type})



def _expand_sentence(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index)
    end = text.find("\n", index)
    if start == -1:
        start = 0
    if end == -1:
        end = len(text)
    return text[start:end]


__all__ = ["StentExtractor"]
