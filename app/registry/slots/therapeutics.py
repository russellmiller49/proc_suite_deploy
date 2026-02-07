"""Therapeutic procedure extractors (Destruction, Dilation, Aspiration)."""

from __future__ import annotations

import re
from typing import Any

from app.coder.dictionary import get_site_pattern_map
from app.common.sectionizer import Section
from app.common.spans import Span
from app.registry.slots.base import SlotResult, section_for_offset
from app.registry.schema import DestructionEvent, EnhancedDilationEvent, AspirationEvent

# Patterns
APC_RE = re.compile(r"argon plasma|APC|electrocautery", re.IGNORECASE)
CRYO_RE = re.compile(r"cryotherapy|cryoablation|cryodebridement", re.IGNORECASE)
LASER_RE = re.compile(r"laser", re.IGNORECASE)

BALLOON_SIZE_RE = re.compile(r"([\d\./]+)\s*(?:mm|cm)\s*balloon", re.IGNORECASE)
PRESSURE_RE = re.compile(r"(\d+)\s*(?:atm|atmospheres)", re.IGNORECASE)

ASPIRATION_RE = re.compile(r"therapeutic (?:aspiration|suction)|mucus plug|toilet", re.IGNORECASE)
VOLUME_RE = re.compile(r"(\d+)\s*(?:ml|cc|mL)", re.IGNORECASE)

class DestructionExtractor:
    slot_name = "destruction_events"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        site_patterns = get_site_pattern_map()
        extracted: list[tuple[int, DestructionEvent, Span]] = []
        
        if not any(x.search(text) for x in (APC_RE, CRYO_RE, LASER_RE)):
            return SlotResult([], [], 0.0)

        # Iterate through sites to find collocated destruction terms
        for site, patterns in site_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    seg_start, seg_end = _sentence_bounds(text, match.start())
                    segment = text[seg_start:seg_end]
                    
                    modality = None
                    if APC_RE.search(segment):
                        modality = "APC/Electrocautery"
                    elif CRYO_RE.search(segment):
                        # Distinguish from cryobiopsy: check for "biopsy" or "sample" nearby?
                        # For now, assume cryotherapy context if destruction patterns match
                        modality = "Cryotherapy"
                    elif LASER_RE.search(segment):
                        modality = "Laser"
                    
                    if modality:
                        span = Span(
                            text=segment.strip(),
                            start=seg_start,
                            end=seg_end,
                            section=section_for_offset(sections, match.start()),
                        )
                        extracted.append((seg_start, DestructionEvent(modality=modality, site=site), span))

        extracted.sort(key=lambda item: item[0])
        events = [item[1] for item in extracted]
        evidence = [item[2] for item in extracted]
        return SlotResult(events, evidence, 0.85 if events else 0.0)


class EnhancedDilationExtractor:
    slot_name = "dilation_events"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        site_patterns = get_site_pattern_map()
        extracted: list[tuple[int, EnhancedDilationEvent, Span]] = []

        if "dilation" not in text.lower() and "dilatation" not in text.lower():
            return SlotResult([], [], 0.0)

        for site, patterns in site_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    seg_start, seg_end = _sentence_bounds(text, match.start())
                    segment = text[seg_start:seg_end]
                    lower_seg = segment.lower()
                    if "dilation" not in lower_seg and "balloon" not in lower_seg:
                        continue
                    
                    # Extract details
                    size_m = BALLOON_SIZE_RE.search(segment)
                    size = f"{size_m.group(1)}mm" if size_m else None
                    
                    press_m = PRESSURE_RE.search(segment)
                    pressure = f"{press_m.group(1)} atm" if press_m else None
                    
                    span = Span(
                        text=segment.strip(),
                        start=seg_start,
                        end=seg_end,
                        section=section_for_offset(sections, match.start()),
                    )
                    extracted.append(
                        (
                            seg_start,
                            EnhancedDilationEvent(site=site, balloon_size=size, inflation_pressure=pressure),
                            span,
                        )
                    )

        extracted.sort(key=lambda item: item[0])
        events = [item[1] for item in extracted]
        evidence = [item[2] for item in extracted]
        return SlotResult(events, evidence, 0.8 if events else 0.0)


class AspirationExtractor:
    slot_name = "aspiration_events"

    def extract(self, text: str, sections: list[Section]) -> SlotResult:
        events: list[AspirationEvent] = []
        evidence: list[Span] = []

        for match in ASPIRATION_RE.finditer(text):
            segment = _expand_sentence(text, match.start())
            
            # Simple volume extraction in the same sentence
            vol_m = VOLUME_RE.search(segment)
            vol = f"{vol_m.group(1)} ml" if vol_m else None
            
            # Attempt site extraction or default to 'Tracheobronchial Tree'
            # This is a simplified approach; robust site linking requires dependency parsing
            site = "Tracheobronchial Tree" 
            
            span = Span(
                text=segment.strip(),
                start=match.start(),
                end=match.end() + len(segment), # Approx
                section=section_for_offset(sections, match.start())
            )
            evidence.append(span)
            events.append(AspirationEvent(site=site, volume=vol, character="Secretions"))

        return SlotResult(events, evidence, 0.7 if events else 0.0)

def _expand_sentence(text: str, index: int) -> str:
    start, end = _sentence_bounds(text, index)
    return text[start:end]


def _sentence_bounds(text: str, index: int) -> tuple[int, int]:
    # Expand to nearest sentence boundary (newline or period)
    # Search backwards
    start = max(
        text.rfind("\n", 0, index),
        text.rfind(".", 0, index)
    )
    if start == -1:
        start = 0
    else:
        start += 1 # Skip the delimiter

    # Search forwards
    end_newline = text.find("\n", index)
    end_period = text.find(".", index)
    
    if end_newline == -1: end_newline = len(text)
    if end_period == -1: end_period = len(text)
    
    end = min(end_newline, end_period)
    
    # Extend end to include the period if it was the delimiter, for readability
    if end < len(text) and text[end] == ".":
        end += 1
        
    return start, end
