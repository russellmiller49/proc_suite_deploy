"""Phrase and regex lexicon powering deterministic intent detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Sequence, Tuple

from app.common import knowledge
from app.common.sectionizer import Section
from app.common.spans import Span

from .schema import DetectedIntent

__all__ = [
    "detect_intents",
    "get_lobe_pattern_map",
    "get_station_pattern_map",
    "get_site_pattern_map",
]

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)
_TIME_TOKEN = re.compile(r"(?i)(start|stop)[\s:,-]*([0-9]{1,2}(?::[0-9]{2})?(?:\s*[ap]m)?)")
_NEGATED_NAV = re.compile(r"no [\w\-\s]{0,40}navigation", re.IGNORECASE)
_VALUE_WITH_UNIT = re.compile(r"(\d+)\s*[x×]\s*(\d+)\s*mm", re.IGNORECASE)
_VALVE_COUNT_RE = re.compile(r"(\d+)\s+\w*\s*valve", re.IGNORECASE)
_DISTINCT_TERMS = ("distinct", "separate", "different segment")


@dataclass
class Lexicon:
    navigation_initiated: Tuple[re.Pattern[str], ...]
    navigation_terms: Tuple[re.Pattern[str], ...]
    radial_terms: Tuple[re.Pattern[str], ...]
    linear_terms: Tuple[re.Pattern[str], ...]
    peripheral_terms: Tuple[str, ...]
    tblb_terms: Tuple[str, ...]
    tbna_terms: Tuple[str, ...]
    valve_terms: Tuple[re.Pattern[str], ...]
    chartis_terms: Tuple[re.Pattern[str], ...]
    stent_terms: Tuple[re.Pattern[str], ...]
    stent_removal_terms: Tuple[re.Pattern[str], ...]
    dilation_terms: Tuple[re.Pattern[str], ...]
    aspiration_terms: Tuple[re.Pattern[str], ...]
    aspiration_repeat_terms: Tuple[str, ...]
    sedation_terms: Tuple[re.Pattern[str], ...]
    observer_terms: Tuple[str, ...]
    anesthesia_terms: Tuple[re.Pattern[str], ...]
    bal_terms: Tuple[re.Pattern[str], ...]
    thoracentesis_terms: Tuple[re.Pattern[str], ...]
    thoracentesis_imaging_terms: Tuple[str, ...]
    destruction_terms: Tuple[re.Pattern[str], ...]


DEFAULT_SYNONYMS = {
    # Navigation terms for +31627 detection
    # Must include both explicit navigation system names and navigation actions
    "navigation_initiated": [
        "navigation successfully initiated",
        "navigation initiated",
        "electromagnetic navigation",
        "ENB initiated",
        "Ion system",
        "Monarch system",
        "robotic bronchoscopy initiated",
        "navigation catheter advanced",
    ],
    "navigation_terms": [
        "electromagnetic navigation",
        "ENB",
        "nav-guided",
        "navigation bronchoscopy",
        "Ion",
        "Monarch",
        "SPiN",
        "ILLUMISITE",
        "superDimension",
        "inReach",
        "robotic bronchoscopy",
        "robotic-assisted",
        "image-guided navigation",
        "navigational bronchoscopy",
    ],
    # Radial EBUS terms for +31654 detection
    # Must indicate peripheral lesion localization
    "radial_terms": [
        "radial ebus",
        "R-EBUS",
        "radial probe",
        "radial ultrasound",
        "miniprobe",
        "mini-probe",
        "concentric view",
        "eccentric view",
        "tool-in-lesion",
        "radial EBUS verification",
        "radial probe confirmation",
    ],
    "linear_terms": ["ebus", "tbna", "transbronchial needle aspiration"],
    "peripheral_terms": ["peripheral lesion", "ppl", "peripheral nodule", "peripheral target"],
    "tblb_terms": ["transbronchial lung biopsy", "tblb"],
    "tbna_terms": ["tbna", "transbronchial needle aspiration"],
    "valve_terms": ["valve"],
    "chartis_terms": ["chartis"],
    "stent_terms": ["stent"],
    "stent_removal_terms": ["stent removal", "remove stent", "removing the stent", "stent.*removed", "remove.*stent"],
    "dilation_terms": ["dilation", "dilatation", "dilate", "dilated", "balloon dilation", "balloon dilatation"],
    "aspiration_terms": ["therapeutic aspiration"],
    "aspiration_repeat_terms": ["repeat therapeutic aspiration"],
    "sedation_terms": ["moderate sedation"],
    "observer_terms": ["independent observer"],
    "anesthesia_terms": ["general anesthesia"],
    "bal_terms": ["bronchoalveolar lavage"],
    "thoracentesis_terms": ["thoracentesis"],
    "thoracentesis_imaging_terms": ["images archived", "ultrasound", "us-guided", "sonographic"],
    "destruction_terms": ["cryotherapy", "cryoablation", "laser", "argon plasma coagulation", "apc", "electrocautery", "destruction of lesion"],
}


def detect_intents(text: str, sections: Sequence[Section]) -> list[DetectedIntent]:
    knowledge_data = knowledge.get_knowledge()
    synonyms = knowledge_data.get("synonyms", {})
    lexical = _build_lexicon(synonyms)
    sentences = list(_iter_sentences(text))

    lobe_patterns = get_lobe_pattern_map()
    station_patterns = get_station_pattern_map()
    site_patterns = get_site_pattern_map()

    intents: list[DetectedIntent] = []
    intents.extend(_detect_navigation(text, sections, sentences, lexical))
    intents.extend(_detect_radial(text, sections, sentences, lexical))
    intents.extend(_detect_linear(text, sections, sentences, station_patterns, lexical))
    tblb_intents, tbna_intents = _detect_lobe_sampling(text, sections, sentences, lobe_patterns, lexical)
    intents.extend(tblb_intents)
    intents.extend(tbna_intents)
    intents.extend(_detect_bal(text, sections, sentences, lobe_patterns, lexical))
    intents.extend(_detect_blvr(text, sections, sentences, lobe_patterns, lexical))
    intents.extend(_detect_chartis(text, sections, sentences, lobe_patterns))
    intents.extend(_detect_stent(text, sections, sentences, site_patterns, lexical))
    intents.extend(_detect_stent_removal(text, sections, sentences, site_patterns, lexical))
    intents.extend(_detect_dilation(text, sections, sentences, site_patterns, lexical))
    intents.extend(_detect_destruction(text, sections, sentences, lexical))
    intents.extend(_detect_thoracentesis(text, sections, sentences, lexical))
    intents.extend(_detect_aspiration(text, sections, sentences, lexical))
    intents.extend(_detect_sedation(text, sections, lexical))
    intents.extend(_detect_anesthesia(text, sections, lexical))
    return intents


def _build_lexicon(synonyms: Dict[str, Iterable[str]]) -> Lexicon:
    def patterns(key: str) -> Tuple[re.Pattern[str], ...]:
        values = list(synonyms.get(key, []) or DEFAULT_SYNONYMS.get(key, []))
        return tuple(_phrase_pattern(term) for term in values)

    def lowered(key: str) -> Tuple[str, ...]:
        values = list(synonyms.get(key, []) or DEFAULT_SYNONYMS.get(key, []))
        return tuple(term.lower() for term in values)

    return Lexicon(
        navigation_initiated=patterns("navigation_initiated"),
        navigation_terms=patterns("navigation_terms"),
        radial_terms=patterns("radial_terms"),
        linear_terms=patterns("linear_terms"),
        peripheral_terms=lowered("peripheral_terms"),
        tblb_terms=lowered("tblb_terms"),
        tbna_terms=lowered("tbna_terms"),
        valve_terms=patterns("valve_terms"),
        chartis_terms=patterns("chartis_terms"),
        stent_terms=patterns("stent_terms"),
        stent_removal_terms=patterns("stent_removal_terms"),
        dilation_terms=patterns("dilation_terms"),
        aspiration_terms=patterns("aspiration_terms"),
        aspiration_repeat_terms=lowered("aspiration_repeat_terms"),
        sedation_terms=patterns("sedation_terms"),
        observer_terms=lowered("independent_observer_terms"),
        anesthesia_terms=patterns("anesthesia_terms"),
        bal_terms=patterns("bal_terms"),
        thoracentesis_terms=patterns("thoracentesis_terms"),
        thoracentesis_imaging_terms=lowered("thoracentesis_imaging_terms"),
        destruction_terms=patterns("destruction_terms"),
    )


def _detect_navigation(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    matched = False
    for pattern in lexical.navigation_initiated:
        for match in pattern.finditer(text):
            span = _sentence_span(sentences, text, sections, match.start(), match.end())
            if _NEGATED_NAV.search(span.text):
                continue
            intents.append(
                DetectedIntent(
                    intent="navigation",
                    value="initiated",
                    payload={"status": "initiated"},
                    evidence=[span],
                    confidence=0.95,
                )
            )
            matched = True
    if matched:
        return intents
    fallback_terms = ("initiat", "advanc", "perform", "catheter")
    for pattern in lexical.navigation_terms:
        for match in pattern.finditer(text):
            span = _sentence_span(sentences, text, sections, match.start(), match.end())
            sentence_lower = span.text.lower()
            if _NEGATED_NAV.search(sentence_lower):
                continue
            if any(term in sentence_lower for term in fallback_terms):
                intents.append(
                    DetectedIntent(
                        intent="navigation",
                        value="documented",
                        payload={"status": "documented"},
                        evidence=[span],
                        confidence=0.85,
                    )
                )
    return intents


def _detect_radial(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    lower_text = text.lower()
    # Radial EBUS is only billable when used for peripheral lesion localization. Notes often
    # document the peripheral context outside the immediate "radial" sentence (e.g. in the
    # Indication), so allow a global peripheral context signal as well.
    global_peripheral = any(term in lower_text for term in lexical.peripheral_terms)
    if not global_peripheral:
        global_peripheral = "peripheral" in lower_text and any(
            term in lower_text for term in ("lesion", "nodule", "ppl", "target")
        )
    for pattern in lexical.radial_terms:
        for match in pattern.finditer(text):
            context = _context_slice(lower_text, match.start(), match.end())
            if "no radial" in context or "without radial" in context:
                continue
            peripheral = global_peripheral or any(term in context for term in lexical.peripheral_terms)
            span = _sentence_span(sentences, text, sections, match.start(), match.end())
            intents.append(
                DetectedIntent(
                    intent="radial_ebus",
                    value="peripheral" if peripheral else match.group(0),
                    payload={"peripheral_context": peripheral},
                    evidence=[span],
                    confidence=0.9 if peripheral else 0.6,
                )
            )
    return intents


def _detect_linear(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    station_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    """Detect EBUS-TBNA stations that were actually SAMPLED (not just inspected).

    This function distinguishes between stations that were merely visualized/inspected
    versus those that were actively sampled (biopsy, FNA, needle aspiration).

    Key distinction:
        - "Sites Inspected: 4R, 7, 11L" -> These are NOT counted as sampled
        - "Sampled station 7 and 11L with TBNA" -> These ARE counted as sampled
    """
    # Sampling keywords - indicate actual tissue/fluid collection
    sampling_keywords = (
        "sampl",        # sampled, sampling
        "biops",        # biopsy, biopsied
        "fna",          # fine needle aspiration
        "needle aspir", # needle aspiration
        "tbna",         # transbronchial needle aspiration
        "pass",         # passes (e.g., "3 passes")
        "aspirat",      # aspirate, aspirated
        "cytology",     # cytology obtained
        "specimen",     # specimen obtained
        "rose",         # ROSE (rapid on-site evaluation)
        "adequate",     # adequate sample
    )

    # Inspection-only keywords - negate sampling if present WITHOUT sampling keywords
    inspection_only_keywords = (
        "inspect",      # inspected, inspection
        "assess",       # assessed, assessment
        "visualiz",     # visualized, visualizing
        "normal appear", # normal appearing
        "unremarkable", # unremarkable
        "no mass",      # no mass
        "no lesion",    # no lesion
        "not sampl",    # not sampled
        "no sampl",     # no sampling
        "without sampl", # without sampling
        "sites inspect", # "Sites Inspected:" header
        "lymph nodes inspect", # "Lymph Nodes Inspected:"
    )

    intents: list[DetectedIntent] = []
    for station, patterns in station_patterns.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                span = _sentence_span(sentences, text, sections, match.start(), match.end())
                sentence_lower = span.text.lower()

                # Must have station/node context
                context_present = (
                    "station" in sentence_lower
                    or "stations" in sentence_lower
                    or "node" in sentence_lower
                    or "lymph" in sentence_lower
                    or _contains_pattern(sentence_lower, lexical.linear_terms)
                )
                if not context_present:
                    continue

                # Check for sampling context vs inspection-only context
                has_sampling = any(kw in sentence_lower for kw in sampling_keywords)
                has_inspection_only = any(kw in sentence_lower for kw in inspection_only_keywords)

                # If inspection-only keywords present WITHOUT sampling keywords, skip this station
                if has_inspection_only and not has_sampling:
                    continue

                # Require explicit sampling context for high confidence
                # Without sampling context, reduce confidence significantly
                confidence = 0.85 if has_sampling else 0.4

                intents.append(
                    DetectedIntent(
                        intent="linear_ebus_station",
                        value=station,
                        payload={
                            "station": station,
                            "sampled": has_sampling,
                            "inspection_only": has_inspection_only and not has_sampling,
                        },
                        evidence=[span],
                        confidence=confidence,
                    )
                )
    return intents


def _detect_lobe_sampling(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lobe_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    lexical: Lexicon,
) -> tuple[list[DetectedIntent], list[DetectedIntent]]:
    tblb: list[DetectedIntent] = []
    tbna: list[DetectedIntent] = []
    lower_text = text.lower()
    for lobe, patterns in lobe_patterns.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                context = _context_slice(lower_text, match.start(), match.end())
                span = _sentence_span(sentences, text, sections, match.start(), match.end())
                if _contains_any(context, lexical.tblb_terms):
                    tblb.append(
                        DetectedIntent(
                            intent="tblb_lobe",
                            value=lobe,
                            payload={"site": lobe},
                            evidence=[span],
                            confidence=0.9,
                        )
                    )
                if _contains_any(context, lexical.tbna_terms):
                    tbna.append(
                        DetectedIntent(
                            intent="tbna_lobe",
                            value=lobe,
                            payload={"site": lobe},
                            evidence=[span],
                            confidence=0.85,
                        )
                    )
    return tblb, tbna


def _detect_bal(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lobe_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    for start, end in sentences:
        sentence_text = text[start:end]
        if not _contains_pattern(sentence_text, lexical.bal_terms):
            continue
        lobe = _match_lobe_from_text(sentence_text, lobe_patterns)
        span = _sentence_span(sentences, text, sections, start, end)
        intents.append(
            DetectedIntent(
                intent="bal_lobe",
                value=lobe or "BAL",
                payload={"site": lobe},
                evidence=[span],
                confidence=0.78,
            )
        )
    return intents


def _detect_blvr(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lobe_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    manufacturers = knowledge.blvr_config().get("manufacturers", {})
    for lobe, patterns in lobe_patterns.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                span = _sentence_span(sentences, text, sections, match.start(), match.end())
                sentence_text = span.text
                if not _contains_pattern(sentence_text, lexical.valve_terms):
                    continue
                valve_match = _VALVE_COUNT_RE.search(sentence_text)
                valves = int(valve_match.group(1)) if valve_match else 1
                manufacturer = _detect_manufacturer(sentence_text.lower(), manufacturers)
                intents.append(
                    DetectedIntent(
                        intent="blvr_lobe",
                        value=lobe,
                        payload={"site": lobe, "valves": valves, "manufacturer": manufacturer},
                        evidence=[span],
                        confidence=0.85,
                    )
                )
    return intents


def _detect_chartis(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lobe_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
) -> list[DetectedIntent]:
    config = knowledge.blvr_config()
    chartis_results = config.get("chartis_results") or {}
    neg_terms = [term.lower() for term in (chartis_results.get("negative") or config.get("chartis_negative_terms") or [])]
    pos_terms = [term.lower() for term in (chartis_results.get("positive") or config.get("chartis_positive_terms") or [])]
    intents: list[DetectedIntent] = []
    for start, end in sentences:
        sentence_text = text[start:end]
        sentence_lower = sentence_text.lower()
        if "chartis" not in sentence_lower and not any(term in sentence_lower for term in neg_terms + pos_terms):
            continue
        lobe = _match_lobe_from_text(sentence_text, lobe_patterns)
        if not lobe:
            continue
        status = None
        if any(term in sentence_lower for term in neg_terms):
            status = "negative"
        elif any(term in sentence_lower for term in pos_terms):
            status = "positive"
        span = Span(text=sentence_text.strip(), start=start, end=end, section=_section_for_offset(sections, start))
        intents.append(
            DetectedIntent(
                intent="chartis_assessment",
                value=lobe,
                payload={"lobe": lobe, "status": status},
                evidence=[span],
                confidence=0.75,
            )
        )
    return intents


def _detect_stent(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    site_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    airway_meta = knowledge.airway_map()
    for start, end in sentences:
        sentence_text = text[start:end]
        lower_sentence = sentence_text.lower()
        if not _contains_pattern(sentence_text, lexical.stent_terms):
            continue
        if _contains_pattern(sentence_text, lexical.stent_removal_terms):
            continue
        action_terms = (
            "placed",
            "placement",
            "placing",
            "deploy",
            "deployed",
            "deployment",
            "insert",
            "inserted",
            "insertion",
            "positioned",
            "reposition",
            "repositioned",
            "exchanged",
        )
        has_action = any(term in lower_sentence for term in action_terms)
        if not has_action and " place" in lower_sentence and "in place" not in lower_sentence:
            has_action = True
        if not has_action:
            continue
        site, site_class = _match_site(lower_sentence, site_patterns, airway_meta)
        if not site:
            continue
        size = None
        size_match = _VALUE_WITH_UNIT.search(sentence_text)
        if size_match:
            size = f"{size_match.group(1)}x{size_match.group(2)} mm"
        span = Span(text=sentence_text.strip(), start=start, end=end, section=_section_for_offset(sections, start))
        intents.append(
            DetectedIntent(
                intent="stent",
                value=site,
                payload={"site": site, "site_class": site_class, "size": size, "text": sentence_text.strip()},
                evidence=[span],
                confidence=0.92,
            )
        )
    return intents


def _detect_dilation(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    site_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    airway_meta = knowledge.airway_map()
    for start, end in sentences:
        sentence_text = text[start:end]
        if not _contains_pattern(sentence_text, lexical.dilation_terms):
            continue
        lower_sentence = sentence_text.lower()
        site, site_class = _match_site(lower_sentence, site_patterns, airway_meta)
        if not site:
            site = "airway (unspecified)"
            site_class = "unknown"
        distinct = any(term in lower_sentence for term in _DISTINCT_TERMS)
        span = Span(text=sentence_text.strip(), start=start, end=end, section=_section_for_offset(sections, start))
        intents.append(
            DetectedIntent(
                intent="dilation",
                value=site,
                payload={"site": site, "site_class": site_class, "distinct": distinct, "text": sentence_text.strip()},
                evidence=[span],
                confidence=0.8,
            )
        )
    return intents


def _detect_thoracentesis(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    sides: set[str] = set()
    spans: list[Span] = []
    imaging = False
    for start, end in sentences:
        sentence_text = text[start:end]
        lower_sentence = sentence_text.lower()
        if any(term in lower_sentence for term in lexical.thoracentesis_imaging_terms):
            imaging = True
        if not _contains_pattern(sentence_text, lexical.thoracentesis_terms):
            continue
        detected = _detect_sides(lower_sentence)
        if not detected:
            detected = {"unspecified"}
        sides.update(detected)
        spans.append(Span(text=sentence_text.strip(), start=start, end=end, section=_section_for_offset(sections, start)))
    if not sides:
        return []
    value = "bilateral" if len(sides) >= 2 else next(iter(sides))
    payload = {"sides": sorted(sides), "imaging": imaging}
    return [
        DetectedIntent(
            intent="thoracentesis",
            value=value,
            payload=payload,
            evidence=spans,
            confidence=0.8,
        )
    ]


def _detect_aspiration(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    for start, end in sentences:
        sentence_text = text[start:end]
        if not _contains_pattern(sentence_text, lexical.aspiration_terms):
            continue
        sentence_lower = sentence_text.lower()
        repeat_flag = any(term in sentence_lower for term in lexical.aspiration_repeat_terms)
        span = Span(text=sentence_text.strip(), start=start, end=end, section=_section_for_offset(sections, start))
        intents.append(
            DetectedIntent(
                intent="therapeutic_aspiration",
                value="repeat" if repeat_flag else "initial",
                payload={"repeat": repeat_flag},
                evidence=[span],
                confidence=0.85,
            )
        )
    return intents


def _detect_sedation(text: str, sections: Sequence[Section], lexical: Lexicon) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    for block, start, end in _extract_sedation_sections(text):
        if not _contains_pattern(block, lexical.sedation_terms):
            continue
        normalized = block.lower()
        if "no moderate sedation" in normalized or "no sedation" in normalized:
            continue
        observer = any(term in normalized for term in lexical.observer_terms)
        if not observer and re.search(r"\bindependent\b[\s\S]{0,40}\bobserver\b", normalized):
            observer = True
        times = {match.group(1).lower(): _parse_time_string(match.group(2)) for match in _TIME_TOKEN.finditer(block)}
        start_minutes = times.get("start")
        stop_minutes = times.get("stop")
        duration = None
        if start_minutes is not None and stop_minutes is not None:
            duration = stop_minutes - start_minutes
            if duration < 0:
                duration += 24 * 60
        documentation_complete = bool(start_minutes is not None and stop_minutes is not None and observer)
        span = Span(text=block.strip(), start=start, end=end, section=_section_for_offset(sections, start))
        intents.append(
            DetectedIntent(
                intent="sedation",
                value="moderate",
                payload={
                    "start_minutes": start_minutes,
                    "stop_minutes": stop_minutes,
                    "duration_minutes": duration,
                    "observer": observer,
                    "documentation_complete": documentation_complete,
                },
                evidence=[span],
                confidence=0.9 if documentation_complete else 0.5,
            )
        )
    return intents


def _detect_anesthesia(text: str, sections: Sequence[Section], lexical: Lexicon) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    for pattern in lexical.anesthesia_terms:
        for match in pattern.finditer(text):
            span = _match_span(text, sections, match)
            intents.append(
                DetectedIntent(
                    intent="anesthesia",
                    value=match.group(0),
                    payload={"type": "general"},
                    evidence=[span],
                    confidence=0.75,
                )
            )
    return intents


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term and term in text for term in terms)


def _contains_pattern(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _match_span(text: str, sections: Sequence[Section], match: re.Match[str]) -> Span:
    return Span(
        text=match.group(0),
        start=match.start(),
        end=match.end(),
        section=_section_for_offset(sections, match.start()),
    )


def _sentence_span(
    sentences: Sequence[Tuple[int, int]],
    text: str,
    sections: Sequence[Section],
    start: int,
    end: int,
) -> Span:
    sentence_range = _find_sentence_range(sentences, start)
    if not sentence_range:
        sentence_range = (start, end)
    begin, finish = sentence_range
    return Span(
        text=text[begin:finish].strip(),
        start=begin,
        end=finish,
        section=_section_for_offset(sections, begin),
    )


def _find_sentence_range(sentences: Sequence[Tuple[int, int]], index: int) -> Tuple[int, int] | None:
    for start, end in sentences:
        if start <= index < end:
            return start, end
    return None


def _iter_sentences(text: str) -> Iterator[Tuple[int, int]]:
    for match in _SENTENCE_RE.finditer(text):
        yield match.start(), match.end()


def _section_for_offset(sections: Sequence[Section], offset: int) -> str | None:
    for section in sections:
        if section.start <= offset < section.end:
            return section.title
    return None


def _context_slice(text: str, start: int, end: int, window: int = 250) -> str:
    return text[max(0, start - window) : min(len(text), end + window)]


def _phrase_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    if " " in term or "-" in term:
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(rf"(?i)\b{escaped}\b")


def _detect_manufacturer(sentence_lower: str, manufacturers: Dict[str, Sequence[str]]) -> str | None:
    for name, aliases in manufacturers.items():
        for alias in aliases:
            if alias.lower() in sentence_lower:
                return name
    return None


def _detect_sides(sentence_lower: str) -> set[str]:
    sides = set()
    if "right" in sentence_lower:
        sides.add("right")
    if "left" in sentence_lower:
        sides.add("left")
    return sides


def _extract_sedation_sections(text: str) -> list[Tuple[str, int, int]]:
    sections: list[Tuple[str, int, int]] = []
    lower = text.lower()
    start_idx = 0
    while True:
        idx = lower.find("sedation", start_idx)
        if idx == -1:
            break
        start = text.rfind("\n\n", 0, idx)
        if start == -1:
            start = 0
        else:
            start += 2
        end = text.find("\n\n", idx)
        if end == -1:
            end = len(text)
        sections.append((text[start:end], start, end))
        start_idx = end
    return sections


def _parse_time_string(value: str | None) -> int | None:
    if not value:
        return None
    token = value.strip().lower()
    ampm = None
    if token.endswith("am") or token.endswith("pm"):
        ampm = token[-2:]
        token = token[:-2].strip()
    hour = 0
    minute = 0
    if ":" in token:
        hour_str, minute_str = token.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    elif len(token) in (3, 4):
        hour = int(token[:-2])
        minute = int(token[-2:])
    else:
        hour = int(token)
    if ampm:
        if hour == 12:
            hour = 0
        if ampm == "pm":
            hour += 12
    return hour * 60 + minute


def _match_site(
    sentence_lower: str,
    site_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    airway_meta: Dict[str, Dict[str, str]]
) -> tuple[str | None, str | None]:
    for site, patterns in site_patterns.items():
        for pattern in patterns:
            if pattern.search(sentence_lower):
                site_class = _site_class(site, airway_meta)
                return site, site_class
    return None, None


def _match_lobe_from_text(sentence_text: str, lobe_patterns: Dict[str, Tuple[re.Pattern[str], ...]]) -> str | None:
    for lobe, patterns in lobe_patterns.items():
        for pattern in patterns:
            if pattern.search(sentence_text):
                return lobe
    return None


def _site_class(site: str, airway_meta: Dict[str, Dict[str, str]]) -> str:
    if site in airway_meta:
        return airway_meta[site].get("class", "unknown").lower()
    lobes = knowledge.lobe_aliases()
    if site in lobes:
        return "lobe"
    return "unknown"


def get_lobe_pattern_map() -> Dict[str, Tuple[re.Pattern[str], ...]]:
    aliases = knowledge.lobe_aliases()
    if not aliases:
        return {
            "RUL": (re.compile(r"right upper lobe", re.IGNORECASE), re.compile(r"\bRUL\b", re.IGNORECASE)),
            "RML": (re.compile(r"right middle lobe", re.IGNORECASE), re.compile(r"\bRML\b", re.IGNORECASE)),
            "RLL": (re.compile(r"right lower lobe", re.IGNORECASE), re.compile(r"\bRLL\b", re.IGNORECASE)),
            "LUL": (re.compile(r"left upper lobe", re.IGNORECASE), re.compile(r"\bLUL\b", re.IGNORECASE)),
            "LLL": (re.compile(r"left lower lobe", re.IGNORECASE), re.compile(r"\bLLL\b", re.IGNORECASE)),
        }
    return {lobe: tuple(_phrase_pattern(alias) for alias in aliases_list) for lobe, aliases_list in aliases.items()}


def get_station_pattern_map() -> Dict[str, Tuple[re.Pattern[str], ...]]:
    aliases = knowledge.station_aliases()
    if not aliases:
        return {
            "4R": (re.compile(r"\b4R\b", re.IGNORECASE), re.compile(r"station\s*4\s*(?:right|R)", re.IGNORECASE)),
            "4L": (re.compile(r"\b4L\b", re.IGNORECASE),),
            "7": (re.compile(r"\b7\b", re.IGNORECASE), re.compile(r"station\s*7", re.IGNORECASE)),
            "11R": (re.compile(r"\b11R\b", re.IGNORECASE),),
            "11L": (re.compile(r"\b11L\b", re.IGNORECASE),),
        }
    return {station: tuple(_phrase_pattern(alias) for alias in aliases_list) for station, aliases_list in aliases.items()}


def get_site_pattern_map() -> Dict[str, Tuple[re.Pattern[str], ...]]:
    patterns: Dict[str, Tuple[re.Pattern[str], ...]] = {}
    airway_map = knowledge.airway_map()
    for site, meta in airway_map.items():
        aliases = meta.get("aliases", [])
        if aliases:
            patterns[site] = tuple(_phrase_pattern(alias) for alias in aliases)
    lobe_patterns = get_lobe_pattern_map()
    patterns.update(lobe_patterns)
    
    # Ensure Trachea is present if not already
    if "Trachea" not in patterns and "TRACHEA" not in patterns:
        patterns["Trachea"] = (re.compile(r"\btrachea\b", re.IGNORECASE),)
        
    return patterns


def _detect_stent_removal(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    site_patterns: Dict[str, Tuple[re.Pattern[str], ...]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    airway_meta = knowledge.airway_map()
    for start, end in sentences:
        sentence_text = text[start:end]
        if not _contains_pattern(sentence_text, lexical.stent_removal_terms):
            continue
        lower_sentence = sentence_text.lower()
        site, site_class = _match_site(lower_sentence, site_patterns, airway_meta)
        span = Span(text=sentence_text.strip(), start=start, end=end, section=_section_for_offset(sections, start))
        intents.append(
            DetectedIntent(
                intent="stent_removal",
                value="removal",
                payload={"text": sentence_text.strip(), "site": site, "site_class": site_class},
                evidence=[span],
                confidence=0.9,
            )
        )
    return intents


def _detect_destruction(
    text: str,
    sections: Sequence[Section],
    sentences: Sequence[Tuple[int, int]],
    lexical: Lexicon,
) -> list[DetectedIntent]:
    intents: list[DetectedIntent] = []
    for start, end in sentences:
        sentence_text = text[start:end]
        if not _contains_pattern(sentence_text, lexical.destruction_terms):
            continue
        span = Span(text=sentence_text.strip(), start=start, end=end, section=_section_for_offset(sections, start))
        intents.append(
            DetectedIntent(
                intent="destruction",
                value="destruction",
                payload={"text": sentence_text.strip()},
                evidence=[span],
                confidence=0.85,
            )
        )
    return intents
