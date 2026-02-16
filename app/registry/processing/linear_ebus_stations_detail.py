"""Deterministic backstop extraction for per-station linear EBUS morphology/sampling detail.

The v3 registry schema supports `registry.granular_data.linear_ebus_stations_detail[]`
as an array of EBUSStationDetail entries. Model-driven extraction often captures
only performed flags/stations_sampled; this module provides a conservative parser
for common templated note formats (e.g., "Station 7: Measured ... by EBUS ...").
"""

from __future__ import annotations

import re
from typing import Any


try:
    from app.ner.entity_types import normalize_station
except Exception:  # pragma: no cover
    normalize_station = None  # type: ignore[assignment]


_EBUS_HINT_RE = re.compile(r"(?i)\b(?:ebus|endobronchial\s+ultrasound|ebus-tbna)\b")

_STATION_HEADER_RE = re.compile(
    r"(?is)\bStation\s+(?P<station>[0-9]{1,2}[LR]?(?:Rs|Ri|Ls|Li)?)\b"
)
_NUMBERED_STATION_HEADER_RE = re.compile(
    r"(?im)^\s*\d{1,2}\s*[.)]\s*(?:station\s*)?(?P<station>[0-9]{1,2}[LR]?(?:Rs|Ri|Ls|Li)?)\b"
)

_GLOBAL_NEEDLE_GAUGE_RE = re.compile(r"(?i)\b(19|21|22|25)\s*[- ]?(?:g|gauge)\b")
_COMPLETENESS_NEEDLE_GAUGE_RE = re.compile(
    r"(?i)\bneedle\s+gauge\s*(?:\(\s*per\s+station\s*\))?\s*[:=]\s*(19|21|22|25)\b"
)
_COMPLETENESS_PASSES_RE = re.compile(
    r"(?i)\bpasses?\s*(?:\(\s*per\s+station\s*\))?\s*[:=]\s*(\d{1,2})\b"
)
_COMPLETENESS_SHORT_AXIS_RE = re.compile(
    r"(?i)\b(?:node\s+size\s+short\s+axis|short[-\s]?axis)(?:\s*\(mm\))?\s*[:=]\s*(\d+(?:\.\d+)?)\b"
)
_COMPLETENESS_LYMPH_RE = re.compile(
    r"(?i)\blymphocytes?\s+present\s*[:=]?\s*(yes|no|true|false)\b"
)

_NON_STATION_STOP_RE = re.compile(
    r"(?im)^\s*(?:"
    r"(?:right|left)\s+(?:upper|middle|lower)\s+lobe\s+mass\s*:|"
    r"(?:right|left)\s+lobe\s+mass\s*:|"
    r"(?:lung\s+)?mass\s*:|"
    r"(?:lung\s+)?lesion\s*:|"
    r"(?:lung\s+)?nodule\s*:"
    r")"
)

_MASS_HEADER_RE = re.compile(
    r"(?im)^\s*(?:(?P<lobe>(?:right|left)\s+(?:upper|middle|lower)\s+lobe)|(?P<side>(?:right|left)\s+lobe)|(?P<lung>lung))?\s*"
    r"(?P<type>mass|lesion|nodule)\s*:\s*(?P<rest>.*)$"
)

_EBUS_MM_RE = re.compile(r"(?i)\bmeasured\s+(?P<mm>\d+(?:\.\d+)?)\s*mm\b.{0,40}?\bby\s+EBUS\b")
_EBUS_AXES_RE = re.compile(
    r"(?i)\bmeasured\s+(?P<a>\d+(?:\.\d+)?)\s*(?:x|by)\s*(?P<b>\d+(?:\.\d+)?)\s*mm\b.{0,40}?\bby\s+EBUS\b"
)
_AXES_FALLBACK_RE = re.compile(
    r"(?i)\b(?P<a>\d+(?:\.\d+)?)\s*(?:x|by)\s*(?P<b>\d+(?:\.\d+)?)\s*mm\b"
)

_STATION_SIZE_LIST_LINE_RE = re.compile(
    r"(?im)^\s*(?P<station>2R|2L|3P|4R|4L|5|7|8|9|10R|10L|11R(?:s|i)?|11L(?:s|i)?|12R|12L)\s*"
    r"[-:]\s*(?P<mm>\d+(?:\.\d+)?)\s*mm\b"
)

_PASSES_RE = re.compile(r"(?i)\b(\d{1,2})\s+passes?\b")
_PASSES_WORD_RE = re.compile(r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+passes?\b")
_NEEDLE_GAUGE_RE = re.compile(r"(?i)\b(19|21|22|25)\s*[- ]?(?:g|gauge)\b")

_SAMPLED_FALSE_RE = re.compile(
    r"(?i)\b(?:"
    r"not\s+biopsied"
    r"|not\s+sampled"
    r"|not\s+aspirated"
    r"|not\s+biopsied\s+due"
    r"|did\s+not\s+have\s+any\s+biops(?:y|ies)\s+target"
    r"|did\s+not\s+have\s+any\s+sampling\s+target"
    r"|no\s+(?:biops(?:y|ies)|sampling)\s+target"
    r")\b"
)
_SAMPLED_TRUE_RE = re.compile(r"(?i)\b(?:biopsied|sampled|aspirated)\b")
_SAMPLED_ACTION_RE = re.compile(r"(?i)\b(?:tbna|fna|needle\s+aspirat|passes?)\b")

_ECHOGENICITY_RE = re.compile(r"(?i)\b(?:heterogeneous|homogeneous)\b")
_SHAPE_RE = re.compile(r"(?i)\b(?:oval|round|irregular)\s+shape\b|\b(?:oval|round|irregular)\b")
_MARGIN_DISTINCT_RE = re.compile(r"(?i)\b(?:sharp|distinct)\s+margins?\b")
_MARGIN_INDISTINCT_RE = re.compile(r"(?i)\bindistinct\s+margins?\b")
_MARGIN_IRREGULAR_RE = re.compile(r"(?i)\birregular\s+margins?\b")

_CHS_PRESENT_RE = re.compile(r"(?i)\b(?:chs|central\s+hilar\s+structure)\b.{0,20}\bpresent\b")
_CHS_ABSENT_RE = re.compile(r"(?i)\b(?:chs|central\s+hilar\s+structure)\b.{0,20}\babsent\b")
_NECROSIS_PRESENT_RE = re.compile(r"(?i)\bnecros(?:is|e|ed)\b")
_NECROSIS_NEG_RE = re.compile(r"(?i)\bno\s+necros(?:is|e)\b")
_CALC_PRESENT_RE = re.compile(r"(?i)\bcalcif(?:ied|ication)\b")
_CALC_NEG_RE = re.compile(r"(?i)\bno\s+calcif(?:ied|ication)\b")

_ROSE_RE = re.compile(r"(?i)\brose\b")

_LYMPH_POS_RE = re.compile(
    r"(?i)\b(?:adequate|present|identified|seen)\b[^.\n]{0,60}\b(?:lymphocytes?|lymphoid\s+tissue)\b"
    r"|\b(?:lymphocytes?|lymphoid\s+tissue)\b[^.\n]{0,60}\b(?:present|identified|seen|adequate)\b"
)
_LYMPH_NEG_RE = re.compile(r"(?i)\b(?:no|without|scant|rare)\s+(?:lymphocytes?|lymphoid\s+tissue)\b")
_BLOOD_ONLY_RE = re.compile(r"(?i)\b(?:blood\s+only|acellular)\b")
_NONDIAGNOSTIC_RE = re.compile(r"(?i)\b(?:nondiagnostic|non-diagnostic|insufficient)\b")


def _maybe_unescape_newlines(text: str) -> str:
    raw = text or ""
    if not raw.strip():
        return raw
    if "\n" in raw or "\r" in raw:
        return raw
    if "\\n" not in raw and "\\r" not in raw:
        return raw
    return raw.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


_WORD_NUMBERS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _normalize_station_token(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if normalize_station is None:
        return raw.upper()
    return normalize_station(raw)


def _normalize_mass_label(match: re.Match[str]) -> str:
    lobe = match.group("lobe")
    side = match.group("side")
    lung = match.group("lung")
    typ = match.group("type") or "mass"

    label_parts: list[str] = []
    if lobe:
        label_parts.append(lobe.strip())
    elif side:
        label_parts.append(side.strip())
    elif lung:
        label_parts.append("lung")
    label_parts.append(typ.strip())

    label = " ".join(label_parts).strip()
    label = re.sub(r"\s+", " ", label).strip()
    return label.title() if label else typ.title()


def _infer_station_sampled_from_text(text: str, station: str) -> bool | None:
    """Best-effort inference of sampled vs measured-only for a station token."""
    if not text or not station:
        return None
    pat = re.compile(rf"(?i)\\b{re.escape(station)}\\b")
    for raw_line in (text or "").splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue
        if not pat.search(line):
            continue
        if _SAMPLED_FALSE_RE.search(line):
            return False
        if _SAMPLED_TRUE_RE.search(line) or _SAMPLED_ACTION_RE.search(line):
            return True
    return None


def _apply_section_to_entry(
    entry: dict[str, Any],
    section: str,
    *,
    section_offset: int = 0,
    global_gauge: int | None,
    global_gauge_evidence: dict[str, Any] | None = None,
    default_sampled: bool | None = None,
) -> None:
    sampled: bool | None = default_sampled
    if _SAMPLED_FALSE_RE.search(section):
        sampled = False
    elif _SAMPLED_TRUE_RE.search(section):
        sampled = True
    if sampled is not None:
        entry["sampled"] = sampled

    # Sizes: prefer explicit "Measured <mm> by EBUS" patterns.
    axes_match = _EBUS_AXES_RE.search(section)
    if axes_match:
        a = _to_float(axes_match.group("a"))
        b = _to_float(axes_match.group("b"))
        if a is not None and b is not None:
            short_val = min(a, b)
            long_val = max(a, b)
            if entry.get("short_axis_mm") is None:
                entry["short_axis_mm"] = short_val
                entry["_short_axis_mm_evidence"] = {
                    "text": (axes_match.group(0) or "").strip(),
                    "start": section_offset + int(axes_match.start()),
                    "end": section_offset + int(axes_match.end()),
                }
            if entry.get("long_axis_mm") is None:
                entry["long_axis_mm"] = long_val
                entry["_long_axis_mm_evidence"] = {
                    "text": (axes_match.group(0) or "").strip(),
                    "start": section_offset + int(axes_match.start()),
                    "end": section_offset + int(axes_match.end()),
                }
    else:
        size_match = _EBUS_MM_RE.search(section)
        if size_match:
            mm = _to_float(size_match.group("mm"))
            if mm is not None:
                if entry.get("short_axis_mm") is None:
                    entry["short_axis_mm"] = mm
                    entry["_short_axis_mm_evidence"] = {
                        "text": (size_match.group(0) or "").strip(),
                        "start": section_offset + int(size_match.start()),
                        "end": section_offset + int(size_match.end()),
                    }
        else:
            # Fallback: some templates omit "by EBUS" but still provide axes.
            axes2 = _AXES_FALLBACK_RE.search(section)
            if axes2:
                a = _to_float(axes2.group("a"))
                b = _to_float(axes2.group("b"))
                if a is not None and b is not None:
                    short_val = min(a, b)
                    long_val = max(a, b)
                    if entry.get("short_axis_mm") is None:
                        entry["short_axis_mm"] = short_val
                        entry["_short_axis_mm_evidence"] = {
                            "text": (axes2.group(0) or "").strip(),
                            "start": section_offset + int(axes2.start()),
                            "end": section_offset + int(axes2.end()),
                        }
                    if entry.get("long_axis_mm") is None:
                        entry["long_axis_mm"] = long_val
                        entry["_long_axis_mm_evidence"] = {
                            "text": (axes2.group(0) or "").strip(),
                            "start": section_offset + int(axes2.start()),
                            "end": section_offset + int(axes2.end()),
                        }

    # Morphology
    shape_raw = None
    shape_match = _SHAPE_RE.search(section)
    if shape_match:
        shape_raw = shape_match.group(0).lower()
    if shape_raw:
        if "oval" in shape_raw:
            entry.setdefault("shape", "oval")
        elif "round" in shape_raw:
            entry.setdefault("shape", "round")
        elif "irregular" in shape_raw:
            entry.setdefault("shape", "irregular")

    if _MARGIN_DISTINCT_RE.search(section):
        entry.setdefault("margin", "distinct")
    elif _MARGIN_INDISTINCT_RE.search(section):
        entry.setdefault("margin", "indistinct")
    elif _MARGIN_IRREGULAR_RE.search(section):
        entry.setdefault("margin", "irregular")

    echo_match = _ECHOGENICITY_RE.search(section)
    if echo_match:
        val = echo_match.group(0).lower()
        if "hetero" in val:
            entry.setdefault("echogenicity", "heterogeneous")
        elif "homo" in val:
            entry.setdefault("echogenicity", "homogeneous")

    if _CHS_PRESENT_RE.search(section):
        entry.setdefault("chs_present", True)
    elif _CHS_ABSENT_RE.search(section):
        entry.setdefault("chs_present", False)

    if _NECROSIS_PRESENT_RE.search(section) and not _NECROSIS_NEG_RE.search(section):
        entry.setdefault("necrosis_present", True)
    if _CALC_PRESENT_RE.search(section) and not _CALC_NEG_RE.search(section):
        entry.setdefault("calcification_present", True)

    # Sampling details: only when explicitly sampled.
    # Avoid "ghost gauge/pass" entries for nodes described as measured-only or not biopsied.
    if sampled is not True:
        return

    gauge = None
    gauge_local = _NEEDLE_GAUGE_RE.search(section)
    if gauge_local:
        gauge = _to_int(gauge_local.group(1))
    elif global_gauge is not None:
        gauge = global_gauge
    if gauge is not None:
        if entry.get("needle_gauge") is None:
            entry["needle_gauge"] = gauge
            if gauge_local:
                entry["_needle_gauge_evidence"] = {
                    "text": (gauge_local.group(0) or "").strip(),
                    "start": section_offset + int(gauge_local.start()),
                    "end": section_offset + int(gauge_local.end()),
                }
            elif global_gauge_evidence:
                entry["_needle_gauge_evidence"] = global_gauge_evidence

    passes_match = _PASSES_RE.search(section)
    if passes_match:
        passes = _to_int(passes_match.group(1))
        if passes is not None:
            if entry.get("number_of_passes") is None:
                entry["number_of_passes"] = passes
                entry["_number_of_passes_evidence"] = {
                    "text": (passes_match.group(0) or "").strip(),
                    "start": section_offset + int(passes_match.start()),
                    "end": section_offset + int(passes_match.end()),
                }
    else:
        word_match = _PASSES_WORD_RE.search(section)
        if word_match:
            passes = _WORD_NUMBERS.get(word_match.group(1).lower())
            if passes is not None:
                if entry.get("number_of_passes") is None:
                    entry["number_of_passes"] = passes
                    entry["_number_of_passes_evidence"] = {
                        "text": (word_match.group(0) or "").strip(),
                        "start": section_offset + int(word_match.start()),
                        "end": section_offset + int(word_match.end()),
                    }

    # ROSE
    if _ROSE_RE.search(section):
        entry.setdefault("rose_performed", True)
        lower = section.lower()
        if "malignan" in lower:
            entry.setdefault("rose_result", "Malignant")
        elif "suspicious" in lower:
            entry.setdefault("rose_result", "Suspicious for malignancy")
        elif "atypical" in lower:
            entry.setdefault("rose_result", "Atypical cells")
        elif "granuloma" in lower:
            entry.setdefault("rose_result", "Granuloma")
        elif "necrosis" in lower and "only" in lower:
            entry.setdefault("rose_result", "Necrosis only")
        elif "nondiagnostic" in lower:
            entry.setdefault("rose_result", "Nondiagnostic")
        elif "adequate" in lower:
            # Many LN templates say "adequate tissue"; map to the canonical LN adequacy label.
            entry.setdefault("rose_result", "Adequate lymphocytes")

    # Lymphocyte adequacy (explicit-only; station-level boolean).
    if "lymphocytes_present" not in entry:
        lymph_match: re.Match[str] | None = None
        if _LYMPH_NEG_RE.search(section):
            lymph_match = _LYMPH_NEG_RE.search(section)
            entry.setdefault("lymphocytes_present", False)
        elif _LYMPH_POS_RE.search(section):
            lymph_match = _LYMPH_POS_RE.search(section)
            entry.setdefault("lymphocytes_present", True)
        elif _ROSE_RE.search(section) and (_BLOOD_ONLY_RE.search(section) or _NONDIAGNOSTIC_RE.search(section)):
            lymph_match = _BLOOD_ONLY_RE.search(section) or _NONDIAGNOSTIC_RE.search(section)
            entry.setdefault("lymphocytes_present", False)

        if lymph_match:
            snippet = (lymph_match.group(0) or "").strip()
            if snippet:
                entry["_lymphocytes_present_evidence"] = {
                    "text": snippet,
                    "start": section_offset + int(lymph_match.start()),
                    "end": section_offset + int(lymph_match.end()),
                }

    # Morphologic impression
    lower = section.lower()
    if "benign" in lower:
        entry.setdefault("morphologic_impression", "benign")
    elif "suspicious" in lower:
        entry.setdefault("morphologic_impression", "suspicious")
    elif "malignan" in lower:
        entry.setdefault("morphologic_impression", "malignant")


def extract_linear_ebus_stations_detail(note_text: str) -> list[dict[str, Any]]:
    """Extract linear EBUS station morphology/sampling details from text."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return []
    if not _EBUS_HINT_RE.search(text):
        return []

    global_gauge: int | None = None
    global_gauge_evidence: dict[str, Any] | None = None
    gauge_match = _GLOBAL_NEEDLE_GAUGE_RE.search(text)
    if not gauge_match:
        gauge_match = _COMPLETENESS_NEEDLE_GAUGE_RE.search(text)
    if gauge_match:
        global_gauge = _to_int(gauge_match.group(1))
        if global_gauge is not None:
            global_gauge_evidence = {
                "text": (gauge_match.group(0) or "").strip(),
                "start": int(gauge_match.start()),
                "end": int(gauge_match.end()),
            }

    global_passes: int | None = None
    global_passes_evidence: dict[str, Any] | None = None
    passes_match = _COMPLETENESS_PASSES_RE.search(text)
    if passes_match:
        global_passes = _to_int(passes_match.group(1))
        if global_passes is not None:
            global_passes_evidence = {
                "text": (passes_match.group(0) or "").strip(),
                "start": int(passes_match.start()),
                "end": int(passes_match.end()),
            }

    global_short_axis_mm: float | None = None
    global_short_axis_evidence: dict[str, Any] | None = None
    short_match = _COMPLETENESS_SHORT_AXIS_RE.search(text)
    if short_match:
        global_short_axis_mm = _to_float(short_match.group(1))
        if global_short_axis_mm is not None:
            global_short_axis_evidence = {
                "text": (short_match.group(0) or "").strip(),
                "start": int(short_match.start()),
                "end": int(short_match.end()),
            }

    global_lymphocytes_present: bool | None = None
    global_lymph_evidence: dict[str, Any] | None = None
    lymph_match = _COMPLETENESS_LYMPH_RE.search(text)
    if lymph_match:
        raw = (lymph_match.group(1) or "").strip().lower()
        if raw in {"yes", "true"}:
            global_lymphocytes_present = True
        elif raw in {"no", "false"}:
            global_lymphocytes_present = False
        if global_lymphocytes_present is not None:
            global_lymph_evidence = {
                "text": (lymph_match.group(0) or "").strip(),
                "start": int(lymph_match.start()),
                "end": int(lymph_match.end()),
            }

    matches = list(_STATION_HEADER_RE.finditer(text))

    by_station: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw_section = text[start:end]
        leading_trim = len(raw_section) - len(raw_section.lstrip())
        section_offset = start + leading_trim
        section = raw_section.strip()
        if section:
            stop_match = _NON_STATION_STOP_RE.search(section)
            if stop_match:
                section = section[: stop_match.start()].strip()
        if not section:
            continue

        station = _normalize_station_token(match.group("station") or "")
        if not station:
            continue

        entry = by_station.get(station)
        if entry is None:
            entry = {"station": station}
            by_station[station] = entry
            order.append(station)

        _apply_section_to_entry(
            entry,
            section,
            section_offset=section_offset,
            global_gauge=global_gauge,
            global_gauge_evidence=global_gauge_evidence,
        )

    # Numbered-list station formats (common in templated "Sites Sampled" sections), e.g.:
    #   1. 11Rs ... 4 passes ... ROSE: ...
    #   2) 7 (subcarinal) ... 3 passes ...
    numbered_matches = list(_NUMBERED_STATION_HEADER_RE.finditer(text))
    for idx, match in enumerate(numbered_matches):
        start = match.start()
        end = numbered_matches[idx + 1].start() if idx + 1 < len(numbered_matches) else len(text)
        raw_section = text[start:end]
        leading_trim = len(raw_section) - len(raw_section.lstrip())
        section_offset = start + leading_trim
        section = raw_section.strip()
        if section:
            stop_match = _NON_STATION_STOP_RE.search(section)
            if stop_match:
                section = section[: stop_match.start()].strip()
        if not section:
            continue

        station = _normalize_station_token(match.group("station") or "")
        if not station:
            continue

        entry = by_station.get(station)
        if entry is None:
            entry = {"station": station}
            by_station[station] = entry
            order.append(station)

        # In numbered station lists, assume sampling unless an explicit negation exists.
        _apply_section_to_entry(
            entry,
            section,
            section_offset=section_offset,
            global_gauge=global_gauge,
            default_sampled=True,
        )

    # Size-only station list lines (common in free-text narratives), e.g.:
    #   11L - 7.8mm
    #   4L - 2.7mm
    for match in _STATION_SIZE_LIST_LINE_RE.finditer(text):
        station = _normalize_station_token(match.group("station") or "")
        if not station:
            continue

        entry = by_station.get(station)
        if entry is None:
            entry = {"station": station}
            by_station[station] = entry
            order.append(station)

        mm = _to_float(match.group("mm"))
        if mm is not None and entry.get("short_axis_mm") is None:
            entry["short_axis_mm"] = mm
            entry["_short_axis_mm_evidence"] = {
                "text": (match.group(0) or "").strip(),
                "start": int(match.start()),
                "end": int(match.end()),
            }

        # Treat bare size lists as measurement-only unless sampling is explicitly stated elsewhere.
        if entry.get("sampled") is None:
            inferred = _infer_station_sampled_from_text(text, station)
            entry["sampled"] = False if inferred is None else bool(inferred)

    # Non-station targets (masses/lesions/nodules) are often documented in the same
    # templated section as stations. Capture them as additional entries so CPT
    # station counts and UI tables reflect the diagnostic target.
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = _MASS_HEADER_RE.match(line or "")
        if not match:
            idx += 1
            continue

        label = _normalize_mass_label(match)
        entry = by_station.get(label)
        if entry is None:
            entry = {"station": label}
            by_station[label] = entry
            order.append(label)

        block_lines = [(line or "").strip()]
        j = idx + 1
        while j < len(lines):
            next_line = lines[j] or ""
            if not next_line.strip():
                break
            if _STATION_HEADER_RE.search(next_line):
                break
            if _MASS_HEADER_RE.match(next_line):
                break
            if re.match(
                r"(?i)^\s*(?:complications?|estimated\s+blood\s+loss|ebl|impression|plan|disposition|follow-?up|post[- ]procedure)\b",
                next_line,
            ):
                break
            block_lines.append(next_line.strip())
            j += 1

        block = "\n".join(block_lines).strip()
        if block:
            _apply_section_to_entry(entry, block, global_gauge=global_gauge)

        idx = j if j > idx else idx + 1

    # Apply completeness-style global defaults when station-level detail is missing.
    if by_station:
        for entry in by_station.values():
            sampled_flag = entry.get("sampled")
            allow_sampling_defaults = sampled_flag is not False

            if allow_sampling_defaults and global_gauge is not None and entry.get("needle_gauge") is None:
                entry["needle_gauge"] = global_gauge
                if global_gauge_evidence:
                    entry.setdefault("_needle_gauge_evidence", global_gauge_evidence)

            if allow_sampling_defaults and global_passes is not None and entry.get("number_of_passes") is None:
                entry["number_of_passes"] = global_passes
                if global_passes_evidence:
                    entry.setdefault("_number_of_passes_evidence", global_passes_evidence)

            if global_short_axis_mm is not None and entry.get("short_axis_mm") is None:
                entry["short_axis_mm"] = global_short_axis_mm
                if global_short_axis_evidence:
                    entry.setdefault("_short_axis_mm_evidence", global_short_axis_evidence)

            if global_lymphocytes_present is not None and entry.get("lymphocytes_present") is None:
                entry["lymphocytes_present"] = global_lymphocytes_present
                if global_lymph_evidence:
                    entry.setdefault("_lymphocytes_present_evidence", global_lymph_evidence)

    # Ensure stable ordering.
    return [by_station[s] for s in order if s in by_station]
