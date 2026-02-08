"""Heuristics for navigation target and cryobiopsy site extraction.

These helpers are used as a deterministic backstop when model-driven extraction
misses per-target/per-site granularity in complex navigation cases.
"""

from __future__ import annotations

import re
from typing import Any


_TARGET_HEADER_RE = re.compile(
    r"(?im)^\s*(?P<header>"
    r"(?:(?:RIGHT|LEFT)\s+(?:UPPER|MIDDLE|LOWER)\s+LOBE\s+TARGET)"
    r"|(?:RUL|RML|RLL|LUL|LLL|LINGULA)\s+TARGET"
    r")\s*:?\s*$"
)

_NODULE_TARGET_HEADER_RE = re.compile(
    r"(?im)^\s*(?P<header>"
    r"(?:RIGHT|LEFT)\s+(?:UPPER|MIDDLE|LOWER)\s+LOBE\b[^\n:]{0,220}?\b(?:NODULE|LESION)\b[^\n:]{0,220}?"
    r")\s*:\s*$"
)

_NUMBERED_TARGET_HEADER_RE = re.compile(
    r"(?im)^\s*Target\s*(?P<num>\d{1,2})\s*[:\-]\s*(?P<header>.+?)\s*$"
)

_TARGET_LOBE_FROM_HEADER: dict[str, str] = {
    "RIGHT UPPER LOBE TARGET": "RUL",
    "RIGHT MIDDLE LOBE TARGET": "RML",
    "RIGHT LOWER LOBE TARGET": "RLL",
    "LEFT UPPER LOBE TARGET": "LUL",
    "LEFT LOWER LOBE TARGET": "LLL",
    "RUL TARGET": "RUL",
    "RML TARGET": "RML",
    "RLL TARGET": "RLL",
    "LUL TARGET": "LUL",
    "LLL TARGET": "LLL",
    "LINGULA TARGET": "Lingula",
}

_ENGAGE_LOCATION_RE = re.compile(
    r"(?is)\bengage(?:d)?\s+the\s+(?P<segment>[^.\n]{3,160}?)\s+of\s+(?:the\s+)?(?P<lobe>RUL|RML|RLL|LUL|LLL|LINGULA)\s*"
    r"\((?P<bronchus>[^)]+)\)"
)

_TARGET_LESION_SIZE_CM_RE = re.compile(r"(?is)\btarget\s+lesion\b[^.\n]{0,80}\b(\d+(?:\.\d+)?)\s*cm\b")
_TARGET_LESION_SIZE_MM_RE = re.compile(r"(?is)\btarget\s+lesion\b[^.\n]{0,80}\b(\d+(?:\.\d+)?)\s*mm\b")

_REBUS_VIEW_RE = re.compile(
    r"(?is)\b(?:radial\s+ebus|rebus)\b.{0,240}?\b(concentric|eccentric|adjacent|not visualized)\b"
)
_REGISTRATION_ERROR_RE = re.compile(
    r"(?is)\bregistration\b[^.\n]{0,200}\berror\b[^.\n]{0,60}\b(\d+(?:\.\d+)?)\s*mm\b"
)
_REGISTRATION_ERROR_FALLBACK_RE = re.compile(r"(?is)\berror\s+of\s+(\d+(?:\.\d+)?)\s*mm\b")
_TARGET_SEGMENT_OF_LOBE_RE = re.compile(
    r"(?is)\btarget\s+lesion\b[^.\n]{0,260}\bin\s+(?:the\s+)?(?P<segment>[A-Za-z][A-Za-z -]{0,60}?Segment)\s+of\s+(?:the\s+)?(?P<lobe>RUL|RML|RLL|LUL|LLL|LINGULA)\b"
)
_BRONCHUS_CODE_RE = re.compile(r"\b([LR]B\d{1,2})\b", re.IGNORECASE)
_NEEDLE_GAUGE_RE = re.compile(r"(?i)\b(19|21|22|25)\s*[- ]?(?:g|gauge)\b")
_PASSES_RE = re.compile(r"(?i)\b(\d{1,2})\s+passes?\b")
_SPECIMEN_COUNT_RE = re.compile(r"(?i)\b(\d{1,2})\s+(?:specimens?|samples?|biops(?:y|ies))\b")

_CT_PART_SOLID_EXPLICIT_RE = re.compile(
    r"(?i)\b(?:"
    r"part[-\s]?solid|semi[-\s]?solid|mixed\s+(?:density|attenuation)|"
    r"solid\s+component|increasing\s+density"
    r")\b"
)
_CT_GROUND_GLASS_RE = re.compile(
    r"(?i)\b(?:ground[-\s]?glass|groundglass|ggo|ggn|non[-\s]?solid|nonsolid)\b"
)
_CT_SUBSOLID_RE = re.compile(r"(?i)\bsub[-\s]?solid|subsolid\b")
_CT_CAVITARY_RE = re.compile(r"(?i)\b(?:cavitary|cavit(?:y|ation))\b")
_CT_CALCIFIED_RE = re.compile(r"(?i)\b(?:calcified|calcification)\b")
_CT_SOLID_CONTEXT_RE = re.compile(
    r"(?i)\bsolid\b[^.\n]{0,60}\b(?:nodule|lesion|mass|opacity)\b"
    r"|\b(?:nodule|lesion|mass|opacity)\b[^.\n]{0,60}\bsolid\b"
)

_PLEURAL_DISTANCE_RE = re.compile(
    r"(?i)\b(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm)\s+(?:from|to)\s+(?:the\s+)?"
    r"pleura(?:l\s+(?:surface|space))?\b"
)
_PLEURAL_ABUTTING_RE = re.compile(
    r"(?i)\b(?:abutting|abuts|touching|contacting|against|pleural[-\s]?based)\b[^.\n]{0,60}\bpleura\b"
    r"|\bpleura\b[^.\n]{0,60}\b(?:abutting|abuts|touching|contacting|against|pleural[-\s]?based)\b"
)

_AIR_BRONCHOGRAM_POS_RE = re.compile(r"(?i)\bair\s+bronchogram(?:s)?\b")
_AIR_BRONCHOGRAM_NEG_RE = re.compile(
    r"(?i)\b(?:no|without|absent)\b[^.\n]{0,60}\bair\s+bronchogram(?:s)?\b"
    r"|\bair\s+bronchogram(?:s)?\b[^.\n]{0,60}\b(?:not\s+present|absent)\b"
)

_SUV_RE = re.compile(r"(?i)\bSUV(?:\s*max)?\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\b")

_BRONCHUS_SIGN_NOT_ASSESSED_RE = re.compile(
    r"(?i)\bbronchus\s+sign\b[^.\n]{0,40}\b(?:not\s+assessed|unknown|indeterminate|n/?a)\b"
)
_BRONCHUS_SIGN_NEG_RE = re.compile(
    r"(?i)\bbronchus\s+sign\b[^.\n]{0,40}\b(?:negative|absent|no|not\s+present)\b"
    r"|\b(?:negative|absent|not\s+present)\b[^.\n]{0,20}\bbronchus\s+sign\b"
)
_BRONCHUS_SIGN_POS_RE = re.compile(
    r"(?i)\bbronchus\s+sign\b[^.\n]{0,40}\b(?:positive|present|yes)\b"
    r"|\b(?:positive|present)\b[^.\n]{0,20}\bbronchus\s+sign\b"
)

_TIL_NEG_RE = re.compile(
    r"(?i)\b(?:tool\s*[- ]?\s*in\s*[- ]?\s*lesion|t\.?i\.?l\.?)\b"
    r"(?:\s+(?:confirmation|confirm(?:ed|ation)?)\b)?\s*[:=]?\s*"
    r"(?:not\s+confirmed|unable\s+to\s+confirm|unconfirmed|negative|failed|not\s+achieved|no)\b"
    r"|\b(?:not\s+confirmed|unable\s+to\s+confirm|unconfirmed|negative|failed|not\s+achieved)\b[^.\n]{0,40}"
    r"\b(?:tool\s*[- ]?\s*in\s*[- ]?\s*lesion|t\.?i\.?l\.?)\b"
)
_TIL_POS_RE = re.compile(
    r"(?i)\b(?:tool\s*[- ]?\s*in\s*[- ]?\s*lesion|t\.?i\.?l\.?)\b"
    r"(?:\s+(?:confirmation|confirm(?:ed|ation)?)\b)?\s*[:=]?\s*"
    r"(?:confirmed|achieved|yes|positive)\b"
    r"|\b(?:confirmed|achieved|yes|positive)\b[^.\n]{0,40}"
    r"\b(?:tool\s*[- ]?\s*in\s*[- ]?\s*lesion|t\.?i\.?l\.?)\b"
)
_TIL_METHOD_CBCT_RE = re.compile(r"(?i)\b(?:cbct|cone[-\s]?beam\s+ct)\b")
_TIL_METHOD_AUG_FLUORO_RE = re.compile(r"(?i)\b(?:augmented\s+fluor(?:oscopy|o)|aug\s+fluoro)\b")
_TIL_METHOD_FLUORO_RE = re.compile(r"(?i)\bfluor(?:oscopy|o)\b")
_TIL_METHOD_REBUS_RE = re.compile(r"(?i)\b(?:radial\s+ebus|r-?ebus|rebus)\b")

_INLINE_TARGET_RE = re.compile(
    r"\bTarget(?:\s+Lesion)?\s*[:\-]\s*(?P<loc>[^\n\r]+)",
    re.IGNORECASE | re.MULTILINE,
)

_FIDUCIAL_SENTENCE_RE = re.compile(r"(?i)\b(fiducial(?:\s+marker)?s?\b[^\n]{0,260})")
_FIDUCIAL_ACTION_RE = re.compile(r"(?i)\b(?:plac(?:ed|ement)|deploy\w*|position\w*|insert\w*)\b")
_NEGATION_RE = re.compile(r"(?i)\b(?:no|not|without|denies|deny)\b")

_CRYO_RE = re.compile(r"(?i)\btransbronchial\s+cryo(?:biopsy|biopsies)\b|\bcryobiops(?:y|ies)\b|\bTBLC\b")
_CRYO_PROBE_SIZE_RE = re.compile(r"(?i)\b(\d(?:\.\d)?)\s*mm\s*cryo\s*probe\b")
_CRYO_FREEZE_RE = re.compile(r"(?i)\bfreeze\s+time\b[^.\n]{0,40}\b(\d{1,2})\s*seconds?\b")
_TOTAL_SAMPLES_RE = re.compile(r"(?i)\btotal\s+(\d{1,2})\s+samples?\b")

_ROSE_LINE_RE = re.compile(r"(?im)^\s*ROSE(?:\s+Result)?\s*[:\-]\s*(?P<result>.+?)\s*$")
_ROSE_HEADER_RE = re.compile(r"(?im)^\s*ROSE(?:\s+Result)?\s*[:\-]\s*$")

_COUNT_WORDS: dict[str, int] = {
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
    "eleven": 11,
    "twelve": 12,
}


def _canonical_header(value: str) -> str:
    raw = (value or "").strip().upper()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def _normalize_lobe(value: str | None) -> str | None:
    if value is None:
        return None
    upper = str(value).strip().upper()
    if upper in {"RUL", "RML", "RLL", "LUL", "LLL"}:
        return upper
    if upper == "LINGULA":
        return "Lingula"
    return None


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: str | None) -> int | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.isdigit():
        try:
            return int(raw)
        except Exception:
            return None
    return _COUNT_WORDS.get(raw.lower())


def _first_line_containing(text: str, pattern: re.Pattern[str]) -> str | None:
    for line in (text or "").splitlines():
        if pattern.search(line):
            return line.strip() or None
    return None


def _detect_ct_characteristics(text: str) -> str | None:
    label, _match = _match_ct_characteristics(text)
    return label


def _match_ct_characteristics(text: str) -> tuple[str | None, re.Match[str] | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None

    # Prefer explicit mixed/part-solid terms before GGO/subsolid.
    match = _CT_PART_SOLID_EXPLICIT_RE.search(raw)
    if match:
        return "Part-solid", match

    match = _CT_GROUND_GLASS_RE.search(raw)
    if match:
        return "Ground-glass", match

    match = _CT_SUBSOLID_RE.search(raw)
    if match:
        return "Part-solid", match

    match = _CT_CAVITARY_RE.search(raw)
    if match:
        return "Cavitary", match

    match = _CT_CALCIFIED_RE.search(raw)
    if match:
        return "Calcified", match

    match = _CT_SOLID_CONTEXT_RE.search(raw)
    if match:
        return "Solid", match

    return None, None


def _extract_distance_from_pleura_mm(text: str) -> tuple[float | None, re.Match[str] | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None

    match = _PLEURAL_ABUTTING_RE.search(raw)
    if match:
        return 0.0, match

    match = _PLEURAL_DISTANCE_RE.search(raw)
    if not match:
        return None, None

    try:
        val = float(match.group("val"))
    except Exception:
        return None, None
    unit = (match.group("unit") or "").lower()
    if unit == "cm":
        val *= 10.0
    if val < 0:
        return None, None
    return val, match


def _extract_air_bronchogram_present(text: str) -> tuple[bool | None, re.Match[str] | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None

    match = _AIR_BRONCHOGRAM_NEG_RE.search(raw)
    if match:
        return False, match

    match = _AIR_BRONCHOGRAM_POS_RE.search(raw)
    if match:
        return True, match

    return None, None


def _extract_pet_suv_max(text: str) -> tuple[float | None, re.Match[str] | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None

    match = _SUV_RE.search(raw)
    if not match:
        return None, None
    try:
        value = float(match.group(1))
    except Exception:
        return None, None
    if value < 0:
        return None, None
    return value, match


def _extract_bronchus_sign(text: str) -> tuple[bool | None, re.Match[str] | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None

    if _BRONCHUS_SIGN_NOT_ASSESSED_RE.search(raw):
        return None, None

    match = _BRONCHUS_SIGN_NEG_RE.search(raw)
    if match:
        return False, match

    match = _BRONCHUS_SIGN_POS_RE.search(raw)
    if match:
        return True, match

    return None, None


def _extract_registration_error_mm(text: str) -> tuple[float | None, re.Match[str] | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None

    match = _REGISTRATION_ERROR_RE.search(raw)
    if match:
        reg_err = _coerce_float(match.group(1))
        if reg_err is not None:
            return reg_err, match

    if re.search(r"(?i)\bregistration\b", raw):
        match2 = _REGISTRATION_ERROR_FALLBACK_RE.search(raw)
        if match2:
            reg_err = _coerce_float(match2.group(1))
            if reg_err is not None:
                return reg_err, match2

    return None, None


def _match_til_confirmation_method(text: str) -> tuple[str | None, re.Match[str] | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None

    # Priority: CBCT > augmented fluoro > fluoro > rEBUS.
    for label, pattern in (
        ("CBCT", _TIL_METHOD_CBCT_RE),
        ("Augmented fluoroscopy", _TIL_METHOD_AUG_FLUORO_RE),
        ("Fluoroscopy", _TIL_METHOD_FLUORO_RE),
        ("Radial EBUS", _TIL_METHOD_REBUS_RE),
    ):
        match = pattern.search(raw)
        if match:
            return label, match

    return None, None


def _extract_tool_in_lesion_confirmation(
    text: str,
) -> tuple[bool | None, re.Match[str] | None, str | None, dict[str, object] | None]:
    """Extract tool-in-lesion confirmation (explicit-only) plus method (when present).

    Returns:
      (tool_in_lesion_confirmed, til_match, confirmation_method, method_evidence_meta)
    """
    raw = (text or "").strip()
    if not raw:
        return None, None, None, None

    neg = _TIL_NEG_RE.search(raw)
    if neg:
        return False, neg, None, None

    pos = _TIL_POS_RE.search(raw)
    if not pos:
        return None, None, None, None

    window_start = max(0, pos.start() - 120)
    window_end = min(len(raw), pos.end() + 240)
    window = raw[window_start:window_end]
    method, method_match = _match_til_confirmation_method(window)

    method_meta: dict[str, object] | None = None
    if method and method_match:
        snippet = (method_match.group(0) or "").strip()
        if snippet:
            method_meta = {
                "text": snippet,
                "start": int(window_start + method_match.start()),
                "end": int(window_start + method_match.end()),
            }

    return True, pos, method, method_meta


_INLINE_TARGET_STOP_WORDS: tuple[str, ...] = (
    "PROCEDURE",
    "INDICATION",
    "TECHNIQUE",
    "DESCRIPTION",
)


def _trim_at_stop_words(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    for stop_word in _INLINE_TARGET_STOP_WORDS:
        match = re.search(rf"(?i)\b{re.escape(stop_word)}\b", text)
        if match:
            text = text[: match.start()].strip()
    return text


def _truncate_location(value: str, *, max_len: int = 100) -> str:
    """Truncate location strings to a safe length (prevents accidental whole-note capture)."""
    cleaned = _trim_at_stop_words(value)
    if len(cleaned) <= max_len:
        return cleaned
    clipped = cleaned[:max_len].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = cleaned[:max_len].strip()
    return clipped


def _maybe_unescape_newlines(text: str) -> str:
    """Convert literal '\\n'/'\\r' sequences into real newlines when the note looks escaped."""
    raw = text or ""
    if not raw.strip():
        return raw
    if "\n" in raw or "\r" in raw:
        return raw
    if "\\n" not in raw and "\\r" not in raw:
        return raw
    return raw.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def _infer_lobe_from_text(text: str) -> str | None:
    upper = (text or "").upper()
    for token in ("RUL", "RML", "RLL", "LUL", "LLL"):
        if re.search(rf"\b{token}\b", upper):
            return token
    if "LINGULA" in upper:
        return "Lingula"
    if "LEFT UPPER" in upper:
        return "LUL"
    if "LEFT LOWER" in upper:
        return "LLL"
    if "RIGHT UPPER" in upper:
        return "RUL"
    if "RIGHT MIDDLE" in upper:
        return "RML"
    if "RIGHT LOWER" in upper:
        return "RLL"
    return None


def _extract_rose_result(text: str) -> str | None:
    raw = text or ""
    if not raw.strip():
        return None
    inline = _ROSE_LINE_RE.search(raw)
    if inline:
        result = (inline.group("result") or "").strip()
        return result or None

    header = _ROSE_HEADER_RE.search(raw)
    if not header:
        return None

    # Common template: "ROSE Result:" on its own line, then the result on the next non-empty line.
    tail = raw[header.end() :]
    for line in tail.splitlines()[:6]:
        candidate = (line or "").strip()
        if candidate:
            return candidate
    return None


def _fiducial_in_section(section_text: str) -> tuple[bool, str | None]:
    """Return (placed, details) based on a conservative fiducial placement check."""
    match = _FIDUCIAL_SENTENCE_RE.search(section_text or "")
    if not match:
        return False, None
    sentence = (match.group(1) or "").strip()
    if not sentence:
        return False, None
    if not _FIDUCIAL_ACTION_RE.search(sentence):
        return False, None
    if _NEGATION_RE.search(sentence):
        return False, None
    return True, sentence


def extract_navigation_targets(note_text: str) -> list[dict[str, Any]]:
    """Extract per-target navigation data from common '... LOBE TARGET' headings.

    Returns a list of dicts compatible with granular_data.navigation_targets.
    """
    text = note_text or ""
    if not text.strip():
        return []

    scan_text = _maybe_unescape_newlines(text)

    matches = list(_TARGET_HEADER_RE.finditer(scan_text))
    match_mode = "lobe_target"
    if not matches:
        matches = list(_NODULE_TARGET_HEADER_RE.finditer(scan_text))
        match_mode = "nodule_header"
    if not matches:
        matches = list(_NUMBERED_TARGET_HEADER_RE.finditer(scan_text))
        match_mode = "numbered_target"
    if not matches:
        # Fallback: support inline patterns like "Target: 20mm nodule in LLL" without
        # relying on explicit "... LOBE TARGET" section headings.
        targets: list[dict[str, Any]] = []
        for match in _INLINE_TARGET_RE.finditer(scan_text):
            raw_loc_full_raw = _trim_at_stop_words(match.group("loc") or "")
            leading_trim = len(raw_loc_full_raw) - len(raw_loc_full_raw.lstrip())
            raw_loc_full = raw_loc_full_raw.strip()
            loc_offset = match.start("loc") + leading_trim
            raw_loc = _truncate_location(raw_loc_full)
            if not raw_loc:
                continue
            if len(raw_loc) <= 2:
                continue

            target: dict[str, Any] = {
                "target_number": len(targets) + 1,
                "target_location_text": raw_loc,
            }
            evidence: dict[str, dict[str, object]] = {}

            lobe = _infer_lobe_from_text(raw_loc_full)
            if lobe:
                target["target_lobe"] = lobe

            lesion_size_mm: float | None = None
            size_match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*cm\b", raw_loc_full)
            if size_match:
                lesion_size_mm = _coerce_float(size_match.group(1))
                if lesion_size_mm is not None:
                    lesion_size_mm *= 10.0
            else:
                size_match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*mm\b", raw_loc_full)
                if size_match:
                    lesion_size_mm = _coerce_float(size_match.group(1))
            if lesion_size_mm is not None:
                target["lesion_size_mm"] = lesion_size_mm

            ct_char, ct_match = _match_ct_characteristics(raw_loc_full)
            if ct_char:
                target["ct_characteristics"] = ct_char
                if ct_match:
                    evidence["ct_characteristics"] = {
                        "text": ct_match.group(0).strip(),
                        "start": int(loc_offset + ct_match.start()),
                        "end": int(loc_offset + ct_match.end()),
                    }

            pleural_mm, pleural_match = _extract_distance_from_pleura_mm(raw_loc_full)
            if pleural_mm is not None:
                target["distance_from_pleura_mm"] = pleural_mm
                if pleural_match:
                    evidence["distance_from_pleura_mm"] = {
                        "text": pleural_match.group(0).strip(),
                        "start": int(loc_offset + pleural_match.start()),
                        "end": int(loc_offset + pleural_match.end()),
                    }

            suv_max, suv_match = _extract_pet_suv_max(raw_loc_full)
            if suv_max is not None:
                target["pet_suv_max"] = suv_max
                if suv_match:
                    evidence["pet_suv_max"] = {
                        "text": suv_match.group(0).strip(),
                        "start": int(loc_offset + suv_match.start()),
                        "end": int(loc_offset + suv_match.end()),
                    }

            air_bronch, air_match = _extract_air_bronchogram_present(raw_loc_full)
            if air_bronch is not None:
                target["air_bronchogram_present"] = air_bronch
                if air_match:
                    evidence["air_bronchogram_present"] = {
                        "text": air_match.group(0).strip(),
                        "start": int(loc_offset + air_match.start()),
                        "end": int(loc_offset + air_match.end()),
                    }

            bronchus_sign, bs_match = _extract_bronchus_sign(raw_loc_full)
            if bronchus_sign is not None:
                target["bronchus_sign"] = bronchus_sign
                if bs_match:
                    evidence["bronchus_sign"] = {
                        "text": bs_match.group(0).strip(),
                        "start": int(loc_offset + bs_match.start()),
                        "end": int(loc_offset + bs_match.end()),
                    }

            til_confirmed, til_match, til_method, til_method_meta = _extract_tool_in_lesion_confirmation(raw_loc_full)
            if til_confirmed is not None:
                target["tool_in_lesion_confirmed"] = til_confirmed
                if til_match:
                    evidence["tool_in_lesion_confirmed"] = {
                        "text": til_match.group(0).strip(),
                        "start": int(loc_offset + til_match.start()),
                        "end": int(loc_offset + til_match.end()),
                    }
                if til_confirmed is True and til_method:
                    target["confirmation_method"] = til_method
                    if til_method_meta:
                        evidence["confirmation_method"] = {
                            "text": str(til_method_meta.get("text") or ""),
                            "start": int(loc_offset + int(til_method_meta.get("start") or 0)),
                            "end": int(loc_offset + int(til_method_meta.get("end") or 0)),
                        }
                    if til_method == "CBCT":
                        target["cbct_til_confirmed"] = True

            if evidence:
                target["_evidence"] = evidence

            targets.append(target)

        # When there is a single target, enrich it with global navigation details found elsewhere
        # in the note (segment/bronchus, registration error, rEBUS view, sampling tools/counts).
        if len(targets) == 1:
            target = targets[0]
            lobe = _normalize_lobe(target.get("target_lobe")) or _infer_lobe_from_text(
                str(target.get("target_location_text") or "")
            )

            seg_match = _TARGET_SEGMENT_OF_LOBE_RE.search(scan_text)
            if seg_match:
                seg_lobe = _normalize_lobe(seg_match.group("lobe")) or lobe
                segment = (seg_match.group("segment") or "").strip() or None
                if seg_lobe:
                    target["target_lobe"] = seg_lobe
                    lobe = seg_lobe
                if segment:
                    target["target_segment"] = segment

            bronchus: str | None = None
            bronchus_matches = [m.group(1).upper() for m in _BRONCHUS_CODE_RE.finditer(scan_text)]
            if bronchus_matches and lobe:
                desired_prefix = "L" if (lobe.startswith("L") or lobe == "Lingula") else "R"
                bronchus = next((b for b in bronchus_matches if b.startswith(desired_prefix)), None)
            elif bronchus_matches:
                bronchus = bronchus_matches[0]

            if lobe and target.get("target_segment"):
                segment = str(target.get("target_segment") or "").strip()
                if bronchus:
                    target["target_location_text"] = f"{lobe} ({bronchus} {segment})"
                else:
                    target["target_location_text"] = f"{lobe} ({segment})"

            if not target.get("ct_characteristics"):
                ct_char = _detect_ct_characteristics(str(target.get("target_location_text") or ""))
                if ct_char:
                    target["ct_characteristics"] = ct_char

            reg_err, reg_match = _extract_registration_error_mm(scan_text)
            if reg_err is not None:
                target["registration_error_mm"] = reg_err
                if reg_match:
                    existing_evidence = target.get("_evidence") if isinstance(target.get("_evidence"), dict) else {}
                    if isinstance(existing_evidence, dict) and "registration_error_mm" not in existing_evidence:
                        existing_evidence["registration_error_mm"] = {
                            "text": reg_match.group(0).strip(),
                            "start": int(reg_match.start()),
                            "end": int(reg_match.end()),
                        }
                        target["_evidence"] = existing_evidence

            rebus_match = _REBUS_VIEW_RE.search(scan_text)
            if rebus_match:
                view = (rebus_match.group(1) or "").strip().title()
                if view:
                    target["rebus_used"] = True
                    target["rebus_view"] = view

            tools: list[str] = []
            tbna_match = re.search(r"(?i)\btransbronchial\s+needle\s+aspiration\b|\btbna\b", scan_text)
            if tbna_match:
                gauge_match = _NEEDLE_GAUGE_RE.search(scan_text[tbna_match.start() : tbna_match.start() + 500])
                gauge = gauge_match.group(1) if gauge_match else None
                tools.append(f"Needle ({gauge}G)" if gauge else "Needle")

                passes_match = _PASSES_RE.search(scan_text[tbna_match.start() : tbna_match.start() + 500])
                if passes_match:
                    try:
                        target["number_of_needle_passes"] = int(passes_match.group(1))
                    except Exception:
                        pass

            cryo_match = _CRYO_RE.search(scan_text)
            if cryo_match:
                window = scan_text[cryo_match.start() : cryo_match.start() + 700]
                probe_size = None
                probe_match = _CRYO_PROBE_SIZE_RE.search(window)
                if probe_match:
                    probe_size = probe_match.group(1)
                if probe_size:
                    tools.append(f"Cryoprobe ({probe_size}mm)")
                else:
                    tools.append("Cryoprobe")

                # Best-effort count for cryobiopsy samples (supports digits or small number words).
                count = None
                m_total = _TOTAL_SAMPLES_RE.search(window)
                if m_total:
                    count = _coerce_int(m_total.group(1))
                if count is None:
                    m_count = re.search(
                        r"(?i)\b(?P<count>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
                        r"[^.\n]{0,40}\b(?:samples?|specimens?|biops(?:y|ies))\b",
                        window,
                    )
                    if m_count:
                        count = _coerce_int(m_count.group("count"))
                if isinstance(count, int):
                    target["number_of_cryo_biopsies"] = count

            forceps_match = re.search(r"(?i)\bforceps\s+biops(?:y|ies)\b|\btransbronchial\s+forceps\s+biops(?:y|ies)\b", scan_text)
            if forceps_match:
                tools.append("Forceps")
                window = scan_text[forceps_match.start() : forceps_match.start() + 500]
                spec_match = _SPECIMEN_COUNT_RE.search(window)
                if spec_match:
                    try:
                        target["number_of_forceps_biopsies"] = int(spec_match.group(1))
                    except Exception:
                        pass

            brush_match = re.search(r"(?i)\bbrushings?\b|\bcytology\s+brush\b", scan_text)
            if brush_match:
                tools.append("Brush")

            if tools:
                target["sampling_tools_used"] = tools

            rose_text = _extract_rose_result(scan_text)
            if rose_text:
                target["rose_performed"] = True
                target["rose_result"] = rose_text[:240]

        if targets:
            return targets

        # Fallback: some templates use navigation prose ("... used to engage the <segment> of RLL (RB6)")
        # without explicit "... TARGET" headings or inline "Target:" markers. Treat each distinct "engage"
        # location as a navigation target.
        engage_matches = list(_ENGAGE_LOCATION_RE.finditer(scan_text))
        if not engage_matches:
            return []

        seen_locations: set[str] = set()
        for engage in engage_matches:
            segment = (engage.group("segment") or "").strip() or None
            lobe = _normalize_lobe(engage.group("lobe")) or None
            bronchus = (engage.group("bronchus") or "").strip()

            if segment and lobe and bronchus:
                location_text = f"{segment} of {lobe} ({bronchus})"
            elif segment and lobe:
                location_text = f"{segment} of {lobe}"
            elif lobe:
                location_text = f"{lobe} target"
            else:
                location_text = "Unknown target"

            location_text = _truncate_location(location_text)
            if not location_text:
                continue

            # Deduplicate identical locations (common when notes repeat the same "engage ..." sentence).
            if location_text.lower() in seen_locations:
                continue
            seen_locations.add(location_text.lower())

            # Local window: look for lesion size near the engage statement.
            window = scan_text[engage.start() : min(len(scan_text), engage.start() + 800)]
            lesion_size_mm: float | None = None
            cm = _TARGET_LESION_SIZE_CM_RE.search(window)
            if cm:
                lesion_size_mm = _coerce_float(cm.group(1))
                if lesion_size_mm is not None:
                    lesion_size_mm *= 10.0
            else:
                mm = _TARGET_LESION_SIZE_MM_RE.search(window)
                if mm:
                    lesion_size_mm = _coerce_float(mm.group(1))

            target: dict[str, Any] = {
                "target_number": len(targets) + 1,
                "target_location_text": location_text,
            }
            if lobe:
                target["target_lobe"] = lobe
            if segment:
                target["target_segment"] = segment
            if lesion_size_mm is not None:
                target["lesion_size_mm"] = lesion_size_mm

            ct_char = _detect_ct_characteristics(window)
            if ct_char:
                target["ct_characteristics"] = ct_char

            targets.append(target)

        return targets

    def _header_label_suffix(header_text: str) -> str | None:
        raw = (header_text or "").strip()
        if not raw:
            return None
        # Prefer explicit "nodule #2" style identifiers.
        m = re.search(r"(?i)\b(?:nodule|lesion)\s*#\s*(\d{1,2})\b", raw)
        if m:
            return f"nodule #{m.group(1)}"
        # Common template: "(labeled as RUL #2)".
        m = re.search(r"(?i)\blabeled\s+as\s+[A-Z]{2,6}\s*#\s*(\d{1,2})\b", raw)
        if m:
            return f"nodule #{m.group(1)}"
        # Last resort: any "# <num>" marker.
        m = re.search(r"(?i)#\s*(\d{1,2})\b", raw)
        if m:
            return f"nodule #{m.group(1)}"
        return None

    targets: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        header_raw = match.group("header") or ""
        header = _canonical_header(header_raw)
        section_start = match.end()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(scan_text)
        section = scan_text[section_start:section_end] if section_end > section_start else ""
        section_leading_trim = len(section) - len(section.lstrip())
        section_offset = section_start + section_leading_trim
        section_text = section.strip()

        header_leading_trim = len(header_raw) - len(header_raw.lstrip())
        header_offset = match.start("header") + header_leading_trim
        header_text = header_raw.strip()

        lobe = _TARGET_LOBE_FROM_HEADER.get(header)
        if match_mode in {"nodule_header", "numbered_target"}:
            lobe = _infer_lobe_from_text(header_raw) or lobe

        segment: str | None = None
        location_text: str | None = header_raw.strip() if match_mode == "numbered_target" else None

        engage = _ENGAGE_LOCATION_RE.search(section)
        if engage:
            segment = (engage.group("segment") or "").strip() or None
            lobe = _normalize_lobe(engage.group("lobe")) or lobe
            bronchus = (engage.group("bronchus") or "").strip()
            if segment and lobe and bronchus:
                location_text = f"{segment} of {lobe} ({bronchus})"
            elif segment and lobe:
                location_text = f"{segment} of {lobe}"

        if not location_text and lobe:
            # Prefer a meaningful sentence when present; otherwise fall back to the header lobe.
            target_line = _first_line_containing(section, re.compile(r"(?i)\btarget\s+lesion\b"))
            location_text = target_line or f"{lobe} target"

        # When we have rich nodule headers, include the nodule identifier to avoid collapsing
        # multiple targets that share the same segment (common in RUL RB1 #1/#2 workflows).
        if match_mode == "nodule_header" and location_text:
            suffix = _header_label_suffix(header_raw)
            if suffix and suffix.lower() not in location_text.lower():
                location_text = f"{location_text} {suffix}"

        lesion_size_mm: float | None = None
        cm = _TARGET_LESION_SIZE_CM_RE.search(section)
        if cm:
            lesion_size_mm = _coerce_float(cm.group(1))
            if lesion_size_mm is not None:
                lesion_size_mm *= 10.0
        else:
            mm = _TARGET_LESION_SIZE_MM_RE.search(section)
            if mm:
                lesion_size_mm = _coerce_float(mm.group(1))

        if lesion_size_mm is None and match_mode == "numbered_target":
            size_match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*cm\b", header_raw)
            if size_match:
                lesion_size_mm = _coerce_float(size_match.group(1))
                if lesion_size_mm is not None:
                    lesion_size_mm *= 10.0
            else:
                size_match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*mm\b", header_raw)
                if size_match:
                    lesion_size_mm = _coerce_float(size_match.group(1))

        rebus_view: str | None = None
        rebus_match = _REBUS_VIEW_RE.search(section_text)
        if rebus_match:
            view = (rebus_match.group(1) or "").strip().title()
            rebus_view = view or None

        fiducial_placed, fiducial_details = _fiducial_in_section(section_text)

        evidence: dict[str, dict[str, object]] = {}

        target: dict[str, Any] = {
            "target_number": idx + 1,
            "target_location_text": _truncate_location(location_text or "Unknown target"),
        }
        if lobe:
            target["target_lobe"] = lobe
        if segment:
            target["target_segment"] = segment
        if lesion_size_mm is not None:
            target["lesion_size_mm"] = lesion_size_mm
        if rebus_view is not None:
            target["rebus_used"] = True
            target["rebus_view"] = rebus_view

        ct_char, ct_match = _match_ct_characteristics(section_text)
        ct_base_offset = section_offset
        if ct_char is None and match_mode in {"nodule_header", "numbered_target"}:
            ct_char, ct_match = _match_ct_characteristics(header_text)
            ct_base_offset = header_offset
        if ct_char is not None:
            target["ct_characteristics"] = ct_char
            if ct_match:
                evidence["ct_characteristics"] = {
                    "text": ct_match.group(0).strip(),
                    "start": int(ct_base_offset + ct_match.start()),
                    "end": int(ct_base_offset + ct_match.end()),
                }

        pleural_mm, pleural_match = _extract_distance_from_pleura_mm(section_text)
        pleural_base_offset = section_offset
        if pleural_mm is None and match_mode in {"nodule_header", "numbered_target"}:
            pleural_mm, pleural_match = _extract_distance_from_pleura_mm(header_text)
            pleural_base_offset = header_offset
        if pleural_mm is not None:
            target["distance_from_pleura_mm"] = pleural_mm
            if pleural_match:
                evidence["distance_from_pleura_mm"] = {
                    "text": pleural_match.group(0).strip(),
                    "start": int(pleural_base_offset + pleural_match.start()),
                    "end": int(pleural_base_offset + pleural_match.end()),
                }

        suv_max, suv_match = _extract_pet_suv_max(section_text)
        suv_base_offset = section_offset
        if suv_max is None and match_mode in {"nodule_header", "numbered_target"}:
            suv_max, suv_match = _extract_pet_suv_max(header_text)
            suv_base_offset = header_offset
        if suv_max is not None:
            target["pet_suv_max"] = suv_max
            if suv_match:
                evidence["pet_suv_max"] = {
                    "text": suv_match.group(0).strip(),
                    "start": int(suv_base_offset + suv_match.start()),
                    "end": int(suv_base_offset + suv_match.end()),
                }

        air_bronch, air_match = _extract_air_bronchogram_present(section_text)
        air_base_offset = section_offset
        if air_bronch is None and match_mode in {"nodule_header", "numbered_target"}:
            air_bronch, air_match = _extract_air_bronchogram_present(header_text)
            air_base_offset = header_offset
        if air_bronch is not None:
            target["air_bronchogram_present"] = air_bronch
            if air_match:
                evidence["air_bronchogram_present"] = {
                    "text": air_match.group(0).strip(),
                    "start": int(air_base_offset + air_match.start()),
                    "end": int(air_base_offset + air_match.end()),
                }

        bronchus_sign, bs_match = _extract_bronchus_sign(section_text)
        bs_base_offset = section_offset
        if bronchus_sign is None and match_mode in {"nodule_header", "numbered_target"}:
            bronchus_sign, bs_match = _extract_bronchus_sign(header_text)
            bs_base_offset = header_offset
        if bronchus_sign is not None:
            target["bronchus_sign"] = bronchus_sign
            if bs_match:
                evidence["bronchus_sign"] = {
                    "text": bs_match.group(0).strip(),
                    "start": int(bs_base_offset + bs_match.start()),
                    "end": int(bs_base_offset + bs_match.end()),
                }

        reg_err, reg_match = _extract_registration_error_mm(section_text)
        reg_base_offset = section_offset
        if reg_err is None and match_mode in {"nodule_header", "numbered_target"}:
            reg_err, reg_match = _extract_registration_error_mm(header_text)
            reg_base_offset = header_offset
        if reg_err is not None:
            target["registration_error_mm"] = reg_err
            if reg_match:
                evidence["registration_error_mm"] = {
                    "text": reg_match.group(0).strip(),
                    "start": int(reg_base_offset + reg_match.start()),
                    "end": int(reg_base_offset + reg_match.end()),
                }

        til_confirmed, til_match, til_method, til_method_meta = _extract_tool_in_lesion_confirmation(section_text)
        til_base_offset = section_offset
        if til_confirmed is None and match_mode in {"nodule_header", "numbered_target"}:
            til_confirmed, til_match, til_method, til_method_meta = _extract_tool_in_lesion_confirmation(header_text)
            til_base_offset = header_offset
        if til_confirmed is not None:
            target["tool_in_lesion_confirmed"] = til_confirmed
            if til_match:
                evidence["tool_in_lesion_confirmed"] = {
                    "text": til_match.group(0).strip(),
                    "start": int(til_base_offset + til_match.start()),
                    "end": int(til_base_offset + til_match.end()),
                }
            if til_confirmed is True and til_method:
                target["confirmation_method"] = til_method
                if til_method_meta:
                    evidence["confirmation_method"] = {
                        "text": str(til_method_meta.get("text") or ""),
                        "start": int(til_base_offset + int(til_method_meta.get("start") or 0)),
                        "end": int(til_base_offset + int(til_method_meta.get("end") or 0)),
                    }
                if til_method == "CBCT":
                    target["cbct_til_confirmed"] = True

        if evidence:
            target["_evidence"] = evidence
        if fiducial_placed:
            target["fiducial_marker_placed"] = True
        if fiducial_details:
            target["fiducial_marker_details"] = fiducial_details

        # Light-touch sampling hints (used for downstream aggregation).
        cryo_match = _CRYO_RE.search(section_text)
        if cryo_match:
            target.setdefault("sampling_tools_used", []).append("Cryoprobe")
            window = section_text[cryo_match.start() : cryo_match.start() + 600]
            samples = _TOTAL_SAMPLES_RE.search(window)
            if samples:
                try:
                    target["number_of_cryo_biopsies"] = int(samples.group(1))
                except Exception:
                    pass

        tbna_match = re.search(r"(?i)\btransbronchial\s+needle\s+aspiration\b|\btbna\b", section_text)
        if tbna_match:
            target.setdefault("sampling_tools_used", []).append("Needle")
            window = section_text[tbna_match.start() : tbna_match.start() + 600]
            samples = _TOTAL_SAMPLES_RE.search(window)
            if samples:
                try:
                    target["number_of_needle_passes"] = int(samples.group(1))
                except Exception:
                    pass

        brush_match = re.search(
            r"(?i)\b(?:transbronchial\s+)?brush(?:ing|ings)?\b|\bcytology\s+brush\b",
            section_text,
        )
        if brush_match:
            target.setdefault("sampling_tools_used", []).append("Brush")

        forceps_match = re.search(
            r"(?i)\bforceps\s+biops(?:y|ies)\b|\btransbronchial\s+forceps\s+biops(?:y|ies)\b",
            section_text,
        )
        if forceps_match:
            target.setdefault("sampling_tools_used", []).append("Forceps")
            window = section_text[forceps_match.start() : forceps_match.start() + 500]
            spec_match = _SPECIMEN_COUNT_RE.search(window)
            if spec_match:
                try:
                    target["number_of_forceps_biopsies"] = int(spec_match.group(1))
                except Exception:
                    pass

        targets.append(target)

    return targets


def extract_cryobiopsy_sites(note_text: str) -> list[dict[str, Any]]:
    """Extract per-site cryobiopsy details from target sections.

    Returns a list of dicts compatible with granular_data.cryobiopsy_sites.
    """
    text = note_text or ""
    if not text.strip():
        return []

    targets = extract_navigation_targets(text)
    if not targets:
        return []

    sites: list[dict[str, Any]] = []
    for target in targets:
        lobe = _normalize_lobe(target.get("target_lobe"))
        if not lobe:
            continue

        # Pull the section associated with this target by re-finding it; keep logic simple and
        # only use per-target cryo flags already detected in extract_navigation_targets.
        if "Cryoprobe" not in (target.get("sampling_tools_used") or []):
            continue

        # Best-effort parse by scanning the whole note for cryo details; when multiple targets
        # exist, these are often uniform (probe size/freeze time).
        probe_size = None
        probe = _CRYO_PROBE_SIZE_RE.search(text)
        if probe:
            probe_size = _coerce_float(probe.group(1))
        freeze = None
        freeze_match = _CRYO_FREEZE_RE.search(text)
        if freeze_match:
            try:
                freeze = int(freeze_match.group(1))
            except Exception:
                freeze = None

        biopsies = target.get("number_of_cryo_biopsies")
        if biopsies is not None:
            try:
                biopsies = int(biopsies)
            except Exception:
                biopsies = None

        site: dict[str, Any] = {
            "site_number": len(sites) + 1,
            "lobe": lobe,
        }
        segment = target.get("target_segment")
        if isinstance(segment, str) and segment.strip():
            site["segment"] = segment.strip()
        if probe_size in {1.1, 1.7, 1.9, 2.4}:
            site["probe_size_mm"] = probe_size
        if isinstance(freeze, int):
            site["freeze_time_seconds"] = freeze
        if isinstance(biopsies, int):
            site["number_of_biopsies"] = biopsies
        sites.append(site)

    return sites


__all__ = ["extract_navigation_targets", "extract_cryobiopsy_sites"]
