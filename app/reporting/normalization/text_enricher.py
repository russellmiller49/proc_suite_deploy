from __future__ import annotations

import re
from typing import Any

from proc_schemas.clinical import ProcedureBundle

from .types import NormalizationNote


def extract_lung_location_hint(text: str) -> str | None:
    """Best-effort location from free text (lobe shorthand)."""
    if not text:
        return None
    upper = text.upper()
    for token in ("RUL", "RML", "RLL", "LUL", "LLL"):
        if re.search(rf"\b{token}\b", upper):
            return token
    if "RIGHT UPPER LOBE" in upper:
        return "RUL"
    if "RIGHT MIDDLE LOBE" in upper:
        return "RML"
    if "RIGHT LOWER LOBE" in upper:
        return "RLL"
    if "LEFT UPPER LOBE" in upper:
        return "LUL"
    if "LEFT LOWER LOBE" in upper:
        return "LLL"
    return None


def extract_bronch_segment_hint(text: str) -> str | None:
    """Best-effort bronchopulmonary segment token (e.g., RB10, LB6, B6)."""
    if not text:
        return None
    upper = text.upper()
    match = re.search(r"\b([RL]B\d{1,2})\b", upper)
    if match:
        return match.group(1)
    match = re.search(r"\bB\d{1,2}\b", upper)
    if match:
        return match.group(0)
    return None


def infer_rebus_pattern(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    if "concentric" in lowered:
        return "Concentric"
    if "eccentric" in lowered:
        return "Eccentric"
    if "adjacent" in lowered:
        return "Adjacent"
    return None


def parse_count(text: str, pattern: str) -> int | None:
    """Parse shorthand counts like 'TBNA x4' or 'Bx x 6'."""
    if not text:
        return None
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except Exception:
        return None
    return value if value >= 0 else None


def parse_operator(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"(?im)^\s*(?:operator|attending)\s*:\s*([^\n]+?)\s*$", text)
    if not match:
        return None
    value = match.group(1).strip()
    value = value.replace("[", "").replace("]", "").strip()
    if not value or value.strip().lower() in ("redacted", "unknown", "n/a", "na"):
        return None
    return value or None


def parse_referred_physician(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"(?im)^\s*(?:cc\s*)?referred\s+physician\s*:\s*([^\n]+?)\s*$", text)
    if not match:
        return None
    value = match.group(1).strip()
    value = value.replace("[", "").replace("]", "").strip()
    if not value or value.strip().lower() in ("redacted", "unknown", "n/a", "na"):
        return None
    return value or None


def parse_service_date(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"(?im)^\s*(?:service\s*date|date\s+of\s+procedure)\s*:\s*([^\n]+?)\s*$", text)
    if not match:
        return None
    value = match.group(1).strip()
    value = value.replace("[", "").replace("]", "").strip()
    if not value or value.strip().lower() in ("redacted", "unknown", "n/a", "na"):
        return None
    return value or None


def parse_bracket_sections(text: str) -> dict[str, str]:
    if not text:
        return {}
    pattern = re.compile(r"(?im)^\s*\[(indication|anesthesia|description|complication(?:s)?|plan)\]\s*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = str(match.group(1) or "").strip().lower()
        if key.startswith("complication"):
            key = "complications"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if body:
            sections[key] = body
    return sections


def infer_side(text: str) -> str | None:
    if not text:
        return None
    has_right = bool(re.search(r"(?i)\bright\b", text))
    has_left = bool(re.search(r"(?i)\bleft\b", text))
    if has_right and not has_left:
        return "right"
    if has_left and not has_right:
        return "left"
    return None


def parse_wll_volumes_liters(text: str) -> tuple[float | None, float | None]:
    if not text:
        return None, None
    match = re.search(
        r"(?i)\b(\d+(?:\.\d+)?)\s*L\s*IN\b\s*/\s*(\d+(?:\.\d+)?)\s*L\s*OUT\b",
        text,
    )
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except Exception:
            return None, None

    instilled = None
    returned = None

    match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*L\s*(?:IN|INSTILLED)\b", text)
    if match:
        try:
            instilled = float(match.group(1))
        except Exception:
            instilled = None
    match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*L\s*(?:OUT|RETURN(?:ED)?)\b", text)
    if match:
        try:
            returned = float(match.group(1))
        except Exception:
            returned = None

    if instilled is None:
        match = re.search(r"(?i)\b(?:in|instilled)\s*(\d+(?:\.\d+)?)\s*L\b", text)
        if match:
            try:
                instilled = float(match.group(1))
            except Exception:
                instilled = None
    if returned is None:
        match = re.search(r"(?i)\b(?:out|returned)\s*(\d+(?:\.\d+)?)\s*L\b", text)
        if match:
            try:
                returned = float(match.group(1))
            except Exception:
                returned = None

    return instilled, returned


def parse_obstruction_pre_post(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"(?i)(\d{1,3})\s*%\s*(?:->|→)\s*(\d{1,3})\s*%", text)
    if match:
        try:
            return int(match.group(1)), int(match.group(2))
        except Exception:
            return None, None

    pre = None
    post = None
    match = re.search(r"(?i)\bpre[-\s]*procedure\b[^\n]{0,80}?(\d{1,3})\s*%\b", text)
    if match:
        try:
            pre = int(match.group(1))
        except Exception:
            pre = None
    match = re.search(r"(?i)\bpost[-\s]*procedure\b[^\n]{0,80}?(\d{1,3})\s*%\b", text)
    if match:
        try:
            post = int(match.group(1))
        except Exception:
            post = None
    return pre, post


def parse_ebl_ml(text: str) -> int | None:
    match = re.search(r"(?i)\bebl\s*[:\\-]?\s*(\d{1,4})\s*m\s*l\b", text)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    match = re.search(r"(?i)\bestimated\s+blood\s+loss\b[^\n]{0,40}?(\d{1,4})\s*m\s*l\b", text)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def parse_dilation_sizes_mm(text: str) -> list[int]:
    if not text:
        return []
    match = re.search(r"(?i)\bdilators?\b[^\n]{0,100}?\b((?:\d{1,2}\s*,\s*)*\d{1,2})\s*mm\b", text)
    if match:
        values = [int(v) for v in re.findall(r"\d{1,2}", match.group(1))]
        return [v for v in values if v > 0]

    if re.search(r"(?i)\bdilat", text) and re.search(r"(?i)\bdilator", text):
        values = [int(v) for v in re.findall(r"(?i)\b(\d{1,2})\s*mm\b", text)]
        return [v for v in values if v > 0]
    return []


def parse_post_dilation_diameter_mm(text: str) -> int | None:
    if not text:
        return None
    for pattern in (
        r"(?i)\bpatent\b[^\n]{0,40}?~?\s*(\d{1,2})\s*mm\b",
        r"(?i)\bopened\s+up\s+to\b[^\n]{0,20}?~?\s*(\d{1,2})\s*mm\b",
        r"(?i)\bopen(?:ed)?\b[^\n]{0,40}?\bto\b[^\n]{0,20}?~?\s*(\d{1,2})\s*mm\b",
    ):
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except Exception:
            value = None
        if value and value > 0:
            return value
    return None


def enrich_from_text(bundle: ProcedureBundle, source_text: str) -> tuple[ProcedureBundle, list[NormalizationNote]]:
    """Best-effort enrichment of a ProcedureBundle using scrubbed source text.

    Guardrails:
    - never deletes user-provided data
    - only fills missing fields
    - no rendering/template selection
    """
    if not source_text:
        return bundle, []

    notes: list[NormalizationNote] = []
    payload = bundle.model_dump(exclude_none=False)

    if payload.get("free_text_hint") in (None, ""):
        payload["free_text_hint"] = source_text
        notes.append(
            NormalizationNote(
                kind="text_enrichment",
                path="/free_text_hint",
                message="Filled free_text_hint from source_text",
                source="text_enricher",
            )
        )

    encounter: dict[str, Any] = payload.get("encounter") or {}

    if encounter.get("attending") in (None, ""):
        attending = parse_operator(source_text)
        if attending:
            encounter["attending"] = attending
            notes.append(
                NormalizationNote(
                    kind="text_enrichment",
                    path="/encounter/attending",
                    message="Parsed attending/operator from source_text",
                    source="text_enricher",
                )
            )

    if encounter.get("referred_physician") in (None, ""):
        referred = parse_referred_physician(source_text)
        if referred:
            encounter["referred_physician"] = referred
            notes.append(
                NormalizationNote(
                    kind="text_enrichment",
                    path="/encounter/referred_physician",
                    message="Parsed referred physician from source_text",
                    source="text_enricher",
                )
            )

    if encounter.get("date") in (None, ""):
        svc_date = parse_service_date(source_text)
        if svc_date:
            encounter["date"] = svc_date
            notes.append(
                NormalizationNote(
                    kind="text_enrichment",
                    path="/encounter/date",
                    message="Parsed service date from source_text",
                    source="text_enricher",
                )
            )

    payload["encounter"] = encounter

    bracket_sections = parse_bracket_sections(source_text)
    if bracket_sections:
        if payload.get("indication_text") in (None, "") and bracket_sections.get("indication"):
            payload["indication_text"] = bracket_sections["indication"]
            notes.append(
                NormalizationNote(
                    kind="text_enrichment",
                    path="/indication_text",
                    message="Filled indication_text from bracket sections",
                    source="text_enricher",
                )
            )
        if payload.get("complications_text") in (None, "") and bracket_sections.get("complications"):
            payload["complications_text"] = bracket_sections["complications"]
            notes.append(
                NormalizationNote(
                    kind="text_enrichment",
                    path="/complications_text",
                    message="Filled complications_text from bracket sections",
                    source="text_enricher",
                )
            )
        if payload.get("impression_plan") in (None, "") and bracket_sections.get("plan"):
            payload["impression_plan"] = bracket_sections["plan"]
            notes.append(
                NormalizationNote(
                    kind="text_enrichment",
                    path="/impression_plan",
                    message="Filled impression_plan from bracket sections",
                    source="text_enricher",
                )
            )

    normalized = ProcedureBundle.model_validate(payload)
    return normalized, notes
