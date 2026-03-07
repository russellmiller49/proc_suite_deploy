"""Deterministic extractors for registry fields.

These extractors use regex patterns to reliably extract structured data
that the LLM often misses or extracts incorrectly. They run BEFORE the
LLM extraction and provide seed data that takes precedence.

Targets fields identified as systematically missing in v2.8 validation:
- patient_age, gender
- asa_class
- sedation_type, airway_type
- primary_indication
- disposition
- institution_name
- bleeding_severity
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.common.spans import Span
from app.registry.normalization import (
    normalize_gender,
    normalize_sedation_type,
    normalize_airway_type,
    normalize_asa_class,
    normalize_bleeding_severity,
)


# =============================================================================
# Attribute Regex (for evidence spans)
# =============================================================================

# Needle gauge patterns (e.g., "19G", "19-gauge", "19 gauge")
NEEDLE_GAUGE_RE = re.compile(r"(?i)\b(19|21|22|25)\s*-?\s*g(?:auge)?\b")

# BAL volume patterns (e.g., "instilled 40 cc", "instilled 100ml")
BAL_VOLUME_RE = re.compile(r"(?i)\binstill(?:ed)?\s*(\d{1,4})\s*(?:ml|cc)\b")

# Device size patterns (e.g., "14 x 40 mm", "14x40mm", "1.7 x 2.4 cm")
DEVICE_SIZE_RE = re.compile(r"(?i)\b(\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\s*(?:mm|cm)?)\b")

# Lesion size patterns (e.g., "4.8 x 3.2 cm", "4.8 by 3.2 cm")
LESION_SIZE_RE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(cm|mm)\b")

# Obstruction percent patterns (e.g., "stenosis 80%", "obstruction is 90%")
OBSTRUCTION_PCT_RE = re.compile(r"(?i)\b(?:occlu|obstruct|stenosis)\w*\s*(?:of|is)?\s*(\d{1,3})\s*%")


def is_negated(token_span_start: int, token_span_end: int, text: str) -> bool:
    """Return True when a nearby negation cue applies to the token span.

    Heuristics:
    - Negation cue within ~30 chars before the token.
    - Negation cue within ~5 tokens before the token.
    """
    raw = text or ""
    if not raw.strip():
        return False
    start = max(0, int(token_span_start or 0))
    end = max(start, int(token_span_end or start))

    # Keep negation scope sentence-local so a previous sentence such as
    # "no secretions." does not negate the next sentence's finding.
    sentence_boundary = max(raw.rfind(".", 0, start), raw.rfind("\n", 0, start), raw.rfind(";", 0, start))
    local_start = sentence_boundary + 1 if sentence_boundary >= 0 else 0

    char_window = raw[max(local_start, start - 60) : start]
    if re.search(r"(?i)\b(?:no|not|without|absent|absence\s+of|negative\s+for|denies?)\b[^.\n]{0,30}$", char_window):
        return True

    # Token-window guardrail: inspect the last ~5 tokens before the span.
    prefix = raw[max(local_start, start - 160) : start]
    tokens = re.findall(r"\b[\w'-]+\b", prefix.lower())
    if tokens:
        recent = tokens[-5:]
        if any(tok in {"no", "not", "without", "absent", "absence", "denies", "denied"} for tok in recent):
            return True

    # Inline "no <term>" style at exact start boundary.
    inline = raw[max(local_start, start - 40) : min(len(raw), end + 10)]
    if re.search(r"(?i)\b(?:no|without|absent)\b[^.\n]{0,24}\b", inline):
        return True
    return False


def extract_demographics(note_text: str) -> Dict[str, Any]:
    """Extract patient age and gender from note header.

    Patterns recognized:
    - "Age: 66 | Sex: M"
    - "52M", "65F"
    - "Date of Birth: 03/15/1959 (65 years old)"
    - "Patient: 72-year-old male"

    Returns:
        Dict with 'patient_age' (int) and 'gender' (str) if found
    """
    result: Dict[str, Any] = {}

    # Pattern 1: "Age: 66 | Sex: M" or "Age: 66, Sex: F"
    age_sex_pattern = r"Age:?\s*(\d+)\s*[|,]\s*Sex:?\s*([MFmf]|Male|Female)"
    match = re.search(age_sex_pattern, note_text, re.IGNORECASE)
    if match:
        result["patient_age"] = int(match.group(1))
        result["gender"] = normalize_gender(match.group(2))
        return result

    # Guardrail: reject age-like captures that are catheter/device units (12F, 14Fr, 8mm, 3cm).
    def _reject_unit_suffix(text: str, idx_after_number: int) -> bool:
        tail = (text or "")[idx_after_number : idx_after_number + 8]
        return bool(re.match(r"(?i)\s*(?:f|fr|mm|cm)\b", tail))

    # Pattern 2: "(65 years old)"
    years_old_pattern = r"\((\d+)\s*years?\s*old\)"
    match = re.search(years_old_pattern, note_text, re.IGNORECASE)
    if match:
        age_raw = int(match.group(1))
        if not _reject_unit_suffix(note_text, match.end(1)):
            result["patient_age"] = age_raw

    # Pattern 3: "72-year-old male/female" (also handles stuttered "66 year old-year-old male")
    year_old_gender_pattern = (
        r"\b(\d{1,3})(?!\s*(?:f|fr|mm|cm)\b)\s*(?:-?\s*year\s*-?\s*old)(?:\s*-\s*year\s*-?\s*old)?\s*"
        r"(male|female|man|woman)\b"
    )
    match = re.search(year_old_gender_pattern, note_text, re.IGNORECASE)
    if match:
        result["patient_age"] = int(match.group(1))
        result["gender"] = normalize_gender(match.group(2))
        return result

    # Pattern 4: "72-year-old" age-only mention.
    year_old_pattern = r"\b(\d{1,3})(?!\s*(?:f|fr|mm|cm)\b)\s*(?:-?\s*year\s*-?\s*old)\b"
    match = re.search(year_old_pattern, note_text, re.IGNORECASE)
    if match:
        result["patient_age"] = int(match.group(1))

    # Pattern 5: Separate age and gender mentions
    # Age
    if "patient_age" not in result:
        # Word-boundary guardrail: avoid anchoring to substrings like "Page 1 of 2".
        age_pattern = r"\b(?:age|pt\s*age|patient\s*age)(?:\s*\(years?\))?\b[\s:]+(\d{1,3})\b"
        match = re.search(age_pattern, note_text or "", re.IGNORECASE)
        if match:
            try:
                age = int(match.group(1))
            except Exception:
                age = None
            if age is not None and 0 <= age <= 120:
                result["patient_age"] = age

    # Gender
    if "gender" not in result:
        gender_pattern = r"\b(?:sex|gender)\b[\s:]+([MFmf]|Male|Female)\b"
        match = re.search(gender_pattern, note_text or "", re.IGNORECASE)
        if match:
            result["gender"] = normalize_gender(match.group(1))

    # Pattern 6: Provation-style headers where labels and values are split across lines.
    #
    # Example:
    #   Age:
    #   Gender:
    #   ...
    #   83
    #   Male
    if "patient_age" not in result or "gender" not in result:
        lines = (note_text or "").splitlines()

        def _is_age_value(line: str) -> int | None:
            raw = (line or "").strip()
            if not re.fullmatch(r"\d{1,3}", raw):
                return None
            try:
                age = int(raw)
            except Exception:
                return None
            return age if 0 <= age <= 120 else None

        def _is_gender_value(line: str) -> str | None:
            raw = (line or "").strip()
            if re.fullmatch(r"(?i)male|female|m|f", raw):
                return normalize_gender(raw)
            return None

        # Prefer an explicit "Age:" label as the anchor.
        for i, line in enumerate(lines):
            if re.fullmatch(r"(?i)\s*age\s*:?\s*", line or ""):
                scan_end = min(len(lines), i + 25)
                for j in range(i + 1, scan_end):
                    if "patient_age" not in result:
                        age = _is_age_value(lines[j])
                        if age is not None:
                            result["patient_age"] = age
                            # Commonly, gender follows shortly after the age value.
                            if "gender" not in result:
                                for k in range(j + 1, min(scan_end, j + 6)):
                                    gender = _is_gender_value(lines[k])
                                    if gender:
                                        result["gender"] = gender
                                        break
                            break

                    if "gender" not in result:
                        gender = _is_gender_value(lines[j])
                        if gender:
                            result["gender"] = gender

                break

    return result


def extract_asa_class(note_text: str) -> Optional[int]:
    """Extract ASA classification from note.

    Patterns recognized:
    - "ASA Classification: II"
    - "ASA: 3"
    - "ASA III-E"

    If not found, returns 3 as default (common for interventional pulmonology).

    Returns:
        ASA class as integer 1-6, or None
    """
    # Pattern 1: "ASA Classification: II", "ASA Class: 3", or "ASA: 3"
    asa_pattern = r"ASA(?:\s+(?:Classification|Class))?[\s:]+([IViv123456]+(?:-E)?)"
    match = re.search(asa_pattern, note_text, re.IGNORECASE)
    if match:
        return normalize_asa_class(match.group(1))

    # Pattern 2: "ASA III" without explicit "Classification"
    asa_pattern2 = r"\bASA\s+([IViv]+(?:-E)?)\b"
    match = re.search(asa_pattern2, note_text)
    if match:
        return normalize_asa_class(match.group(1))

    # Default to 3 if ASA not documented (common for IP procedures)
    # This matches the v2.8 synthetic data behavior
    return 3


def extract_sedation_airway(note_text: str) -> Dict[str, Any]:
    """Extract sedation type and airway type from procedure context.

    Logic:
    - OR cases with "general anesthesia" -> sedation_type="General", airway_type="ETT"
    - Bedside with local only -> sedation_type="Local Only", airway_type="Native"
    - IV sedation (midazolam/fentanyl) -> sedation_type="Moderate", airway_type="Native"
    - MAC -> sedation_type="MAC", airway_type depends on context

    Returns:
        Dict with 'sedation_type' and/or 'airway_type' if determinable
    """
    result: Dict[str, Any] = {}
    note_lower = note_text.lower()

    moderate_indicators = [
        "moderate sedation",
        "conscious sedation",
        "iv sedation",
        "midazolam",
        "fentanyl",
        "versed",
    ]
    moderate_present = any(ind in note_lower for ind in moderate_indicators)
    proceduralist_moderate_context = bool(
        moderate_present
        and (
            re.search(r"\bno\s+anesthesiologist\s+present\b|\bwithout\s+anesthesiologist\b", note_lower)
            or re.search(
                r"\b(?:attending|proceduralist|operator|physician)\b[^.\n]{0,80}\bperformed\s+(?:own\s+)?sedation\b",
                note_lower,
            )
            or re.search(
                r"\bmoderate\s+sedation\b[^.\n]{0,120}\bby\b[^.\n]{0,60}\b(?:the\s+)?(?:attending|proceduralist|operator|physician)\b",
                note_lower,
            )
        )
    )

    if proceduralist_moderate_context:
        result["sedation_type"] = "Moderate"
        if re.search(r"\bett\b|endotracheal tube|intubated", note_lower):
            result["airway_type"] = "ETT"
        elif re.search(r"\blma\b|laryngeal mask", note_lower):
            result["airway_type"] = "LMA"
        else:
            result["airway_type"] = "Native"
        return result

    # Check for general anesthesia indicators
    ga_indicators = [
        "general anesthesia",
        "general anesthetic",
        "under general",
        "ga with",
        "propofol/fentanyl/rocuronium",
        "jet ventilation",
    ]

    if any(ind in note_lower for ind in ga_indicators):
        result["sedation_type"] = "General"

        # Check airway type for GA
        if "rigid bronchoscop" in note_lower:
            # Rigid bronchoscopy - airway managed by scope
            pass  # Leave airway_type unset
        elif re.search(r"\bett\b|endotracheal tube|intubated", note_lower):
            result["airway_type"] = "ETT"
        elif re.search(r"\blma\b|laryngeal mask", note_lower):
            result["airway_type"] = "LMA"
        elif re.search(r"\bi-?gel\b", note_lower):
            result["airway_type"] = "iGel"
        else:
            result["airway_type"] = "ETT"  # Default for GA

        return result

    # Check for MAC
    if re.search(r"\bmac\b|monitored anesthesia care", note_lower):
        result["sedation_type"] = "MAC"
        if not re.search(r"\bett\b|intubated|laryngeal mask", note_lower):
            result["airway_type"] = "Native"
        return result

    if any(ind in note_lower for ind in moderate_indicators):
        # Don't set if also has GA indicators (already handled above)
        if "sedation_type" not in result:
            result["sedation_type"] = "Moderate"
            result["airway_type"] = "Native"
        return result

    # Check for local only (thoracentesis, bedside procedures)
    local_indicators = [
        "local anesthesia only",
        "local only",
        "1% lidocaine",
        "lidocaine infiltration",
        "under local",
    ]

    bedside_procedures = [
        "thoracentesis",
        "bedside chest tube",
        "bedside procedure",
        "bedside bronchoscopy",
    ]

    is_local = any(ind in note_lower for ind in local_indicators)
    is_bedside = any(proc in note_lower for proc in bedside_procedures)

    if is_local or (is_bedside and "sedation_type" not in result):
        if "general" not in note_lower and "moderate" not in note_lower:
            result["sedation_type"] = "Local Only"
            result["airway_type"] = "Native"

    return result


def extract_institution_name(note_text: str) -> Optional[str]:
    """Extract institution name from note header.

    Patterns recognized:
    - "Institution: Sacred Heart Medical Center"
    - "** St. Mary's Teaching Hospital, Chicago, IL"
    - "Hospital: Mayo Clinic"

    Returns:
        Institution name string or None
    """
    # Pattern 1: "Institution: Name"
    inst_pattern = r"Institution:\s*(.+?)(?:\n|$)"
    match = re.search(inst_pattern, note_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: "Hospital: Name"
    hosp_pattern = r"Hospital:\s*(.+?)(?:\n|$)"
    match = re.search(hosp_pattern, note_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 3: "** Hospital Name, City, State" at start of note
    header_pattern = r"^\*{1,2}\s*(.+?(?:Hospital|Medical Center|Clinic|Institute|Health).+?)(?:\n|$)"
    match = re.search(header_pattern, note_text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip().strip("*").strip()

    return None


def extract_primary_indication(note_text: str) -> Optional[str]:
    """Extract primary indication from INDICATION section.

    Looks for INDICATION, CLINICAL SUMMARY, or REASON FOR PROCEDURE sections.

    Returns:
        Indication text or None
    """
    text = _maybe_unescape_newlines(note_text or "")

    # Anchor to true section headers (prefer explicit colon) so strings like
    # "INDICATION FOR OPERATION" do not get parsed as "FOR OPERATION: ..."
    indication_patterns = [
        r"(?is)\bINDICATION(?:S)?\s+FOR\s+(?:OPERATION|PROCEDURE|EXAM(?:INATION)?)\s*:\s*"
        r"(.+?)(?=\n\s*\n|\n\s*[A-Z][A-Z0-9 /()'&\.-]{2,}:\s*|\Z)",
        r"(?ims)^\s*INDICATION(?:S)?(?:\s+FOR\s+(?:OPERATION|PROCEDURE|EXAM(?:INATION)?))?\s*:\s*"
        r"(.+?)(?=^\s*[A-Z][A-Z0-9 /()'&\.-]{2,}:\s*|\n\s*\n|\Z)",
        r"(?ims)^\s*(?:CLINICAL SUMMARY|CLINICAL INDICATION)\s*:\s*"
        r"(.+?)(?=^\s*[A-Z][A-Z0-9 /()'&\.-]{2,}:\s*|\n\s*\n|\Z)",
        r"(?ims)^\s*REASON FOR (?:PROCEDURE|EXAM(?:INATION)?)\s*:\s*"
        r"(.+?)(?=^\s*[A-Z][A-Z0-9 /()'&\.-]{2,}:\s*|\n\s*\n|\Z)",
    ]

    for pattern in indication_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match:
            raw = match.group(1).strip()
            # Drop common "Target:" lines that often follow the indication header.
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            lines = [ln for ln in lines if not re.match(r"(?i)^target(?:s)?\\b\\s*:?", ln)]
            indication = " ".join(lines).strip()

            # Remove common leading header residue and consent boilerplate.
            indication = re.sub(r"(?i)^for\s+operation\s*:\s*", "", indication).strip()
            indication = re.sub(r"(?i)^\[REDACTED\]\s*", "", indication).strip()
            indication = re.split(
                r"(?i)\b(?:"
                r"the\s+nature,\s*purpose,\s*risks?,\s*benefits?\s+and\s+alternatives?"
                r"|risks?,\s*benefits?,\s*(?:and\s+)?alternatives?"
                r"|patient(?:\s+or\s+surrogate)?\s+indicated\s+a\s+wish\s+to\s+proceed"
                r"|informed\s+consent\s+was\s+signed"
                r"|consent\s+was\s+signed"
                r")\b",
                indication,
                maxsplit=1,
            )[0].strip(" .,:;-\t")
            indication = re.split(
                r"(?i)\b(?:ebus[-\s]*findings|procedure\s+in\s+detail|preoperative\s+diagnosis|postoperative\s+diagnosis|anesthesia|monitoring)\b",
                indication,
                maxsplit=1,
            )[0].strip(" .,:;-\t")

            # For templated "X year old ... who presents with Y", keep Y.
            presents_match = re.search(r"(?i)\bpresents?\s+with\b", indication)
            if presents_match:
                tail = indication[presents_match.end() :].strip(" .,:;-\t")
                if tail:
                    indication = tail

            # Clean up whitespace
            indication = re.sub(r"\s+", " ", indication)
            if not indication:
                continue
            if re.fullmatch(
                r"(?i)(diagnostic(?:\s+and\s+staging)?|staging(?:\s+and\s+diagnostic)?|diagnostic\s*/\s*staging)",
                indication,
            ):
                continue
            # Limit length
            if len(indication) > 500:
                indication = indication[:500] + "..."
            return indication

    return None


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


def extract_bronchus_sign(note_text: str) -> Optional[bool]:
    """Extract CT bronchus sign polarity when explicitly documented.

    Air bronchogram phrasing is treated as bronchus sign polarity.

    Returns:
        True when positive/present, False when negative/absent, None when not documented/indeterminate.
    """
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return None

    # If explicitly "not assessed"/unknown, treat as missing (do not infer).
    if re.search(
        r"(?i)\bbronchus\s+sign\b[^.\n]{0,40}\b(?:not\s+assessed|unknown|indeterminate|n/?a)\b",
        text,
    ):
        return None

    candidates: list[tuple[int, bool]] = []
    patterns: list[tuple[bool, str]] = [
        # Negative first (also catches "not present")
        (
            False,
            r"\bbronchus\s+sign\b[^.\n]{0,40}\b(?:negative|absent|no|not\s+present)\b",
        ),
        (False, r"\b(?:negative|absent|not\s+present)\b[^.\n]{0,20}\bbronchus\s+sign\b"),
        (False, r"\bno\b[^.\n]{0,20}\bbronchus\s+sign\b"),
        # Positive
        (True, r"\bbronchus\s+sign\b[^.\n]{0,40}\b(?:positive|present|yes)\b"),
        (True, r"\b(?:positive|present)\b[^.\n]{0,20}\bbronchus\s+sign\b"),
        # Air bronchogram phrasing treated as bronchus sign
        (False, r"\b(?:no|without|absen(?:t|ce))\b[^.\n]{0,60}\bair\s+bronchogram(?:s)?\b"),
        (False, r"\bair\s+bronchogram(?:s)?\b[^.\n]{0,60}\b(?:not\s+present|absen(?:t|ce))\b"),
        (True, r"\bair\s+bronchogram(?:s)?\b"),
    ]

    for value, pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            candidates.append((match.start(), value))

    if not candidates:
        return None

    # Prefer earliest mention; if ties, earlier patterns win (stable sort).
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def extract_ecog(note_text: str) -> Dict[str, Any]:
    """Extract ECOG/Zubrod performance status when explicitly documented.

    Returns:
        Dict with `ecog_score` (int 0-4) OR `ecog_text` (e.g., "0-1") when a range is documented.
    """
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return {}

    pattern = re.compile(
        r"(?i)\b(?:ECOG|Zubrod)(?:\s*/\s*(?:ECOG|Zubrod))?\b"
        r"(?:\s*performance\s*status|\s*PS)?\s*[:=]?\s*"
        r"(?P<val>[0-4](?:\s*(?:-|–|/|to)\s*[0-4])?)\b"
    )
    match = pattern.search(text)
    if not match:
        return {}

    raw_val = (match.group("val") or "").strip()
    if not raw_val:
        return {}

    val_norm = raw_val.replace("–", "-")
    val_norm = re.sub(r"\s+", " ", val_norm).strip()
    val_norm = re.sub(r"\s*(?:to|/)\s*", "-", val_norm, flags=re.IGNORECASE)
    val_norm = re.sub(r"\s*-\s*", "-", val_norm)

    if "-" in val_norm:
        return {"ecog_text": val_norm}

    if val_norm.isdigit():
        score = int(val_norm)
        if 0 <= score <= 4:
            return {"ecog_score": score}

    return {}


def extract_disposition(note_text: str) -> Optional[str]:
    """Extract patient disposition from note.

    Patterns recognized:
    - "Extubated in OR, stable, admit overnight"
    - "Transferred to floor"
    - "Outpatient discharge"
    - "ICU admission"

    Returns:
        Disposition string or None
    """
    note_lower = note_text.lower()

    # Check for common disposition patterns
    if "icu admission" in note_lower or "admitted to icu" in note_lower:
        return "ICU admission"

    if "pacu" in note_lower and "floor" in note_lower:
        return "PACU then floor"

    if "admit overnight" in note_lower or "overnight observation" in note_lower:
        return "Admit overnight for observation"

    if "transfer" in note_lower and "floor" in note_lower:
        return "Transferred to floor"

    if "outpatient" in note_lower and ("discharge" in note_lower or "released" in note_lower):
        return "Outpatient discharge"

    if "home" in note_lower and "discharge" in note_lower:
        return "Discharged home"

    if "extubated" in note_lower:
        if "stable" in note_lower:
            return "Extubated, stable, transferred to recovery"

    # Look for explicit DISPOSITION section
    disp_pattern = r"DISPOSITION[\s:]+(.+?)(?=\n|$)"
    match = re.search(disp_pattern, note_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _normalize_outcomes_disposition(note_text: str) -> str | None:
    """Normalize disposition to the v3 outcomes.disposition enum values."""
    text = note_text or ""
    lower = text.lower()
    if not lower.strip():
        return None

    # ICU
    if re.search(r"(?i)\b(?:micu|sicu|icu)\b", lower) or "critical care" in lower:
        return "ICU admission"

    # Explicit inpatient routing
    if re.search(r"(?i)\balready\s+inpatient\b", lower):
        if re.search(r"(?i)\btransfer(?:red)?\b[^.\n]{0,80}\bicu\b", lower):
            return "Already inpatient - transfer to ICU"
        if re.search(r"(?i)\breturn(?:ed)?\b[^.\n]{0,80}\bfloor\b", lower):
            return "Already inpatient - return to floor"

    # Observation
    if re.search(r"(?i)\b(?:overnight\s+observation|obs(?:ervation)?\b|observation\s+unit)\b", lower):
        return "Observation unit"

    # Floor admission
    if re.search(r"(?i)\b(?:admit(?:ted)?|transfer(?:red)?)\b[^.\n]{0,80}\b(?:floor|ward|telemetry)\b", lower):
        return "Floor admission"

    # Outpatient discharge
    if re.search(r"(?i)\b(?:outpatient)\b", lower) and re.search(r"(?i)\bdischarg(?:e|ed)\b", lower):
        return "Outpatient discharge"
    if re.search(r"(?i)\b(?:discharg(?:e|ed)|dc)\b[^.\n]{0,80}\bhome\b", lower):
        return "Outpatient discharge"
    if re.search(r"(?i)\bdischarg(?:e|ed)\b", lower) and re.search(r"(?i)\bhome\b", lower):
        return "Outpatient discharge"
    if re.search(r"(?i)\bdischarg(?:e|ed)\b", lower) and re.search(r"(?i)\bif\s+stable\b", lower):
        return "Outpatient discharge"

    return None


def extract_follow_up_plan_text(note_text: str) -> str | None:
    """Extract a concise follow-up/disposition/plan free-text block."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return None

    header_re = re.compile(
        r"(?im)^\s*(?P<header>disposition|follow-?up|recommendations?|impression/plan|plan)\s*:\s*(?P<rest>.*)$"
    )
    stop_header_re = re.compile(r"(?m)^\s*[A-Z][A-Z0-9 /-]{2,}\s*:\s*")

    blocks: list[str] = []
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx] or ""
        match = header_re.match(line)
        if not match:
            idx += 1
            continue

        rest = (match.group("rest") or "").strip()
        buf: list[str] = []
        if rest:
            buf.append(rest)

        j = idx + 1
        while j < len(lines):
            nxt = lines[j] or ""
            if not nxt.strip():
                break
            if header_re.match(nxt):
                break
            if stop_header_re.match(nxt) and not re.search(r"(?i)^\s*(?:follow-?up|plan|recommendations?|disposition)\b", nxt):
                break
            buf.append(nxt.strip())
            j += 1

        if buf:
            blocks.append(" ".join(buf).strip())

        idx = j if j > idx else idx + 1

    if not blocks:
        return None

    merged = " ".join(blocks)
    merged = re.sub(r"\s+", " ", merged).strip()
    if not merged:
        return None
    return merged[:900]


def extract_procedure_completed(note_text: str) -> tuple[bool | None, str | None]:
    """Return (procedure_completed, procedure_aborted_reason)."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return None, None

    aborted_reason = None
    aborted_match = re.search(
        r"(?i)\b(?:procedure\s+)?(?:aborted|terminated)\b[^.\n]{0,120}",
        text,
    )
    if aborted_match:
        snippet = aborted_match.group(0).strip()
        reason_match = re.search(
            r"(?i)\b(?:due\s+to|because|secondary\s+to)\b\s*(?P<reason>[^.\n]{3,200})",
            snippet,
        )
        if reason_match:
            aborted_reason = reason_match.group("reason").strip()
        else:
            aborted_reason = snippet[:200]
        return False, aborted_reason

    # Conservative "completed" markers.
    if re.search(
        r"(?i)\b(?:procedure\s+was\s+completed|procedure\s+completed|completed\s+without\s+complications?)\b",
        text,
    ):
        return True, None
    if re.search(r"(?i)\bpatient\s+tolerated\s+the\s+procedure\s+well\b", text):
        return True, None
    if re.search(r"(?i)\bno\s+(?:immediate\s+)?complications?\b", text):
        return True, None
    if re.search(r"(?i)\bthe\s+patient\s+was\s+stable\b", text) and re.search(r"(?i)\btransferred\b", text):
        return True, None

    return None, None


def extract_outcomes(note_text: str) -> Dict[str, Any]:
    """Extract v3 outcomes fields when explicitly supported by the note."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return {}

    outcomes: dict[str, Any] = {}

    plan = extract_follow_up_plan_text(text)
    if plan:
        outcomes["follow_up_plan_text"] = plan

    completed, aborted_reason = extract_procedure_completed(text)
    if completed is not None:
        outcomes["procedure_completed"] = completed
    if aborted_reason:
        outcomes["procedure_aborted_reason"] = aborted_reason

    disp = _normalize_outcomes_disposition(text)
    if disp:
        outcomes["disposition"] = disp

    return {"outcomes": outcomes} if outcomes else {}


def extract_bleeding_severity(note_text: str) -> Optional[str]:
    """Extract bleeding severity from note.

    Returns:
        'None', 'Mild', 'Mild (<50mL)', 'Moderate', 'Severe', or None
    """
    note_lower = note_text.lower()

    # Check for explicit bleeding mentions
    if "no bleeding" in note_lower or "no significant bleeding" in note_lower:
        return "None"

    if (
        "minimal bleeding" in note_lower
        or "minor bleeding" in note_lower
        or "trace bleeding" in note_lower
        or "scant bleeding" in note_lower
    ):
        return "None"

    # Check for EBL (estimated blood loss)
    ebl_pattern = r"(?:ebl|estimated blood loss|blood loss)[\s:]*<?\s*(\d+)\s*(?:ml|cc)?"
    match = re.search(ebl_pattern, note_lower)
    if match:
        ebl = int(match.group(1))
        # Hard gate: very low EBL values are common and should not be treated as a bleeding complication.
        if ebl < 10:
            return "None"
        if ebl < 50:
            return "Mild (<50mL)"
        elif ebl < 200:
            return "Moderate"
        else:
            return "Severe"

    if "mild bleeding" in note_lower:
        return "Mild"

    if "moderate bleeding" in note_lower:
        return "Moderate"

    if "severe bleeding" in note_lower or "massive bleeding" in note_lower:
        return "Severe"

    # Default to None if not explicitly mentioned
    return "None"


_BLEEDING_INTERVENTION_PATTERNS: list[tuple[str, str]] = [
    ("Cold saline", r"\b(?:cold|iced)\s+saline\b"),
    ("Epinephrine", r"\b(?:epinephrine|epi)\b"),
    ("Balloon tamponade", r"\bballoon\s+tamponade\b"),
    ("Electrocautery", r"\belectrocautery\b|\bcauteriz(?:e|ed|ation)\b|\bcoagulat(?:e|ed|ion)\b"),
    ("APC", r"\bapc\b|argon\s+plasma"),
    ("Tranexamic acid", r"\btranexamic\s+acid\b|\btxa\b"),
    ("Bronchial blocker", r"\bbronchial\s+blocker\b|\bendobronchial\s+blocker\b"),
    ("Transfusion", r"\btransfus(?:ion|ed)\b|\bprbc\b|\bpacked\s+red\b"),
    ("Embolization", r"\bemboliz(?:ation|ed)\b"),
    ("Surgery", r"\bsurgery\b|\bthoracotomy\b"),
]


def extract_bleeding_intervention_required(note_text: str) -> list[str] | None:
    """Extract bleeding interventions as schema enum values.

    This is intentionally conservative: it only flags bleeding as a complication
    when an intervention to control bleeding is explicitly documented.
    """
    text = note_text or ""
    lowered = text.lower()

    # Explicit negations: don't infer interventions.
    if re.search(r"\bno\s+(?:immediate\s+)?complications\b", lowered):
        return None
    if re.search(r"\bno\s+(?:significant\s+)?bleeding\b", lowered):
        return None
    if (
        re.search(r"(?i)\b(?:complications?\s*:\s*none\s+procedural|no\s+procedural\s+complications)\b", text)
        and re.search(r"(?i)\b(?:indication|pre\s*dx|post\s*dx|diagnosis)\b[^.\n]{0,160}\bhemoptysis\b", text)
        and not re.search(
            r"(?i)\b(?:biops(?:y|ies|ied)|cryobiops(?:y|ies)|tbna|needle\s+aspiration|brushings?|debridement)\b",
            text,
        )
    ):
        return None

    hits: list[str] = []
    for label, pattern in _BLEEDING_INTERVENTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(label)

    # Dedupe while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in hits:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    return deduped or None


def extract_providers(note_text: str) -> Dict[str, Any]:
    """Extract provider information from note.

    Returns dict matching expected schema:
    {
        "attending_name": str or None,
        "fellow_name": str or None,
        "assistant_name": str or None,
        "assistant_role": str or None,
        "trainee_present": bool or None,
    }
    """
    result: Dict[str, Any] = {
        "attending_name": None,
        "fellow_name": None,
        "assistant_name": None,
        "assistant_role": None,
        "trainee_present": None,
    }

    # Pattern for attending
    attending_patterns = [
        # Avoid false-positive capture of header words like "Participation"
        r"(?:Attending\s+Participation|Attending\s+Physician\s+Participation)\s*:\s*(?:\*{1,2}\s*)?(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:Attending\s+Physician|Attending|Primary\s+Operator)\s*:\s*(?:\*{1,2}\s*)?(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"\*{2}\s*Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]

    for pattern in attending_patterns:
        match = re.search(pattern, note_text)
        if match:
            name = match.group(1).strip()
            # Keep leading ** if present in original (per v2.8 data format)
            if "** " in note_text[max(0, match.start()-5):match.start()]:
                name = f"** {name}"
            elif "**" in note_text[max(0, match.start()-3):match.start()]:
                name = f"** {name}"
            result["attending_name"] = name
            break

    # Pattern for fellow
    fellow_patterns = [
        r"(?:Fellow|IP Fellow|Pulmonary Fellow)[\s:]+(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]

    for pattern in fellow_patterns:
        match = re.search(pattern, note_text)
        if match:
            result["fellow_name"] = match.group(1).strip()
            result["trainee_present"] = True
            break

    # Check for trainee presence
    trainee_indicators = ["fellow", "resident", "trainee", "pgy"]
    note_lower = note_text.lower()
    if any(ind in note_lower for ind in trainee_indicators):
        result["trainee_present"] = True

    # Pattern for assistant
    assistant_patterns = [
        r"(?:Assistant|Assist(?:ed)? by)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s*(?:RN|RT|Tech|PA|NP)?",
    ]

    for pattern in assistant_patterns:
        match = re.search(pattern, note_text)
        if match:
            result["assistant_name"] = match.group(1).strip()
            # Try to determine role
            role_match = re.search(r"(RN|RT|Tech|PA|NP|Resident)", note_text[match.start():match.end()+20])
            if role_match:
                result["assistant_role"] = role_match.group(1)
            break

    return result


# =============================================================================
# PROCEDURE EXTRACTORS (Phase 7)
# =============================================================================

# BAL detection patterns
BAL_PATTERNS = [
    r"\bbroncho[-\s]?alveolar\s+lavage\b",
    r"\bbronchial\s+alveolar\s+lavage\b",
    r"\bBAL\b(?!\s*score)",  # BAL but not "BAL score"
]

# Endobronchial biopsy detection patterns
ENDOBRONCHIAL_BIOPSY_PATTERNS = [
    r"\bendobronchial\s+biops",
    # Common typo seen in copied notes: "enbobronchial forcep biopsies".
    r"\ben(?:do|bo)bronchial\s+forcep?s?\s+biops",
    r"\bbiops(?:y|ied|ies)\b[^.\n]{0,60}\bendobronchial\b",
    r"\blesions?\s+were\s+biopsied\b",
    r"\bebbx\b",
    # Notes sometimes document endobronchial forceps biopsies without the keyword
    # "endobronchial" (e.g., cavity/mycetoma within an airway segment).
    r"\bforceps\s+biops(?:y|ies)\b[^.\n]{0,140}\b(?:cavity|endobronch|airway|bronch(?:us|ial)|trachea|carina|mainstem)\b",
    r"\b(?:cavity|endobronch|airway|bronch(?:us|ial)|trachea|carina|mainstem)\b[^.\n]{0,140}\bforceps\s+biops(?:y|ies)\b",
    # EBUS intranodal mini-forceps (often documented without the word "biopsy")
    r"\bmini[-\s]?forceps\b[^.\n]{0,240}\b(?:ebus|endobronchial\s+ultrasound|lymph\s+node|station)\b",
    r"\b(?:ebus|endobronchial\s+ultrasound|lymph\s+node|station)\b[^.\n]{0,240}\bmini[-\s]?forceps\b",
]

# Transbronchial biopsy detection patterns (parenchyma / peripheral lung).
#
# Guardrail: some EBUS templates misuse "transbronchial biopsies" language for
# nodal sampling; the extractor filters those contexts.
TRANSBRONCHIAL_BIOPSY_PATTERNS = [
    r"\btransbronchial\s+(?:lung\s+)?biops(?:y|ies)\b",
    r"\btransbronchial\s+forceps\s+biops(?:y|ies)\b",
    r"\btbbx\b",
    r"\btblb\b",
]

# Radial EBUS detection patterns (rEBUS for peripheral lesion localization)
RADIAL_EBUS_PATTERNS = [
    r"\bradial\s+ebus\b",
    r"\bradial\s+endobronchial\s+ultrasound\b",
    r"\br-?ebus\b",
    r"\brp-?ebus\b",
    r"\bminiprobe\b",
    r"\bradial\s+probe\b",
    # Some notes document rEBUS as "radial ultrasound" without the EBUS token.
    r"\bradial\s+ultrasound\b",
]

# EUS-B detection patterns (endoscopic ultrasound via EBUS bronchoscope)
EUS_B_PATTERNS = [
    r"\beus-?b\b",
    r"\beusb\b",
    r"\beus[- ]?b[- ]?fna\b",
    r"\besophageal\s+approach\b",
    r"\btransesophageal\b",
]

# Cryotherapy / tumor destruction patterns (31641 family)
CRYOTHERAPY_PATTERNS = [
    r"\bcryotherap(?:y|ies)\b",
    r"\bcryo(?:therapy|debulk(?:ing)?)\b",
    r"\bcryo\s*spray\b",
    r"\bcryospray\b",
]
CRYOPROBE_PATTERN = r"\bcryo\s*probe\b"
CRYOBIOPSY_PATTERN = r"\bcryo\s*biops(?:y|ies)\b|\bcryobiops(?:y|ies)\b"

# Rigid bronchoscopy patterns (31640/31641 family)
RIGID_BRONCHOSCOPY_PATTERNS = [
    r"\brigid\s+bronchoscop",  # bronchoscopy/bronchoscope/bronchoscopic
    r"\brigid\s+optic\b",
    r"\brigid\s+scope\b",
]

# Linear EBUS patterns (EBUS-TBNA)
LINEAR_EBUS_PATTERNS = [
    r"\blinear\s+ebus\b",
    r"\bconvex(?:-probe)?\s+(?:ebus|endobronchial\s+ultrasound)\b",
    r"\bebus[-\s]?tbna\b",
    r"\bendobronchial\s+ultrasound[-\s]guided\b[^.]{0,80}\b(?:tbna|needle)\b",
]

# Navigation / robotic bronchoscopy patterns
NAVIGATIONAL_BRONCHOSCOPY_PATTERNS = [
    r"\bnavigational\s+bronchoscopy\b",
    r"\bnavigation\s+bronchoscopy\b",
    r"\belectromagnetic\s+navigation\b",
    r"\bEMN\b",
    r"\bENB\b",
    r"\bion\b[^.\n]{0,40}\bbronchoscop",
    r"\bmonarch\b[^.\n]{0,40}\bbronchoscop",
    r"\brobotic\b[^.\n]{0,40}\bbronchoscop",
    r"\brobotic\b[^.\n]{0,40}\bbronch",
    r"\bgalaxy\b[^.\n]{0,40}\bbronch",
    r"\bnoah\b[^.\n]{0,40}\bbronch",
    r"\bsuperdimension\b",
    r"\billumisite\b",
    r"\bveran\b",
    r"\bspin(?:drive)?\b",
]

# TBNA (conventional) patterns
TBNA_CONVENTIONAL_PATTERNS = [
    r"\btbna\b",
    r"\btransbronchial\s+needle\s+aspiration\b",
    r"\btransbronchial\s+needle\b",
]

# Brushings patterns
BRUSHINGS_PATTERNS = [
    r"\bbrushings?\b",
    r"\bcytology\s+brush(?:ings?)?\b",
    r"\bbronchial\s+brushing(?:s)?\b",
    r"\bbronchoscopic\s+brush(?:ings?)?\b",
    # Some templates document "brush" without the plural "brushings".
    r"\b(?:cytology|bronch(?:ial|oscopic))\s+brush\b",
    r"\bbrush\b[^.\n]{0,60}\b(?:cytolog|specimen|sample|pass(?:es)?|sent\s+for)\b",
]

# Mechanical debulking / excision patterns (31640 family)
MECHANICAL_DEBULKING_PATTERNS = [
    r"\bmechanical\s+debulk(?:ing)?\b",
    r"\bdebulk(?:ed|ing)?\b",
    r"\b(?:tumou?r|lesion|mass)\b[^.\n]{0,160}\b(?:resect|excise|excision|core\s*out|remove(?:d)?|debulk(?:ed|ing)?)\b",
    r"\b(?:resect|excise|excision|core\s*out|remove(?:d)?)\b[^.\n]{0,160}\b(?:tumou?r|lesion|mass)\b",
    r"\b(?:snare|microdebrider|microdebrid\w*|rigid\s+coring)\b[^.\n]{0,220}\b(?:en\s+bloc|resect|excise|excision|remove(?:d)?|debulk(?:ed|ing)?)\b",
    r"\b(?:en\s+bloc)\b[^.\n]{0,220}\b(?:resect|excise|excision|remove(?:d)?|debulk(?:ed|ing)?)\b",
]

# Bronchopleural fistula (BPF) glue/sealant patterns (therapeutic bronchoscopy).
BPF_SEALANT_PATTERNS = [
    r"\b(?:bronchopleural|broncho-pleural|broncho\s*pleural)\s+fistula\b",
    r"\b(?:alveolar|alveolo)\s*[-\s]?pleural\s+fistula\b",
    r"\bbpf\b",
    r"\bapf\b",
]

BPF_SEALANT_AGENT_PATTERNS = [
    r"\b(?:tisseel|glue|sealant|fibrin\s+glue|cyanoacrylate)\b",
    r"\bveno-?\s*seal\b",
]

# Transbronchial cryobiopsy patterns
TRANSBRONCHIAL_CRYOBIOPSY_PATTERNS = [
    r"\btransbronchial\s+cryo\b",
    r"\bcryo\s*biops(?:y|ies)\b",
    r"\bcryobiops(?:y|ies)\b",
    r"\bTBLC\b",
]

# Peripheral ablation patterns (MWA/RFA/cryoablation)
PERIPHERAL_ABLATION_PATTERNS = [
    r"\bavuecue\b",
    r"\bmicrowave\s+catheter\b",
    r"\bmicrowave\s+ablation\b",
    r"\bmwa\b",
    r"\bradiofrequency\s+ablation\b",
    r"\brf\s+ablation\b",
    r"\brfa\b",
    r"\bcryoablation\b",
    r"\bcryo\s*ablation\b",
]

# Thermal ablation patterns (APC/laser/electrocautery)
THERMAL_ABLATION_PATTERNS = [
    r"\bapc\b",
    r"\bargon\s+plasma\b",
    r"\belectrocautery\b",
    r"\bcauteriz(?:e|ed|ation)\b",
    r"\blaser\b",
    r"\bthermal\s+ablation\b",
]

# Chest ultrasound patterns (76604 family)
CHEST_ULTRASOUND_PATTERNS = [
    r"\bchest\s+ultrasound\s+findings\b",
    r"\bultrasound,\s*chest\b",
    r"\bchest\s+ultrasound\b",
    r"\b76604\b",
]

CHEST_ULTRASOUND_IMAGE_DOC_PATTERNS = [
    r"\bimage\s+saved\s+and\s+printed\b",
    r"\bimage\s+saved\b",
    r"\bwith\s+image\s+documentation\b",
]

# Thoracentesis patterns
THORACENTESIS_PATTERNS = [
    r"\bthoracentesis\b",
    r"\bpleural\s+tap\b",
]

# Chest tube / pleural drainage catheter patterns
CHEST_TUBE_PATTERNS = [
    r"\bpigtail\s+catheter\b",
    r"\bchest\s+tube\b",
    r"\btube\s+thoracostomy\b",
]

# Indwelling pleural catheter (IPC / tunneled) patterns
IPC_PATTERNS = [
    r"\bpleurx\b",
    r"\baspira\b",
    r"\btunne(?:l|ll)ed\s+pleural\s+catheter\b",
    r"\btunnel(?:ing)?\s+pleural\s+catheter\b",
    r"\bindwelling\s+pleural\s+catheter\b",
    r"\bipc\b[^.\n]{0,30}\b(?:catheter|drain)\b",
    r"\brocket\b[^.\n]{0,40}\b(?:ipc|catheter|pleur)\b",
    r"\btunne(?:l|ll)ed\s+catheter\b",
    r"\btunnel(?:ing)?\s+catheter\b",
    r"\btunneling\s+device\b",
]

# Therapeutic aspiration patterns (exclude routine suction)
THERAPEUTIC_ASPIRATION_PATTERNS = [
    r"\btherapeutic\s+aspiration\b",
    # Common OCR/typo variant seen in PDF-extracted notes.
    r"\bthereapeutic\s+aspiration\b",
    r"\btherapeutic\s+suction(?:ing)?\b",
    r"\bmucus\s+plug\s+(?:removal|aspiration|extracted|suctioned|cleared)\b",
    r"\b(?:large|tenacious|obstructing)\s+(?:mucus\s+)?plug\b[^.\n]{0,80}\b(?:extract(?:ed|ion)?|remov(?:ed|al)|suction(?:ed|ing)?|clear(?:ed|ing)?)\b",
    r"\b(?:large\s+)?(?:blood\s+)?clot\s+(?:removal|aspiration|extracted|suctioned|cleared)\b",
    r"\b(?:blood\s+)?clot\b[^.\n]{0,60}\b(?:was\s+)?(?:successfully\s+)?(?:removed|evacuated|extracted)\b",
    r"\bairway\s+(?:cleared|cleared\s+of)\s+(?:mucus|secretions|blood|clot)\b",
    r"\b(?:copious|large\s+amount\s+of|thick|tenacious|purulent|bloody|blood-tinged)\s+secretions?\b[^.]{0,80}\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|cleared|remov(?:ed|al))\b",
    r"\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|cleared|remov(?:ed|al))\b[^.]{0,80}\b(?:copious|large\s+amount\s+of|thick|tenacious|purulent|bloody|blood-tinged)\s+secretions?\b",
]

ROUTINE_SUCTION_PATTERNS = [
    r"\broutine\s+suction(?:ing)?\b",
    r"\bminimal\s+secretions?\s+(?:suctioned|cleared|noted)\b",
    r"\bmild\s+secretions?\s+(?:suctioned|cleared|noted)\b",
    r"\bstandard\s+suction(?:ing)?\b",
    r"\bscant\s+secretions?\b",
    r"\bsmall\s+amount\s+of\s+secretions?\b",
]

# Airway stent patterns (31636/31638 family)
AIRWAY_STENT_DEVICE_PATTERNS = [
    r"\bstent\b",
    r"\by-?\s*stent\b",
    r"\bsilicone\s+stent\b",
    r"\bmetal(?:lic)?\s+stent\b",
    r"\bdumon\b",
    r"\baero(?:stent)?\b",
    r"\baerstent\b",
    r"\bultraflex\b",
    r"\bbonostent\b",
    r"\bsems\b",
    # Some bronchoscopy notes refer to an occlusive airway device as a "vascular plug"
    # without using the term "stent"; this is still billed under 31638 when revised.
    r"\bvascular\s+plug\b",
    r"\bendobronchial\s+plug\b",
]

AIRWAY_STENT_PLACEMENT_PATTERNS = [
    r"\b(?:place(?:d)?|deploy(?:ed)?|insert(?:ed)?|positioned|deliver(?:ed)?|deploy(?:ment)?)\b",
    r"\b(?:placement|insertion)\b",
]

AIRWAY_STENT_REMOVAL_PATTERNS = [
    r"\b(?:remov(?:e|ed|al)|retriev(?:e|ed|al)|extract(?:ed)?|explant(?:ed)?|remov(?:ing)|pull(?:ed)?)\b",
    r"\b(?:grasp(?:ed)?|peel(?:ed)?).{0,20}\b(?:out|off|remove(?:d)?)\b",
]

# Airway dilation patterns (31630 family)
AIRWAY_DILATION_PATTERNS = [
    r"\bballoon\s+dilat",
    r"\bdilat\w*\b[^.\n]{0,60}\bballoon\b",
    r"\bballoon\b[^.\n]{0,60}\bdilat",
    r"\bcre\s+balloon\b",
    r"\bdilatational\s+balloon\b",
    r"\bmustang\s+balloon\b",
]

# Foreign body removal patterns (31635 family)
FOREIGN_BODY_REMOVAL_PATTERNS = [
    r"\bforeign\s+body\s+remov",
    r"\bforeign\s+body\b[^.\n]{0,60}\b(?:remov|retriev|extract|grasp)\w*",
    r"\bretriev(?:e|ed|al)\b[^.\n]{0,60}\bforeign\s+body\b",
    r"\b(?:fracture[dm]|broken|migrated)\s+(?:piece|fragment|segment|portion)\s+of\s+(?:the\s+)?stent\b",
    r"\bstent\s+(?:fragment|piece)\s+was\s+(?:removed|retrieved|extracted)\b",
]

# Percutaneous tracheal puncture / transtracheal access patterns (31612 family).
TRACHEAL_PUNCTURE_PATTERNS = [
    r"\btranstracheal\s+(?:injection|aspiration)\b",
    r"\b(?:trachea|tracheal\s+wall|anterior\s+tracheal\s+wall)\b[^.\n]{0,80}\bpunctur\w*\b",
    r"\bpunctur\w*\b[^.\n]{0,80}\b(?:trachea|tracheal\s+wall)\b",
    r"\b\d{1,2}\s*(?:g|ga|gauge)\b[^.\n]{0,60}\bangiocat(?:h|heter)\b[^.\n]{0,80}\bpunctur\w*\b",
    r"\bangiocat(?:h|heter)\b[^.\n]{0,80}\bpunctur\w*\b[^.\n]{0,80}\b(?:trachea|tracheal\s+wall)\b",
]

# Balloon occlusion / endobronchial blocker patterns (31634 family).
BALLOON_OCCLUSION_PATTERNS = [
    r"\bballoon\s+occlusion\b",
    r"\bserial\s+occlusion\b",
    r"\bocclusion\b[^.\n]{0,80}\b(?:endobronchial\s+blocker|blocker|uniblocker|arndt|ardnt|fogarty)\b",
    r"\b(?:endobronchial\s+blocker|uniblocker|arndt|ardnt|fogarty)\b[^.\n]{0,80}\bocclu\w*\b",
    r"\b(?:endobronchial\s+blocker|uniblocker)\b[^.\n]{0,80}\bballoon\b[^.\n]{0,80}\b(?:inflated|deflated|inflate|deflate|occlu)\w*\b",
]

# BLVR (endobronchial valve) patterns (31647 family)
BLVR_PATTERNS = [
    r"\b(spiration|zephyr)\b",
    r"\b(endobronchial|bronchial)\s+valve\b",
    r"\bvalve\s+(?:deployment|placement|insertion)\b",
    r"\bolympus\b[^.\n]{0,40}\bvalve\b",
    r"\b(?:lung\s+volume\s+reduction|bronchoscopic\s+lung\s+volume\s+reduction)\b",
    r"\bchartis\b",
]

_CPT_LINE_PATTERN = re.compile(r"^\s*\d{5}\b")
_PROCEDURE_DETAIL_SECTION_PATTERN = re.compile(
    r"(?im)^\s*(?:procedure\s+in\s+detail|description\s+of\s+procedure|procedure\s+description)\s*:?"
)
_SECTION_HEADING_INLINE_RE = re.compile(
    r"(?im)^\s*(?P<header>[A-Za-z][A-Za-z /()_-]{0,80})\s*:\s*(?P<rest>.*)$"
)
_SECTION_HEADING_STANDALONE_RE = re.compile(
    r"(?im)^\s*(?P<header>[A-Z][A-Z0-9 /()_-]{1,80})\s*$"
)
_NON_PROCEDURAL_HEADINGS: tuple[str, ...] = (
    "PLAN",
    "IMPRESSION/PLAN",
    "IMPRESSION / PLAN",
    "ASSESSMENT/PLAN",
    "ASSESSMENT / PLAN",
    "RECOMMENDATION",
    "RECOMMENDATIONS",
    "ASSESSMENT",
)

WHOLE_LUNG_LAVAGE_PATTERNS = [
    r"\bwhole\s+lung\s+lavage\b",
    r"\bwll\b",
]

BRONCHIAL_THERMOPLASTY_PATTERNS = [
    r"\bbronchial\s+thermoplasty\b",
    r"\bthermoplasty\s+catheter\b",
    r"\brf\s+activations?\b",
]


def _normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).upper()


def _is_non_procedural_heading(header: str) -> bool:
    if header in _NON_PROCEDURAL_HEADINGS:
        return True
    return any(
        header.startswith(prefix)
        for token in _NON_PROCEDURAL_HEADINGS
        for prefix in (f"{token} ", f"{token}/", f"{token} -")
    )

DIAGNOSTIC_BRONCHOSCOPY_PATTERNS = [
    r"\bthe\s+airway\s+was\s+inspected\b",
    r"\bairway\s+was\s+inspected\b",
    r"\binitial\s+airway\s+inspection\s+findings\b",
    r"\bbronchoscope\b[^.\n]{0,80}\b(?:introduc|advance|insert)\w*\b",
    r"\b(?:introduc|advance|insert)\w*\b[^.\n]{0,80}\bbronchoscope\b",
    r"\bbronchoscopy\b[^.\n]{0,80}\b(?:perform|completed)\w*\b",
    r"\bdynamic\s+bronchoscopy\b",
    r"\bbronchoscopy\b[^.\n]{0,80}\bdynamic\s+assessment\b",
    r"\bforced\s+expiratory\s+maneuver\b",
]

_CHECKBOX_TOKEN_RE = re.compile(
    r"(?im)(?<!\d)(?P<val>[01])\s*[^\w\n]{0,6}\s*(?P<label>[A-Za-z][A-Za-z /()_-]{0,80})"
)


def _checkbox_selected(note_text: str, *, label_patterns: list[str]) -> bool | None:
    """Return True/False if checkbox-style selection is present, else None.

    Supports templates that encode options as "1 <Label>" / "0 <Label>" where the
    separator may be a dash, bullet, or zero-width character.
    """
    if not note_text:
        return None

    compiled = [re.compile(pat, re.IGNORECASE) for pat in label_patterns]
    selected = False
    deselected = False
    for match in _CHECKBOX_TOKEN_RE.finditer(note_text):
        try:
            val = int(match.group("val"))
        except Exception:
            continue
        label = (match.group("label") or "").strip()
        if not label:
            continue
        if not any(p.search(label) for p in compiled):
            continue
        if val == 1:
            selected = True
        elif val == 0:
            deselected = True

    if selected:
        return True
    if deselected:
        return False
    return None


def _strip_cpt_definition_lines(text: str) -> str:
    """Remove template/definition lines that start with a 5-digit CPT code."""
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        if _CPT_LINE_PATTERN.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _preferred_procedure_detail_text(note_text: str) -> tuple[str, bool]:
    """Return (preferred_text, used_detail_section).

    If the note contains a distinct procedure-detail section, return only the
    text after that header to avoid matching planned/consent/template blocks.
    """
    text = note_text or ""
    match = _PROCEDURE_DETAIL_SECTION_PATTERN.search(text)
    if not match:
        return text, False

    # Slice after the header token (keeps same-line content if present).
    tail = text[match.end() :]

    # Stop at non-procedural headings (e.g., IMPRESSION/PLAN), which frequently contain
    # future/planned procedures that should not trigger performed flags.
    stop_at: int | None = None
    for heading_match in _SECTION_HEADING_INLINE_RE.finditer(tail):
        header = _normalize_heading(heading_match.group("header") or "")
        if not header:
            continue
        if _is_non_procedural_heading(header):
            stop_at = heading_match.start()
            break

    # Some templates use standalone headings without ":" (e.g., "IMPRESSION / PLAN").
    for heading_match in _SECTION_HEADING_STANDALONE_RE.finditer(tail):
        header = _normalize_heading(heading_match.group("header") or "")
        if not header:
            continue
        if not _is_non_procedural_heading(header):
            continue
        heading_start = heading_match.start()
        if stop_at is None or heading_start < stop_at:
            stop_at = heading_start
            break

    if stop_at is not None and stop_at >= 0:
        tail = tail[:stop_at]

    return tail.lstrip("\r\n "), True


def _extract_ln_stations_from_text(note_text: str) -> list[str]:
    """Extract IASLC lymph node station tokens from free text.

    This is a conservative backstop used when NER station extraction fails.
    It requires station context (e.g., 'station 7', '11L lymph node') to
    avoid false positives from unrelated numbers (e.g., '5-7 days').
    """
    text_lower = (note_text or "").lower()
    if not text_lower.strip():
        return []

    try:
        from app.ner.entity_types import normalize_station
    except Exception:
        normalize_station = None  # type: ignore[assignment]

    sampling_hint_re = re.compile(
        r"\b(?:tbna|fna|aspirat|biops|sampled|sampling|needle|passes?|core|forceps)\b",
        re.IGNORECASE,
    )
    sampling_negation_re = re.compile(
        r"\b(?:"
        r"not\s+(?:sampled|biopsied|aspirated)"
        r"|site\s+was\s+not\s+sampled"
        r"|without\s+biops"
        r"|no\s+biops(?:y|ies)"
        r"|did\s+not\s+have\s+any\s+biops(?:y|ies)\s+target"
        r"|did\s+not\s+have\s+any\s+sampling\s+target"
        r"|no\s+(?:biops(?:y|ies)|sampling)\s+target"
        r"|biops(?:y|ies)\s+were\s+not\s+taken"
        r"|not\b[^.\n]{0,40}\bperform\b[^.\n]{0,80}\b(?:transbronchial\s+)?(?:sampling|sample|tbna|fna|aspirat|biops)\w*"
        r"|decision\b[^.\n]{0,80}\bnot\b[^.\n]{0,40}\bperform\b[^.\n]{0,80}\b(?:transbronchial\s+)?(?:sampling|sample|tbna|fna|aspirat|biops)\w*"
        r")\b",
        re.IGNORECASE,
    )
    station_context_re = re.compile(r"\b(?:station(?:s)?|stn|level|ln|node(?:s)?|lymph)\b", re.IGNORECASE)
    station_token_re = re.compile(
        r"(?<![0-9A-Z])(2R|2L|3p|4R|4L|5|7|8|9|10R|10L|11R(?:S|I)?|11L(?:S|I)?|12R|12L)(?![0-9A-Z])",
        re.IGNORECASE,
    )
    table_header_re = re.compile(
        r"(?i)\bstation\s*:?\s*(?:ebus\s+size|short\s+axis|size\s*\(mm\))[^.\n]{0,120}\bnumber\s+of\s+passes\b"
    )
    table_row_re = re.compile(
        r"(?i)^\s*(?:2R|2L|3P|4R|4L|5|7|8|9|10R|10L|11R(?:S|I)?|11L(?:S|I)?|12R|12L|(?:right|left)(?:\s+(?:upper|middle|lower)\s+lobe)?\s+(?:mass|lesion|nodule)|lung\s+(?:mass|lesion|nodule)|mass|lesion|nodule)\b"
    )

    stations: list[str] = []

    for raw_line in (note_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if table_header_re.search(line) or table_row_re.match(line):
            continue

        if not sampling_hint_re.search(line):
            continue
        if sampling_negation_re.search(line):
            continue

        tokens = [m.group(1) for m in station_token_re.finditer(line)]
        has_subcarinal = bool(re.search(r"\bsubcarinal\b", line, re.IGNORECASE))
        if not tokens and not has_subcarinal:
            continue

        alpha_station_present = any(any(ch.isalpha() for ch in tok) for tok in tokens)

        for match in station_token_re.finditer(line):
            candidate = match.group(1)
            if not candidate:
                continue
            candidate_norm = normalize_station(candidate) if normalize_station is not None else candidate.strip().upper()
            if not candidate_norm:
                continue

            # Avoid interpreting bare digits (e.g., "7") as stations when they look like counts ("7 passes").
            if candidate_norm.isdigit() and not alpha_station_present:
                prefix = line[max(0, match.start() - 20) : match.start()]
                if re.search(r"(?i)\b(?:site|case|patient)\s+#?\s*$", prefix):
                    continue
                if not station_context_re.search(prefix):
                    continue

            if candidate_norm not in stations:
                stations.append(candidate_norm)

        if has_subcarinal and "7" not in stations:
            stations.append("7")

    return stations


def extract_bal(note_text: str) -> Dict[str, Any]:
    """Extract BAL (bronchoalveolar lavage) procedure indicator.

    Returns:
        Dict with 'bal' fields populated when detected, empty dict otherwise
    """
    note_text = _maybe_unescape_newlines(note_text or "")
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text = preferred_text or ""
    text_lower = text.lower()

    for pattern in BAL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            # Check for negation
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,40}" + pattern
            if not re.search(negation_check, text_lower, re.IGNORECASE):
                bal: dict[str, Any] = {"performed": True}

                # Location: prefer narrative phrasing over specimen/headers.
                loc_match = re.search(
                    r"(?i)\b(?:bronch(?:ial)?\s+alveolar\s+lavage|broncho[-\s]?alveolar\s+lavage|BAL)\b"
                    r"[^.\n]{0,60}\b(?:was\s+)?performed\b[^.\n]{0,60}\b(?:at|in)\b\s+(?P<loc>[^.\n]{3,140})",
                    text,
                )
                if loc_match:
                    loc = (loc_match.group("loc") or "").strip().strip(" ,;:-")
                    if loc and loc.lower() != "bal":
                        bal["location"] = loc

                instilled_match = re.search(
                    r"(?i)\b(?:instilled|infused)\s+(?P<num>\d{1,4})\s*(?:cc|ml)\b",
                    text,
                )
                if instilled_match:
                    try:
                        bal["volume_instilled_ml"] = int(instilled_match.group("num"))
                    except Exception:
                        pass
                else:
                    instilled_match = re.search(
                        r"(?i)\b(?P<num>\d{1,4})\s*(?:cc|ml)\b[^.\n]{0,40}\b(?:ns\s+)?(?:instilled|infused)\b",
                        text,
                    )
                    if instilled_match:
                        try:
                            bal["volume_instilled_ml"] = int(instilled_match.group("num"))
                        except Exception:
                            pass

                recovered_match = re.search(
                    r"(?i)\b(?:return(?:ed)?|recovered|suction\s*returned)\s+(?:with\s+)?(?P<num>\d{1,4})\s*(?:cc|ml)\b",
                    text,
                )
                if recovered_match:
                    try:
                        bal["volume_recovered_ml"] = int(recovered_match.group("num"))
                    except Exception:
                        pass
                else:
                    recovered_match = re.search(
                        r"(?i)\b(?P<num>\d{1,4})\s*(?:cc|ml)\b(?:\s*(?:was|were|is|are))?\s*(?:return(?:ed)?|recovered)\b",
                        text,
                    )
                    if recovered_match:
                        try:
                            bal["volume_recovered_ml"] = int(recovered_match.group("num"))
                        except Exception:
                            pass

                return {"bal": bal}
    return {}


def extract_whole_lung_lavage(note_text: str) -> Dict[str, Any]:
    """Extract whole-lung lavage and keep it distinct from BAL."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return {}

    lowered = text.lower()
    explicit_wll = any(re.search(pattern, lowered, re.IGNORECASE) for pattern in WHOLE_LUNG_LAVAGE_PATTERNS)
    contextual_wll = bool(
        "lavage" in lowered
        and (
            re.search(r"(?i)\b(?:pap|pulmonary\s+alveolar\s+proteinosis)\b", text)
            or re.search(r"(?i)\bdouble[-\s]+lumen\s+tube\b", text)
            or re.search(r"(?i)\blung\s+isolation\b", text)
            or re.search(r"(?i)\beffluent\b[^.\n]{0,80}\bclear", text)
        )
    )
    if not explicit_wll and not contextual_wll:
        return {}

    proc: dict[str, Any] = {"performed": True}

    side_match = re.search(
        r"(?i)\bwhole\s+lung\s+lavage\b[^.\n]{0,80}\((right|left)\s+lung\)"
        r"|\b(?:right|left)\s+lung\b[^.\n]{0,80}\b(?:was\s+)?lavaged\b",
        text,
    )
    if side_match:
        side = next((group for group in side_match.groups() if group), None)
        if side:
            proc["side"] = side.capitalize()

    volume_match = re.search(
        r"(?i)\btotal(?:\s+lavage)?\s+volume\b[^0-9]{0,20}(?P<vol>\d+(?:\.\d+)?)\s*l\b",
        text,
    )
    if volume_match:
        try:
            proc["total_volume_liters"] = float(volume_match.group("vol"))
        except Exception:
            pass

    cycles_match = re.search(r"(?i)\b(?P<count>\d+)\s+(?:lavage\s+)?cycles?\b", text)
    if cycles_match:
        try:
            proc["cycles"] = int(cycles_match.group("count"))
        except Exception:
            pass

    if re.search(r"(?i)\b(?:pap|pulmonary\s+alveolar\s+proteinosis)\b", text):
        proc["indication"] = "PAP"

    return {"whole_lung_lavage": proc}


def extract_bronchial_thermoplasty(note_text: str) -> Dict[str, Any]:
    """Extract bronchial thermoplasty sessions and treated lobes."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return {}

    lowered = text.lower()
    explicit_thermoplasty = any(
        re.search(pattern, lowered, re.IGNORECASE) for pattern in BRONCHIAL_THERMOPLASTY_PATTERNS
    )
    rf_context = bool(
        re.search(r"(?i)\brf\s+activations?\b", text)
        and re.search(r"(?i)\b(?:segmental|subsegmental|airway|bronch(?:i|us))\b", text)
    )
    if not explicit_thermoplasty and not rf_context:
        return {}

    proc: dict[str, Any] = {"performed": True}

    session_match = re.search(r"(?i)\bsession\s*(?P<num>\d+)\b", text)
    if session_match:
        try:
            proc["session_number"] = int(session_match.group("num"))
        except Exception:
            pass

    locations = _extract_lung_locations_from_text(text)
    if locations:
        proc["areas_treated"] = locations

    activations_match = re.search(r"(?i)\btotal\s+activations?\s*:\s*(?P<count>\d+)\b", text)
    if not activations_match:
        activations_match = re.search(r"(?i)\b(?P<count>\d+)\s+activations?\b", text)
    if activations_match:
        try:
            proc["number_of_activations"] = int(activations_match.group("count"))
        except Exception:
            pass

    return {"bronchial_thermoplasty": proc}


def is_true_therapeutic_aspiration(
    note_text: str,
    context_spans: list[tuple[int, int]],
) -> bool:
    """Gate therapeutic aspiration to clinically corroborated contexts."""
    text = note_text or ""
    lowered = text.lower()
    if not lowered.strip():
        return False

    contexts: list[str] = []
    for start, end in context_spans:
        s = max(0, int(start) - 240)
        e = min(len(text), int(end) + 240)
        if e > s:
            contexts.append(text[s:e])
    context_text = " ".join(contexts) if contexts else text
    context_lower = context_text.lower()

    obstruction_re = re.compile(
        r"(?i)\b(?:obstruct\w*|occlu\w*|plug\w*|impaction|large\s+burden|copious|tenacious|purulent|blood\s+clot)\b"
        r"|\bthick\s+(?:mucus|mucous|secretions?)\b"
    )
    material_clearance_re = re.compile(
        r"(?i)\b(?:mucus|mucous|secretions?|blood|clot(?:s)?|plug(?:s)?)\b[^.\n]{0,100}"
        r"\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?|remov(?:ed|al)?)\b"
        r"|\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?|remov(?:ed|al)?)\b[^.\n]{0,100}"
        r"\b(?:mucus|mucous|secretions?|blood|clot(?:s)?|plug(?:s)?)\b"
    )
    therapeutic_phrase = bool(
        re.search(r"(?i)\bthere?apeutic\s+(?:aspiration|suction(?:ing)?)\b", context_text)
    )
    location_mentions = {
        m.group(0).lower()
        for m in re.finditer(
            r"(?i)\b(?:trachea|carina|right\s+mainstem|left\s+mainstem|bronchus\s+intermedius|rul|rml|rll|lul|lll|lingula)\b",
            context_text,
        )
    }
    multi_site_therapeutic = therapeutic_phrase and len(location_mentions) >= 2

    has_corroboration = bool(
        obstruction_re.search(context_text)
        or material_clearance_re.search(context_text)
        or multi_site_therapeutic
    )
    if not has_corroboration:
        return False

    # Boilerplate filter: suppress templated concluding line when earlier findings are scant/minimal.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s and s.strip()]
    therapeutic_indices = [
        idx for idx, sentence in enumerate(sentences)
        if re.search(r"(?i)\bthere?apeutic\s+(?:aspiration|suction(?:ing)?)\b", sentence)
    ]
    if therapeutic_indices:
        first_idx = therapeutic_indices[0]
        earlier_text = " ".join(sentences[:first_idx]).lower()
        therapeutic_text = " ".join(sentences[idx] for idx in therapeutic_indices).lower()
        early_scant = bool(
            re.search(
                r"(?i)\b(?:scant|minimal|mild|small\s+amount\s+of)\b[^.\n]{0,40}\b(?:mucus|mucous|secretions?)\b",
                earlier_text,
            )
        )
        early_high_burden = bool(obstruction_re.search(earlier_text))
        template_like = bool(
            re.search(
                r"(?i)\b(?:following|after)\s+confirmation\s+of\s+hemostasis\b[^.]{0,140}\bthere?apeutic\s+aspiration\b",
                therapeutic_text,
            )
            or re.search(r"(?i)\bthere?apeutic\s+aspiration\s+of\s+all\s+endobronchial\s+secretions\b", therapeutic_text)
        )
        in_final_sentence = max(therapeutic_indices) >= max(0, len(sentences) - 2)
        if (template_like or in_final_sentence) and early_scant and not early_high_burden:
            return False

    return True


def extract_therapeutic_aspiration(note_text: str) -> Dict[str, Any]:
    """Extract therapeutic aspiration procedure indicator.

    Distinguishes therapeutic aspiration (mucus plug removal, clot removal)
    from routine suctioning which is not separately billable.

    Returns:
        Dict with 'therapeutic_aspiration': {'performed': True, 'material': <str>}
        if therapeutic aspiration detected, empty dict otherwise
    """
    note_text = _maybe_unescape_newlines(note_text or "")
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text or note_text)
    text = preferred_text or ""
    text_lower = text.lower()

    # Check for routine suction first (exclude these)
    for pattern in ROUTINE_SUCTION_PATTERNS:
        if re.search(pattern, text_lower):
            return {}

    search_texts = [text]
    if note_text != text:
        search_texts.append(note_text)

    detail_hint_text = text_lower
    for candidate_text in search_texts:
        candidate_lower = candidate_text.lower()
        for pattern in THERAPEUTIC_ASPIRATION_PATTERNS:
            match = re.search(pattern, candidate_lower, re.IGNORECASE)
            if not match:
                continue

            # Check for negation
            negation_check = r"\b(?:no|not|without)\b[^.\n]{0,40}" + pattern
            if re.search(negation_check, candidate_lower, re.IGNORECASE):
                continue
            if not is_true_therapeutic_aspiration(candidate_text, [(int(match.start()), int(match.end()))]):
                continue

            # Determine material type from a local window around the match
            window = 250
            window_start = max(0, match.start() - window)
            window_end = min(len(candidate_lower), match.end() + window)
            local_window = candidate_lower[window_start:window_end]

            material = "Other"
            if "purulent" in local_window or "pus" in local_window:
                material = "Purulent secretions"
            elif any(token in local_window for token in ("mucus", "mucous", "plug")):
                material = "Mucus plug"
            elif any(token in local_window for token in ("blood", "clot", "bloody", "blood-tinged")):
                material = "Blood/clot"
            elif "secretions" in local_window:
                material = "Mucus plug"

            # If the local window is sparse/header-only, prefer detail-section cues.
            if material == "Other":
                if "purulent" in detail_hint_text or "pus" in detail_hint_text:
                    material = "Purulent secretions"
                elif any(token in detail_hint_text for token in ("mucus", "mucous", "plug", "secretions")):
                    material = "Mucus plug"
            elif material == "Blood/clot":
                # Avoid downgrading explicit clot removal to mucus-only just because secretions
                # appear elsewhere in the detail section.
                if "clot" not in local_window and "blood clot" not in local_window:
                    if "purulent" in detail_hint_text or "pus" in detail_hint_text:
                        material = "Purulent secretions"
                    elif any(token in detail_hint_text for token in ("mucus", "mucous", "plug", "secretions")):
                        material = "Mucus plug"

            location: str | None = None
            loc_match = re.search(
                r"(?i)\btherapeutic\s+aspiration\b[^.\n]{0,140}\bclean\s+out\b\s+(?:the\s+)?(?P<loc>[^.\n]{3,400}?)\s+\bfrom\b",
                candidate_text,
            )
            if loc_match:
                candidate = (loc_match.group("loc") or "").strip().strip(" ,;:-")
                if candidate:
                    location = candidate
            if not location:
                seg_match = re.search(
                    r"(?is)\bsegments?\s+cleared\s*:\s*(?P<locs>[^\n]{8,600})",
                    candidate_text,
                )
                if seg_match:
                    locs = (seg_match.group("locs") or "").strip()
                    locs = re.sub(r"\s+", " ", locs).strip(" .;:-")
                    if locs:
                        location = locs
            if not location:
                secretions_match = re.search(
                    r"(?is)\bsecretions?\b[^.\n]{0,260}\bthroughout\b[^.\n]{0,20}(?P<locs>[^.\n]{3,220})\.[^.\n]{0,140}\btherapeutic\s+aspiration\b",
                    candidate_text,
                )
                if secretions_match:
                    locs = (secretions_match.group("locs") or "").strip()
                    locs = re.sub(r"\s+", " ", locs).strip(" ,;:-")
                    if locs:
                        location = locs
            if not location:
                extracted_match = re.search(
                    r"(?i)\b(?:plug|mucus|clot|secretions?)\b[^.\n]{0,140}\b(?:extract(?:ed|ion)?|removed|suctioned|cleared)\b"
                    r"[^.\n]{0,40}\bfrom\s+(?P<loc>[^.\n]{2,120})",
                    candidate_text,
                )
                if extracted_match:
                    candidate_loc = (extracted_match.group("loc") or "").strip().strip(" ,;:-")
                    candidate_loc = re.split(r"(?i)\bwith\b|\busing\b", candidate_loc, maxsplit=1)[0].strip(" ,;:-")
                    if candidate_loc:
                        location = candidate_loc

            result = {"therapeutic_aspiration": {"performed": True}}
            result["therapeutic_aspiration"]["material"] = material
            if location:
                result["therapeutic_aspiration"]["location"] = location
            return result
    return {}


def extract_intubation(note_text: str) -> Dict[str, Any]:
    """Extract emergency endotracheal intubation (31500) indicator.

    This is intentionally conservative to avoid coding routine anesthesia intubation.
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    fiberoptic = bool(re.search(r"\bfiber\s*optic\s+intubat(?:ion|ed|ing)\b", text_lower)) or bool(
        re.search(r"\bfiberoptic\s+intubat(?:ion|ed|ing)\b", text_lower)
    )
    endotracheal_intubation = bool(re.search(r"\bendotracheal\s+intubat(?:ion|ed|ing)\b", text_lower))
    ett_placed = bool(
        re.search(
            r"\b(?:ett|endotracheal\s+tube)\b[^.\n]{0,60}\b(?:was\s+)?(?:placed|inserted)\b",
            text_lower,
        )
    )
    intubation_mentioned = bool(re.search(r"\bintubat(?:ion|ed|ing)\b", text_lower))

    # Guardrail: only treat as 31500-eligible when explicitly special/emergent.
    emergency_context = bool(re.search(r"\b(?:emergent|emergency|code\s+blue|crash)\b", text_lower))
    difficult_context = bool(
        re.search(r"\b(?:difficult\s+airway|failed\s+intubation|multiple\s+attempts)\b", text_lower)
    )
    selective_context = bool(
        re.search(
            r"\b(?:selective|mainstem)\b[^.\n]{0,80}\bintubat(?:ion|ed|ing)\b|\bintubat(?:ion|ed|ing)\b[^.\n]{0,80}\b(?:mainstem|bronchus)\b",
            text_lower,
            re.IGNORECASE,
        )
        or re.search(r"\binto\s+the\s+(?:right|left)\s+main(?:\s*|-)?stem\b", text_lower, re.IGNORECASE)
    )
    procedural_verb_context = bool(
        re.search(
            r"\bintubat(?:ion|ed|ing)\b[^.\n]{0,80}\b(?:perform(?:ed)?|place(?:d)?|insert(?:ed)?|advance(?:d)?)\b"
            r"|\b(?:perform(?:ed)?|place(?:d)?|insert(?:ed)?|advance(?:d)?)\b[^.\n]{0,80}\bintubat(?:ion|ed|ing)\b",
            text_lower,
            re.IGNORECASE,
        )
    )

    tube_match = re.search(
        r"\b(?P<size>\d{1,2}(?:\.\d+)?)\s*(?:mm\s*)?(?:(?P<mlt>mlt)\s*)?(?:ett|endotracheal\s+tube)\b"
        r"|\b(?P<size2>\d{1,2}(?:\.\d+)?)\s*(?P<mlt2>mlt)\b",
        text_lower,
    )
    tube_size = None
    if tube_match:
        size = tube_match.group("size") or tube_match.group("size2")
        mlt = tube_match.group("mlt") or tube_match.group("mlt2")
        if size:
            tube_size = f"{size} MLT" if mlt else size

    route = None
    if re.search(r"\bvia\s+oral\b|\boral\s+pathway\b|\borotracheal\b", text_lower):
        route = "Oral"
    elif re.search(r"\bvia\s+nasal\b|\bnasal\s+pathway\b|\bnasotracheal\b", text_lower):
        route = "Nasal"

    special_context = fiberoptic or emergency_context or difficult_context or selective_context
    if not special_context:
        return {}
    if not (intubation_mentioned or endotracheal_intubation or ett_placed):
        return {}
    if selective_context and not (fiberoptic or endotracheal_intubation or ett_placed or procedural_verb_context):
        return {}

    proc: dict[str, Any] = {"performed": True}
    if fiberoptic:
        proc["method"] = "Fiberoptic"
    elif endotracheal_intubation:
        proc["method"] = "Endotracheal"
    if route:
        proc["route"] = route
    if tube_size:
        proc["tube_size"] = tube_size

    return {"intubation": proc}


def _has_airway_stent_action(text_lower: str, *, action_patterns: list[str]) -> bool:
    """Return True if text documents an airway stent action (placement/removal)."""
    if not text_lower:
        return False
    device_hit = any(re.search(pat, text_lower, re.IGNORECASE) for pat in AIRWAY_STENT_DEVICE_PATTERNS)
    if not device_hit:
        return False
    return any(re.search(pat, text_lower, re.IGNORECASE) for pat in action_patterns)


def _stent_action_window_hit(text_lower: str, *, verbs: list[str], max_chars: int = 80) -> bool:
    """Return True if a stent device keyword and an action verb co-occur nearby."""
    if not text_lower:
        return False

    device = (
        r"(?:stent|y-?\s*stent|dumon|aero(?:stent)?|aerstent|ultraflex|sems|"
        r"silicone\s+stent|metal(?:lic)?\s+stent|bonostent|vascular\s+plug|endobronchial\s+plug)"
    )
    verb_union = "|".join(verbs)
    span = max(0, int(max_chars))
    patterns = [
        rf"\b{device}\b[^.\n]{{0,{span}}}\b(?:{verb_union})\w*\b",
        rf"\b(?:{verb_union})\w*\b[^.\n]{{0,{span}}}\b{device}\b",
    ]
    return any(re.search(p, text_lower, re.IGNORECASE) for p in patterns)


_STENT_PLACEMENT_NEGATION_PATTERNS = [
    r"\bdecision\b[^.\n]{0,80}\bnot\b[^.\n]{0,40}\b(?:place|insert|deploy|perform)\w*\b[^.\n]{0,80}\bstent\b",
    r"\bno\s+additional\s+stents?\b[^.\n]{0,40}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\b",
    r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,40}\bstent(?:s)?\b[^.\n]{0,40}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\b",
    r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,80}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\w*\b[^.\n]{0,80}\bstent(?:s)?\b",
    r"\b(?:refus(?:ed|al)|reluctan(?:t|ce)|hesitan(?:t|cy)|did\s+not\s+want)\b[^.\n]{0,80}\bstent(?:s)?\b[^.\n]{0,80}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\b",
]

_STENT_PLACEMENT_VERBS_RE = re.compile(
    r"\b(place|placed|placement|deploy|deployed|insert|inserted|advance|advanced|seat|seated|positioned)\b",
    re.IGNORECASE,
)
_STENT_REMOVAL_VERBS_RE = re.compile(
    r"\b(remov|retriev|extract|explant|pull|grasp|peel)\w*\b",
    re.IGNORECASE,
)

_STENT_BRAND_PATTERNS: dict[str, tuple[str, str | None]] = {
    "Dumon": (r"\bdumon\b", "Silicone - Dumon"),
    "Novatech": (r"\bnovatec(?:h)?\b", "Silicone - Novatech"),
    "Ultraflex": (r"\bultraflex\b", "Other"),
    "Aero": (r"\baero(?:stent)?\b|\baerstent\b", "Other"),
    "Atrium iCast": (r"\b(?:atrium\s+)?icast\b", "Other"),
}


def _stent_placement_negated(text_lower: str) -> bool:
    return any(re.search(pat, text_lower, re.IGNORECASE) for pat in _STENT_PLACEMENT_NEGATION_PATTERNS)


def _select_stent_brand(text_lower: str, action: str | None) -> tuple[str | None, str | None]:
    candidates: list[dict[str, object]] = []
    for brand, (pattern, stent_type) in _STENT_BRAND_PATTERNS.items():
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if not matches:
            continue
        placement_hits = 0
        removal_hits = 0
        stent_context_hits = 0
        for match in matches:
            window_start = max(0, match.start() - 80)
            window_end = min(len(text_lower), match.end() + 80)
            window = text_lower[window_start:window_end]
            # Guardrail: require stent context near the brand token. This prevents
            # instrument-only matches (e.g., "Dumon ... bronchoscope") from
            # hallucinating a stent brand/type.
            if "stent" not in window:
                continue
            if "bronchoscop" in window and "stent" not in window:
                continue
            stent_context_hits += 1
            if _STENT_PLACEMENT_VERBS_RE.search(window):
                placement_hits += 1
            if _STENT_REMOVAL_VERBS_RE.search(window):
                removal_hits += 1
        if stent_context_hits == 0:
            continue
        candidates.append(
            {
                "brand": brand,
                "stent_type": stent_type,
                "placement_hits": placement_hits,
                "removal_hits": removal_hits,
            }
        )

    if not candidates:
        return None, None

    if action and action.lower().startswith("remov"):
        best = max(candidates, key=lambda c: (c["removal_hits"], c["placement_hits"]))
    else:
        best = max(candidates, key=lambda c: (c["placement_hits"], c["removal_hits"]))

    return best["brand"], best["stent_type"]


def classify_stent_action(note_text: str) -> Dict[str, Any]:
    """Classify airway stent action with pre-existing stent gating."""
    full_text = note_text or ""
    full_lower = full_text.lower()
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text = preferred_text or ""
    text_lower = text.lower()
    if not text_lower.strip():
        return {}

    placement = bool(
        _stent_action_window_hit(
            text_lower,
            verbs=["placed", "insert", "inserted", "deploy", "deployed", "implant", "advanced"],
        )
        or re.search(
            r"(?i)\bto\s+place\b[^.\n]{0,80}\b(?:stent|bonostent|vascular\s+plug|endobronchial\s+plug)\b",
            text,
        )
        or re.search(
            r"(?i)\b(?:new|another|replacement|custom)\b[^.\n]{0,80}\b(?:stent|bonostent|vascular\s+plug|endobronchial\s+plug)\b",
            text,
        )
        or re.search(
            r"(?i)\b(?:bonostent|stent|vascular\s+plug|endobronchial\s+plug)\b[^.\n]{0,100}\b(?:was\s+)?(?:placed|inserted|deployed)\b"
            r"|\b(?:placed|inserted|deployed)\b[^.\n]{0,100}\b(?:bonostent|stent|vascular\s+plug|endobronchial\s+plug)\b",
            text,
        )
    )
    removal = bool(
        _stent_action_window_hit(
            text_lower,
            verbs=["remov", "retriev", "extract", "explant", "pull", "withdraw", "exchange", "replace"],
        )
        or re.search(
            r"(?i)\b(?:stent(?:s)?|bonostent)\b[^.\n]{0,120}\b(?:removed|retrieved|extracted|withdrawn)\b",
            text,
        )
    )
    if _stent_placement_negated(text_lower) and not removal:
        placement = False

    explicit_exchange = bool(
        re.search(r"(?i)\b(?:exchange|exchanged|revision|reposition(?:ing)?|revis(?:ed|ion))\b", text)
        or re.search(r"(?i)\b(?:replace|replaced|replacement)\b[^.\n]{0,80}\bstent\b", text)
    )

    # If stent/plug removal is documented before any same-session placement event,
    # treat as an exchange/revision of an existing airway device.
    first_removal_event = re.search(
        r"(?i)\b(?:stent(?:s)?|bonostent|vascular\s+plug|endobronchial\s+plug)\b[^.\n]{0,120}"
        r"\b(?:remov(?:e|ed|al)|retriev(?:e|ed|al)|extract(?:ed|ion)?|withdrawn|explant(?:ed|ation)?)\b"
        r"|\b(?:remov(?:e|ed|al)|retriev(?:e|ed|al)|extract(?:ed|ion)?|withdrawn|explant(?:ed|ation)?)\b[^.\n]{0,120}"
        r"\b(?:stent(?:s)?|bonostent|vascular\s+plug|endobronchial\s+plug)\b",
        text,
    )
    first_placement_event = re.search(
        r"(?i)\b(?:stent(?:s)?|bonostent|vascular\s+plug|endobronchial\s+plug)\b[^.\n]{0,120}"
        r"\b(?:placed|inserted|deployed|implanted)\b"
        r"|\b(?:placed|inserted|deployed|implanted)\b[^.\n]{0,120}\b(?:stent(?:s)?|bonostent|vascular\s+plug|endobronchial\s+plug)\b"
        r"|\bto\s+place\b[^.\n]{0,120}\b(?:stent(?:s)?|bonostent|vascular\s+plug|endobronchial\s+plug)\b",
        text,
    )
    removal_before_placement = bool(
        first_removal_event
        and first_placement_event
        and int(first_removal_event.start()) < int(first_placement_event.start())
    )
    if removal_before_placement:
        explicit_exchange = True

    early_text = full_lower[:2600]
    preexisting_stent = bool(
        re.search(
            r"(?i)\b(?:known|existing|prior|previously|pre[-\s]?existing)\b[^.\n]{0,120}\b(?:stent|bonostent)\b",
            early_text,
        )
        or re.search(
            r"(?i)\b(?:stent|bonostent)\b[^.\n]{0,120}\b(?:placed|placement)\b[^.\n]{0,80}\b(?:\d+\s*(?:day|week|month|year)s?\s+ago|ago)\b",
            early_text,
        )
        or re.search(r"(?i)\bafter\s+placement\b", early_text)
        or removal_before_placement
    )

    if placement and removal and (preexisting_stent or explicit_exchange):
        action_type = "revision"
    elif placement:
        action_type = "placement"
    elif removal:
        action_type = "removal"
    elif explicit_exchange:
        action_type = "revision"
    else:
        action_type = "assessment_only"

    stent_type: str | None = None
    if re.search(r"(?i)\by-?\s*stent\b", text):
        stent_type = "Y-Stent"
    else:
        _brand, brand_type = _select_stent_brand(text_lower, action_type)
        stent_type = brand_type

    size_candidates: list[str] = []
    for match in re.finditer(
        r"(?i)\b\d+(?:\.\d+)?\s*(?:x|×)\s*\d+(?:\.\d+)?(?:\s*(?:x|×)\s*\d+(?:\.\d+)?)?\s*(?:mm|cm)?\b",
        text,
    ):
        raw = re.sub(r"\s+", "", (match.group(0) or "").strip())
        window = text_lower[max(0, match.start() - 80) : min(len(text_lower), match.end() + 80)]
        if raw and "stent" in window and raw not in size_candidates:
            size_candidates.append(raw)
    limb_lengths_mm: list[int] = []
    for match in re.finditer(
        r"(?i)\b(?:tracheal|right\s+mainstem|left\s+mainstem)\s+limb\b[^.\n]{0,30}\b(\d{1,3})\s*mm\b",
        text,
    ):
        try:
            val = int(match.group(1))
        except Exception:
            continue
        if 1 <= val <= 200 and val not in limb_lengths_mm:
            limb_lengths_mm.append(val)

    return {
        "action_type": action_type,
        "placement": placement,
        "removal": removal,
        "exchange": bool(placement and removal and (preexisting_stent or explicit_exchange)),
        "preexisting_stent": preexisting_stent,
        "stent_type": stent_type,
        "sizes": size_candidates,
        "limb_lengths_mm": limb_lengths_mm,
    }


def extract_airway_dilation(note_text: str) -> Dict[str, Any]:
    """Extract airway dilation indicator (balloon dilation)."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    def _max_balloon_diameter_from_text(raw_text: str) -> float | None:
        best: float | None = None
        for sentence in re.split(r"(?<=[.!?\n])\s+", raw_text or ""):
            local = sentence.lower()
            if "balloon" not in local:
                continue

            candidates: list[float] = []

            for seq_match in re.finditer(r"\b\d{1,2}(?:\.\d+)?(?:\s*[/-]\s*\d{1,2}(?:\.\d+)?){1,4}\b", local):
                for token in re.findall(r"\d{1,2}(?:\.\d+)?", seq_match.group(0) or ""):
                    try:
                        val = float(token)
                    except Exception:
                        continue
                    if 4 <= val <= 25:
                        candidates.append(val)

            for mm_match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*mm\b", local):
                prefix = local[max(0, mm_match.start() - 4) : mm_match.start()]
                if "x" in prefix or "×" in prefix:
                    continue
                try:
                    val = float(mm_match.group(1))
                except Exception:
                    continue
                if 4 <= val <= 25:
                    candidates.append(val)

            for to_match in re.finditer(r"\bto\s+(\d+(?:\.\d+)?)\s*mm\b", local):
                try:
                    val = float(to_match.group(1))
                except Exception:
                    continue
                if 4 <= val <= 25:
                    candidates.append(val)

            if candidates:
                local_best = max(candidates)
                if best is None or local_best > best:
                    best = local_best

        return best

    for pattern in AIRWAY_DILATION_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if not match:
            continue
        negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
        if re.search(negation_check, text_lower, re.IGNORECASE):
            continue
        proc: dict[str, Any] = {"performed": True, "method": "Balloon"}
        window_start = max(0, match.start() - 200)
        window_end = min(len(text_lower), match.end() + 240)
        window = text_lower[window_start:window_end]

        balloon_diameter_mm: float | None = None
        explicit_max_diameter_documented = False

        # Prefer explicitly documented maximal diameter over balloon kit size sequences.
        max_diam_match = re.search(
            r"(?i)\b(?:max(?:imal|imum)|up\s+to)\b[^.\n]{0,60}\b(?:diameter|size)\b[^.\n]{0,60}\b(?P<diam>\d+(?:\.\d+)?)\s*mm\b",
            window,
        )
        if max_diam_match:
            try:
                val = float(max_diam_match.group("diam"))
                if 4 <= val <= 25:
                    balloon_diameter_mm = val
                    explicit_max_diameter_documented = True
            except Exception:
                balloon_diameter_mm = None

        # Common dilation balloon documentation: "<diam sequence> x <length> mm balloon"
        # Example: "Merit 6-7-8 x 20 mm balloon" -> diameter max=8mm; length=20mm (not stored in schema).
        if balloon_diameter_mm is None:
            seq_x_match = re.search(
                r"(?i)\b(?P<seq>\d{1,2}(?:\s*[/-]\s*\d{1,2}){1,4})\s*(?:x|×|by)\s*(?P<len>\d+(?:\.\d+)?)\s*mm\b"
                r"[^.]{0,60}\bballoon\b",
                window,
            )
            if seq_x_match:
                try:
                    nums = [float(n) for n in re.findall(r"\d{1,2}(?:\.\d+)?", seq_x_match.group("seq") or "")]
                    nums = [n for n in nums if 4 <= n <= 25]
                    if nums:
                        balloon_diameter_mm = max(nums)
                except Exception:
                    balloon_diameter_mm = None

        # Next-most common: "<diam> x <length> mm balloon"
        if balloon_diameter_mm is None:
            dim_x_match = re.search(
                r"(?i)\b(?P<diam>\d+(?:\.\d+)?)\s*(?:mm|cm)?\s*(?:x|×|by)\s*(?P<len>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm)\b"
                r"[^.]{0,60}\bballoon\b",
                window,
            )
            if dim_x_match:
                try:
                    diam = float(dim_x_match.group("diam"))
                    unit = (dim_x_match.group("unit") or "mm").lower().strip()
                    if unit == "cm":
                        diam *= 10.0
                    if 4 <= diam <= 25:
                        balloon_diameter_mm = diam
                except Exception:
                    balloon_diameter_mm = None

        # Fallback: pick an mm value in range, avoiding obvious "x <length> mm" length tokens.
        if balloon_diameter_mm is None:
            candidates: list[float] = []
            for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*mm\b", window):
                prefix = window[max(0, m.start() - 3) : m.start()]
                if "x" in prefix or "×" in prefix:
                    continue
                try:
                    val = float(m.group(1))
                except Exception:
                    continue
                if 4 <= val <= 25:
                    candidates.append(val)
            if candidates:
                balloon_diameter_mm = max(candidates)

        # Capture balloon size sequences like "6/7/8 balloon" when mm values are absent.
        if balloon_diameter_mm is None:
            seq_match = re.search(r"\b(\d{1,2})(?:\s*[/-]\s*(\d{1,2}))+\b", window)
            if seq_match and re.search(r"\bballoon\b", window):
                try:
                    nums = [float(n) for n in re.findall(r"\d{1,2}", seq_match.group(0) or "")]
                    nums = [n for n in nums if 4 <= n <= 25]
                    if nums:
                        balloon_diameter_mm = max(nums)
                except Exception:
                    balloon_diameter_mm = None

        if balloon_diameter_mm is not None:
            proc["balloon_diameter_mm"] = balloon_diameter_mm

        # Multi-dilation notes can contain >1 balloon event; retain the maximum
        # explicitly documented diameter when it is larger than the local window.
        global_max_diameter = _max_balloon_diameter_from_text(preferred_text or "")
        if global_max_diameter is not None and not explicit_max_diameter_documented:
            existing = proc.get("balloon_diameter_mm")
            if not isinstance(existing, (int, float)) or float(global_max_diameter) > float(existing):
                proc["balloon_diameter_mm"] = float(global_max_diameter)

        # Best-effort location for downstream bundling (RUL/RML/RLL/LUL/LLL etc).
        location_map: tuple[tuple[str, str], ...] = (
            ("RUL", r"(?i)\b(?:rul|right\s+upper(?:\s+lobe)?)\b"),
            ("RML", r"(?i)\b(?:rml|right\s+middle(?:\s+lobe)?)\b"),
            ("RLL", r"(?i)\b(?:rll|right\s+lower(?:\s+lobe)?)\b"),
            ("LUL", r"(?i)\b(?:lul|left\s+upper(?:\s+lobe)?)\b"),
            ("LLL", r"(?i)\b(?:lll|left\s+lower(?:\s+lobe)?)\b"),
            ("Lingula", r"(?i)\blingula\b"),
            ("RMS", r"(?i)\b(?:rms|right\s+main(?:\s*|-)?stem)\b"),
            ("LMS", r"(?i)\b(?:lms|left\s+main(?:\s*|-)?stem)\b"),
            ("BI", r"(?i)\b(?:bi|bronchus\s+intermedius)\b"),
            ("Trachea", r"(?i)\btrachea(?:l)?\b"),
        )
        best_loc: str | None = None
        best_pos = -1
        for loc, loc_re in location_map:
            for m_loc in re.finditer(loc_re, window):
                if m_loc.start() >= best_pos:
                    best_pos = m_loc.start()
                    best_loc = loc
        if best_loc:
            proc["location"] = best_loc

        # Pre/post diameter heuristics.
        pre_match = re.search(r"(?i)\bdown\s+to\s*(\d+(?:\.\d+)?)\s*mm\b", window)
        if pre_match:
            try:
                proc["pre_dilation_diameter_mm"] = float(pre_match.group(1))
            except Exception:
                pass
        post_match = re.search(r"(?i)\bdilat\w*\b[^.\n]{0,40}\bto\s*(\d+(?:\.\d+)?)\s*mm\b", window)
        if post_match:
            try:
                proc["post_dilation_diameter_mm"] = float(post_match.group(1))
            except Exception:
                pass

        # Target anatomy heuristic: stent expansion vs stenosis/stricture dilation.
        try:
            if re.search(
                r"(?i)\bstent\b[^.\n]{0,80}\b(?:expand|dilat)\w*\b|\b(?:expand|dilat)\w*\b[^.\n]{0,80}\bstent\b",
                window,
            ):
                proc["target_anatomy"] = "Stent expansion"
            elif re.search(r"(?i)\b(?:stenos|strictur|narrow|web)\w*\b", window):
                proc["target_anatomy"] = "Stenosis"
        except Exception:
            pass
        return {"airway_dilation": proc}

    return {}


def extract_airway_stent(note_text: str) -> Dict[str, Any]:
    """Extract airway stent indicator(s) with conservative action guesses.

    Notes:
    - If both placement and removal are present, mark the *primary* stent event as a revision
      and set airway_stent_removal=True so 31638 can be derived.
    - If removal is present without placement, set airway_stent_removal=True so
      31638 can be derived.
    - If both a new stent placement and a revision of existing stents are documented,
      emit `airway_stent` for the placement and `airway_stent_revision` for the revision
      (schema must support the latter).
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}
    classified = classify_stent_action(note_text) or {}
    classified_action_type = str(classified.get("action_type") or "").strip().lower()
    preexisting_stent = bool(classified.get("preexisting_stent"))
    classified_sizes = [
        str(v).strip() for v in (classified.get("sizes") or []) if isinstance(v, str) and str(v).strip()
    ]

    def _has_nonnegated_strong_stent_placement(text: str) -> bool:
        if not text:
            return False
        strong_verbs = ["deploy", "insert", "advance", "seat", "deliver", "implant"]
        device = (
            r"(?:stent|y-?\s*stent|dumon|aero(?:stent)?|aerstent|ultraflex|sems|"
            r"silicone\s+stent|metal(?:lic)?\s+stent|bonostent|vascular\s+plug|endobronchial\s+plug)"
        )
        verb_union = "|".join(strong_verbs)
        span = 40
        patterns = [
            rf"\b{device}\b[^.\n]{{0,{span}}}\b(?:{verb_union})\w*\b",
            rf"\b(?:{verb_union})\w*\b[^.\n]{{0,{span}}}\b{device}\b",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                window_start = max(0, m.start() - 160)
                window_end = min(len(text), m.end() + 140)
                window = text[window_start:window_end]
                if re.search(r"(?i)\b(?:previously|prior)\b", window):
                    if not re.search(r"(?i)\b(?:new|custom|replacement|another|additional)\b", window):
                        continue
                if _stent_placement_negated(window.lower()):
                    continue
                # Hypothetical-only guard ("would insert a stent", "plan to insert stent")
                if re.search(
                    r"(?i)\b(?:consider(?:ed|ation)?|discussion|plan(?:ned)?|would|if\s+needed|may)\b",
                    window,
                ) and not re.search(r"(?i)\b(?:successfully|was)\b[^.\n]{0,60}\b(?:deployed|inserted)\b", window):
                    continue
                return True
        return False

    def _infer_airway_site_last(raw: str) -> str | None:
        if not raw:
            return None
        patterns: tuple[tuple[str, str], ...] = (
            ("Trachea", r"(?i)\btrachea(?:l)?\b"),
            ("Carina (Y)", r"(?i)\bcarina\b|\by-?\s*stent\b"),
            ("Right mainstem", r"(?i)\b(?:rms|right\s+main(?:\s*|-)?stem)\b"),
            ("Left mainstem", r"(?i)\b(?:lms|left\s+main(?:\s*|-)?stem)\b"),
            ("Bronchus intermedius", r"(?i)\b(?:bi|bronchus\s+intermedius)\b"),
            ("RUL", r"(?i)\b(?:rul|right\s+upper(?:\s+lobe)?)\b"),
            ("RML", r"(?i)\b(?:rml|right\s+middle(?:\s+lobe)?)\b"),
            ("RLL", r"(?i)\b(?:rll|right\s+lower(?:\s+lobe)?)\b"),
            ("LUL", r"(?i)\b(?:lul|left\s+upper(?:\s+lobe)?)\b"),
            ("LLL", r"(?i)\b(?:lll|left\s+lower(?:\s+lobe)?)\b"),
            ("Lingula", r"(?i)\blingula\b"),
        )
        best_loc: str | None = None
        best_pos = -1
        for loc, pat in patterns:
            for m in re.finditer(pat, raw):
                if m.start() >= best_pos:
                    best_pos = m.start()
                    best_loc = loc
        return best_loc

    def _sentence_window_around(raw: str, pos: int, *, radius: int = 260) -> str:
        if not raw:
            return ""
        start = max(0, pos - radius)
        end = min(len(raw), pos + radius)
        left_boundary = max(raw.rfind(".", 0, pos), raw.rfind("\n", 0, pos))
        if left_boundary >= start:
            start = left_boundary + 1
        right_boundary_candidates = [b for b in (raw.find(".", pos), raw.find("\n", pos)) if b != -1]
        if right_boundary_candidates:
            right_boundary = min(right_boundary_candidates)
            if right_boundary <= end:
                end = right_boundary
        return raw[start:end].strip()

    def _infer_airway_site_for_stent_context(raw: str, pos: int) -> str | None:
        if not raw:
            return None
        sentence = _sentence_window_around(raw, pos)
        if sentence:
            segments = [seg.strip() for seg in re.split(r"(?<=[.;])\s+|\n+", sentence) if seg.strip()]
            placement_segments = [
                seg
                for seg in segments
                if re.search(r"(?i)\bstent\b", seg)
                and re.search(
                    r"(?i)\b(?:deploy(?:ed|ment)?|insert(?:ed|ion)?|place(?:d|ment)?|advance(?:d|ment)?|implant(?:ed|ation)?|reposition(?:ed|ing)?)\b",
                    seg,
                )
            ]
            for segment in reversed(placement_segments or segments):
                inferred = _infer_airway_site_last(segment)
                if inferred:
                    return inferred

        around = raw[max(0, pos - 180) : min(len(raw), pos + 220)]
        inferred = _infer_airway_site_last(around)
        if inferred:
            return inferred

        lookback = raw[max(0, pos - 700) : pos]
        return _infer_airway_site_last(lookback)

    def _has_nonhistorical_placed_stent(text: str) -> bool:
        """Detect new-stent placement from 'placed' while excluding history-only phrasing.

        Example to exclude: "previously placed stent in place" (existing device, not new placement).
        """
        if not text:
            return False
        placed_re = re.compile(
            r"(?i)\bplaced\b[^.\n]{0,60}\b(?:stent|bonostent)\b|\b(?:stent|bonostent)\b[^.\n]{0,60}\bplaced\b"
        )
        for match in placed_re.finditer(text):
            window_start = max(0, match.start() - 140)
            window_end = min(len(text), match.end() + 120)
            window = text[window_start:window_end]
            if re.search(r"(?i)\b(?:previously|prior)\b", window):
                # Allow explicit "new/custom/replacement" placements even in a sentence that
                # also references a prior device.
                if not re.search(r"(?i)\b(?:new|custom|replacement|another|additional)\b", window):
                    continue
            # Avoid treating generic positioning phrases as placement evidence.
            if re.search(r"(?i)\bstent\b[^.\n]{0,80}\bin\s+(?:good\s+)?position\b|\bstent\b[^.\n]{0,80}\bin\s+place\b", window):
                if not re.search(r"(?i)\b(?:deploy|insert|advance|seat|deliver|implant)\w*\b", window):
                    continue
            if _stent_placement_negated(window.lower()):
                continue
            return True
        return False

    placement_negated = _stent_placement_negated(text_lower)
    # Prefer proximity-based evidence of actual action (avoids history-only mentions).
    strong_placement_window_hit = _has_nonnegated_strong_stent_placement(text_lower)
    placed_stent_hit = _has_nonhistorical_placed_stent(text_lower)
    has_placement = strong_placement_window_hit or placed_stent_hit

    # Removal/exchange detection must be conservative: "bronchoscope was removed and the stent advanced"
    # is a common placement workflow and should not be treated as stent removal.
    removal_window_hit = _stent_action_window_hit(
        text_lower,
        # Prefer stronger, stent-specific cues; handle generic "removed" only when stent is the subject.
        verbs=["retriev", "extract", "explant", "exchang", "replac"],
    )
    removed_stent_hit = bool(
        re.search(
            r"(?i)\b(?:"
            r"stent(?:s)?\b[^.\n]{0,80}\b(?:remov(?:e|ed|al)|retriev(?:e|ed|al)|extract(?:ed|ion)?|explant(?:ed|ation)?|exchange(?:d)?|replac(?:ed|ement)?)\b"
            r"|"
            r"(?:remov(?:e|ed|al)|retriev(?:e|ed|al)|extract(?:ed|ion)?|explant(?:ed|ation)?|exchange(?:d)?|replac(?:ed|ement)?)\b[^.\n]{0,80}\bstent(?:s)?\b"
            r")",
            text_lower,
        )
    )
    pull_out_hit = bool(re.search(r"(?i)\bstent\b[^.\n]{0,40}\bpull(?:ed)?\s+out\b", text_lower))
    has_removal = bool(removal_window_hit or removed_stent_hit or pull_out_hit)

    revision_window_hit = _stent_action_window_hit(
        text_lower,
        verbs=["revis", "reposition", "adjust", "manipulat", "exchang", "replac"],
    )
    proximally_hint = bool(
        re.search(
            r"\b(?:bring|brought|pull|pulled|move|moved|advance|advanced)\b[^.\n]{0,80}\bproxim(?:al|ally)\b",
            text_lower,
            re.IGNORECASE,
        )
    )
    revision_hint = revision_window_hit or proximally_hint
    # If removal is explicitly documented without placement, treat as removal
    # rather than revision/repositioning.
    if has_removal and not has_placement:
        revision_hint = False

    # Classification guardrail: do not let same-session remove/reinsert steps
    # override an otherwise clear new placement when no pre-existing stent exists.
    if classified_action_type == "placement":
        has_placement = True
        if not preexisting_stent:
            has_removal = False
            revision_hint = False
    elif classified_action_type == "removal":
        has_placement = False
        has_removal = True
        revision_hint = False
    elif classified_action_type == "revision":
        revision_hint = True

    if not has_placement and not has_removal and not revision_hint:
        return {}

    # Exclude explicit history-only removal (e.g., "stent removed 2 years ago").
    removal_history = bool(
        re.search(
            r"\bstent\b[^.\n]{0,40}\bremoved\b[^.\n]{0,40}\b(?:year|yr|month|day)s?\s+ago",
            text_lower,
        )
        or re.search(r"\b(?:history|prior|previous)\b[^.\n]{0,80}\bstent\b", text_lower)
        or re.search(r"\bold\s+stent\b", text_lower)
    )
    if removal_history and not has_placement:
        return {}

    # Negation guard: placement-only mentions.
    if placement_negated and not has_placement and not has_removal and not revision_hint:
        return {}

    existing_revision_hint = bool(
        re.search(r"(?i)\b(?:existing|prior|previous|known|left[- ]sided)\b[^.\n]{0,140}\bstent", text_lower)
        or re.search(r"(?i)\bin[- ]stent\b[^.\n]{0,120}\bsecretions?\b", text_lower)
        or re.search(r"(?i)\bstent(?:s)?\b[^.\n]{0,120}\b(?:revis|revision|reposition|adjust)", text_lower)
        or preexisting_stent
    )
    has_existing_revision = bool(revision_hint and existing_revision_hint)

    def _best_stent_size_candidate(text: str) -> tuple[str, float | None, float | None, int] | None:
        size_re = re.compile(
            r"(?i)\b(?P<diam>\d+(?:\.\d+)?)\s*(?:(?P<unit1>mm|cm)\s*)?(?:x|×|by)\s*"
            r"(?P<len>\d+(?:\.\d+)?)\s*(?P<unit2>mm|cm)\b"
        )
        size_no_unit_re = re.compile(
            r"(?i)\b(?P<diam>\d{1,2}(?:\.\d+)?)\s*(?:x|×|by)\s*(?P<len>\d{1,3}(?:\.\d+)?)\b"
        )
        y_re = re.compile(
            r"(?i)\b(?P<a>\d+(?:\.\d+)?)\s*[-/x×]\s*(?P<b>\d+(?:\.\d+)?)\s*[-/x×]\s*(?P<c>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm)\b"
        )
        candidates: list[tuple[int, int, str, float | None, float | None]] = []
        for m in size_re.finditer(text or ""):
            raw = (m.group(0) or "").strip()
            if not raw:
                continue
            tight_start = max(0, m.start() - 80)
            tight_end = min(len(text), m.end() + 80)
            tight = (text or "")[tight_start:tight_end].lower()
            if "stent" not in tight:
                continue
            ctx_start = max(0, m.start() - 120)
            ctx_end = min(len(text), m.end() + 160)
            ctx = (text or "")[ctx_start:ctx_end].lower()

            try:
                diameter = float(m.group("diam"))
                length = float(m.group("len"))
            except Exception:
                diameter = None
                length = None

            unit1 = (m.group("unit1") or m.group("unit2") or "mm").lower().strip()
            unit2 = (m.group("unit2") or "mm").lower().strip()
            if diameter is not None and unit1 == "cm":
                diameter *= 10.0
            if length is not None and unit2 == "cm":
                length *= 10.0

            # Score: prefer explicit placement/deployment context and later (narrative) mentions.
            score = 0
            if re.search(r"(?i)\b(?:deployed|deploy(?:ment)?|placed|placement|insert(?:ed|ion)?|implant(?:ed|ation)?)\b", ctx):
                score += 2
            if re.search(r"(?i)\b(?:existing|prior|previous|known)\b", ctx):
                score -= 1
            if diameter is not None and length is not None and 6 <= diameter <= 25 and 10 <= length <= 100:
                score += 1

            candidates.append((score, m.start(), raw, diameter, length))

        for m in size_no_unit_re.finditer(text or ""):
            raw = (m.group(0) or "").strip()
            if not raw:
                continue
            tight_start = max(0, m.start() - 80)
            tight_end = min(len(text), m.end() + 80)
            tight = (text or "")[tight_start:tight_end].lower()
            if "stent" not in tight:
                continue
            ctx_start = max(0, m.start() - 140)
            ctx_end = min(len(text), m.end() + 180)
            ctx = (text or "")[ctx_start:ctx_end].lower()
            if re.search(r"(?i)\b(?:fr|french|gauge)\b", ctx):
                continue

            try:
                diameter = float(m.group("diam"))
                length = float(m.group("len"))
            except Exception:
                continue
            if not (6 <= diameter <= 25 and 10 <= length <= 100):
                continue

            score = 0
            if re.search(r"(?i)\b(?:deployed|deploy(?:ment)?|placed|placement|insert(?:ed|ion)?|implant(?:ed|ation)?)\b", ctx):
                score += 2
            if re.search(r"(?i)\b(?:existing|prior|previous|known)\b", ctx):
                score -= 2
            score += 1
            candidates.append((score, m.start(), raw, diameter, length))

        for m in y_re.finditer(text or ""):
            raw = (m.group(0) or "").strip()
            if not raw:
                continue
            tight_start = max(0, m.start() - 80)
            tight_end = min(len(text), m.end() + 80)
            tight = (text or "")[tight_start:tight_end].lower()
            if "stent" not in tight:
                continue
            ctx_start = max(0, m.start() - 120)
            ctx_end = min(len(text), m.end() + 160)
            ctx = (text or "")[ctx_start:ctx_end].lower()

            try:
                a = float(m.group("a"))
                b = float(m.group("b"))
                c = float(m.group("c"))
            except Exception:
                continue
            unit = (m.group("unit") or "mm").lower().strip()
            if unit == "cm":
                a *= 10.0
                b *= 10.0
                c *= 10.0
            diameter = max(a, b, c)
            length = None

            score = 0
            if re.search(r"(?i)\b(?:deployed|deploy(?:ment)?|placed|placement|insert(?:ed|ion)?|implant(?:ed|ation)?)\b", ctx):
                score += 2
            if re.search(r"(?i)\b(?:existing|prior|previous|known)\b", ctx):
                score -= 1
            if 6 <= diameter <= 25:
                score += 1

            candidates.append((score, m.start(), raw, diameter, length))

        if not candidates:
            return None
        best = max(candidates, key=lambda item: (item[0], item[1]))
        return best[2], best[3], best[4], best[1]

    def _apply_brand_type(proc: dict[str, Any], *, action: str) -> None:
        if re.search(r"\by-?\s*stent\b", text_lower):
            proc.setdefault("stent_type", "Y-Stent")
        brand, stent_type = _select_stent_brand(text_lower, action)
        if brand and not proc.get("stent_brand"):
            proc["stent_brand"] = brand
        if stent_type and not proc.get("stent_type"):
            proc["stent_type"] = stent_type

    result: dict[str, Any] = {}

    # Primary event: prefer capturing placement details when present.
    if has_placement:
        placement_proc: dict[str, Any] = {"performed": True, "action": "Placement"}
        _apply_brand_type(placement_proc, action="Placement")

        size_candidate = _best_stent_size_candidate(preferred_text or "")
        if size_candidate:
            raw_size, diameter, length, pos = size_candidate
            if raw_size and not placement_proc.get("device_size"):
                placement_proc["device_size"] = raw_size
            if diameter is not None and length is not None and 6 <= diameter <= 25 and 10 <= length <= 100:
                placement_proc["diameter_mm"] = diameter
                placement_proc["length_mm"] = length
            inferred = _infer_airway_site_for_stent_context(preferred_text or "", pos)
            if inferred and not placement_proc.get("location"):
                placement_proc["location"] = inferred
        elif classified_sizes and not placement_proc.get("device_size"):
            placement_proc["device_size"] = classified_sizes[0]

        if re.search(r"(?i)\b(?:deployed|placed|inserted)\b[^.\n]{0,120}\b(?:in\s+good\s+position|well\s+positioned|good\s+position)\b", preferred_text or ""):
            placement_proc["deployment_successful"] = True

        if not placement_proc.get("location"):
            m = re.search(r"(?i)\bstent\b", preferred_text or "")
            if m:
                window = (preferred_text or "")[m.start() : min(len(preferred_text or ""), m.start() + 320)]
                inferred = _infer_airway_site_last(window)
                if inferred:
                    placement_proc["location"] = inferred
        result["airway_stent"] = placement_proc

        # Secondary: existing-stent revision documented alongside a new placement.
        if has_existing_revision and not has_removal:
            revision_proc: dict[str, Any] = {"performed": True, "action": "Revision/Repositioning"}
            # Best-effort: left-sided stents are usually LMS/LUL/LLL; keep coarse.
            if re.search(r"(?i)\b(?:lms|left\s+main(?:\s*|-)?stem)\b", preferred_text or "") or re.search(
                r"(?i)\bleft[-\s]?sided\b", preferred_text or ""
            ):
                revision_proc["location"] = "Left mainstem"
            result["airway_stent_revision"] = revision_proc

    # If no placement, fall back to a single conservative stent action classification.
    if not has_placement:
        proc: dict[str, Any] = {"performed": True}
        if has_removal:
            proc["action"] = "Removal"
            proc["airway_stent_removal"] = True
        elif revision_hint:
            proc["action"] = "Revision/Repositioning"
        _apply_brand_type(proc, action=str(proc.get("action") or ""))

        # Existing-stent context (assessment/obstruction/cleaning): capture stent size/location
        # when explicitly documented so downstream evidence/coding has the details.
        size_candidate = _best_stent_size_candidate(preferred_text or "")
        if size_candidate:
            raw_size, diameter, length, pos = size_candidate
            if raw_size and not proc.get("device_size"):
                proc["device_size"] = raw_size
            if diameter is not None and not proc.get("diameter_mm"):
                proc["diameter_mm"] = diameter
            if length is not None and not proc.get("length_mm"):
                proc["length_mm"] = length
            if not proc.get("location"):
                inferred = _infer_airway_site_for_stent_context(preferred_text or "", pos)
                if inferred:
                    proc["location"] = inferred
        elif classified_sizes and not proc.get("device_size"):
            proc["device_size"] = classified_sizes[0]

        if not proc.get("location"):
            # Best-effort: infer from the immediate stent mention window (avoid using full-note
            # "last location" which can be polluted by other later airway targets).
            m = re.search(r"(?i)\bstent\b", preferred_text or "")
            if m:
                window = (preferred_text or "")[m.start() : min(len(preferred_text or ""), m.start() + 280)]
                inferred = _infer_airway_site_last(window)
                if inferred:
                    proc["location"] = inferred

        result["airway_stent"] = proc

    # Exchange/removal + placement in the same session: represent as revision on the primary stent.
    if has_placement and has_removal:
        if preexisting_stent or classified_action_type == "revision" or bool(classified.get("exchange")):
            proc = result.get("airway_stent") if isinstance(result.get("airway_stent"), dict) else {"performed": True}
            proc["performed"] = True
            proc["action"] = "Revision/Repositioning"
            proc["airway_stent_removal"] = True
            result["airway_stent"] = proc
            result.pop("airway_stent_revision", None)
        else:
            # Same-session trial/deploy/remove/redeploy of a new stent should remain placement.
            proc = result.get("airway_stent")
            if isinstance(proc, dict):
                proc["performed"] = True
                proc["action"] = "Placement"
                proc.pop("airway_stent_removal", None)
                result["airway_stent"] = proc

    # Populate action_type for stability in non-validated downstream consumers.
    for key in ("airway_stent", "airway_stent_revision"):
        proc = result.get(key)
        if not isinstance(proc, dict):
            continue
        action = str(proc.get("action") or "").strip()
        if action == "Placement":
            proc["action_type"] = "placement"
        elif action == "Removal":
            proc["action_type"] = "removal"
        elif action == "Revision/Repositioning":
            proc["action_type"] = "revision"
        elif action == "Assessment only":
            proc["action_type"] = "assessment_only"

    return result or {}


def extract_balloon_occlusion(note_text: str) -> Dict[str, Any]:
    """Extract balloon occlusion / endobronchial blocker workflow details."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    if not any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in BALLOON_OCCLUSION_PATTERNS):
        return {}

    # Negation guard.
    if re.search(
        r"(?i)\b(?:no|not|without|declined|deferred)\b[^.\n]{0,80}"
        r"\b(?:balloon\s+occlusion|serial\s+occlusion|endobronchial\s+blocker|blocker|uniblocker|arndt|ardnt|fogarty)\b",
        text_lower,
    ):
        return {}

    proc: dict[str, Any] = {"performed": True}

    # Device size (Fr) near blocker terms.
    size_re = re.compile(
        r"(?i)\b(?P<size>\d+(?:\.\d+)?)\s*(?:fr|french)\b[^.\n]{0,80}"
        r"\b(?:endobronchial\s+blocker|blocker|uniblocker|arndt|ardnt|fogarty)\b"
        r"|\b(?:endobronchial\s+blocker|blocker|uniblocker|arndt|ardnt|fogarty)\b[^.\n]{0,80}"
        r"\b(?P<size2>\d+(?:\.\d+)?)\s*(?:fr|french)\b"
    )
    size_candidates: list[tuple[int, int, str]] = []
    for match in size_re.finditer(preferred_text or ""):
        raw = (match.group(0) or "").strip()
        if not raw:
            continue
        window_start = max(0, match.start() - 120)
        window_end = min(len(preferred_text or ""), match.end() + 120)
        window = (preferred_text or "")[window_start:window_end].lower()
        score = 0
        if re.search(r"(?i)\b(?:placed|inserted|advanced|positioned)\b", window):
            score += 2
        if re.search(r"(?i)\b(?:removed|withdrawn|deflated|old|previous|prior)\b", window):
            score -= 1
        size_candidates.append((score, match.start(), raw))
    if size_candidates:
        # Prefer the strongest placement-context match; tie-breaker favors later mentions.
        best = max(size_candidates, key=lambda item: (item[0], item[1]))
        proc["device_size"] = best[2][:120]

    # Occlusion location (best-effort, verbatim).
    loc_match = re.search(
        r"(?i)\b(?:balloon\s+)?occlu(?:sion|ded|ding)\b[^.\n]{0,80}\b(?:of|in|at)\b\s+(?P<loc>[^.;\n]{3,120})",
        preferred_text or "",
    ) or re.search(
        r"(?i)\b(?:endobronchial\s+blocker|blocker|uniblocker|arndt|ardnt|fogarty)\b[^.\n]{0,120}"
        r"\b(?:positioned|placed|advanced)\b[^.\n]{0,120}\b(?:in|at)\b\s+(?P<loc>[^.;\n]{3,120})",
        preferred_text or "",
    )
    if loc_match:
        loc_raw = (loc_match.group("loc") or "").strip().strip(" ,;:-")
        if loc_raw and re.search(
            r"(?i)\b(?:trachea|main\s*stem|mainstem|bronch(?:us|ial)|carina|lob(?:e|ar)|segment|bi|lingula)\b",
            loc_raw,
        ):
            proc["occlusion_location"] = loc_raw[:180]

    # Air leak result near occlusion workflows (best-effort, verbatim).
    leak_match = re.search(
        r"(?i)\b(?:air\s*)?leak\b[^.\n]{0,80}\b(?:resolved|cessation|ceased|stopped|persist(?:ed|ent)|continued|ongoing|present)\b[^.\n]{0,80}",
        preferred_text or "",
    )
    if leak_match:
        val = (leak_match.group(0) or "").strip()
        if val:
            proc["air_leak_result"] = val[:220]

    return {"balloon_occlusion": proc}


def extract_blvr(note_text: str) -> Dict[str, Any]:
    """Extract BLVR (endobronchial valve) indicator.

    Conservative by default: only fires on high-signal valve/BLVR terms.
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    pal_valve_localization_hit = bool(
        re.search(r"(?i)\b(?:pal|persistent\s+air\s+leak|air\s+leak|pneumothorax)\b", preferred_text)
        and re.search(r"(?i)\bbronchoscop\w*\b|\bballoon\s+occlusion\b|\bocclusion\b", preferred_text)
        and re.search(
            r"(?i)\b(?:\d{1,2}|one|two|three|four|five|six)\s+valves?\b[^.\n]{0,80}\b(?:deploy(?:ed|ment)?|place(?:d|ment)?|insert(?:ed|ion)?)\b"
            r"|\b(?:deploy(?:ed|ment)?|place(?:d|ment)?|insert(?:ed|ion)?)\b[^.\n]{0,80}\b(?:\d{1,2}|one|two|three|four|five|six)\s+valves?\b",
            preferred_text,
        )
    )
    blvr_hit = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in BLVR_PATTERNS) or pal_valve_localization_hit
    if not blvr_hit:
        return {}

    proc: dict[str, Any] = {"performed": True}

    if "zephyr" in text_lower:
        proc["valve_type"] = "Zephyr (Pulmonx)"
    elif "spiration" in text_lower:
        proc["valve_type"] = "Spiration (Olympus)"

    # Important: "valves ... well placed"/"previously placed valves" are inspection-only and
    # must not be treated as a valve placement procedure.
    placed_token = r"(?<!well\s)(?<!previously\s)(?<!prior\s)(?<!already\s)placed\b"
    planning_re = re.compile(
        r"(?i)\b(?:plan(?:s|ned)?|scheduled|next\s+(?:week|procedure|time)|at\s+next\s+procedure|future|later|defer(?:red)?|separate\s+procedure|to\s+be\s+performed|will\s+proceed)\b"
    )
    placement_strong_re = re.compile(
        r"(?i)\b(?:deploy(?:ed|ment)?|insert(?:ed|ion)?|deliver(?:ed)?|implant(?:ed|ation)?)\b"
    )
    placement_noun_re = re.compile(r"(?i)\bplacement\b")
    placement_completed_re = re.compile(r"(?i)\b(?:performed|completed|successful(?:ly)?)\b")

    placement_performed = False
    placement_planned_only = False
    for sentence in re.split(r"(?:\n+|(?<=[.!?])\s+)", preferred_text or ""):
        sent = (sentence or "").strip()
        if not sent:
            continue
        sent_lower = sent.lower()
        if "valve" not in sent_lower and "zephyr" not in sent_lower and "spiration" not in sent_lower:
            continue
        is_planned = bool(planning_re.search(sent))
        has_strong = bool(
            placement_strong_re.search(sent)
            or re.search(rf"(?i)\bvalves?\b[^.\n]{{0,80}}\b{placed_token}", sent)
            or re.search(rf"(?i)\b{placed_token}[^.\n]{{0,80}}\bvalves?\b", sent)
        )
        has_noun_completed = bool(placement_noun_re.search(sent) and placement_completed_re.search(sent))

        if (has_strong or has_noun_completed) and not is_planned:
            placement_performed = True
            break
        if (has_strong or placement_noun_re.search(sent)) and is_planned:
            placement_planned_only = True

    removal_present = bool(
        re.search(
            r"\bvalve\b[^.\n]{0,80}\b(?:remov|retriev|extract|explant)\w*\b",
            text_lower,
            re.IGNORECASE,
        )
    )
    if placement_performed:
        proc["procedure_type"] = "Valve placement"
    elif removal_present:
        proc["procedure_type"] = "Valve removal"
    elif "chartis" in text_lower:
        proc["procedure_type"] = "Valve assessment"
    elif placement_planned_only:
        # A planned/scheduled valve placement mention is not evidence of a completed placement.
        # Keep BLVR only if Chartis/removal evidence is present; otherwise, skip.
        return {}

    # If we only see generic valve mentions with no procedure action (placement/removal/Chartis),
    # do not assert BLVR was performed in this session.
    if not proc.get("procedure_type"):
        return {}

    # BLVR specificity (high-yield): infer target segments + final deployed valve count
    # from explicit segment tokens (e.g., RB9/RB10, LB1+2). Avoid counting valves
    # that were removed and not replaced during the same session.
    segment_token_re = re.compile(r"(?i)\b([RL]B\s*\d{1,2}(?:\s*[+/]\s*\d{1,2})?)\b")
    placement_verb_re = re.compile(r"(?i)\b(?:place(?:d|ment)?|deploy(?:ed|ment)?|insert(?:ed|ion)?|deliver(?:ed)?)\b")
    removal_verb_re = re.compile(r"(?i)\b(?:remov(?:e|ed|al)?|retriev(?:e|ed|al)?|extract(?:ed|ion)?|explant(?:ed|ation)?|withdrawn|pull(?:ed)?)\b")
    replace_verb_re = re.compile(r"(?i)\b(?:replac(?:e|ed|ement)?|exchang(?:e|ed|ing)?)\b")

    def _normalize_segment(raw: str) -> str:
        token = (raw or "").strip().upper()
        token = re.sub(r"\s+", "", token)
        token = token.replace("/", "+")
        return token

    def _segment_to_lobe(seg: str) -> str | None:
        match = re.match(r"^(?P<side>[RL])B(?P<num>\d{1,2})", seg)
        if not match:
            return None
        side = match.group("side")
        try:
            num = int(match.group("num"))
        except Exception:
            return None
        if side == "R":
            if 1 <= num <= 3:
                return "RUL"
            if 4 <= num <= 5:
                return "RML"
            if 6 <= num <= 10:
                return "RLL"
        if side == "L":
            if 4 <= num <= 5:
                return "Lingula"
            if 1 <= num <= 3:
                return "LUL"
            if 6 <= num <= 10:
                return "LLL"
        return None

    states: dict[str, str] = {}
    for raw_sentence in re.split(r"(?:\n+|(?<=[.!?])\s+)", preferred_text or ""):
        sentence = (raw_sentence or "").strip()
        if not sentence:
            continue
        segments = [_normalize_segment(tok) for tok in segment_token_re.findall(sentence)]
        if not segments:
            continue
        sentence_lower = sentence.lower()
        # Require valve context (prevents unrelated RB/LB numbers from steering BLVR detail).
        if "valve" not in sentence_lower and "zephyr" not in sentence_lower and "spiration" not in sentence_lower:
            continue
        has_place = bool(placement_verb_re.search(sentence))
        has_remove = bool(removal_verb_re.search(sentence))
        has_replace = bool(replace_verb_re.search(sentence))

        # If both placement and removal appear in the same sentence, assume the end
        # state is removal unless a replacement/exchange is explicitly stated.
        if has_remove and not has_replace and (has_place or not has_place):
            for seg in segments:
                states[seg] = "removed"
            continue
        if has_place or has_replace:
            for seg in segments:
                states[seg] = "deployed"

    deployed_segments = sorted([s for s, state in states.items() if state == "deployed"])
    removed_segments = sorted([s for s, state in states.items() if state == "removed"])
    final_segments = deployed_segments or removed_segments

    if final_segments and not proc.get("segments_treated"):
        proc["segments_treated"] = final_segments
    if final_segments and not proc.get("number_of_valves"):
        proc["number_of_valves"] = len(final_segments)
    if not proc.get("target_lobe") and final_segments:
        lobes = {l for l in (_segment_to_lobe(s) for s in final_segments) if l}
        if len(lobes) == 1:
            proc["target_lobe"] = next(iter(lobes))

    if proc.get("procedure_type") == "Valve placement" and not proc.get("number_of_valves"):
        word_to_int = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
        }
        count_match = re.search(
            r"(?i)\b(?P<count>\d{1,2}|one|two|three|four|five|six)\s+valves?\b[^.\n]{0,40}\b(?:deploy(?:ed|ment)?|place(?:d|ment)?|insert(?:ed|ion)?)\b"
            r"|\b(?:deploy(?:ed|ment)?|place(?:d|ment)?|insert(?:ed|ion)?)\b[^.\n]{0,40}\b(?P<count2>\d{1,2}|one|two|three|four|five|six)\s+valves?\b",
            preferred_text,
        )
        if count_match:
            raw_count = (count_match.group("count") or count_match.group("count2") or "").strip().lower()
            try:
                proc["number_of_valves"] = int(raw_count)
            except Exception:
                parsed = word_to_int.get(raw_count)
                if parsed is not None:
                    proc["number_of_valves"] = parsed

    if not proc.get("target_lobe"):
        lobes = _extract_lung_locations_from_text(preferred_text)
        if len(lobes) == 1:
            proc["target_lobe"] = lobes[0]

    # Collateral ventilation assessment is schema-constrained (Chartis-focused).
    # Only populate when Chartis is explicitly mentioned; rely on evidence text to
    # support 31634 in non-Chartis balloon occlusion workflows.
    if proc.get("collateral_ventilation_assessment") is None and "chartis" in text_lower:
        if re.search(r"(?i)\bno/minimal\s+collateral\s+ventilation\b", text_lower) or re.search(
            r"(?i)\bno\s+collateral\s+ventilation\b", text_lower
        ):
            proc["collateral_ventilation_assessment"] = "Chartis negative"
        elif re.search(r"(?i)\bcollateral\s+ventilation\b[^.\n]{0,40}\bpresent\b", text_lower):
            proc["collateral_ventilation_assessment"] = "Chartis positive"
        else:
            proc["collateral_ventilation_assessment"] = "Chartis indeterminate"

    return {"blvr": proc}


def extract_diagnostic_bronchoscopy(note_text: str) -> Dict[str, Any]:
    """Extract diagnostic bronchoscopy (31622 family).

    Purpose: backstop cases where the only bronchoscopy service is airway inspection.
    Avoid firing from consent/indication text by preferring the procedure-detail section.
    """
    preferred_text, used_detail_section = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    full_lower = (note_text or "").lower()
    if not text_lower.strip():
        return {}

    # Hard negations: aborted/not performed.
    if re.search(
        r"(?i)\b(?:procedure\s+aborted|bronchoscopy\s+aborted|bronchoscopy\s+not\s+performed|unable\s+to\s+perform\s+bronchoscopy)\b",
        text_lower,
    ):
        return {}

    # Require some evidence of intraprocedural airway inspection / scope use.
    hits = any(re.search(pat, text_lower, re.IGNORECASE) for pat in DIAGNOSTIC_BRONCHOSCOPY_PATTERNS)
    if not hits:
        return {}

    # If we didn't find a procedure-detail section, be conservative: require a
    # strong inspection cue or explicit procedure-header 31622 context.
    strong_inspection_cue = bool(
        re.search(
            r"(?i)\b(?:"
            r"the\s+airway\s+was\s+inspected|"
            r"initial\s+airway\s+inspection\s+findings|"
            r"tracheobronchial\s+tree\s+was\s+examined|"
            r"dynamic\s+bronchoscopy|"
            r"dynamic\s+assessment|"
            r"forced\s+expiratory\s+maneuver|"
            r"cpap\s+titration\s+during\s+bronchoscopy"
            r")\b",
            text_lower,
        )
    )
    header_31622_cue = bool(
        re.search(r"(?im)^\s*31622\s*:\s*bronchoscopy\s+only\b", note_text or "")
        or re.search(
            r"(?is)\bprocedures?\s+performed\s*:\b[^.\n]{0,240}\b31622\b[^.\n]{0,120}\bbronchoscopy\s+only\b",
            note_text or "",
        )
    )
    if not used_detail_section and not (strong_inspection_cue or header_31622_cue):
        return {}

    # Ensure we're in bronchoscopy context, not generic airway exam wording.
    # (The scope context is often in the header/instrument section, while the
    # detail section just says "airway was inspected".)
    if "bronchoscop" not in full_lower and "bronchoscope" not in full_lower:
        return {}

    return {"diagnostic_bronchoscopy": {"performed": True}}


def extract_foreign_body_removal(note_text: str) -> Dict[str, Any]:
    """Extract foreign body removal indicator.
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    if re.search(
        r"(?i)\bstent\b[^.\n]{0,140}\b(?:remov|retriev|extract|withdraw|grasp|en\s+bloc)\w*\b"
        r"|\b(?:remov|retriev|extract|withdraw|grasp)\w*\b[^.\n]{0,140}\bstent\b",
        preferred_text or "",
    ):
        return {}

    match: re.Match[str] | None = None
    for pat in FOREIGN_BODY_REMOVAL_PATTERNS:
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            match = m
            break

    if match is None:
        # Some notes mention a stent removal without explicitly calling it a
        # "foreign body." Treat removal-only stent cases as foreign body removal,
        # but skip when placement is clearly documented (likely an exchange).
        stent_removal_re = re.compile(
            r"(?i)\bstent\b[^.\n]{0,120}\b(?:remov|retriev|extract|pull|grasp|peel|withdraw)\w*\b"
            r"|\b(?:remov|retriev|extract|pull|grasp|peel|withdraw)\w*\b[^.\n]{0,120}\bstent\b"
        )
        m = stent_removal_re.search(text_lower)
        if not m:
            return {}
        placement_hit = _stent_action_window_hit(
            text_lower,
            verbs=["place", "deploy", "insert", "advance", "seat", "deliver", "implant"],
            max_chars=40,
        )
        if placement_hit:
            return {}
        match = m

    proc: dict[str, Any] = {"performed": True}
    window_start = max(0, match.start() - 160)
    window_end = min(len(text_lower), match.end() + 160)
    local = text_lower[window_start:window_end]
    if re.search(r"\bforceps\b", local):
        proc["retrieval_tool"] = "Forceps"
    elif re.search(r"\bbasket\b", local):
        proc["retrieval_tool"] = "Basket"
    elif re.search(r"\bcryoprobe\b|\bcryo\s+probe\b", local):
        proc["retrieval_tool"] = "Cryoprobe"
    elif re.search(r"\bsnare\b", local):
        proc["retrieval_tool"] = "Snare"

    return {"foreign_body_removal": proc}


def extract_therapeutic_injection(note_text: str) -> Dict[str, Any]:
    """Extract endobronchial therapeutic instillation/injection (e.g., amphotericin).

    Conservative guardrails:
    - Exclude topical anesthesia (lidocaine/xylocaine) and plain saline-only instillations.
    - Require explicit instillation/injection language plus a medication cue or dose.
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text = preferred_text or ""
    if not text.strip():
        return {}

    instill_re = re.compile(r"(?i)\b(?:instill(?:ed|ation)?|inject(?:ed|ion)?)\b")
    med_dose_re = re.compile(
        r"(?i)\b(?P<med>[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*)*)\s+"
        r"(?P<dose>\d{1,4}\s*(?:mg|mcg|ug|g))\b"
    )
    dose_med_re = re.compile(
        r"(?i)\b(?P<dose>\d{1,4}\s*(?:mg|mcg|ug|g))\s+(?P<med>[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*)*)\b"
    )
    vol_re = re.compile(r"(?i)\b(?P<vol>\d{1,4})\s*(?:cc|ml)\b")
    exclude_re = re.compile(r"(?i)\b(?:lidocaine|xylocaine|saline|normal\s+saline|ns)\b")
    bal_context_re = re.compile(r"(?i)\b(?:bal|bronchoalveolar|bronchial\s+alveolar|lavage)\b")
    med_fallback_re = re.compile(
        r"(?i)\b(?:amphotericin|antibiotic|antifungal|gentamicin|tobramycin|vancomycin|"
        r"tranexamic\s+acid|txa|epinephrine)\b"
    )

    candidates: list[dict[str, Any]] = []
    for sentence in re.split(r"(?:\n+|(?<=[.!?])\s+)", text):
        if not sentence or not instill_re.search(sentence):
            continue
        if exclude_re.search(sentence) and not re.search(r"(?i)\b(?:amphotericin|antibiot|antifungal)\b", sentence):
            # Likely topical anesthesia or lavage-only context.
            continue

        med = None
        dose = None
        m_med = med_dose_re.search(sentence)
        if m_med:
            med = (m_med.group("med") or "").strip()
            dose = (m_med.group("dose") or "").strip()
        else:
            m_rev = dose_med_re.search(sentence)
            if m_rev:
                med = (m_rev.group("med") or "").strip()
                dose = (m_rev.group("dose") or "").strip()

            # Fallback: medication name without explicit dose (e.g., "amphotericin was instilled")
            m_simple = med_fallback_re.search(sentence)
            if m_simple:
                med = (m_simple.group(0) or "").strip()

        vol_match = vol_re.search(sentence)
        volume_ml: float | None = None
        if vol_match:
            try:
                volume_ml = float(int(vol_match.group("vol")))
            except Exception:
                volume_ml = None

        # Avoid misclassifying BAL instillation volumes as therapeutic injections
        # unless a medication/dose is explicitly documented.
        if bal_context_re.search(sentence) and not (med or dose):
            continue

        # Guardrail: volume-only instillation sentences are usually BAL/flush/hemostasis
        # artifacts and should not trigger 31573 without an explicit medication/dose cue.
        if not (med or dose):
            continue

        proc: dict[str, Any] = {"performed": True}
        if med:
            proc["medication"] = med
        if dose:
            proc["dose"] = dose
        if volume_ml is not None:
            proc["volume_ml"] = volume_ml
        locations = _extract_lung_locations_from_text(sentence)
        if locations:
            proc["location"] = locations[0]
        candidates.append(proc)

    if not candidates:
        return {}

    if len(candidates) == 1:
        return {"therapeutic_injection": candidates[0]}

    # Merge multiple instillations into a single summary object.
    meds: list[str] = []
    doses: list[str] = []
    volumes: list[float] = []
    locations: list[str] = []
    for cand in candidates:
        med = cand.get("medication")
        if isinstance(med, str) and med and med not in meds:
            meds.append(med)
        dose = cand.get("dose")
        if isinstance(dose, str) and dose and dose not in doses:
            doses.append(dose)
        vol = cand.get("volume_ml")
        if isinstance(vol, (int, float)) and vol > 0:
            volumes.append(float(vol))
        loc = cand.get("location")
        if isinstance(loc, str) and loc:
            locations.append(loc)

    merged: dict[str, Any] = {"performed": True}
    if meds:
        merged["medication"] = "; ".join(meds)
    if doses:
        merged["dose"] = "; ".join(doses)
    if len(set(volumes)) == 1:
        merged["volume_ml"] = volumes[0]
    if locations:
        # Prefer a single consistent location if available.
        unique_locs = []
        for loc in locations:
            if loc not in unique_locs:
                unique_locs.append(loc)
        if len(unique_locs) == 1:
            merged["location"] = unique_locs[0]

    return {"therapeutic_injection": merged}

    return {}


def extract_endobronchial_biopsy(note_text: str) -> Dict[str, Any]:
    """Extract endobronchial (airway) biopsy indicator.

    This is distinct from transbronchial biopsy (parenchyma).
    """
    full_text = note_text or ""
    preferred_text, used_detail_section = _preferred_procedure_detail_text(full_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text = preferred_text or full_text
    text_lower = text.lower()
    if not text_lower.strip():
        return {}

    explicit_endobronchial_biopsy = bool(
        re.search(r"(?i)\bendobronchial\s+biops|\bebbx\b", text)
    )
    peripheral_workflow = bool(
        re.search(
            r"(?i)\b(?:peripheral|nodule|lung\s+(?:nodule|lesion|mass)|pulmonary\s+(?:nodule|lesion|mass)|"
            r"navigation|navigational|electromagnetic\s+navigation|\benb\b|ion\b|robotic|"
            r"radial\s+(?:ebus|probe|ultrasound)|rebus|miniprobe|"
            r"transbronchial\s+(?:lung\s+)?biops|tbbx|tblb)\b",
            text,
        )
    )
    no_endobronchial_disease = bool(
        re.search(
            r"(?i)\bno\s+endobronchial\s+(?:lesions?|tumou?rs?|mass(?:es)?)\b",
            text,
        )
    )
    if no_endobronchial_disease and peripheral_workflow and not explicit_endobronchial_biopsy:
        return {}

    narrative_start = 0
    if not used_detail_section:
        narrative_cue = re.search(
            r"(?i)\b(?:procedure\s*:|procedure\s+description\s*:|following\s+intravenous\s+medications|bronchoscope\s+was\s+introduced)\b",
            text,
        )
        if narrative_cue:
            narrative_start = int(narrative_cue.start())

    def _extract_airway_locations_from_text(text: str) -> list[str]:
        raw = text or ""
        lowered = raw.lower()
        locations: list[str] = []

        def add(value: str) -> None:
            if value and value not in locations:
                locations.append(value)

        if re.search(r"\blower\s+trachea\b", lowered):
            add("Lower Trachea")
        if re.search(r"\btrachea(?:l)?\b", lowered) and "Lower Trachea" not in locations:
            add("Trachea")
        if re.search(r"\b(?:main\s+carina|carina)\b", lowered):
            add("Carina")
        if re.search(r"\b(?:rms|right\s+main(?:\s*|-)?stem)\b", lowered):
            add("Right mainstem")
        if re.search(r"\b(?:lms|left\s+main(?:\s*|-)?stem)\b", lowered):
            add("Left mainstem")
        if re.search(r"\b(?:bronchus\s+intermedius)\b", lowered):
            add("Bronchus intermedius")

        return locations

    def _extract_sentence_scoped_locations(sentence_text: str) -> list[str]:
        sentence = re.sub(r"\s+", " ", (sentence_text or "").strip())
        if not sentence:
            return []
        match = re.search(r"(?i)\bwas\s+performed\s+(?:at|in|on)\s+(?P<loc>[^.\n;]+)", sentence)
        if not match:
            return []
        raw_loc = (match.group("loc") or "").strip()
        raw_loc = re.split(
            r"(?i)\b(?:using|with|samples?|specimens?|lesion|biops(?:y|ies)|sent\s+for)\b",
            raw_loc,
            maxsplit=1,
        )[0].strip(" ,;:-")
        if not raw_loc:
            return []
        return [raw_loc]

    def _extract_sample_count(text: str) -> int | None:
        window = text or ""
        if not window.strip():
            return None
        word_to_int = {
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
        numeric_re = re.compile(
            r"(?i)\b(?P<num>\d{1,2})\s+(?:samples|specimens)\b[^.\n]{0,40}\b(?:were\s+)?(?:obtained|taken|collected)\b"
        )
        word_re = re.compile(
            r"(?i)\b(?P<word>one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:samples|specimens)\b[^.\n]{0,40}\b(?:were\s+)?(?:obtained|taken|collected)\b"
        )
        m = numeric_re.search(window)
        if m:
            try:
                val = int(m.group("num"))
                if 1 <= val <= 50:
                    return val
            except Exception:
                return None
        m2 = word_re.search(window)
        if m2:
            val = word_to_int.get((m2.group("word") or "").strip().lower())
            if val:
                return val
        return None

    for pattern in ENDOBRONCHIAL_BIOPSY_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            if not used_detail_section and narrative_start > 0 and int(match.start()) < narrative_start:
                continue
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            proc: dict[str, Any] = {"performed": True}
            # Prefer locations in the same clause to avoid picking up nearby bleeding/airway survey sites.
            start = int(match.start())
            end = int(match.end())
            left_boundary = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
            clause_start = left_boundary + 1 if left_boundary != -1 else 0
            next_period = text.find(".", end)
            next_nl = text.find("\n", end)
            right_candidates = [pos for pos in (next_period, next_nl) if pos != -1]
            clause_end = min(right_candidates) if right_candidates else len(text)
            clause = text[clause_start:clause_end]
            clause_lower = clause.lower()

            # Guardrail: avoid classifying peripheral/radial/fluoro lesion sampling
            # as endobronchial biopsy unless there is explicit airway context.
            peripheral_context = bool(
                re.search(
                    r"(?i)\b(?:peripheral|radial|fluoro(?:scop)?y|sheath|concentric|eccentric|nodule|parenchym|subsegment)\b",
                    clause_lower,
                )
            )
            airway_context = bool(
                re.search(
                    r"(?i)\b(?:endobronch|airway|trachea|carina|mainstem|bronchus\s+intermedius|mucosa)\b",
                    clause_lower,
                )
            )
            if peripheral_context and not airway_context:
                continue

            sentence_scoped_locations = _extract_sentence_scoped_locations(clause)
            if sentence_scoped_locations:
                proc["locations"] = sentence_scoped_locations
            else:
                airway_locations = _extract_airway_locations_from_text(clause)
                locations = _extract_lung_locations_from_text(clause)
                combined_locations = airway_locations + [loc for loc in locations if loc not in airway_locations]
                if combined_locations:
                    proc["locations"] = combined_locations
                else:
                    # Some notes localize the airway segment in the preceding sentence
                    # (e.g., "...right lower lobe... Multiple forceps biopsies...").
                    context_start = max(0, clause_start - 280)
                    context_window = text[context_start:clause_end]
                    context_lung_locations = _extract_lung_locations_from_text(context_window)
                    if context_lung_locations:
                        proc["locations"] = [context_lung_locations[-1]]
                    else:
                        context_airway_locations = _extract_airway_locations_from_text(context_window)
                        if context_airway_locations:
                            proc["locations"] = [context_airway_locations[-1]]

            # Sample count often appears in the next sentence ("Five samples were obtained.").
            trailing_window = text[clause_start : min(len(text), clause_end + 260)]
            sample_count = _extract_sample_count(trailing_window)
            if sample_count is not None:
                proc["number_of_samples"] = sample_count
            return {"endobronchial_biopsy": proc}

    return {}


def extract_radial_ebus(note_text: str) -> Dict[str, Any]:
    """Extract radial EBUS indicator (peripheral lesion localization)."""
    text_lower = (note_text or "").lower()
    for pattern in RADIAL_EBUS_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            # Guardrail: "radial ultrasound" can appear outside bronchoscopy; require
            # bronchoscopy/probe context in a local window for this variant.
            if "radial\\s+ultrasound" in pattern:
                local = text_lower[max(0, match.start() - 160) : min(len(text_lower), match.end() + 220)]
                if not re.search(r"(?i)\b(?:bronchoscop|bronchoscope|probe|miniprobe|r-?ebus|rebus)\b", local):
                    continue
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            proc: dict[str, Any] = {"performed": True}

            # Probe position/view (explicit-only; do not infer from tool lists).
            view_match = re.search(
                r"(?i)\b(?:radial\s+ebus|r-?ebus|rebus)\b[^.\n]{0,240}\b"
                r"(concentric|eccentric|adjacent|not\s+visualized|no\s+view|absent|aerated\s+lung)\b",
                note_text or "",
            )
            if view_match:
                raw_view = (view_match.group(1) or "").strip().lower()
                if "concentric" in raw_view:
                    proc["probe_position"] = "Concentric"
                elif "eccentric" in raw_view:
                    proc["probe_position"] = "Eccentric"
                elif "adjacent" in raw_view:
                    proc["probe_position"] = "Adjacent"
                elif raw_view:
                    proc["probe_position"] = "Not visualized"
            else:
                local = (note_text or "")[max(0, match.start() - 220) : min(len(note_text or ""), match.end() + 320)]
                if re.search(
                    r"(?i)\b(?:attempt(?:ed|ing)?|unable|could\s+not|cannot|can't|unsuccessful|failed)\b"
                    r"[^.\n]{0,140}\b(?:visualiz|see|find|locat|identify)\w*\b",
                    local,
                ):
                    proc["probe_position"] = "Not visualized"

            return {"radial_ebus": proc}
    return {}


def extract_eus_b(note_text: str) -> Dict[str, Any]:
    """Extract EUS-B indicator (endoscopic ultrasound via EBUS bronchoscope)."""
    text_lower = (note_text or "").lower()
    for pattern in EUS_B_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            return {"eus_b": {"performed": True}}
    return {}


def extract_cryotherapy(note_text: str) -> Dict[str, Any]:
    """Extract cryotherapy (tumor destruction/stenosis relief) indicator."""
    preferred_text, used_detail = _preferred_procedure_detail_text(note_text)
    if used_detail:
        preferred_text = _strip_cpt_definition_lines(preferred_text)
    else:
        preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = preferred_text.lower()
    for pattern in CRYOTHERAPY_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            # Guardrail: avoid future/planned recommendation language.
            prefix = text_lower[max(0, match.start() - 100) : match.start()]
            suffix = text_lower[match.end() : min(len(text_lower), match.end() + 100)]
            planning_context = bool(
                re.search(
                    r"(?i)\b(?:consider(?:ed|ation)?|recommend(?:ed|ation)?|plan(?:ned)?|future|next)\b",
                    prefix,
                )
                or re.search(r"(?i)\b(?:at\s+next\s+intervention|if\s+needed)\b", suffix)
            )
            if planning_context:
                continue
            return {"cryotherapy": {"performed": True}}

    if re.search(CRYOPROBE_PATTERN, text_lower, re.IGNORECASE) and not re.search(
        CRYOBIOPSY_PATTERN, text_lower, re.IGNORECASE
    ):
        specimen_any_re = re.compile(r"\b(?:specimen(?:s)?|histolog(?:y|ic))\b", re.IGNORECASE)
        specimen_none_re = re.compile(r"\bspecimen(?:s)?\b[^.\n]{0,60}\b(?:none|n/?a|na)\b", re.IGNORECASE)
        biopsy_token_re = re.compile(r"\bbiops(?:y|ies|ied)\b", re.IGNORECASE)
        sent_to_path_re = re.compile(r"\bsent\b[^.\n]{0,40}\bpatholog(?:y|ic)\b", re.IGNORECASE)
        therapeutic_context_re = re.compile(
            r"\b(?:ablat|destroy|debulk|devitaliz|recanaliz|stenos(?:is|ed)|tumou?r)\w*\b",
            re.IGNORECASE,
        )
        # Cryoprobe is also used therapeutically for cryo-extraction (e.g., clot / mucus plug removal).
        cryo_extraction_context_re = re.compile(
            r"\b(?:clot|mucus\s+plug|plug)\b[^\n]{0,60}\b(?:remov|extract|retriev|debrid)\w*\b"
            r"|\b(?:remov|extract|retriev|debrid)\w*\b[^\n]{0,60}\b(?:clot|mucus\s+plug|plug)\b",
            re.IGNORECASE,
        )

        saw_biopsy_only = False
        saw_therapeutic = False

        for cryo_match in re.finditer(CRYOPROBE_PATTERN, preferred_text, re.IGNORECASE):
            local = preferred_text[max(0, cryo_match.start() - 240) : min(len(preferred_text), cryo_match.end() + 260)]
            local_lower = local.lower()

            specimen_context = bool(specimen_any_re.search(local_lower)) and not bool(specimen_none_re.search(local_lower))
            biopsy_context = bool(
                biopsy_token_re.search(local_lower) or specimen_context or sent_to_path_re.search(local_lower)
            )
            therapeutic_context = bool(therapeutic_context_re.search(local_lower) or cryo_extraction_context_re.search(local_lower))

            if therapeutic_context:
                saw_therapeutic = True
            if biopsy_context and not therapeutic_context:
                saw_biopsy_only = True

        if saw_biopsy_only and not saw_therapeutic:
            return {}
        # Generic cryoprobe mentions without explicit therapeutic intent are common in diagnostic
        # cryobiopsy workflows; do not assert cryotherapy unless therapeutic context is present.
        if not saw_therapeutic:
            return {}
        return {"cryotherapy": {"performed": True}}

    return {}


def extract_rigid_bronchoscopy(note_text: str) -> Dict[str, Any]:
    """Extract rigid bronchoscopy indicator."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    for pattern in RIGID_BRONCHOSCOPY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            proc: dict[str, Any] = {"performed": True}

            # Best-effort rigid scope size (inner diameter, mm)
            size_match = re.search(
                r"(?i)\b(\d+(?:\.\d+)?)\s*-?\s*mm\b[^.\n]{0,40}\b(?:non[-\s]?ventilating\s+)?(?:rigid\s+)?(?:tracheoscope|bronch(?:oscope|oscop)?|scope|barrel)\b",
                preferred_text or "",
            )
            if not size_match:
                size_match = re.search(
                    r"(?i)\brigid(?:\s+bronch(?:oscope|oscop))?\b[^.\n]{0,40}\b(\d+(?:\.\d+)?)\s*-?\s*mm\b",
                    preferred_text or "",
                )
            if size_match:
                try:
                    proc["rigid_scope_size"] = float(size_match.group(1))
                except Exception:
                    pass

            return {"rigid_bronchoscopy": proc}

    return {}


def extract_navigational_bronchoscopy(note_text: str) -> Dict[str, Any]:
    """Extract navigational/robotic bronchoscopy indicator."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    for pattern in NAVIGATIONAL_BRONCHOSCOPY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            return {"navigational_bronchoscopy": {"performed": True}}
    return {}


def extract_tbna_conventional(note_text: str) -> Dict[str, Any]:
    """Extract conventional TBNA indicator."""
    preferred_text, used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text) if used_detail else _strip_cpt_definition_lines(preferred_text)
    raw_text = preferred_text or ""
    if not raw_text.strip():
        return {}

    ebus_context_re = re.compile(
        r"\b(?:ebus|endobronchial\s+ultrasound|convex\s+probe|ebus[-\s]?tbna)\b",
        re.IGNORECASE,
    )
    # Radial EBUS is commonly used for peripheral lesion localization; TBNA in a
    # radial-EBUS paragraph is often peripheral/lung TBNA (not nodal EBUS-TBNA).
    radial_context_re = re.compile(
        r"\b(?:radial\s+ebus|radial\s+probe|r-?ebus|rp-?ebus|miniprobe)\b",
        re.IGNORECASE,
    )
    nodal_ebus_context_re = re.compile(
        r"\b(?:endobronchial\s+ultrasound|convex\s+probe|ebus[-\s]?tbna|linear\s+ebus)\b",
        re.IGNORECASE,
    )
    transthoracic_core_context_re = re.compile(
        r"(?i)\b(?:transthoracic|percutaneous|coaxial|core\s+needle\s+biops(?:y|ies)|pleural[-\s]?based|chest\s+wall)\b"
    )
    bronchoscopic_workflow_re = re.compile(r"(?i)\bbronchoscop|bronchoscope|ebus|radial|navigation|robotic\b")
    note_has_nodal_ebus = bool(nodal_ebus_context_re.search(raw_text))

    def _local_context(text: str, start: int, end: int, before_lines: int = 4, after_lines: int = 4) -> str:
        line_start = start
        for _ in range(before_lines + 1):
            prev_nl = text.rfind("\n", 0, line_start)
            if prev_nl == -1:
                line_start = 0
                break
            line_start = prev_nl
        if line_start != 0:
            line_start += 1

        line_end = end
        for _ in range(after_lines + 1):
            next_nl = text.find("\n", line_end)
            if next_nl == -1:
                line_end = len(text)
                break
            line_end = next_nl + 1

        return text[line_start:line_end]

    nodal_stations: set[str] = set()
    peripheral_hit = False
    peripheral_targets: list[str] = []

    def _extract_peripheral_target(text: str) -> str | None:
        raw = text or ""
        if not raw:
            return None
        # Prefer explicit segment wording (e.g., "left upper lobe posterior segment").
        seg_match = re.search(
            r"(?i)\b(?:left|right)\s+(?:upper|middle|lower)\s+lobe\b[^.\n]{0,40}\b(?:anterior|posterior|apical|lateral|medial)\s+segment\b",
            raw,
        )
        if seg_match:
            return seg_match.group(0).strip()
        abbrev_match = re.search(
            r"(?i)\b(?:rul|rml|rll|lul|lll)\b[^.\n]{0,40}\b(?:segment|seg)\b",
            raw,
        )
        if abbrev_match:
            return abbrev_match.group(0).strip()
        lobes = _extract_lung_locations_from_text(raw)
        if lobes:
            return lobes[0]
        if re.search(r"(?i)\b(?:mass|lesion|nodule)\b", raw):
            return "Lung Mass"
        return None

    for pattern in TBNA_CONVENTIONAL_PATTERNS:
        for match in re.finditer(pattern, raw_text, re.IGNORECASE):
            # Treat TBNA mentions inside an EBUS paragraph as EBUS-TBNA, not conventional TBNA.
            lookback_start = max(0, match.start() - 800)
            paragraph_break = raw_text.rfind("\n\n", lookback_start, match.start())
            if paragraph_break != -1:
                lookback_start = paragraph_break + 2
            ebus_lookback = raw_text[lookback_start:match.start()]
            ebus_lookahead = raw_text[match.end() : min(len(raw_text), match.end() + 40)]
            if ebus_context_re.search(ebus_lookback) or ebus_context_re.search(ebus_lookahead):
                # Allow peripheral TBNA described in a radial-EBUS navigation paragraph
                # or explicitly labeled as conventional TBNA after EBUS scope removal.
                local = _local_context(raw_text, match.start(), match.end(), before_lines=4, after_lines=4)
                if not radial_context_re.search(local):
                    if not re.search(r"(?i)\bconventional\s+tbna\b", local) and not re.search(
                        r"(?i)\bebus\s+bronchoscope\b[^.\n]{0,160}\bwithdrawn\b", local
                    ) and not re.search(r"(?i)\btherapeutic\s+bronchoscope\b", local):
                        continue

            before = raw_text[max(0, match.start() - 120) : match.start()]
            if re.search(r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}$", before, re.IGNORECASE):
                continue

            # Prefer station tokens on the same line as the TBNA mention; this avoids
            # "bleeding" stations from specimen logs or later EBUS station lists.
            line_start = raw_text.rfind("\n", 0, match.start())
            line_start = 0 if line_start == -1 else line_start + 1
            line_end = raw_text.find("\n", match.end())
            line_end = len(raw_text) if line_end == -1 else line_end
            line = raw_text[line_start:line_end]
            stations = _extract_ln_stations_from_text(line)

            if not stations:
                # Some templates put the station token on the preceding line; allow a
                # one-line lookback but ignore specimen headings.
                prev_end = line_start - 1
                if prev_end > 0:
                    prev_start = raw_text.rfind("\n", 0, prev_end)
                    prev_start = 0 if prev_start == -1 else prev_start + 1
                    prev_line = raw_text[prev_start:prev_end]
                    if prev_line and not re.search(r"(?i)\bspecimen", prev_line):
                        stations = _extract_ln_stations_from_text(prev_line)
            if stations:
                # When the note contains nodal EBUS language anywhere, station-based TBNA mentions
                # are more likely EBUS-TBNA than conventional (prevents phantom tbna_conventional).
                if note_has_nodal_ebus:
                    continue
                for station in stations:
                    if station:
                        nodal_stations.add(str(station).upper().strip())
                continue

            # Guardrail: in nodal EBUS templates, "Biopsy Tools: TBNA" lines are part of
            # station result blocks and should not imply peripheral/lung TBNA.
            if note_has_nodal_ebus:
                local = _local_context(raw_text, match.start(), match.end(), before_lines=3, after_lines=3)
                if re.search(r"(?i)\bbiopsy\s+tools\s*:\s*tbna\b", local):
                    continue

            peripheral_hit = True
            target = _extract_peripheral_target(line)
            if target and target not in peripheral_targets:
                peripheral_targets.append(target)

    # Some peripheral sampling notes document "needle biopsies" without explicitly
    # naming TBNA; treat these as peripheral TBNA when the workflow context supports it.
    for match in re.finditer(r"\bneedle\s+biops(?:y|ies)\b", raw_text, re.IGNORECASE):
        before = raw_text[max(0, match.start() - 120) : match.start()]
        if re.search(r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}$", before, re.IGNORECASE):
            continue

        local = _local_context(raw_text, match.start(), match.end(), before_lines=4, after_lines=4)
        # Avoid nodal EBUS blocks; those are handled under linear_ebus.
        if nodal_ebus_context_re.search(local) or _extract_ln_stations_from_text(local):
            continue
        if transthoracic_core_context_re.search(local) and not bronchoscopic_workflow_re.search(local):
            continue

        # Require a peripheral workflow signal (radial EBUS, navigation, fluoroscopy, etc).
        if not (
            radial_context_re.search(local)
            or re.search(
                r"(?i)\b(?:navigat(?:ion|ional)|ion\b|robotic|superdimension|fluoro(?:scop)?y|lesion|nodule|mass)\b",
                local,
            )
        ):
            continue
        if transthoracic_core_context_re.search(local) and not bronchoscopic_workflow_re.search(local):
            continue

        peripheral_hit = True
        target = _extract_peripheral_target(local)
        if target and target not in peripheral_targets:
            peripheral_targets.append(target)

    result: dict[str, Any] = {}
    if nodal_stations:
        result["tbna_conventional"] = {"performed": True, "stations_sampled": sorted(nodal_stations)}
    if peripheral_hit:
        targets = peripheral_targets or ["Lung Mass"]
        result["peripheral_tbna"] = {"performed": True, "targets_sampled": targets}
    return result


def extract_linear_ebus(note_text: str) -> Dict[str, Any]:
    """Extract linear EBUS-TBNA indicator with station backfill when present."""
    preferred_text, used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    # Avoid misclassifying radial-only EBUS notes as linear EBUS.
    radial_only = bool(
        re.search(
            r"\b(?:radial\s+ebus|radial\s+endobronchial\s+ultrasound|r-?ebus|rebus|miniprobe|radial\s+probe)\b",
            text_lower,
            re.IGNORECASE,
        )
    )

    stations = _extract_ln_stations_from_text(preferred_text)

    if stations and re.search(r"\b(?:ebus|endobronchial\s+ultrasound)\b", text_lower, re.IGNORECASE):
        return {"linear_ebus": {"performed": True, "stations_sampled": stations}}

    for pattern in LINEAR_EBUS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            if radial_only and not stations:
                return {}
            payload: dict[str, Any] = {"performed": True}
            if stations:
                payload["stations_sampled"] = stations
            return {"linear_ebus": payload}

    return {}


def _extract_lung_locations_from_text(text: str) -> list[str]:
    text_lower = (text or "").lower()
    locations: list[str] = []

    def add(value: str) -> None:
        if value and value not in locations:
            locations.append(value)

    abbrev_patterns = {
        r"\brul\b": "RUL",
        r"\brml\b": "RML",
        r"\brll\b": "RLL",
        r"\blul\b": "LUL",
        r"\blll\b": "LLL",
    }
    for pattern, lobe in abbrev_patterns.items():
        if re.search(pattern, text_lower):
            add(lobe)

    if re.search(r"\blingula\b", text_lower):
        add("Lingula")

    sided_patterns = {
        r"\bright\s+upper(?:\s+lobe)?\b": "RUL",
        r"\bright\s+middle(?:\s+lobe)?\b": "RML",
        r"\bright\s+lower(?:\s+lobe)?\b": "RLL",
        r"\bleft\s+upper(?:\s+lobe)?\b": "LUL",
        r"\bleft\s+lower(?:\s+lobe)?\b": "LLL",
    }
    for pattern, lobe in sided_patterns.items():
        if re.search(pattern, text_lower):
            add(lobe)

    if re.search(r"\bupper\s+lobe\b", text_lower) and not any(loc in locations for loc in ("RUL", "LUL")):
        if re.search(r"\bright\b", text_lower):
            add("RUL")
        elif re.search(r"\bleft\b", text_lower):
            add("LUL")
        else:
            add("Upper lobe")

    if re.search(r"\bmiddle\s+lobe\b", text_lower) and "RML" not in locations:
        if re.search(r"\bright\b", text_lower):
            add("RML")
        else:
            add("Middle lobe")

    if re.search(r"\blower\s+lobe\b", text_lower) and not any(loc in locations for loc in ("RLL", "LLL")):
        if re.search(r"\bright\b", text_lower):
            add("RLL")
        elif re.search(r"\bleft\b", text_lower):
            add("LLL")
        else:
            add("Lower lobe")

    return locations


def extract_brushings(note_text: str) -> Dict[str, Any]:
    """Extract bronchial brushings indicator."""
    text_lower = (note_text or "").lower()
    for pattern in BRUSHINGS_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            prefix = text_lower[max(0, match.start() - 120) : match.start()]
            boundary = max(prefix.rfind("."), prefix.rfind("\n"))
            if boundary != -1:
                prefix = prefix[boundary + 1 :]
            if re.search(r"\b(?:no|not|without|declined|deferred)\b", prefix, re.IGNORECASE):
                continue

            brushings: dict[str, Any] = {"performed": True}
            window_start = max(0, match.start() - 50)
            window_end = min(len(note_text), match.end() + 50)
            locations = _extract_lung_locations_from_text(note_text[window_start:window_end])
            if locations:
                brushings["locations"] = locations

            return {"brushings": brushings}
    return {}


def extract_mechanical_debulking(note_text: str) -> Dict[str, Any]:
    """Extract mechanical debulking / excision indicator.

    Conservative guardrail: do not treat mucus/secretions clearance as tissue debulking.
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text = preferred_text or ""
    text_lower = text.lower()
    if not text_lower.strip():
        return {}

    # Guardrail: "debulking" can be used loosely for mucus/secretions clearance.
    mucus_only_context = bool(
        re.search(r"(?i)\b(?:mucus|mucous|secretions?|mucus\s+plug|blood\s+clots?)\b", text_lower)
        and not re.search(
            r"(?i)\b(?:tumou?r|lesion|mass|endobronchial|obstruct|stenos|granulation|recanaliz)\w*\b",
            text_lower,
        )
    )
    if mucus_only_context:
        return {}

    obstructing_material_context = bool(
        re.search(
            r"(?i)\b(?:fungal(?:-appearing)?|fungal\s+ball|necrotic|endobronchial\s+debris|obstructing\s+material|slough|debris)\b",
            text_lower,
        )
        and re.search(r"(?i)\b(?:obstruct(?:ing|ion)|occlud|endobronchial|lesion|airway)\b", text_lower)
        and re.search(r"(?i)\b(?:forceps|snare|microdebrider|rigid\s+coring)\b", text_lower)
        and re.search(r"(?i)\b(?:remov(?:ed|al)?|debulk(?:ed|ing)?|excise(?:d|ion)?|resect(?:ed|ion)?)\b", text_lower)
        and not re.search(r"(?i)\b(?:foreign\s+body|dental\s+fragment|coin|tooth|aspirat(?:ed|ion))\b", text_lower)
        and not re.search(r"(?i)\b(?:mucous?|secretions?|mucus\s+plug|blood\s+clot)\b", text_lower)
    )
    if obstructing_material_context:
        proc: dict[str, Any] = {"performed": True}
        locations = _extract_lung_locations_from_text(text)
        if locations:
            proc["locations"] = locations
        return {"mechanical_debulking": proc}

    for pattern in MECHANICAL_DEBULKING_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            prefix = text_lower[max(0, match.start() - 120) : match.start()]
            suffix = text_lower[match.end() : min(len(text_lower), match.end() + 120)]

            # Negation / future intent guardrail
            if re.search(r"(?i)\b(?:no|not|without|declined|deferred)\b", prefix):
                continue
            if re.search(
                r"(?i)\b(?:consider(?:ed|ation)?|recommend(?:ed|ation)?|plan(?:ned)?|future|next)\b",
                prefix,
            ) or re.search(r"(?i)\b(?:at\s+next\s+intervention|if\s+needed)\b", suffix):
                continue

            # Sentence-level guard: avoid "debulking of mucus/secretions" false positives.
            sent_start = max(
                text_lower.rfind(".", 0, match.start()),
                text_lower.rfind("!", 0, match.start()),
                text_lower.rfind("?", 0, match.start()),
                text_lower.rfind("\n", 0, match.start()),
            )
            sent_start = 0 if sent_start < 0 else sent_start + 1
            next_punct = re.search(r"[.!?\n]", text_lower[match.end() :])
            sent_end = len(text_lower) if next_punct is None else match.end() + next_punct.start()
            sentence = text_lower[sent_start:sent_end]
            if re.search(r"(?i)\b(?:mucous|mucus|secretions?|plug|clot|blood\s+clot)\b", sentence) and not re.search(
                r"(?i)\b(?:tumou?r|lesion|mass|endobronchial|obstruct|stenos|granulation|web|fungating|recanaliz)\w*\b",
                sentence,
            ):
                continue
            # Avoid treating biopsy removal language as debulking.
            if re.search(r"(?i)\bbiops(?:y|ies)\b", sentence) and not re.search(
                r"(?i)\b(?:debulk|resect|excise|excision|core\s*out|snare|microdebrid|rigid\s+coring)\b",
                sentence,
            ):
                continue
            if re.search(r"(?i)\b(?:lesion|mass)\b[^.\n]{0,60}\bremoved\b", sentence) and not re.search(
                r"(?i)\b(?:debulk|resect|excise|excision|core\s*out|snare|microdebrid|rigid\s+coring|tumou?r)\b",
                sentence,
            ):
                prev_window = text_lower[max(0, sent_start - 200) : sent_start]
                if re.search(r"(?i)\bbiops(?:y|ies)\b", prev_window):
                    continue

            window = text_lower[max(0, match.start() - 220) : min(len(text_lower), match.end() + 220)]
            if not re.search(
                r"(?i)\b(?:tumou?r|lesion|mass|endobronchial|obstruct|stenos|granulation|web|fungating|recanaliz)\w*\b",
                window,
            ):
                continue

            proc: dict[str, Any] = {"performed": True}
            locations = _extract_lung_locations_from_text(window)
            if locations:
                proc["locations"] = locations

            return {"mechanical_debulking": proc}

    return {}


def extract_bpf_sealant(note_text: str) -> Dict[str, Any]:
    """Extract bronchopleural fistula (BPF) glue/sealant intervention indicator."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text = preferred_text or ""
    text_lower = text.lower()
    if not text_lower.strip():
        return {}

    has_agent = any(re.search(p, text_lower, re.IGNORECASE) for p in BPF_SEALANT_AGENT_PATTERNS)
    has_action = bool(
        re.search(r"(?i)\b(?:instill(?:ed|ation)?|appl(?:y|ied|ication)|inject(?:ed|ion)?)\b", text_lower)
    )
    has_bpf = any(re.search(p, text_lower, re.IGNORECASE) for p in BPF_SEALANT_PATTERNS)
    has_occlusion_context = bool(
        re.search(r"(?i)\b(?:block|occlud|seal|plug|subsegment|segment|fistula|air\s+leak)\b", text_lower)
    )
    if not (has_bpf or has_occlusion_context):
        return {}
    if not (has_agent and has_action):
        return {}

    # Negation guardrail
    if re.search(
        r"(?i)\b(?:no|not|without|declined|deferred)\b[^.\n]{0,100}\b(?:glue|sealant|veno)\b",
        text_lower,
    ):
        return {}

    proc: dict[str, Any] = {"performed": True}
    if re.search(r"(?i)\btisseel\b", text):
        proc["sealant_type"] = "Tisseel"
    elif re.search(r"(?i)\bveno-?\s*seal\b", text):
        proc["sealant_type"] = "Veno-seal"
    elif re.search(r"(?i)\bfibrin\s+glue\b", text):
        proc["sealant_type"] = "Fibrin glue"
    elif re.search(r"(?i)\bcyanoacrylate\b", text):
        proc["sealant_type"] = "Cyanoacrylate"
    elif re.search(r"(?i)\bsealant\b", text):
        proc["sealant_type"] = "Sealant"
    elif re.search(r"(?i)\bglue\b", text):
        proc["sealant_type"] = "Glue"

    # Volume (cc/mL) near sealant mention (e.g., "Tisseel 2cc", "2 mL of fibrin glue").
    vol_match = re.search(
        r"(?i)\b(?P<vol>\d+(?:\.\d+)?)\s*(?:cc|ml)\b[^.\n]{0,40}\b(?:tisseel|fibrin\s+glue|cyanoacrylate|veno-?\s*seal|sealant|glue)\b"
        r"|\b(?:tisseel|fibrin\s+glue|cyanoacrylate|veno-?\s*seal|sealant|glue)\b[^.\n]{0,40}\b(?P<vol2>\d+(?:\.\d+)?)\s*(?:cc|ml)\b",
        text,
    )
    if vol_match:
        raw = vol_match.group("vol") or vol_match.group("vol2")
        try:
            vol_val = float(raw)
        except Exception:
            vol_val = None
        if vol_val is not None and 0 < vol_val <= 50:
            proc["volume_ml"] = vol_val

    return {"bpf_sealant": proc}


def extract_transbronchial_cryobiopsy(note_text: str) -> Dict[str, Any]:
    """Extract transbronchial cryobiopsy indicator."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    raw_text = preferred_text or ""
    text_lower = raw_text.lower()

    nodal_context_re = re.compile(r"(?i)\b(?:intranodal|lymph\s+node|stations?|mediastin(?:al|um)|hilar)\b")
    access_tract_re = re.compile(r"(?i)\b(?:tract|tunnel|needle\s*knife|access)\b")
    pulmonary_context_re = re.compile(
        r"(?i)\b(?:transbronchial|lung|pulmonary|parenchymal|nodule|mass|lesion|segment(?:s)?|subsegment(?:s)?|rul|rml|rll|lul|lll|rb\d{1,2}|lb\d{1,2})\b"
    )

    for pattern in TRANSBRONCHIAL_CRYOBIOPSY_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue

            local = raw_text[max(0, match.start() - 180) : min(len(raw_text), match.end() + 240)]

            if nodal_context_re.search(local) or _extract_ln_stations_from_text(local):
                # Intranodal cryobiopsy (EBUS-assisted access) should not be treated as
                # transbronchial lung cryobiopsy (31628 family). Map to endobronchial_biopsy
                # with Cryoprobe so CPT derivation uses 31625 rather than 31628.
                biopsy: dict[str, Any] = {"performed": True, "forceps_type": "Cryoprobe"}

                stations = _extract_ln_stations_from_text(raw_text)
                if stations:
                    biopsy["locations"] = [stations[0]]

                sample_match = re.search(
                    r"(?i)\b(?:obtained|taken|performed)\b[^.\n]{0,80}\bx\s*(?P<n>\d{1,2})\b",
                    raw_text,
                )
                if sample_match:
                    try:
                        biopsy["number_of_samples"] = int(sample_match.group("n"))
                    except Exception:
                        pass
                elif re.search(r"(?i)\bx\s*3\b", local):
                    biopsy["number_of_samples"] = 3

                return {"endobronchial_biopsy": biopsy}

            # Generic cryobiopsy tokens require pulmonary context; otherwise they are too ambiguous.
            if "transbronchial" not in match.group(0).lower() and "tblc" not in match.group(0).lower():
                if not pulmonary_context_re.search(local):
                    continue
                # "tract created" + nodal context suggests intranodal workflow; do not label as lung cryobiopsy.
                if access_tract_re.search(local) and nodal_context_re.search(raw_text):
                    continue

            return {"transbronchial_cryobiopsy": {"performed": True}}

    # Backstop: some notes describe diagnostic cryobiopsy using "cryoprobe biopsies"
    # without the token "cryobiopsy". Treat this as transbronchial cryobiopsy when
    # biopsy/pathology context is explicit and therapeutic/destructive intent is absent.
    cryo_matches = list(re.finditer(CRYOPROBE_PATTERN, raw_text, re.IGNORECASE))
    if cryo_matches:
        therapeutic_context_re = re.compile(
            r"\b(?:ablat|destroy|debulk|devitaliz|recanaliz|stenos(?:is|ed)|obstruction|tumou?r)\w*\b",
            re.IGNORECASE,
        )
        specimen_any_re = re.compile(r"\b(?:specimen(?:s)?|histolog(?:y|ic))\b", re.IGNORECASE)
        specimen_none_re = re.compile(r"\bspecimen(?:s)?\b[^.\n]{0,60}\b(?:none|n/?a|na)\b", re.IGNORECASE)
        biopsy_token_re = re.compile(r"\bbiops(?:y|ies|ied)\b", re.IGNORECASE)
        sent_to_path_re = re.compile(r"\bsent\b[^.\n]{0,40}\bpatholog(?:y|ic)\b", re.IGNORECASE)
        ild_context_re = re.compile(r"\b(?:ild|uip|nsip|interstitial\s+lung)\b", re.IGNORECASE)
        freeze_context_re = re.compile(r"\bfreeze\b", re.IGNORECASE)
        sample_context_re = re.compile(r"\b(?:sample(?:s)?|site\s*\d+)\b", re.IGNORECASE)

        for cryo_match in cryo_matches:
            local = raw_text[max(0, cryo_match.start() - 240) : min(len(raw_text), cryo_match.end() + 260)]
            local_lower = local.lower()

            specimen_context = bool(specimen_any_re.search(local_lower)) and not bool(specimen_none_re.search(local_lower))
            biopsy_context = bool(
                biopsy_token_re.search(local_lower) or specimen_context or sent_to_path_re.search(local_lower)
            )
            pulmonary_context = bool(pulmonary_context_re.search(local))
            therapeutic_context = bool(therapeutic_context_re.search(local_lower))

            if biopsy_context and pulmonary_context and not therapeutic_context:
                return {"transbronchial_cryobiopsy": {"performed": True}}

        # ILD cryobiopsy backstop: some bullet-style notes omit biopsy/pathology tokens but
        # still describe cryobiopsy via cryoprobe + freeze cycles + sampled sites.
        if (
            ild_context_re.search(raw_text)
            and freeze_context_re.search(raw_text)
            and sample_context_re.search(raw_text)
            and pulmonary_context_re.search(raw_text)
            and not therapeutic_context_re.search(text_lower)
        ):
            return {"transbronchial_cryobiopsy": {"performed": True}}
    return {}


def extract_transbronchial_biopsy(note_text: str) -> Dict[str, Any]:
    """Extract transbronchial (parenchymal/peripheral) biopsy indicator.

    Guardrails:
    - Exclude EBUS-nodal sampling blocks that misuse "transbronchial biopsies"
      language (common in templates).
    - Exclude cryobiopsy mentions (handled by extract_transbronchial_cryobiopsy).
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    raw_text = preferred_text or ""
    if not raw_text.strip():
        return {}

    ebus_context_re = re.compile(
        r"\b(?:ebus|endobronchial\s+ultrasound|lymph\s+node|stations?|subcarinal|paratracheal|hilar)\b",
        re.IGNORECASE,
    )
    cryo_context_re = re.compile(r"\b(?:cryo(?:biops(?:y|ies)|probe)|tbbc|tblc)\b", re.IGNORECASE)

    # Heuristic: peripheral workflows often document "tissue biopsies ... using forceps"
    # without the explicit "transbronchial" keyword (esp. with navigation/radial EBUS).
    peripheral_workflow = bool(
        re.search(
            r"(?i)\b(?:fluoro(?:scop)?y|radial\s+ebus|radial\s+probe|r-?ebus|navigat(?:ion|ional)|ion\b|robotic|superdimension)\b",
            raw_text,
        )
    )
    if peripheral_workflow:
        for match in re.finditer(
            r"\b(?:tissue\s+)?biops(?:y|ies)\b[^.\n]{0,140}\bforceps\b|\bforceps\b[^.\n]{0,140}\bbiops(?:y|ies)\b",
            raw_text,
            re.IGNORECASE,
        ):
            before = raw_text[max(0, match.start() - 120) : match.start()]
            if re.search(r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}$", before, re.IGNORECASE):
                continue

            local = raw_text[max(0, match.start() - 220) : min(len(raw_text), match.end() + 220)]
            if cryo_context_re.search(local):
                continue
            if ebus_context_re.search(local):
                if _extract_ln_stations_from_text(local):
                    continue
                if re.search(r"(?i)\bendobronchial\s+ultrasound\b[^.\n]{0,120}\bbiops", local):
                    continue

            proc: dict[str, Any] = {"performed": True}
            detail_window = raw_text[max(0, match.start() - 260) : min(len(raw_text), match.end() + 420)]

            locations: list[str] = []
            for lobe in _extract_lung_locations_from_text(detail_window):
                if lobe and lobe not in locations:
                    locations.append(lobe)
            if locations:
                proc["locations"] = locations

            if re.search(r"(?i)\bforceps\b", detail_window) and not cryo_context_re.search(detail_window):
                proc["forceps_type"] = "Standard"

            return {"transbronchial_biopsy": proc}

    for pattern in TRANSBRONCHIAL_BIOPSY_PATTERNS:
        for match in re.finditer(pattern, raw_text, re.IGNORECASE):
            before = raw_text[max(0, match.start() - 120) : match.start()]
            if re.search(r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}$", before, re.IGNORECASE):
                continue

            local = raw_text[max(0, match.start() - 220) : min(len(raw_text), match.end() + 220)]
            if cryo_context_re.search(local):
                continue
            if ebus_context_re.search(local):
                # If the local context has explicit nodal station tokens, treat it as nodal sampling.
                if _extract_ln_stations_from_text(local):
                    continue
                # "Endobronchial ultrasound guided ..." is almost always nodal EBUS-TBNA in templates.
                if re.search(r"(?i)\bendobronchial\s+ultrasound\b[^.\n]{0,120}\bbiops", local):
                    continue

            proc: dict[str, Any] = {"performed": True}

            detail_window = raw_text[max(0, match.start() - 260) : min(len(raw_text), match.end() + 420)]

            # Best-effort location list extraction for TBBx blocks.
            locations: list[str] = []
            loc_match = re.search(
                r"(?i)\btransbronchial\s+(?:lung\s+)?biops(?:y|ies)\b[^.\n]{0,320}\bat\s+(?P<locs>[^.\n]{4,500})",
                raw_text,
            )
            if loc_match:
                locs_text = (loc_match.group("locs") or "").strip()
                locs_text = re.sub(r"(?i)\btotal\s+\d{1,3}\s+samples?\b.*$", "", locs_text).strip(" ,;:-")
                for part in re.split(r"(?i)\s*,\s*|\s+\band\b\s+", locs_text):
                    candidate = re.sub(r"\s+", " ", part.strip(" ,;:-"))
                    if not candidate:
                        continue
                    if re.search(
                        r"(?i)\b(?:segment|subsegment|rb\d{1,2}|lb\d{1,2}|lobe|bronch(?:us|i|ial)|mainstem|trachea|carina|lingula)\b",
                        candidate,
                    ):
                        if candidate not in locations:
                            locations.append(candidate)

            if not locations:
                for lobe in _extract_lung_locations_from_text(detail_window):
                    if lobe and lobe not in locations:
                        locations.append(lobe)

            if locations:
                proc["locations"] = locations

            # Explicit sample count (when documented).
            sample_match = re.search(
                r"(?i)\btotal\s+(?P<n>\d{1,3})\s+samples?\s+(?:were\s+)?collected\b",
                detail_window,
            )
            if not sample_match:
                sample_match = re.search(
                    r"(?i)\b(?P<n>\d{1,3})\s+samples?\s+(?:were\s+)?collected\b",
                    detail_window,
                )
            if sample_match:
                try:
                    proc["number_of_samples"] = int(sample_match.group("n"))
                except Exception:
                    pass

            # Forceps family (cryo handled separately by cryobiopsy extractor).
            if not cryo_context_re.search(detail_window):
                if re.search(r"(?i)\balligator\s+forceps\b", detail_window):
                    proc["forceps_type"] = "Standard"
                elif re.search(r"(?i)\bforceps\b", detail_window):
                    proc["forceps_type"] = "Standard"

            return {"transbronchial_biopsy": proc}

    return {}


def extract_peripheral_ablation(note_text: str) -> Dict[str, Any]:
    """Extract peripheral ablation indicator with modality when possible."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    negation = re.search(
        r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}\b"
        r"(?:ablation|mwa|rfa|cryoablation)\b",
        text_lower,
        re.IGNORECASE,
    )
    if negation:
        return {}

    has_mwa = bool(
        re.search(r"\bmicrowave\s+ablation\b", text_lower, re.IGNORECASE)
        or re.search(r"\bmwa\b", text_lower, re.IGNORECASE)
        or re.search(r"\bavuecue\b", text_lower, re.IGNORECASE)
        or re.search(r"\bmicrowave\s+catheter\b", text_lower, re.IGNORECASE)
    )
    has_rfa = bool(
        re.search(r"\bradiofrequency\s+ablation\b", text_lower, re.IGNORECASE)
        or re.search(r"\brf\s+ablation\b", text_lower, re.IGNORECASE)
        or re.search(r"\brfa\b", text_lower, re.IGNORECASE)
    )
    has_cryo = bool(
        re.search(r"\bcryoablation\b", text_lower, re.IGNORECASE)
        or re.search(r"\bcryo\s*ablation\b", text_lower, re.IGNORECASE)
    )

    if not (has_mwa or has_rfa or has_cryo):
        return {}

    peripheral_context = bool(
        re.search(
            r"\b(?:peripheral|nodule|lesion|mass|parenchym|target\s+lesion|lung\s+nodule|pulmonary\s+nodule|"
            r"navigation|navigational|robotic|ion|cbct|cone\s*beam|tool[- ]?in[- ]?lesion)\b",
            text_lower,
            re.IGNORECASE,
        )
    )
    endobronchial_context = bool(
        re.search(
            r"\b(?:endobronch|airway|trachea|carina|main(?:\s*|-)?stem|bronch(?:us|ial)|stenos|stricture)\b",
            text_lower,
            re.IGNORECASE,
        )
    )

    # Avoid misclassifying central airway ablation as peripheral lung ablation.
    if endobronchial_context and not peripheral_context:
        return {}
    # "Endobronchial cryoablation"/cryotherapy for CAO should map to thermal/airway
    # interventions, not peripheral nodule ablation.
    if has_cryo and not (has_mwa or has_rfa):
        if re.search(r"\bendobronchial\s+cryoablation\b", text_lower, re.IGNORECASE):
            return {}

    proc: dict[str, Any] = {"performed": True}
    if has_mwa:
        proc["modality"] = "Microwave"
    elif has_rfa:
        proc["modality"] = "Radiofrequency"
    elif has_cryo:
        proc["modality"] = "Cryoablation"

    return {"peripheral_ablation": proc}


def extract_thermal_ablation(note_text: str) -> Dict[str, Any]:
    """Extract thermal ablation indicator (APC/laser/electrocautery)."""
    preferred_text, used_detail = _preferred_procedure_detail_text(note_text)
    if used_detail:
        preferred_text = _strip_cpt_definition_lines(preferred_text)
    else:
        preferred_text = _strip_cpt_definition_lines(preferred_text)
    raw_text = preferred_text or ""
    text_lower = raw_text.lower()

    pleural_context_re = re.compile(r"(?i)\b(?:thoracoscopy|pleuroscopy|pleural|pleura|pleuroscop)\b")
    nodal_access_context_re = re.compile(
        r"(?i)\b(?:needle\s*knife|rx\s+needle\s*knife|tract|tunnel|access)\b.{0,160}\b(?:node|station|intranodal|lymph\s+node)\b"
        r"|\b(?:node|station|intranodal|lymph\s+node)\b.{0,160}\b(?:tract|tunnel|access)\b"
    )
    therapeutic_intent_re = re.compile(
        r"(?i)\b(?:tumou?r|lesion|mass|obstruct|stenos|web|granulation|recanaliz|debulk|destruct|ablat|hemostas|bleed)\w*\b"
    )
    airway_context_re = re.compile(
        r"(?i)\b(?:bronchoscop|airway|endobronch|trachea|carina|main(?:\s*|-)?stem|bronch(?:us|ial))\b"
    )

    modalities: list[str] = []
    performed_hit = False

    for pattern in THERMAL_ABLATION_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue

            local = raw_text[max(0, match.start() - 220) : min(len(raw_text), match.end() + 260)]

            # Pleural/thoracoscopic electrocautery should not map to endobronchial ablation.
            if pleural_context_re.search(local):
                continue

            # Needle-knife tract creation for nodal access is not tumor destruction.
            if nodal_access_context_re.search(local):
                continue

            is_apc_or_laser = bool(
                re.search(
                    r"(?i)\bapc\b|\bargon\s+plasma\b|\blaser\b|nd:?yag|\bco2\b|\bdiode\b",
                    local,
                )
            )
            if not is_apc_or_laser:
                # For generic electrocautery mentions, require airway + therapeutic intent.
                if not (airway_context_re.search(local) and therapeutic_intent_re.search(local)):
                    continue

            performed_hit = True
            local_modalities: list[str] = []
            if re.search(r"\bapc\b|\bargon\s+plasma\b", local, re.IGNORECASE):
                local_modalities.append("APC")
            if re.search(r"\belectrocautery\b|\bcauteriz", local, re.IGNORECASE):
                local_modalities.append("Electrocautery")
            if re.search(r"\bdiode\b", local, re.IGNORECASE):
                local_modalities.append("Laser (Diode)")
            if re.search(r"\bnd:?yag\b", local, re.IGNORECASE):
                local_modalities.append("Laser (Nd:YAG)")
            if re.search(r"\bco2\b", local, re.IGNORECASE):
                local_modalities.append("Laser (CO2)")

            for modality in local_modalities:
                if modality not in modalities:
                    modalities.append(modality)

    if not performed_hit:
        return {}

    proc: dict[str, Any] = {"performed": True}
    if modalities:
        proc["modality"] = modalities
    return {"thermal_ablation": proc}


def extract_percutaneous_tracheostomy(note_text: str) -> Dict[str, Any]:
    """Extract percutaneous tracheostomy indicator.

    Conservative: requires explicit tracheostomy procedure language.
    """
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()

    change_cue = re.search(
        r"(?i)\btrach(?:eostomy)?\b[^.\n]{0,60}\b(?:change|exchange|tube\s+change|changed)\b|\bafter\s+establishment\b[^.\n]{0,60}\btract\b",
        text_lower,
    )
    if not change_cue:
        removed_tube = re.search(
            r"(?i)\b(?:trach(?:eostomy)?|tracheostomy)\s+tube\b[^.\n]{0,120}\b(?:removed|exchanged|changed)\b",
            text_lower,
        )
        placed_new_tube = re.search(
            r"(?i)\bnew\b[^.\n]{0,60}\b(?:trach(?:eostomy)?|tracheostomy)\s+tube\b[^.\n]{0,120}\b(?:placed|inserted)\b",
            text_lower,
        )
        if removed_tube and placed_new_tube:
            change_cue = True
    if change_cue:
        return {}

    patterns = [
        r"\bpercutaneous\s+(?:dilatational\s+)?tracheostomy\b",
        r"\bperc\s+trach\b",
        r"\btracheostomy\b[^.\n]{0,60}\b(?:performed|created)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if not match:
            continue
        negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
        if re.search(negation_check, text_lower, re.IGNORECASE):
            continue

        proc: dict[str, Any] = {"performed": True}
        if "open" in match.group(0).lower():
            proc["method"] = "open"
        elif "percutaneous" in match.group(0).lower() or "perc trach" in match.group(0).lower():
            proc["method"] = "percutaneous"

        if re.search(r"\bportex\b", text_lower):
            proc["device_name"] = "Portex"
        elif re.search(r"\bshiley\b", text_lower):
            proc["device_name"] = "Shiley"

        return {"percutaneous_tracheostomy": proc}

    # Puncture-only (e.g., angiocath puncture for suture fixation / transtracheal access)
    # is *not* a tracheostomy creation. Downstream CPT logic may still derive 31612
    # via evidence reconciliation, but we do not mark percutaneous_tracheostomy here.
    if any(re.search(pat, text_lower, re.IGNORECASE) for pat in TRACHEAL_PUNCTURE_PATTERNS):
        return {}

    return {}


ESTABLISHED_TRACH_ROUTE_PATTERNS = [
    r"\bvia\s+(?:(?:the|an?)\s+)?(?:existing\s+)?trach(?:eostomy)?\b",
    r"\bthrough\s+(?:(?:the|an?)\s+)?(?:existing\s+)?trach(?:eostomy)?\b",
    r"\bvia\s+(?:(?:the|an?)\s+)?(?:existing\s+)?trach(?:eostomy)?\s+tube\b",
    r"\bbronchoscopy\b[^.\n]{0,80}\bvia\b[^.\n]{0,40}\b(?:existing\s+)?trach(?:eostomy)?\s+tube\b",
    r"\bbronchoscopy\b[^.\n]{0,80}\bthrough\b[^.\n]{0,40}\btrach(?:eostomy)?\s+tube\b",
    r"\btrach(?:eostomy)?\s+(?:stoma|tube)\b[^.\n]{0,40}\b(?:used|accessed|entered|through)\b",
    r"\bbronchoscope\b[^.\n]{0,60}\btrach(?:eostomy)?\b",
    r"\bestablished\s+trach(?:eostomy)?\b",
]

ESTABLISHED_TRACH_NEW_PATTERNS = [
    r"\bpercutaneous\s+(?:dilatational\s+)?tracheostomy\b",
    r"\bopen\s+tracheostomy\b",
    r"\btracheostomy\b[^.\n]{0,60}\b(?:performed|placed|inserted|created)\b",
    r"\bnew\s+trach(?:eostomy)?\b",
    r"\btrach(?:eostomy)?\s+(?:created|placed|inserted)\b",
    r"\btracheostomy\s+creation\b",
]


def extract_established_tracheostomy_route(note_text: str) -> Dict[str, Any]:
    """Detect bronchoscopy via an established tracheostomy route."""
    preferred_text, _used_detail = _preferred_procedure_detail_text(note_text)
    preferred_text = _strip_cpt_definition_lines(preferred_text)
    text_lower = (preferred_text or "").lower()
    if not text_lower.strip():
        return {}

    immature_or_newly_reestablished = bool(
        re.search(
            r"(?i)\b(?:immature|early|fresh)\s+tract\b"
            r"|\bnot\s+yet\s+epithelialized\b"
            r"|\baccidental\s+decannulat\w*\b"
            r"|\b(?:day|pod)\s*(?:0?[1-9]|1[0-4])\b[^.\n]{0,80}\btrach(?:eostomy)?\b"
            r"|\bpartially\s+closed\b[^.\n]{0,60}\btract\b"
            r"|\burgent\s+(?:tube\s+)?reinsertion\b",
            text_lower,
        )
    )

    change_cue = re.search(
        r"(?i)\btrach(?:eostomy)?\b[^.\n]{0,60}\b(?:change|exchange|tube\s+change|changed)\b|\bafter\s+establishment\b[^.\n]{0,60}\btract\b",
        text_lower,
    )
    if not change_cue:
        removed_tube = re.search(
            r"(?i)\b(?:trach(?:eostomy)?|tracheostomy)\s+tube\b[^.\n]{0,120}\b(?:removed|exchanged|changed)\b",
            text_lower,
        )
        placed_new_tube = re.search(
            r"(?i)\bnew\b[^.\n]{0,60}\b(?:trach(?:eostomy)?|tracheostomy)\s+tube\b[^.\n]{0,120}\b(?:placed|inserted)\b",
            text_lower,
        )
        if removed_tube and placed_new_tube:
            change_cue = True
    if immature_or_newly_reestablished:
        return {}
    if change_cue:
        return {"established_tracheostomy_route": True}

    for pattern in ESTABLISHED_TRACH_NEW_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return {}

    # Negation guardrail: avoid false positives from explicit "no trach" language, but
    # do not treat phrases like "Not assessed due to ... tracheostomy tube" as negation.
    if re.search(r"\b(?:no|without)\b[^.\n]{0,60}\btrach(?:eostomy)?\b", text_lower) or re.search(
        r"\bnot\s+(?:an?\s+)?(?:existing\s+)?trach(?:eostomy)?\b",
        text_lower,
    ):
        return {}

    for pattern in ESTABLISHED_TRACH_ROUTE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return {"established_tracheostomy_route": True}

    return {}


def extract_neck_ultrasound(note_text: str) -> Dict[str, Any]:
    """Extract neck ultrasound indicator (often pre-tracheostomy vascular mapping)."""
    text_lower = (note_text or "").lower()
    patterns = [
        r"\bneck\s+ultrasound\b",
        r"\bultrasound\s+of\s+(?:the\s+)?neck\b",
    ]

    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            return {"neck_ultrasound": {"performed": True}}

    return {}


def _extract_checked_side(note_text: str, header: str) -> str | None:
    """Extract Right/Left/Bilateral side from checkbox-style lines."""
    if not note_text:
        return None

    header_lower = header.lower()
    for line in note_text.splitlines():
        if header_lower not in line.lower():
            continue
        if re.search(r"(?i)\b1\D{0,6}bilateral\b", line):
            return "Bilateral"
        if re.search(r"(?i)\b1\D{0,6}left\b", line):
            return "Left"
        if re.search(r"(?i)\b1\D{0,6}right\b", line):
            return "Right"
    return None


def _extract_checked_option(note_text: str, header: str, options: list[str]) -> str | None:
    """Extract a checked option label from checkbox-style lines.

    Example pattern: "Volume: 0 None 1 Minimal 0 Small ..."
    """
    if not note_text:
        return None
    header_lower = header.lower()
    for line in note_text.splitlines():
        if header_lower not in line.lower():
            continue
        for option in options:
            if re.search(rf"(?i)\b1\D{{0,10}}{re.escape(option)}\b", line):
                return option
    return None


def extract_chest_ultrasound(note_text: str) -> Dict[str, Any]:
    """Extract chest ultrasound indicator (76604 family).

    Conservative: requires explicit "CHEST ULTRASOUND FINDINGS" or CPT 76604 context.
    """
    text = note_text or ""
    if not re.search("|".join(CHEST_ULTRASOUND_PATTERNS), text, re.IGNORECASE):
        return {}

    proc: dict[str, Any] = {"performed": True}

    if re.search("|".join(CHEST_ULTRASOUND_IMAGE_DOC_PATTERNS), text, re.IGNORECASE):
        proc["image_documentation"] = True

    hemithorax = _extract_checked_side(note_text, "Hemithorax")
    if hemithorax is not None:
        proc["hemithorax"] = hemithorax

    volume = _extract_checked_option(note_text, "Volume", ["None", "Minimal", "Small", "Moderate", "Large"])
    if volume is not None:
        proc["effusion_volume"] = volume

    echogenicity = _extract_checked_option(
        note_text, "Echogenicity", ["Anechoic", "Hypoechoic", "Isoechoic", "Hyperechoic"]
    )
    if echogenicity is not None:
        proc["effusion_echogenicity"] = echogenicity

    loculations = _extract_checked_option(note_text, "Loculations", ["None", "Thin", "Thick"])
    if loculations is not None:
        proc["effusion_loculations"] = loculations

    diaphragm = _extract_checked_option(note_text, "Diaphragmatic Motion", ["Normal", "Diminished", "Absent"])
    if diaphragm is not None:
        proc["diaphragmatic_motion"] = diaphragm

    lung_pre = _extract_checked_option(note_text, "Lung sliding before", ["Present", "Absent"])
    if lung_pre is not None:
        proc["lung_sliding_pre"] = lung_pre

    lung_post = _extract_checked_option(note_text, "Lung sliding post", ["Present", "Absent"])
    if lung_post is not None:
        proc["lung_sliding_post"] = lung_post

    consolidation = _extract_checked_option(note_text, "Lung consolidation/atelectasis", ["Present", "Absent"])
    if consolidation is not None:
        proc["lung_consolidation_present"] = consolidation == "Present"

    pleura = _extract_checked_option(note_text, "Pleura", ["Normal", "Thick", "Nodular"])
    if pleura is not None:
        proc["pleura_characteristics"] = pleura

    # Free-text findings often include key disease-burden/outcome statements.
    finding_lines: list[str] = []
    for raw_line in (note_text or "").splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue
        if re.search(r"(?i)\bno\s+drainable\s+fluid\b", line):
            finding_lines.append(line)
            continue
        if re.search(r"(?i)\batelectasis\b", line) or re.search(r"(?i)\bcollection\b", line):
            finding_lines.append(line)
            continue
    if finding_lines:
        summary = " ".join(finding_lines)
        proc["impression_text"] = summary[:500]

    plan_lines: list[str] = []
    for raw_line in (note_text or "").splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue
        if re.search(r"(?i)\b(?:d/c|dc|discontinue|remove|removed)\b[^.\n]{0,40}\bchest\s+tube\b", line):
            plan_lines.append(line)
    if plan_lines:
        proc["plan_text"] = " ".join(plan_lines)[:300]

    return {"chest_ultrasound": proc}


def extract_thoracentesis(note_text: str) -> Dict[str, Any]:
    """Extract thoracentesis indicators for pleural procedures."""
    text = note_text or ""
    if not re.search("|".join(THORACENTESIS_PATTERNS), text, re.IGNORECASE):
        return {}

    thora: dict[str, Any] = {"performed": True}

    side_match = re.search(r"(?im)^\s*(left|right|bilateral)\s+thoracentesis\b", text)
    if not side_match:
        side_match = re.search(r"(?im)^\s*entry\s+site:\s*(left|right|bilateral)\b", text)
    if not side_match:
        side_match = re.search(r"(?i)\bthoracentesis\b[^.\n]{0,60}\b(left|right|bilateral)\b", text)
    if side_match:
        thora["side"] = side_match.group(1).capitalize()

    if re.search(r"(?i)\bultrasound[-\s]*(?:guided|guidance)\b", text):
        thora["guidance"] = "Ultrasound"
    elif re.search(r"(?i)\blandmark\b|\bblind\b", text):
        thora["guidance"] = "None/Landmark"

    if re.search(
        r"(?i)\btherapeutic\b[^.\n]{0,60}\bthoracentesis\b|\bthoracentesis\b[^.\n]{0,60}\btherapeutic\b",
        text,
    ):
        thora["indication"] = "Therapeutic"
    elif re.search(
        r"(?i)\bdiagnostic\b[^.\n]{0,60}\bthoracentesis\b|\bthoracentesis\b[^.\n]{0,60}\bdiagnostic\b",
        text,
    ):
        thora["indication"] = "Diagnostic"

    return {"thoracentesis": thora}


def extract_pleural_biopsy(note_text: str) -> Dict[str, Any]:
    """Extract percutaneous transthoracic/core pleural-biopsy workflows."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return {}

    explicit_transthoracic = bool(
        re.search(
            r"(?i)\b(?:ultrasound|ct)[-\s]+guided\b[^.\n]{0,120}\btransthoracic\b[^.\n]{0,80}\b(?:core\s+needle\s+)?biops(?:y|ies)\b"
            r"|\b(?:us|u/s)[-\s]+guided\b[^.\n]{0,120}\btransthoracic\b[^.\n]{0,80}\b(?:core\s+needle\s+)?biops(?:y|ies)\b"
            r"|\btransthoracic\b[^.\n]{0,80}\bbiops(?:y|ies)\b[^.\n]{0,40}\b(?:core|cores)\b"
            r"|\btransthoracic\s+biops(?:y|ies)\b[^.\n]{0,120}\b(?:ultrasound|us|u/s|ct)\b"
            r"|\btransthoracic\b[^.\n]{0,120}\bcore\s+needle\s+biops(?:y|ies)\b"
            r"|\bcore\s+needle\s+biops(?:y|ies)\b[^.\n]{0,120}\b(?:transthoracic|coaxial)\b",
            text,
        )
    )
    abrams_or_trucut = bool(re.search(r"(?i)\b(?:abrams|tru[-\s]?cut)\b", text))
    if not explicit_transthoracic and not abrams_or_trucut:
        return {}

    proc: dict[str, Any] = {"performed": True}
    lowered = text.lower()
    if "ultrasound" in lowered or "u/s" in lowered:
        proc["guidance"] = "Ultrasound"
    elif re.search(r"(?i)\bct\b|\bcomputed\s+tomography\b", text):
        proc["guidance"] = "CT"

    if re.search(r"(?i)\babrams\b", text):
        proc["needle_type"] = "Abrams needle"
    elif re.search(r"(?i)\btru[-\s]?cut\b", text):
        proc["needle_type"] = "Tru-cut"
    else:
        proc["needle_type"] = "Cutting needle"

    samples_match = re.search(
        r"(?i)\bobtain(?:ed)?\s+(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:core\s+)?samples?\b"
        r"|\b(?P<count2>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:core\s+)?(?:samples?|cores?)\b(?:[^.\n]{0,40}\b(?:were\s+)?obtained\b)?",
        text,
    )
    if samples_match:
        raw_count = (samples_match.group("count") or samples_match.group("count2") or "").strip().lower()
        try:
            proc["number_of_samples"] = int(raw_count)
        except Exception:
            word_to_int = {
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
            parsed = word_to_int.get(raw_count)
            if parsed is not None:
                proc["number_of_samples"] = parsed

    if re.search(r"(?i)\b(right|rll|rml|rul)\b", text):
        proc["side"] = "Right"
    elif re.search(r"(?i)\b(left|lll|lul)\b", text):
        proc["side"] = "Left"

    return {"pleural_biopsy": proc}


def extract_chest_tube(note_text: str) -> Dict[str, Any]:
    """Extract chest tube / pleural drainage catheter insertion (32556/32557/32551 family)."""
    text = note_text or ""
    text_lower = text.lower()

    has_pigtail = re.search(r"(?i)\bpigtail\s+catheter\b", text) is not None
    has_chest_tube = re.search(r"(?i)\bchest\s+tube\b", text) is not None
    has_insertion = (
        re.search(r"(?i)\b(insert(?:ed)?|placed|placement|insertion|introduc(?:e|ed))\b", text) is not None
    )
    has_incision = re.search(r"(?i)\bincision\b|\bincised\b|\bcut\s+down\b", text) is not None

    maintenance_only = False
    if (has_pigtail or has_chest_tube) and not has_insertion and not has_incision:
        maintenance_only = bool(
            re.search(
                r"(?is)\bexisting\b[^.\n]{0,80}\b(?:chest\s+tube|pigtail\s+catheter)\b",
                text,
            )
            or re.search(
                r"(?is)\b(?:chest\s+tube|pigtail\s+catheter)\b[^.\n]{0,120}\b(?:left\s+in\s+place|remain(?:s|ed)?\s+in\s+place|to\s+suction|on\s+suction|connected\s+to\s+suction)\b",
                text,
            )
        )

    if not ((has_pigtail and has_insertion) or (has_chest_tube and has_insertion) or maintenance_only):
        return {}

    proc: dict[str, Any] = {"performed": True, "action": "Insertion"}
    if maintenance_only:
        proc["action"] = "Repositioning"

    side = _extract_checked_side(note_text, "Entry Site") or _extract_checked_side(
        note_text, "Hemithorax"
    )
    if side in {"Left", "Right"}:
        proc["side"] = side

    if not maintenance_only and ("pleural effusion" in text_lower or re.search(r"(?i)\beffusion\b", text)):
        proc["indication"] = "Effusion drainage"

    if has_pigtail:
        proc["tube_type"] = "Pigtail"
    elif re.search(r"(?i)\blarge\s+bore\b|\bsurgical\b", text):
        proc["tube_type"] = "Surgical/Large bore"
    elif re.search(r"(?i)\bstraight\b", text):
        proc["tube_type"] = "Straight"

    size_match = None
    for line in text.splitlines():
        if not re.search(r"(?i)\bsize\s*:", line):
            continue
        # Prefer checkbox-style selection like "1 14Fr" (avoid matching "12FR").
        size_match = re.search(r"(?i)\b1\D{1,6}(\d{1,2})\s*fr\b", line)
        if size_match:
            break
    if not size_match:
        size_match = re.search(r"(?i)\b1\D{1,6}(\d{1,2})\s*fr\b", text)
    if not size_match:
        size_match = re.search(r"(?i)\b(\d{1,2})\s*fr\b", text)
    if size_match:
        try:
            proc["tube_size_fr"] = int(size_match.group(1))
        except ValueError:
            pass

    if not maintenance_only:
        # Scope guidance detection to the chest-tube procedural neighborhood so
        # unrelated imaging mentions (e.g., radial ultrasound for lung lesions)
        # do not override explicit pleural fluoroscopy guidance.
        guidance_text = text
        anchor = re.search(r"(?i)\b(?:pigtail\s+catheter|chest\s+tube|thoracostomy|pleural\s+space|yueh)\b", text)
        if anchor:
            guidance_text = text[max(0, anchor.start() - 320) : min(len(text), anchor.end() + 520)]

        if re.search(r"(?i)\bfluoro(?:scopy|scopic)?\b", guidance_text):
            proc["guidance"] = "Fluoroscopy"
        elif re.search(r"(?i)\bct\b|\bcomputed tomography\b", guidance_text):
            proc["guidance"] = "CT"
        elif re.search(r"(?i)\bultrasound[-\s]*(?:guided|guidance)\b", guidance_text):
            proc["guidance"] = "Ultrasound"
        elif re.search(r"(?i)\bultrasound\b", guidance_text) and re.search(
            r"(?i)\b(?:pleural|chest\s+tube|pigtail|thoracostomy|catheter|yueh)\b",
            guidance_text,
        ):
            proc["guidance"] = "Ultrasound"

    return {"chest_tube": proc}


def extract_chest_tube_removal(note_text: str) -> Dict[str, Any]:
    """Extract chest tube removal events (distinct from insertion)."""
    text = note_text or ""
    if not text.strip():
        return {}

    removal_re = re.compile(
        r"(?is)\b(?:d/c|dc|discontinu(?:e|ed|ation)|remove(?:d|al)?|pull(?:ed)?|withdrawn)\b"
        r"[^.\n]{0,120}\b(?:chest\s+tube|tube\s+thoracostomy|thoracostomy\s+tube|pigtail\s+catheter|pleural\s+catheter)\b"
        r"|\b(?:chest\s+tube|tube\s+thoracostomy|thoracostomy\s+tube|pigtail\s+catheter|pleural\s+catheter)\b[^.\n]{0,120}"
        r"\b(?:removed|pulled|withdrawn|discontinued|d/c)\b"
    )
    if not removal_re.search(text):
        return {}

    proc: dict[str, Any] = {"performed": True}
    side = (
        _extract_checked_side(note_text, "Entry Site")
        or _extract_checked_side(note_text, "Hemithorax")
        or _extract_checked_side(note_text, "Side")
    )
    if side in {"Left", "Right"}:
        proc["side"] = side
    return {"chest_tube_removal": proc}


def extract_ipc(note_text: str) -> Dict[str, Any]:
    """Extract indwelling pleural catheter (IPC / tunneled pleural catheter)."""
    text = note_text or ""
    text_lower = text.lower()

    checkbox = _checkbox_selected(
        note_text,
        label_patterns=[
            r"tunne(?:l|ll)ed\s+pleural\s+catheter",
            r"indwelling\s+pleural\s+catheter",
            r"\bipc\b",
            r"\bpleurx\b",
            r"\baspira\b",
        ],
    )
    if checkbox is False:
        return {}

    matched_pattern: str | None = None
    for pattern in IPC_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            negation_check = r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}" + pattern
            if re.search(negation_check, text_lower, re.IGNORECASE):
                continue
            matched_pattern = pattern
            break

    if matched_pattern is None:
        fallback_header = bool(
            re.search(r"(?is)\bprocedure\s*:\s*[^\n]{0,240}\b32552\b", text)
            or re.search(r"(?i)\b32552\b", text)
            or re.search(r"(?i)\bRemoval\s+of\s+indwelling\s+tunneled\s+pleural\s+catheter\b", text)
        )
        if not fallback_header:
            return {}
        proc: dict[str, Any] = {"performed": True, "action": "Removal", "tunneled": True, "catheter_brand": "Other"}
        side = _extract_checked_side(note_text, "Side")
        if side in {"Left", "Right"}:
            proc["side"] = side
        return {"ipc": proc}

    if matched_pattern.startswith(r"\bipc\b") and not re.search(r"(?i)\b(?:pleur|effusion)\b", text):
        return {}

    if matched_pattern in {
        r"\btunne(?:l|ll)ed\s+catheter\b",
        r"\btunnel(?:ing)?\s+catheter\b",
    } and not re.search(
        r"(?i)\b(?:pleur|effusion)\b", text
    ):
        return {}

    if matched_pattern == r"\btunneling\s+device\b" and not re.search(r"(?i)\b(?:pleur|effusion)\b", text):
        return {}

    proc: dict[str, Any] = {"performed": True}
    if matched_pattern == r"\btunneling\s+device\b":
        proc["action"] = "Insertion"

    device = r"(?:pleurx|aspira|tunne(?:l|ll)ed\s+pleural\s+catheter|tunnel(?:ing)?\s+pleural\s+catheter|tunne(?:l|ll)ed\s+catheter|tunnel(?:ing)?\s+catheter|indwelling\s+pleural\s+catheter|ipc)"

    # Guardrail: do not treat mentions of a pre-existing catheter being left in place
    # (e.g., during pleuroscopy) as an IPC insertion/removal procedure.
    if re.search(
        rf"(?i)\b(?:previously|prior|existing|pre[-\s]?existing|already)\b[^\n]{{0,120}}\b{device}\b[^\n]{{0,180}}\b(?:left|remain(?:ed|s)?|kept)\s+in\s+place\b",
        text,
    ) or re.search(
        rf"(?i)\b{device}\b[^\n]{{0,120}}\b(?:previously|prior|existing|pre[-\s]?existing|already)\b[^\n]{{0,180}}\b(?:left|remain(?:ed|s)?|kept)\s+in\s+place\b",
        text,
    ):
        return {}

    def _action_window_hit(*, verbs: list[str]) -> bool:
        verb_union = "|".join(verbs)
        patterns = [
            rf"\b{device}\b[^.\n]{{0,80}}\b(?:{verb_union})\w*\b",
            rf"\b(?:{verb_union})\w*\b[^.\n]{{0,80}}\b{device}\b",
        ]
        return any(re.search(p, text_lower, re.IGNORECASE) for p in patterns)

    insertion_hit = _action_window_hit(
        verbs=["placement", "place", "insert", "insertion", "tunnel", "seldinger", "introduc", "advance"]
    )
    removal_hit = _action_window_hit(verbs=["remov", "pull", "extract", "retriev", "exchange"])
    if removal_hit and not insertion_hit:
        proc["action"] = "Removal"
    elif insertion_hit:
        proc["action"] = "Insertion"

    side = (
        _extract_checked_side(note_text, "Entry Site")
        or _extract_checked_side(note_text, "Hemithorax")
        or _extract_checked_side(note_text, "Side")
    )
    if side in {"Left", "Right"}:
        proc["side"] = side
    else:
        right = re.search(rf"(?i)\b(rt|right)\b[^.\n]{{0,40}}\b{device}\b", text) is not None
        left = re.search(rf"(?i)\b(lt|left)\b[^.\n]{{0,40}}\b{device}\b", text) is not None
        if right and not left:
            proc["side"] = "Right"
        elif left and not right:
            proc["side"] = "Left"

    if re.search(r"(?i)\bpleurx\b", text):
        proc["catheter_brand"] = "PleurX"
        proc["tunneled"] = True
    elif re.search(r"(?i)\baspira\b", text):
        proc["catheter_brand"] = "Aspira"
        proc["tunneled"] = True
    elif re.search(r"(?i)\brocket\b", text):
        proc["catheter_brand"] = "Rocket"
        proc["tunneled"] = True
    else:
        proc["catheter_brand"] = "Other"

    if re.search(r"(?i)\btunne(?:l|ll)ed\b", text) or re.search(r"(?i)\bindwelling\b", text):
        proc["tunneled"] = True

    if re.search(r"(?i)\bmalignant\b", text) and re.search(r"(?i)\beffusion\b", text):
        proc["indication"] = "Malignant effusion"

    return {"ipc": proc}


def extract_pleurodesis(note_text: str) -> Dict[str, Any]:
    """Extract pleurodesis signals (32560/32650 family)."""
    text = note_text or ""
    text_lower = text.lower()

    checkbox = _checkbox_selected(
        note_text,
        label_patterns=[
            r"\bpleurodesis\b",
            r"chemical\s+pleurodesis",
            r"\btalc\b",
            r"\bdoxycycline\b",
        ],
    )
    if checkbox is False:
        return {}

    if re.search(r"(?i)\b(?:no|not|without)\b[^.\n]{0,60}\bpleurodesis\b", text):
        return {}

    has_code = re.search(r"(?i)\b(?:32560|32650)\b", text) is not None
    has_word = re.search(r"(?i)\bpleurodesis\b", text) is not None
    has_agent = re.search(
        r"(?i)\b(?:talc|doxycycline|bleomycin|povidone-iodine|silver\s+nitrate)\b",
        text,
    ) is not None
    has_instillation = re.search(
        r"(?i)\b(?:instill(?:ed|ation)?|slurry|poudrage|insufflat(?:ed|ion)?|sclerosing\s+agent)\b",
        text,
    ) is not None

    if not (has_code or has_word or (has_agent and has_instillation)):
        return {}

    proc: dict[str, Any] = {"performed": True}

    if re.search(r"(?i)\bpoudrage\b|\binsufflat", text):
        proc["method"] = "Chemical - poudrage"
    elif re.search(r"(?i)\bslurry\b|\binstill", text) or re.search(
        r"(?i)\bthrough\b[^.\n]{0,60}\bchest\s+tube\b", text
    ):
        proc["method"] = "Chemical - slurry"

    if "doxycycline" in text_lower:
        proc["agent"] = "Doxycycline"
    elif "talc" in text_lower:
        proc["agent"] = "Talc"
        dose_match = re.search(
            r"(?i)\b(\d+(?:\.\d+)?)\s*(?:g|grams)\b[^.\n]{0,60}\btalc\b|\btalc\b[^.\n]{0,60}\b(\d+(?:\.\d+)?)\s*(?:g|grams)\b",
            text,
        )
        if dose_match:
            raw_dose = dose_match.group(1) or dose_match.group(2)
            try:
                dose_val = float(raw_dose)
            except (TypeError, ValueError):
                dose_val = None
            if dose_val is not None and 1 <= dose_val <= 10:
                proc["talc_dose_grams"] = dose_val
    elif "bleomycin" in text_lower:
        proc["agent"] = "Bleomycin"
    elif "povidone" in text_lower and "iodine" in text_lower:
        proc["agent"] = "Povidone-iodine"
    elif "silver nitrate" in text_lower:
        proc["agent"] = "Silver nitrate"

    if "malignant" in text_lower and "effusion" in text_lower:
        proc["indication"] = "Malignant effusion"
    elif "pneumothorax" in text_lower:
        proc["indication"] = "Recurrent pneumothorax"
    elif "recurrent" in text_lower and "effusion" in text_lower:
        proc["indication"] = "Recurrent benign effusion"

    return {"pleurodesis": proc}


def run_deterministic_extractors(note_text: str) -> Dict[str, Any]:
    """Run all deterministic extractors and return combined seed data.

    This function should be called before LLM extraction to provide
    reliable seed data for commonly missed fields.

    Args:
        note_text: Raw procedure note text

    Returns:
        Dict of extracted field values
    """
    seed_data: Dict[str, Any] = {}
    note_text = _maybe_unescape_newlines(note_text or "")

    # Demographics
    demographics = extract_demographics(note_text)
    seed_data.update(demographics)

    # ASA class
    asa = extract_asa_class(note_text)
    if asa is not None:
        seed_data["asa_class"] = asa

    # Sedation and airway
    sedation_airway = extract_sedation_airway(note_text)
    seed_data.update(sedation_airway)

    # Institution
    institution = extract_institution_name(note_text)
    if institution:
        seed_data["institution_name"] = institution

    # Primary indication
    indication = extract_primary_indication(note_text)
    if indication:
        seed_data["primary_indication"] = indication

    # Clinical context (explicit-only fields)
    bronchus_sign = extract_bronchus_sign(note_text)
    if bronchus_sign is not None:
        seed_data["bronchus_sign"] = bronchus_sign

    ecog = extract_ecog(note_text)
    if ecog:
        seed_data.update(ecog)

    # Disposition
    disposition = extract_disposition(note_text)
    if disposition:
        seed_data["disposition"] = disposition

    outcomes_data = extract_outcomes(note_text)
    if outcomes_data:
        seed_data.update(outcomes_data)

    # Bleeding severity
    bleeding = extract_bleeding_severity(note_text)
    if bleeding:
        seed_data["bleeding_severity"] = bleeding

    bleeding_interventions = extract_bleeding_intervention_required(note_text)
    if bleeding_interventions:
        seed_data["bleeding_intervention_required"] = bleeding_interventions

    # Providers
    providers = extract_providers(note_text)
    # Only include provider fields that were actually extracted
    for key, value in providers.items():
        if value is not None:
            seed_data.setdefault("providers", {})[key] = value

    # Procedure extractors (Phase 7)
    # BAL
    bal_data = extract_bal(note_text)
    if bal_data:
        seed_data.setdefault("procedures_performed", {}).update(bal_data)

    wll_data = extract_whole_lung_lavage(note_text)
    if wll_data:
        seed_data.setdefault("procedures_performed", {}).update(wll_data)

    # Therapeutic aspiration
    ta_data = extract_therapeutic_aspiration(note_text)
    if ta_data:
        seed_data.setdefault("procedures_performed", {}).update(ta_data)

    # Therapeutic instillation/injection (e.g., amphotericin)
    inj_data = extract_therapeutic_injection(note_text)
    if inj_data:
        seed_data.setdefault("procedures_performed", {}).update(inj_data)

    # Emergency endotracheal intubation (31500)
    intubation_data = extract_intubation(note_text)
    if intubation_data:
        seed_data.setdefault("procedures_performed", {}).update(intubation_data)

    dilation_data = extract_airway_dilation(note_text)
    if dilation_data:
        seed_data.setdefault("procedures_performed", {}).update(dilation_data)

    stent_data = extract_airway_stent(note_text)
    if stent_data:
        seed_data.setdefault("procedures_performed", {}).update(stent_data)

    balloon_occ_data = extract_balloon_occlusion(note_text)
    if balloon_occ_data:
        seed_data.setdefault("procedures_performed", {}).update(balloon_occ_data)

    blvr_data = extract_blvr(note_text)
    if blvr_data:
        seed_data.setdefault("procedures_performed", {}).update(blvr_data)

    diagnostic_bronch_data = extract_diagnostic_bronchoscopy(note_text)
    if diagnostic_bronch_data:
        seed_data.setdefault("procedures_performed", {}).update(diagnostic_bronch_data)

    foreign_body_data = extract_foreign_body_removal(note_text)
    if foreign_body_data:
        seed_data.setdefault("procedures_performed", {}).update(foreign_body_data)

    # Endobronchial biopsy
    ebx_data = extract_endobronchial_biopsy(note_text)
    if ebx_data:
        seed_data.setdefault("procedures_performed", {}).update(ebx_data)

    # Transbronchial biopsy
    tbbx_data = extract_transbronchial_biopsy(note_text)
    if tbbx_data:
        seed_data.setdefault("procedures_performed", {}).update(tbbx_data)

    radial_ebus_data = extract_radial_ebus(note_text)
    if radial_ebus_data:
        seed_data.setdefault("procedures_performed", {}).update(radial_ebus_data)

    eus_b_data = extract_eus_b(note_text)
    if eus_b_data:
        seed_data.setdefault("procedures_performed", {}).update(eus_b_data)

    linear_ebus_data = extract_linear_ebus(note_text)
    if linear_ebus_data:
        seed_data.setdefault("procedures_performed", {}).update(linear_ebus_data)

    cryotherapy_data = extract_cryotherapy(note_text)
    if cryotherapy_data:
        seed_data.setdefault("procedures_performed", {}).update(cryotherapy_data)

    mechanical_debulking_data = extract_mechanical_debulking(note_text)
    if mechanical_debulking_data:
        seed_data.setdefault("procedures_performed", {}).update(mechanical_debulking_data)

    rigid_bronch_data = extract_rigid_bronchoscopy(note_text)
    if rigid_bronch_data:
        seed_data.setdefault("procedures_performed", {}).update(rigid_bronch_data)

    nav_data = extract_navigational_bronchoscopy(note_text)
    if nav_data:
        seed_data.setdefault("procedures_performed", {}).update(nav_data)

    tbna_data = extract_tbna_conventional(note_text)
    if tbna_data:
        seed_data.setdefault("procedures_performed", {}).update(tbna_data)

    brushings_data = extract_brushings(note_text)
    if brushings_data:
        seed_data.setdefault("procedures_performed", {}).update(brushings_data)

    cryobiopsy_data = extract_transbronchial_cryobiopsy(note_text)
    if cryobiopsy_data:
        seed_data.setdefault("procedures_performed", {}).update(cryobiopsy_data)

    peripheral_ablation_data = extract_peripheral_ablation(note_text)
    if peripheral_ablation_data:
        seed_data.setdefault("procedures_performed", {}).update(peripheral_ablation_data)

    thermal_ablation_data = extract_thermal_ablation(note_text)
    if thermal_ablation_data:
        seed_data.setdefault("procedures_performed", {}).update(thermal_ablation_data)

    thermoplasty_data = extract_bronchial_thermoplasty(note_text)
    if thermoplasty_data:
        seed_data.setdefault("procedures_performed", {}).update(thermoplasty_data)

    bpf_sealant_data = extract_bpf_sealant(note_text)
    if bpf_sealant_data:
        seed_data.setdefault("procedures_performed", {}).update(bpf_sealant_data)

    # Percutaneous tracheostomy
    trach_data = extract_percutaneous_tracheostomy(note_text)
    if trach_data:
        seed_data.setdefault("procedures_performed", {}).update(trach_data)

    established_trach = extract_established_tracheostomy_route(note_text)
    if established_trach:
        seed_data.update(established_trach)

    # Neck ultrasound
    neck_us_data = extract_neck_ultrasound(note_text)
    if neck_us_data:
        seed_data.setdefault("procedures_performed", {}).update(neck_us_data)

    # Chest ultrasound
    chest_us_data = extract_chest_ultrasound(note_text)
    if chest_us_data:
        seed_data.setdefault("procedures_performed", {}).update(chest_us_data)

    # Pleural: thoracentesis
    thoracentesis_data = extract_thoracentesis(note_text)
    if thoracentesis_data:
        seed_data.setdefault("pleural_procedures", {}).update(thoracentesis_data)

    pleural_biopsy_data = extract_pleural_biopsy(note_text)
    if pleural_biopsy_data:
        seed_data.setdefault("pleural_procedures", {}).update(pleural_biopsy_data)

    # Backfill thoracentesis guidance when a separately documented chest ultrasound
    # is extracted (common templating: "76604 chest ultrasound" + "thoracentesis").
    try:
        pleural = seed_data.get("pleural_procedures")
        procs = seed_data.get("procedures_performed")
        if isinstance(pleural, dict) and isinstance(procs, dict):
            thora = pleural.get("thoracentesis")
            chest_us = procs.get("chest_ultrasound")
            if (
                isinstance(thora, dict)
                and thora.get("performed") is True
                and not str(thora.get("guidance") or "").strip()
                and isinstance(chest_us, dict)
                and chest_us.get("performed") is True
            ):
                explicit_no_us = bool(
                    re.search(
                        r"(?i)\bthoracentesis\b[^.\n]{0,120}\b(?:without|no)\b[^.\n]{0,60}\bultrasound\b",
                        note_text or "",
                    )
                )
                explicit_landmark = bool(re.search(r"(?i)\b(?:landmark|blind)\b", note_text or ""))
                if not explicit_no_us and not explicit_landmark:
                    thora["guidance"] = "Ultrasound"
    except Exception:
        pass

    # Pleural: chest tube / pleural drainage catheter
    chest_tube_data = extract_chest_tube(note_text)
    if chest_tube_data:
        seed_data.setdefault("pleural_procedures", {}).update(chest_tube_data)

    # Pleural: chest tube removal (distinct from insertion)
    chest_tube_removal_data = extract_chest_tube_removal(note_text)
    if chest_tube_removal_data:
        seed_data.setdefault("pleural_procedures", {}).update(chest_tube_removal_data)

    # Pleural: indwelling pleural catheter (IPC / tunneled pleural catheter)
    ipc_data = extract_ipc(note_text)
    if ipc_data:
        seed_data.setdefault("pleural_procedures", {}).update(ipc_data)

    pleurodesis_data = extract_pleurodesis(note_text)
    if pleurodesis_data:
        seed_data.setdefault("pleural_procedures", {}).update(pleurodesis_data)

    # ---------------------------------------------------------------------
    # Evidence spans for extracted attribute values (UI highlighting)
    # ---------------------------------------------------------------------
    evidence: dict[str, list[Span]] = {}

    def _add_first_match(field_path: str, pattern: re.Pattern[str], text: str) -> None:
        if not field_path:
            return
        if field_path in evidence:
            return
        match = pattern.search(text or "")
        if not match:
            return
        evidence.setdefault(field_path, []).append(
            Span(
                text=match.group(0).strip(),
                start=int(match.start()),
                end=int(match.end()),
                confidence=0.9,
            )
        )

    def _add_bal_volume_span(volume: object) -> None:
        try:
            vol_int = int(float(volume))  # handles int/float/str numerics
        except Exception:
            return
        instilled_re = re.compile(rf"(?i)\binstill(?:ed)?\s*{vol_int}\s*(?:ml|cc)\b")
        _add_first_match(
            "procedures_performed.bal.volume_instilled_ml",
            instilled_re,
            note_text,
        )

    def _add_stent_device_size_span(device_size: str) -> None:
        normalized_target = re.sub(r"\s+", "", (device_size or "")).lower()
        if not normalized_target:
            return

        best: re.Match[str] | None = None
        for match in DEVICE_SIZE_RE.finditer(note_text or ""):
            normalized = re.sub(r"\s+", "", (match.group(1) or "")).lower()
            if normalized == normalized_target:
                best = match
                break
            if best is None:
                best = match
        if best is None:
            return

        evidence.setdefault("procedures_performed.airway_stent.device_size", []).append(
            Span(
                text=best.group(0).strip(),
                start=int(best.start()),
                end=int(best.end()),
                confidence=0.9,
            )
        )

    def _add_airway_dilation_target_span(target: str) -> None:
        if target == "Stent expansion":
            pat = re.compile(r"(?i)\b(?:dilat\w*|balloon)\b[^.\n]{0,140}\bstent\b|\bstent\b[^.\n]{0,140}\b(?:dilat\w*|balloon)\b")
        elif target == "Stenosis":
            pat = re.compile(
                r"(?i)\b(?:dilat\w*|balloon)\b[^.\n]{0,140}\b(?:stenosis|stricture|lesion)\b"
                r"|\b(?:stenosis|stricture|lesion)\b[^.\n]{0,140}\b(?:dilat\w*|balloon)\b"
            )
        else:
            return
        _add_first_match("procedures_performed.airway_dilation.target_anatomy", pat, note_text)

    def _add_balloon_occlusion_literal(field_path: str, literal: object) -> None:
        if not field_path or field_path in evidence:
            return
        if not isinstance(literal, str) or not literal.strip():
            return
        match = re.search(re.escape(literal.strip()), note_text or "", re.IGNORECASE)
        if not match:
            return
        evidence.setdefault(field_path, []).append(
            Span(
                text=match.group(0).strip(),
                start=int(match.start()),
                end=int(match.end()),
                confidence=0.9,
            )
        )

    try:
        procs = seed_data.get("procedures_performed")
        if isinstance(procs, dict):
            bal = procs.get("bal")
            if isinstance(bal, dict) and bal.get("performed") is True:
                if bal.get("volume_instilled_ml") is not None:
                    _add_bal_volume_span(bal.get("volume_instilled_ml"))

            stent = procs.get("airway_stent")
            if isinstance(stent, dict) and stent.get("performed") is True:
                ds = stent.get("device_size")
                if isinstance(ds, str) and ds.strip():
                    _add_stent_device_size_span(ds)

            dilation = procs.get("airway_dilation")
            if isinstance(dilation, dict) and dilation.get("performed") is True:
                target = dilation.get("target_anatomy")
                if isinstance(target, str) and target.strip():
                    _add_airway_dilation_target_span(target)

            balloon_occ = procs.get("balloon_occlusion")
            if isinstance(balloon_occ, dict) and balloon_occ.get("performed") is True:
                _add_balloon_occlusion_literal(
                    "procedures_performed.balloon_occlusion.occlusion_location",
                    balloon_occ.get("occlusion_location"),
                )
                _add_balloon_occlusion_literal(
                    "procedures_performed.balloon_occlusion.air_leak_result",
                    balloon_occ.get("air_leak_result"),
                )
                _add_balloon_occlusion_literal(
                    "procedures_performed.balloon_occlusion.device_size",
                    balloon_occ.get("device_size"),
                )
    except Exception:
        # Evidence is best-effort; do not fail deterministic extraction.
        pass

    if evidence:
        seed_data["evidence"] = evidence

    return seed_data


__all__ = [
    "run_deterministic_extractors",
    "extract_demographics",
    "extract_asa_class",
    "extract_sedation_airway",
    "extract_institution_name",
    "extract_primary_indication",
    "extract_bronchus_sign",
    "extract_ecog",
    "extract_disposition",
    "extract_follow_up_plan_text",
    "extract_procedure_completed",
    "extract_outcomes",
    "extract_bleeding_severity",
    "extract_bleeding_intervention_required",
    "extract_providers",
    "extract_bal",
    "extract_whole_lung_lavage",
    "extract_bronchial_thermoplasty",
    "extract_therapeutic_aspiration",
    "is_true_therapeutic_aspiration",
    "extract_therapeutic_injection",
    "extract_airway_dilation",
    "extract_airway_stent",
    "classify_stent_action",
    "extract_balloon_occlusion",
    "extract_blvr",
    "extract_foreign_body_removal",
    "extract_endobronchial_biopsy",
    "extract_transbronchial_biopsy",
    "extract_radial_ebus",
    "extract_eus_b",
    "extract_cryotherapy",
    "extract_navigational_bronchoscopy",
    "extract_tbna_conventional",
    "extract_brushings",
    "extract_transbronchial_cryobiopsy",
    "extract_peripheral_ablation",
    "extract_thermal_ablation",
    "extract_percutaneous_tracheostomy",
    "extract_established_tracheostomy_route",
    "extract_neck_ultrasound",
    "extract_chest_ultrasound",
    "extract_thoracentesis",
    "extract_pleural_biopsy",
    "extract_chest_tube",
    "extract_chest_tube_removal",
    "extract_ipc",
    "extract_pleurodesis",
    "is_negated",
    "BAL_PATTERNS",
    "ENDOBRONCHIAL_BIOPSY_PATTERNS",
    "TRANSBRONCHIAL_BIOPSY_PATTERNS",
    "RADIAL_EBUS_PATTERNS",
    "EUS_B_PATTERNS",
    "CRYOTHERAPY_PATTERNS",
    "NAVIGATIONAL_BRONCHOSCOPY_PATTERNS",
    "TBNA_CONVENTIONAL_PATTERNS",
    "BRUSHINGS_PATTERNS",
    "TRANSBRONCHIAL_CRYOBIOPSY_PATTERNS",
    "PERIPHERAL_ABLATION_PATTERNS",
    "THERMAL_ABLATION_PATTERNS",
    "AIRWAY_STENT_DEVICE_PATTERNS",
    "AIRWAY_STENT_PLACEMENT_PATTERNS",
    "AIRWAY_STENT_REMOVAL_PATTERNS",
    "AIRWAY_DILATION_PATTERNS",
    "FOREIGN_BODY_REMOVAL_PATTERNS",
    "BLVR_PATTERNS",
    "BALLOON_OCCLUSION_PATTERNS",
    "TRACHEAL_PUNCTURE_PATTERNS",
    "ESTABLISHED_TRACH_ROUTE_PATTERNS",
    "ESTABLISHED_TRACH_NEW_PATTERNS",
    "CHEST_ULTRASOUND_PATTERNS",
    "THORACENTESIS_PATTERNS",
    "CHEST_TUBE_PATTERNS",
    "IPC_PATTERNS",
]
