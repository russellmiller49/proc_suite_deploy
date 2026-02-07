"""Field-specific normalization for registry extraction outputs.

Implementation note: this module is a package to allow submodules for targeted postprocess utilities.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List
import re
from datetime import datetime
import logging

from app.registry.schema import RegistryRecord

logger = logging.getLogger(__name__)


# Valid EBUS lymph node stations - canonical format
VALID_EBUS_STATIONS = frozenset({
    "2R", "2L", "4R", "4L", "7", "10R", "10L", "11R", "11L",
    # Also accept numeric-only for station 7
})

# Pattern to match valid station format
STATION_PATTERN = re.compile(r"^(2R|2L|4R|4L|7|10R|10L|11R|11L)$", re.IGNORECASE)

# Deterministic station ordering for stable outputs (and tests).
_EBUS_STATION_SORT_ORDER = (
    "2R",
    "2L",
    "3P",
    "4R",
    "4L",
    "7",
    "10R",
    "10L",
    "11R",
    "11L",
    "12R",
    "12L",
    "LUNG MASS",
    "OTHER",
)
_EBUS_STATION_SORT_INDEX = {s: i for i, s in enumerate(_EBUS_STATION_SORT_ORDER)}


def sort_ebus_stations(stations: list[str] | set[str]) -> list[str]:
    """Return stations in deterministic order with de-duping."""
    seen: set[str] = set()
    normalized: list[str] = []
    for station in stations:
        if not station:
            continue
        token = str(station).strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return sorted(normalized, key=lambda s: (_EBUS_STATION_SORT_INDEX.get(s, 999), s))

# Canonical ROSE result values (schema enum)
# Schema enum: ["Adequate - malignant", "Adequate - benign lymphocytes",
#               "Adequate - granulomas", "Adequate - other", "Inadequate", "Not performed"]
ROSE_RESULT_CANONICAL = {
    "malignant": "Adequate - malignant",
    "adequate - malignant": "Adequate - malignant",
    "benign": "Adequate - benign lymphocytes",
    "adequate - benign lymphocytes": "Adequate - benign lymphocytes",
    "nondiagnostic": "Inadequate",
    "non-diagnostic": "Inadequate",
    "non diagnostic": "Inadequate",
    "inadequate": "Inadequate",
    "insufficient": "Inadequate",
    "granuloma": "Adequate - granulomas",
    "granulomatous": "Adequate - granulomas",
    "adequate - granulomas": "Adequate - granulomas",
    "atypical": "Adequate - other",
    "atypical cells": "Adequate - other",
    "atypical cells present": "Adequate - other",
    "atypical lymphoid": "Adequate - other",
    "atypical lymphoid proliferation": "Adequate - other",
    "adequate - other": "Adequate - other",
    "not performed": "Not performed",
}

# ROSE result priority for deriving global result (higher = more significant)
ROSE_RESULT_PRIORITY = {
    "Adequate - malignant": 6,
    "Adequate - other": 5,  # Covers atypical cases
    "Adequate - granulomas": 4,
    "Adequate - benign lymphocytes": 3,
    "Inadequate": 2,
    "Not performed": 1,
}

__all__ = [
    "normalize_sedation_type",
    "normalize_airway_type",
    "map_pleural_guidance",
    "normalize_pleural_procedure",
    "normalize_pleural_side",
    "normalize_pleural_intercostal_space",
    "postprocess_patient_mrn",
    "normalize_procedure_date",
    "normalize_disposition",
    "postprocess_asa_class",
    "normalize_final_diagnosis_prelim",
    "normalize_stent_type",
    "normalize_stent_location",
    "normalize_stent_deployment_method",
    "normalize_ebus_rose_result",
    "normalize_ebus_station_rose_result",
    "derive_global_ebus_rose_result",
    "normalize_ebus_needle_gauge",
    "normalize_ebus_needle_type",
    "normalize_ebus_stations_detail",
    "normalize_elastography_pattern",
    "normalize_list_field",
    "normalize_anesthesia_agents",
    "normalize_ebus_stations",
    "normalize_linear_ebus_stations",
    "normalize_nav_sampling_tools",
    "normalize_follow_up_plan",
    "normalize_cao_location",
    "normalize_cao_tumor_location",
    "normalize_cpt_codes",
    "normalize_assistant_names",
    "normalize_ablation_modality",
    "normalize_airway_device_size",
    "normalize_nav_registration_method",
    "normalize_assistant_name_single",
    "normalize_ventilation_mode",
    "normalize_procedure_setting",
    "normalize_bronch_location_lobe",
    "normalize_attending_name",
    "normalize_provider_role",
    "normalize_immediate_complications",
    "normalize_radiographic_findings",
    "normalize_navigation_platform",
    "normalize_valve_type",
    "validate_station_format",
    "VALID_EBUS_STATIONS",
    "ROSE_RESULT_CANONICAL",
    "ROSE_RESULT_PRIORITY",
    "POSTPROCESSORS",
    # Granular data processing
    "process_granular_data",
    # Registry-record-level guardrails
    "populate_ebus_node_events_fallback",
    "sanitize_ebus_events",
    "cull_hollow_ebus_claims",
    "reconcile_ebus_sampling_from_specimen_log",
    "enrich_ebus_node_event_sampling_details",
    "enrich_ebus_node_event_outcomes",
    "enrich_linear_ebus_needle_gauge",
    "enrich_eus_b_sampling_details",
    "enrich_medical_thoracoscopy_biopsies_taken",
    "enrich_bal_from_procedure_detail",
]


def _coerce_to_text(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        raw = raw[0]
    text = str(raw).strip()
    return text or None


def normalize_sedation_type(raw: Any) -> str | None:
    """Normalize sedation type to schema enum values.

    Schema enum: ["Moderate", "Deep", "General", "MAC", "Local Only", "Topical Only"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.lower()

    mapping = {
        "moderate": "Moderate",
        "moderate sedation": "Moderate",
        "conscious sedation": "Moderate",
        "deep": "Deep",
        "deep sedation": "Deep",
        "general": "General",
        "general anesthesia": "General",
        "ga": "General",
        "local": "Local Only",
        "local anesthesia": "Local Only",
        "local only": "Local Only",
        "topical": "Topical Only",
        "topical only": "Topical Only",
        "topical anesthesia": "Topical Only",
        # MAC is its own enum value in the schema
        "monitored anesthesia care": "MAC",
        "mac": "MAC",
        "mac anesthesia": "MAC",
    }

    if text in mapping:
        val = mapping[text]
    else:
        if "general" in text and "anesth" in text:
            val = "General"
        elif "monitored anesthesia care" in text or " mac " in text or re.search(r"\bmac\b", text):
            val = "MAC"
        elif "deep" in text and "sedat" in text:
            val = "Deep"
        elif "conscious" in text and "sedat" in text:
            val = "Moderate"
        elif "moderate" in text and "sedat" in text:
            val = "Moderate"
        elif "local" in text and "anesth" in text:
            val = "Local Only"
        elif "topical" in text:
            val = "Topical Only"
        else:
            lowered_allowed = {
                "moderate": "Moderate",
                "deep": "Deep",
                "general": "General",
                "mac": "MAC",
                "local only": "Local Only",
                "topical only": "Topical Only",
            }
            val = lowered_allowed.get(text, None)

    allowed = {"Moderate", "Deep", "General", "MAC", "Local Only", "Topical Only"}
    return val if val in allowed else None


def normalize_airway_type(raw: Any) -> str | None:
    """Normalize airway type to schema enum values.

    Schema enum: ["Native", "ETT", "Tracheostomy", "LMA", "iGel"]
    NOTE: "Rigid Bronchoscope" is NOT a valid airway_type in the schema.
    Rigid bronchoscopy is a procedure type, not an airway management device.
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if text in {"", "none", "n/a", "na", "null"}:
        return None

    mapping = {
        "native": "Native",
        "native airway": "Native",
        "native airway with bite block": "Native",
        "natural": "Native",
        "natural airway": "Native",
        "spontaneous": "Native",
        "lma": "LMA",
        "laryngeal mask": "LMA",
        "laryngeal mask airway": "LMA",
        "ett": "ETT",
        "endotracheal": "ETT",
        "endotracheal tube": "ETT",
        "et tube": "ETT",
        "trach": "Tracheostomy",
        "tracheostomy": "Tracheostomy",
        "tracheostomy tube": "Tracheostomy",
        "igel": "iGel",
        "i-gel": "iGel",
        "i gel": "iGel",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching - order matters (more specific first)
    if "igel" in text or "i-gel" in text or "i gel" in text:
        return "iGel"
    if "trach" in text:
        return "Tracheostomy"
    if "lma" in text or "laryngeal mask" in text:
        return "LMA"
    if "ett" in text or "endotracheal" in text or "et tube" in text:
        return "ETT"
    if "native" in text or "natural airway" in text or "bite block" in text:
        return "Native"

    # Validate against allowed values
    allowed = {"Native", "ETT", "Tracheostomy", "LMA", "iGel"}
    title_cased = text.title()
    if title_cased in allowed:
        return title_cased

    return None


def map_pleural_guidance(raw_value: Any) -> str | None:
    text_raw = _coerce_to_text(raw_value)
    if text_raw is None:
        return None
    text = text_raw.lower()

    if text in {"ultrasound", "us", "u/s", "ultrasound-guided", "ultrasound guidance"}:
        return "Ultrasound"
    if text in {"ct", "ct-guided", "ct guidance", "computed tomography"}:
        return "CT"
    if text in {"blind", "no imaging", "unguided"}:
        return "Blind"

    if "ultrasound" in text or "sonograph" in text:
        return "Ultrasound"
    if "ct-guid" in text or " ct " in text or "computed tomography" in text:
        return "CT"
    if "no imaging" in text or "without imaging" in text or "blind" in text or "no image guidance" in text:
        return "Blind"

    return None


def normalize_pleural_procedure(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.lower()

    canonical = {
        "thoracentesis": "Thoracentesis",
        "chest tube": "Chest Tube",
        "chest tube removal": "Chest Tube Removal",
        "tunneled catheter": "Tunneled Catheter",
        "tunnelled catheter": "Tunneled Catheter",
        "tunneled catheter exchange": "Tunneled Catheter Exchange",
        "tunnelled catheter exchange": "Tunneled Catheter Exchange",
        "ipc drainage": "IPC Drainage",
        "medical thoracoscopy": "Medical Thoracoscopy",
        "chemical pleurodesis": "Chemical Pleurodesis",
    }
    if text in canonical:
        return canonical[text]

    # Map common brand names/abbreviations for tunneled catheters
    if any(k in text for k in ["pleurx", "aspira", "denver catheter", "ipc", "indwelling pleural catheter"]):
        if "exchange" in text or "replac" in text:
            return "Tunneled Catheter Exchange"
        if "drain" in text and "place" not in text and "insert" not in text:
            return "IPC Drainage"
        return "Tunneled Catheter"

    if "pleurodesis" in text:
        return "Chemical Pleurodesis"

    if "tunneled" in text or "tunnelled" in text:
        if "catheter" in text or "pleural catheter" in text:
            if "exchange" in text or "replac" in text:
                return "Tunneled Catheter Exchange"
            return "Tunneled Catheter"

    if "thoracoscopy" in text or "pleuroscopy" in text:
        return "Medical Thoracoscopy"

    if any(k in text for k in ["chest tube", "pleural drain", "pleural tube", "pigtail", "intercostal drain", "icd"]):
        if "remov" in text:
            return "Chest Tube Removal"
        return "Chest Tube"

    if "thoracentesis" in text or "pleural tap" in text:
        return "Thoracentesis"

    for k, v in canonical.items():
        if k in text:
            return v

    return None


def normalize_pleural_side(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if text in {"r", "rt", "right", "right-sided", "right side"}:
        return "Right"
    if text in {"l", "lt", "left", "left-sided", "left side"}:
        return "Left"
    if text.startswith("r"):
        return "Right"
    if text.startswith("l"):
        return "Left"
    return None


def normalize_pleural_intercostal_space(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.lower()
    match = re.search(r"(\\d{1,2})(?:st|nd|rd|th)?", text)
    if not match:
        return text_raw.strip() or None
    num = int(match.group(1))
    if 10 <= num % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")
    return f"{num}{suffix}"


def normalize_elastography_pattern(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    cleaned = text_raw.strip().rstrip(".;,")
    return cleaned or None


def postprocess_patient_mrn(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    raw_text = str(text_raw).strip()
    if not raw_text:
        return None

    # Remove surrounding quotes/punctuation that sometimes wrap IDs
    raw_text = raw_text.strip().strip("\"'").strip(",:;")

    # Reject obvious dates
    date_patterns = [
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",
        r"^\d{1,2}-\d{1,2}-\d{2,4}$",
        r"^\d{4}-\d{1,2}-\d{1,2}$",
    ]
    for pat in date_patterns:
        if re.match(pat, raw_text):
            return None

    # Strip common labels if present
    labeled = re.search(r"(?:MRN|Medical Record Number|Patient ID|Pt ID|ID)\s*[:#-]?\s*(.+)$", raw_text, re.IGNORECASE)
    candidate = labeled.group(1).strip() if labeled else raw_text
    candidate = candidate.strip().strip(",:;")

    # Reject obvious phone-like strings only when separators are present
    if re.search(r"[-()\s]", candidate) and re.fullmatch(r"[\d\s().+-]{6,}", candidate) and not re.search(r"[A-Za-z]", candidate):
        return None

    return candidate or None


def normalize_procedure_date(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    raw = text_raw.strip()
    if raw.lower() == "null" or raw == "":
        return None

    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if iso_match:
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return raw
        except ValueError:
            return None

    candidates = [
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%m/%d/%y",
        "%m-%d-%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    date_like = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", raw)
    if date_like:
        text = date_like.group(1)
        for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"]:
            try:
                dt = datetime.strptime(text, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    # Handle compact formats like 03Nov17 or 3Nov2017
    compact = re.search(r"(\d{1,2}[A-Za-z]{3}\d{2,4})", raw)
    if compact:
        text = compact.group(1)
        for fmt in ["%d%b%Y", "%d%b%y"]:
            try:
                dt = datetime.strptime(text, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    return None


def normalize_disposition(raw: Any) -> str | None:
    """Normalize disposition to schema enum values.

    Schema enum: ["Outpatient discharge", "Observation unit", "Floor admission",
                  "ICU admission", "Already inpatient - return to floor",
                  "Already inpatient - transfer to ICU", "Transfer to another facility",
                  "OR", "Death"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text:
        return None

    mapping_exact = {
        # Outpatient discharge
        "discharge home": "Outpatient discharge",
        "home": "Outpatient discharge",
        "outpatient discharge": "Outpatient discharge",
        "outpatient": "Outpatient discharge",
        "discharged": "Outpatient discharge",
        "pacu recovery": "Outpatient discharge",  # PACU then home = outpatient
        "pacu": "Outpatient discharge",
        "recovery": "Outpatient discharge",
        # Observation unit
        "observation unit": "Observation unit",
        "observation": "Observation unit",
        "obs": "Observation unit",
        "23 hour obs": "Observation unit",
        "23-hour observation": "Observation unit",
        # Floor admission
        "floor admission": "Floor admission",
        "admit to floor": "Floor admission",
        "floor": "Floor admission",
        # ICU admission
        "icu admission": "ICU admission",
        "admit to icu": "ICU admission",
        "icu": "ICU admission",
        # Already inpatient - return to floor
        "already inpatient - return to floor": "Already inpatient - return to floor",
        "return to floor": "Already inpatient - return to floor",
        "back to floor": "Already inpatient - return to floor",
        # Already inpatient - transfer to ICU
        "already inpatient - transfer to icu": "Already inpatient - transfer to ICU",
        "transfer to icu": "Already inpatient - transfer to ICU",
        # Transfer to another facility
        "transfer to another facility": "Transfer to another facility",
        "transfer": "Transfer to another facility",
        "transferred": "Transfer to another facility",
        # OR
        "or": "OR",
        "operating room": "OR",
        "to or": "OR",
        # Death
        "death": "Death",
        "expired": "Death",
        "deceased": "Death",
    }
    if text in mapping_exact:
        return mapping_exact[text]

    # ICU keywords (new admission or transfer)
    icu_keywords = [
        "micu",
        "sicu",
        "cticu",
        "ccu",
        "neuro icu",
        "burn icu",
        "intensive care",
        "critical care",
    ]
    if any(k in text for k in icu_keywords):
        if "already inpatient" in text or "transfer" in text:
            return "Already inpatient - transfer to ICU"
        return "ICU admission"
    if "icu" in text:
        if "already inpatient" in text or "transfer" in text:
            return "Already inpatient - transfer to ICU"
        return "ICU admission"

    # Floor keywords
    floor_keywords = [
        "admit to floor",
        "to the floor",
        "medicine floor",
        "surgery floor",
        "inpatient ward",
        "telemetry",
        "step-down",
        "step down",
        "intermediate care",
        "imcu",
        "admit to medicine",
        "admit to oncology",
        "admit to hospitalist",
    ]
    if any(k in text for k in floor_keywords):
        if "already inpatient" in text or "return" in text:
            return "Already inpatient - return to floor"
        return "Floor admission"

    # Observation keywords
    obs_keywords = [
        "admit for observation",
        "admitted for observation",
        "observation status",
        "obs unit",
    ]
    if any(k in text for k in obs_keywords):
        return "Observation unit"

    # Home/outpatient discharge keywords
    home_keywords = [
        "discharge home",
        "discharged home",
        "dc home",
        "sent home",
        "to home",
        "home after",
        "go home",
        "going home",
        "return home",
        "ok for discharge",
        "ok to discharge",
        "same-day discharge",
        "same day discharge",
        "to pacu",
        "post-anesthesia care",
        "post anesthesia care",
        "recovery room",
        "phase i recovery",
        "postop recovery",
        "short stay recovery",
        "day surgery recovery",
        "sds recovery",
    ]
    if any(k in text for k in home_keywords):
        return "Outpatient discharge"

    # Transfer keywords
    if "transfer" in text and "icu" not in text:
        return "Transfer to another facility"

    return None


def postprocess_asa_class(raw_text: Any) -> int | None:
    text_raw = _coerce_to_text(raw_text)
    if text_raw is None or not str(text_raw).strip():
        return None
    cleaned = str(text_raw).strip().upper()
    # Extract first digit if present
    m = re.search(r"([1-5])", cleaned)
    if m:
        return int(m.group(1))
    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    for roman, val in roman_map.items():
        if re.search(rf"\b{roman}\b", cleaned):
            return val
    return None


def normalize_final_diagnosis_prelim(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    t = text_raw.strip().lower()
    if not t:
        return None
    mapping = {
        "malignancy": "Malignancy",
        "malignant": "Malignancy",
        "cancer": "Malignancy",
        "infectious": "Infectious",
        "infection": "Infectious",
        "granulomatous": "Granulomatous",
        "granuloma": "Granulomatous",
        "non-diagnostic": "Non-diagnostic",
        "nondiagnostic": "Non-diagnostic",
        "non diagnostic": "Non-diagnostic",
        "other": "Other",
    }
    if t in mapping:
        return mapping[t]
    if "granulom" in t:
        return "Granulomatous"
    if "infect" in t:
        return "Infectious"
    if "benign" in t:
        return "Benign"
    if "malig" in t or "carcinoma" in t or "adenocarcinoma" in t:
        return "Malignancy"
    if "non" in t and "diagnostic" in t:
        return "Non-diagnostic"
    return None


def normalize_stent_type(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    t = text_raw.strip().lower()
    if not t:
        return None
    mapping = {
        "silicone-dumon": "Silicone-Dumon",
        "dumon": "Silicone-Dumon",
        "silicone y-stent": "Silicone Y-Stent",
        "silicone-y-stent": "Silicone-Y-Stent",
        "y stent": "Silicone Y-Stent",
        "hybrid": "Hybrid",
        "metallic-covered": "Metallic-Covered",
        "covered metallic": "Metallic-Covered",
        "metallic-uncovered": "Metallic-Uncovered",
        "uncovered metallic": "Metallic-Uncovered",
    }
    if t in mapping:
        return mapping[t]
    if "dumon" in t:
        return "Silicone-Dumon"
    if "y" in t and "stent" in t:
        return "Silicone Y-Stent"
    if "covered" in t and "metal" in t:
        return "Metallic-Covered"
    if "uncovered" in t and "metal" in t:
        return "Metallic-Uncovered"
    if "hybrid" in t:
        return "Hybrid"
    if "metal" in t:
        return "Metallic-Uncovered"
    return "Other" if t else None


def normalize_stent_location(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    t = text_raw.strip().lower()
    if not t:
        return None
    if "trache" in t:
        return "Trachea"
    if "mainstem" in t or "main stem" in t or "left main" in t or "right main" in t:
        return "Mainstem"
    if "lob" in t or "rul" in t or "rml" in t or "rll" in t or "lul" in t or "lll" in t:
        return "Lobar"
    return None


def normalize_stent_deployment_method(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    t = text_raw.strip().lower()
    if not t:
        return None
    if "rigid" in t:
        return "Rigid"
    if "flex" in t or "wire" in t:
        return "Flexible over Wire"
    return None


def validate_station_format(station: str) -> str | None:
    """Validate and normalize a station name to canonical format.

    Returns the canonical station name (uppercase) if valid, None otherwise.
    Valid stations: 2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L
    """
    if not station:
        return None
    cleaned = station.strip().upper()
    # Remove common prefixes
    cleaned = re.sub(r"^(STATION|STN|NODE)[\s:]*", "", cleaned, flags=re.IGNORECASE).strip()

    # Extract the first plausible station token, allowing common sub-station suffixes:
    # - "11Rs"/"11Ri" -> "11R"
    match = re.search(
        r"(?<![0-9A-Z])(2R|2L|4R|4L|7|10R|10L|11R|11L)(?:[SI])?(?![0-9A-Z])",
        cleaned,
    )
    if not match:
        return None

    token = match.group(1).upper()

    # Station 7 is ambiguous; require explicit context unless it is exactly "7".
    if token == "7":
        if cleaned == "7":
            return "7"
        if "SUBCARINAL" in cleaned or "STATION" in station.upper():
            return "7"
        return None

    return token if STATION_PATTERN.match(token) else None


def normalize_ebus_station_rose_result(raw: Any) -> str | None:
    """Normalize a per-station ROSE result to canonical form.

    Schema enum: ["Adequate - malignant", "Adequate - benign lymphocytes",
                  "Adequate - granulomas", "Adequate - other", "Inadequate",
                  "Not performed"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"null", "none", "n/a", "na", ""}:
        return None

    # Direct mapping to schema enum values
    mapping = {
        # Adequate - malignant
        "adequate - malignant": "Adequate - malignant",
        "malignant": "Adequate - malignant",
        "carcinoma": "Adequate - malignant",
        "adenocarcinoma": "Adequate - malignant",
        "squamous": "Adequate - malignant",
        "small cell": "Adequate - malignant",
        "positive for malignancy": "Adequate - malignant",
        "suspicious for malignancy": "Adequate - malignant",
        # Adequate - benign lymphocytes
        "adequate - benign lymphocytes": "Adequate - benign lymphocytes",
        "benign": "Adequate - benign lymphocytes",
        "benign lymphocytes": "Adequate - benign lymphocytes",
        "reactive lymphocytes": "Adequate - benign lymphocytes",
        "lymphocytes": "Adequate - benign lymphocytes",
        "reactive": "Adequate - benign lymphocytes",
        # Adequate - granulomas
        "adequate - granulomas": "Adequate - granulomas",
        "granuloma": "Adequate - granulomas",
        "granulomas": "Adequate - granulomas",
        "granulomatous": "Adequate - granulomas",
        # Adequate - other
        "adequate - other": "Adequate - other",
        "adequate": "Adequate - other",
        "atypical": "Adequate - other",
        "atypical cells": "Adequate - other",
        "atypical cells present": "Adequate - other",
        "atypical lymphoid proliferation": "Adequate - other",
        # Inadequate
        "inadequate": "Inadequate",
        "nondiagnostic": "Inadequate",
        "non-diagnostic": "Inadequate",
        "insufficient": "Inadequate",
        # Not performed
        "not performed": "Not performed",
        "not done": "Not performed",
        "rose not available": "Not performed",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "malignan" in text or "carcinoma" in text or "adenocarcinoma" in text or "positive for" in text:
        return "Adequate - malignant"
    if "granulom" in text:
        return "Adequate - granulomas"
    if "non" in text and "diagnostic" in text:
        return "Inadequate"
    if "insufficient" in text or "inadequate" in text:
        return "Inadequate"
    if "not performed" in text or "not done" in text:
        return "Not performed"
    if "atypical" in text:
        return "Adequate - other"
    if "benign" in text or "reactive" in text or "lymphocyte" in text:
        return "Adequate - benign lymphocytes"

    return None


def derive_global_ebus_rose_result(station_details: list[dict[str, Any]] | None) -> str | None:
    """Derive a global EBUS ROSE result from per-station detail.

    Rules (per specification):
    - If all stations have the same result: return that result
    - If any station is Malignant: return "Malignant"
    - If mixture (e.g., benign + nondiagnostic): return "Mixed (station results)"
    - If no station-level ROSE data: return None

    Args:
        station_details: List of station detail dicts with optional 'rose_result' key

    Returns:
        Derived global ROSE result or None
    """
    if not station_details:
        return None

    # Collect all non-null rose results
    rose_by_station: list[tuple[str, str]] = []
    for detail in station_details:
        station = detail.get("station")
        rose = detail.get("rose_result")
        if station and rose:
            rose_by_station.append((station, rose))

    if not rose_by_station:
        return None

    # Get unique results
    unique_results = set(r for _, r in rose_by_station)

    if len(unique_results) == 1:
        # All stations have the same result
        return rose_by_station[0][1]

    # Multiple different results - check for malignant (highest priority)
    for _, result in rose_by_station:
        if result and "malignant" in result.lower():
            return "Malignant"

    # Build mixed summary: "Mixed (11L Nondiagnostic; 4R Benign)"
    parts = [f"{station} {result}" for station, result in rose_by_station]
    return f"Mixed ({'; '.join(parts)})"


def normalize_ebus_rose_result(raw: Any) -> str | None:
    """Normalize global EBUS ROSE result.

    NOTE: This normalizer should generally NOT be used for global ebus_rose_result
    when per-station ROSE data is available. Use derive_global_ebus_rose_result instead.
    This function normalizes the raw string value if one is provided directly.
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"null", "none", "n/a", "na", ""}:
        return None

    # If it looks like a derived "Mixed" result, preserve it
    if text.startswith("mixed"):
        return raw.strip() if isinstance(raw, str) else str(raw).strip()

    # Otherwise normalize like a single result
    return normalize_ebus_station_rose_result(raw)


def normalize_ebus_needle_gauge(raw: Any) -> int | None:
    """Normalize EBUS needle gauge to schema enum integer value.

    Schema enum: [19, 21, 22, 25]
    Handles strings like "22G", "22 gauge", or just "22".
    """
    if raw is None:
        return None

    # If already an int in valid range, return it
    if isinstance(raw, int) and raw in {19, 21, 22, 25}:
        return raw

    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null", ""}:
        return None

    # Extract number from strings like "22G", "22 gauge", "22-gauge"
    match = re.search(r"(\d+)", text)
    if match:
        gauge = int(match.group(1))
        if gauge in {19, 21, 22, 25}:
            return gauge

    return None


def normalize_ebus_needle_type(raw: Any) -> str | None:
    """Normalize EBUS needle type to schema enum values.

    Schema enum: ["Standard FNA", "Core/ProCore", "Acquire"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null", ""}:
        return None

    mapping = {
        # Standard FNA
        "standard": "Standard FNA",
        "standard fna": "Standard FNA",
        "fna": "Standard FNA",
        "fine needle": "Standard FNA",
        "fine needle aspiration": "Standard FNA",
        # Core/ProCore
        "core": "Core/ProCore",
        "procore": "Core/ProCore",
        "core/procore": "Core/ProCore",
        "pro-core": "Core/ProCore",
        # Acquire
        "acquire": "Acquire",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "procore" in text or "pro-core" in text or "pro core" in text:
        return "Core/ProCore"
    if "core" in text:
        return "Core/ProCore"
    if "acquire" in text:
        return "Acquire"
    if "fna" in text or "fine needle" in text or "standard" in text:
        return "Standard FNA"

    return None


def _parse_size_mm(value: Any) -> float | None:
    """Convert size strings/numbers to millimeters."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _coerce_to_text(value)
    if text is None:
        return None
    lowered = text.lower()
    # Handle two-dimension strings like 1.2 x 0.8 cm (use the smaller dimension)
    dim_match = re.findall(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)(?:\s*(mm|cm))?", lowered)
    if dim_match:
        dim1, dim2, unit = dim_match[0]
        unit = unit or "mm"
        vals = []
        for dim in (dim1, dim2):
            try:
                num = float(dim)
                vals.append(num * 10 if unit.startswith("c") else num)
            except Exception:
                continue
        return min(vals) if vals else None

    match = re.search(r"(\d+(?:\.\d+)?)(?:\s*(mm|cm))?", lowered)
    if match:
        try:
            val = float(match.group(1))
            unit = match.group(2) or "mm"
            return val * 10 if unit.startswith("c") else val
        except Exception:
            return None
    return None


def _normalize_morphology_value(text: str | None, mapping: dict[str, str]) -> str | None:
    if text is None:
        return None
    val = text.strip().lower()
    if val in mapping:
        return mapping[val]
    for key, normalized in mapping.items():
        if key in val:
            return normalized
    return None


def normalize_ebus_stations_detail(raw: Any) -> list[dict[str, Any]] | None:
    """Normalize station-level detail entries and pad morphology fields with nulls.

    Rules:
    - Only include entries with valid station names (2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L)
    - Normalize station names to uppercase canonical form
    - Do not invent morphology data - leave as null if not explicitly in source
    - Normalize ROSE results using normalize_ebus_station_rose_result
    """
    if raw is None:
        return None
    entries: list[Any]
    if isinstance(raw, list):
        entries = raw
    else:
        entries = [raw]

    normalized: list[dict[str, Any]] = []
    shape_map = {
        "oval": "oval",
        "elliptical": "oval",
        "elongated": "oval",
        "round": "round",
        "spherical": "round",
        "irregular": "irregular",
        "lobulated": "irregular",
        "asymmetric": "irregular",
    }
    margin_map = {
        "distinct": "distinct",
        "well-defined": "distinct",
        "well-circumscribed": "distinct",
        "clear margin": "distinct",
        "sharp": "distinct",
        "indistinct": "indistinct",
        "ill-defined": "indistinct",
        "blurred": "indistinct",
        "poorly defined": "indistinct",
        "irregular": "irregular",
        "spiculated": "irregular",
    }
    echo_map = {
        "homogeneous": "homogeneous",
        "uniform": "homogeneous",
        "heterogeneous": "heterogeneous",
        "mixed": "heterogeneous",
        "non-uniform": "heterogeneous",
    }
    appearance_allowed = {"benign", "malignant", "indeterminate"}

    for entry in entries:
        if isinstance(entry, dict):
            # Validate station name - skip entries with invalid stations
            raw_station = entry.get("station")
            station = validate_station_format(str(raw_station)) if raw_station else None
            if not station:
                # Skip entries without valid station
                continue

            size_mm = _parse_size_mm(entry.get("size_mm"))
            passes = entry.get("passes")

            # Use the per-station ROSE normalizer
            rose_result = normalize_ebus_station_rose_result(entry.get("rose_result"))

            # Only normalize morphology if explicit text is provided - don't invent
            shape = _normalize_morphology_value(_coerce_to_text(entry.get("shape")), shape_map)
            margin = _normalize_morphology_value(_coerce_to_text(entry.get("margin")), margin_map)
            echogenicity = _normalize_morphology_value(_coerce_to_text(entry.get("echogenicity")), echo_map)

            chs_raw = entry.get("chs_present")
            if isinstance(chs_raw, str):
                chs_lower = chs_raw.strip().lower()
                if chs_lower in {"true", "yes", "present"}:
                    chs_present = True
                elif chs_lower in {"false", "no", "absent"}:
                    chs_present = False
                else:
                    chs_present = None
            else:
                chs_present = bool(chs_raw) if isinstance(chs_raw, bool) else None

            appearance = _coerce_to_text(entry.get("appearance_category"))
            appearance = appearance.lower() if appearance else None
            if appearance not in appearance_allowed:
                appearance = None

            normalized.append(
                {
                    "station": station,  # Already validated and uppercase
                    "size_mm": size_mm,
                    "passes": passes if passes is None or isinstance(passes, int) else None,
                    "shape": shape,
                    "margin": margin,
                    "echogenicity": echogenicity,
                    "chs_present": chs_present,
                    "appearance_category": appearance,
                    "rose_result": rose_result,
                }
            )
        elif isinstance(entry, str):
            # Attempt minimal parsing from a string like "11L 5.4mm benign"
            station_match = re.search(r"\b(2r|2l|4r|4l|7|10r|10l|11r|11l)\b", entry, re.IGNORECASE)
            if not station_match:
                # Skip entries without valid station pattern
                continue
            station = station_match.group(1).upper()
            size_mm = _parse_size_mm(entry)
            normalized.append(
                {
                    "station": station,
                    "size_mm": size_mm,
                    "passes": None,
                    "shape": None,
                    "margin": None,
                    "echogenicity": None,
                    "chs_present": None,
                    "appearance_category": None,
                    "rose_result": normalize_ebus_station_rose_result(entry),
                }
            )
        else:
            continue

    return normalized or None


def normalize_list_field(raw: Any) -> List[str] | None:
    """Convert comma-separated strings or mixed input to a list of strings."""
    if raw is None:
        return None
    if isinstance(raw, list):
        # Already a list, clean and return
        return [str(item).strip() for item in raw if item and str(item).strip()]
    if isinstance(raw, str):
        # Comma-separated string - split and clean
        if not raw.strip():
            return None
        items = [item.strip() for item in raw.split(",") if item.strip()]
        return items if items else None
    # Try to convert to string and split
    try:
        s = str(raw).strip()
        if not s:
            return None
        items = [item.strip() for item in s.split(",") if item.strip()]
        return items if items else None
    except Exception:
        return None


def normalize_anesthesia_agents(raw: Any) -> List[str] | None:
    """Normalize anesthesia agents list, handling comma-separated strings."""
    result = normalize_list_field(raw)
    if result is None:
        return None
    # Normalize common variations
    normalized = []
    agent_mapping = {
        "propofol": "Propofol",
        "fentanyl": "Fentanyl",
        "midazolam": "Midazolam",
        "rocuronium": "Rocuronium",
        "succinylcholine": "Succinylcholine",
        "remifentanil": "Remifentanil",
        "sevoflurane": "Sevoflurane",
        "isoflurane": "Isoflurane",
        "desflurane": "Desflurane",
    }
    for agent in result:
        agent_lower = agent.lower().strip()
        normalized_agent = agent_mapping.get(agent_lower, agent.strip())
        if normalized_agent and normalized_agent not in normalized:
            normalized.append(normalized_agent)
    return normalized if normalized else None


def normalize_ebus_stations(raw: Any) -> List[str] | None:
    """Normalize EBUS stations list, handling comma-separated strings.

    Only includes stations that match valid EBUS lymph node station patterns:
    2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L

    Invalid or unrecognized station names are filtered out.
    """
    def _extract_station_candidates(text: str) -> list[str]:
        return [
            match.group(0)
            for match in re.finditer(
                r"(?:station|stn|node)?\s*(2R|2L|4R|4L|7|10R|10L|11R|11L)(?:[sSiI])?\b",
                text,
                re.IGNORECASE,
            )
        ]

    candidates: list[str] = []
    if raw is None:
        return None

    if isinstance(raw, list):
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            hits = _extract_station_candidates(text)
            if hits:
                candidates.extend(hits)
            else:
                candidates.append(text)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        hits = _extract_station_candidates(text)
        if hits:
            candidates = hits
        else:
            candidates = [item.strip() for item in text.split(",") if item.strip()]
    else:
        text = str(raw).strip()
        if not text:
            return None
        hits = _extract_station_candidates(text)
        candidates = hits if hits else [text]

    # Clean and validate station format - only include valid stations
    normalized: list[str] = []
    for station in candidates:
        validated = validate_station_format(station)
        if validated and validated not in normalized:
            normalized.append(validated)
    return normalized if normalized else None


def normalize_linear_ebus_stations(raw: Any) -> List[str] | None:
    """Alias for linear stations; keeps normalization consistent with ebus_stations_sampled."""
    return normalize_ebus_stations(raw)


def normalize_nav_sampling_tools(raw: Any) -> List[str] | None:
    """Normalize navigation sampling tools list, handling comma-separated strings."""
    result = normalize_list_field(raw)
    if result is None:
        return None
    # Normalize tool names
    normalized = []
    tool_mapping = {
        "forceps": "Forceps",
        "needle": "Needle",
        "brush": "Brush",
        "cryoprobe": "Cryoprobe",
        "cryo": "Cryoprobe",
        "cryobiopsy": "Cryoprobe",
    }
    for tool in result:
        tool_lower = tool.lower().strip()
        normalized_tool = tool_mapping.get(tool_lower, tool.strip().title())
        if normalized_tool and normalized_tool not in normalized:
            normalized.append(normalized_tool)
    return normalized if normalized else None


def normalize_follow_up_plan(raw: Any) -> List[str] | None:
    """Normalize follow-up plan list, handling comma-separated strings."""
    result = normalize_list_field(raw)
    if result is None:
        return None
    # Clean and return as-is (follow-up plans are free text)
    normalized = [item.strip() for item in result if item.strip()]
    return normalized if normalized else None


def normalize_cao_location(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip()
    if not text:
        return None
    
    # Allowed values: ["None", "Trachea", "Mainstem", "Lobar"]
    # Hierarchy of severity/centrality: Trachea > Mainstem > Lobar > None
    
    # Check for presence of keywords in the text
    lower_text = text.lower()
    if "trachea" in lower_text:
        return "Trachea"
    if "mainstem" in lower_text:
        return "Mainstem"
    if "lobar" in lower_text:
        return "Lobar"
    if "none" in lower_text:
        return "None"
        
    # If exact match with Enum
    if text in {"Trachea", "Mainstem", "Lobar", "None"}:
        return text
        
    return None

def normalize_cao_tumor_location(raw: Any) -> str | None:
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip()
    if not text:
        return None
        
    # Allowed: ["Trachea", "RMS", "LMS", "Bronchus Intermedius", "Lobar", "RUL", "RML", "RLL", "LUL", "LLL", "Mainstem"]
    
    lower_text = text.lower()
    
    # Priority 1: Trachea
    if "trachea" in lower_text:
        return "Trachea"
        
    # Priority 2: Mainstems
    if "rms" in text or "right mainstem" in lower_text:
        return "RMS"
    if "lms" in text or "left mainstem" in lower_text:
        return "LMS"
    if "mainstem" in lower_text: # Generic mainstem if side not specified or both
        return "Mainstem"
        
    # Priority 3: Bronchus Intermedius
    if "bronchus intermedius" in lower_text or "bi" in lower_text.split(): # 'bi' might be risky as substring
        return "Bronchus Intermedius"
        
    # Priority 4: Lobes
    lobes = {
        "rul": "RUL", "right upper": "RUL",
        "rml": "RML", "right middle": "RML",
        "rll": "RLL", "right lower": "RLL",
        "lul": "LUL", "left upper": "LUL",
        "lll": "LLL", "left lower": "LLL",
        "lobar": "Lobar"
    }
    
    for key, val in lobes.items():
        if key in lower_text:
            return val
            
    return None


def normalize_assistant_names(raw: Any) -> List[str] | None:
    """Normalize assistant names, handling both single strings and lists.

    Supports:
    - Single string: "Dr. Smith" -> ["Dr. Smith"]
    - Comma-separated: "Dr. Smith, Dr. Jones" -> ["Dr. Smith", "Dr. Jones"]
    - List: ["Dr. Smith", "Dr. Jones"] -> ["Dr. Smith", "Dr. Jones"]
    - Legacy assistant_name field migration
    """
    if raw is None:
        return None

    # If already a list, clean each item
    if isinstance(raw, list):
        normalized = []
        for item in raw:
            if item is None:
                continue
            name = str(item).strip()
            if name and name.lower() not in {"none", "n/a", "na", "null", ""}:
                normalized.append(name)
        return normalized if normalized else None

    # If it's a string, check for comma separation
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"none", "n/a", "na", "null"}:
            return None

        # Check for common separators
        if "," in text or ";" in text:
            items = re.split(r"[,;]", text)
            normalized = [item.strip() for item in items if item.strip()]
            return normalized if normalized else None

        # Single name
        return [text]

    # Try to convert other types
    try:
        text = str(raw).strip()
        if text and text.lower() not in {"none", "n/a", "na", "null"}:
            return [text]
    except Exception:
        pass

    return None


def normalize_assistant_name_single(raw: Any) -> str | None:
    """Normalize assistant_name (singular) to a string.

    Handles list input by taking the first non-empty entry.
    """
    names = normalize_assistant_names(raw)
    if not names:
        return None
    return names[0]


def normalize_nav_registration_method(raw: Any) -> str | None:
    text = _coerce_to_text(raw)
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in {"auto", "automatic"}:
        return "Automatic"
    if lowered in {"manual"}:
        return "Manual"
    # Title-case fallback for unexpected casing
    if text.strip():
        candidate = text.strip().title()
        if candidate in {"Manual", "Automatic"}:
            return candidate
    return None


def normalize_airway_device_size(raw: Any) -> str | None:
    """Normalize airway device size to string format.

    Handles:
    - Integer input: 12 -> "12"
    - Float input: 7.5 -> "7.5"
    - String input: "7.5 ETT" -> "7.5 ETT"
    """
    if raw is None:
        return None

    # Convert numbers to strings
    if isinstance(raw, (int, float)):
        # Format float nicely (remove trailing .0)
        if isinstance(raw, float) and raw == int(raw):
            return str(int(raw))
        return str(raw)

    # Handle string input
    text = str(raw).strip()
    if not text or text.lower() in {"none", "n/a", "na", "null"}:
        return None

    return text


def normalize_ablation_modality(raw: Any) -> str | None:
    """Normalize ablation modality to match schema enum values.

    Handles:
    - List input: Takes first valid item (e.g., ['Radiofrequency', 'Cryoablation'] -> "Radiofrequency (RFA)")
    - String input: Maps to canonical enum value
    - Common abbreviations: RFA, MWA, cryo, etc.

    Schema enum: ["Microwave (MWA)", "Radiofrequency (RFA)", "Cryoablation", "Laser", "Brachytherapy"]
    """
    text_raw = _coerce_to_text(raw)  # _coerce_to_text already takes first item from list
    if text_raw is None:
        return None

    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null"}:
        return None

    # Mapping from various inputs to canonical enum values
    mapping = {
        # Microwave
        "microwave": "Microwave (MWA)",
        "microwave (mwa)": "Microwave (MWA)",
        "mwa": "Microwave (MWA)",
        "microwave ablation": "Microwave (MWA)",
        # Radiofrequency
        "radiofrequency": "Radiofrequency (RFA)",
        "radiofrequency (rfa)": "Radiofrequency (RFA)",
        "rfa": "Radiofrequency (RFA)",
        "rf ablation": "Radiofrequency (RFA)",
        "radiofrequency ablation": "Radiofrequency (RFA)",
        # Cryoablation
        "cryoablation": "Cryoablation",
        "cryo": "Cryoablation",
        "cryotherapy": "Cryoablation",
        "cryo ablation": "Cryoablation",
        # Laser
        "laser": "Laser",
        "laser ablation": "Laser",
        # Brachytherapy
        "brachytherapy": "Brachytherapy",
        "brachy": "Brachytherapy",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching for partial matches
    if "microwave" in text or "mwa" in text:
        return "Microwave (MWA)"
    if "radiofrequency" in text or "rfa" in text:
        return "Radiofrequency (RFA)"
    if "cryo" in text:
        return "Cryoablation"
    if "laser" in text:
        return "Laser"
    if "brachy" in text:
        return "Brachytherapy"

    return None


def normalize_ventilation_mode(raw: Any) -> str | None:
    """Normalize ventilation mode to schema enum values.

    Schema enum: ["Spontaneous", "Controlled Mechanical Ventilation", "Jet Ventilation"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null"}:
        return None

    mapping = {
        # Controlled variants
        "volume control": "Controlled Mechanical Ventilation",
        "pressure control": "Controlled Mechanical Ventilation",
        "controlled mechanical ventilation": "Controlled Mechanical Ventilation",
        "controlled": "Controlled Mechanical Ventilation",
        "mechanical ventilation": "Controlled Mechanical Ventilation",
        "ippv": "Controlled Mechanical Ventilation",
        "cmv": "Controlled Mechanical Ventilation",
        # Spontaneous variants
        "spontaneous": "Spontaneous",
        "spontaneous ventilation": "Spontaneous",
        "spontaneous ventilation with supplemental oxygen": "Spontaneous",
        "spontaneous ventilation on supplemental oxygen": "Spontaneous",
        "spontaneous ventilation with pressure support": "Spontaneous",
        "pressure support": "Spontaneous",
        "cpap": "Spontaneous",
        # Jet variants
        "jet ventilation": "Jet Ventilation",
        "jet": "Jet Ventilation",
        "hfjv": "Jet Ventilation",
        "high frequency jet ventilation": "Jet Ventilation",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "jet" in text:
        return "Jet Ventilation"
    if "spontaneous" in text or "pressure support" in text:
        return "Spontaneous"
    if "controlled" in text or "volume control" in text or "pressure control" in text or "mechanical" in text:
        return "Controlled Mechanical Ventilation"

    return None


def normalize_procedure_setting(raw: Any) -> str | None:
    """Normalize procedure setting to schema enum values.

    Schema enum: ["Bronchoscopy Suite", "Operating Room", "ICU", "Bedside", "Office/Clinic"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null"}:
        return None

    mapping = {
        "bronchoscopy suite": "Bronchoscopy Suite",
        "bronchoscopy room": "Bronchoscopy Suite",
        "bronch suite": "Bronchoscopy Suite",
        "hybrid bronchoscopy suite": "Bronchoscopy Suite",
        "procedure room": "Bronchoscopy Suite",
        "endoscopy suite": "Bronchoscopy Suite",
        "operating room": "Operating Room",
        "or": "Operating Room",
        "main or": "Operating Room",
        "surgery": "Operating Room",
        "icu": "ICU",
        "intensive care unit": "ICU",
        "micu": "ICU",
        "sicu": "ICU",
        "ccu": "ICU",
        "bedside": "Bedside",
        "at bedside": "Bedside",
        "patient room": "Bedside",
        "office/clinic": "Office/Clinic",
        "office": "Office/Clinic",
        "clinic": "Office/Clinic",
        "outpatient clinic": "Office/Clinic",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "bronchoscopy" in text or "bronch" in text or "endoscopy" in text:
        return "Bronchoscopy Suite"
    if "operating room" in text or text == "or":
        return "Operating Room"
    if "icu" in text or "intensive care" in text:
        return "ICU"
    if "bedside" in text or "patient room" in text:
        return "Bedside"
    if "office" in text or "clinic" in text:
        return "Office/Clinic"

    return None


def normalize_bronch_location_lobe(raw: Any) -> str | None:
    """Normalize bronchoscopy location lobe to schema enum values.

    Schema enum: ["RUL", "RML", "RLL", "LUL", "LLL", "Central"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null"}:
        return None

    mapping = {
        # Right upper lobe
        "rul": "RUL",
        "right upper lobe": "RUL",
        "right upper": "RUL",
        # Right middle lobe
        "rml": "RML",
        "right middle lobe": "RML",
        "right middle": "RML",
        # Right lower lobe
        "rll": "RLL",
        "right lower lobe": "RLL",
        "right lower": "RLL",
        # Left upper lobe
        "lul": "LUL",
        "left upper lobe": "LUL",
        "left upper": "LUL",
        # Left lower lobe
        "lll": "LLL",
        "left lower lobe": "LLL",
        "left lower": "LLL",
        # Central airways
        "central": "Central",
        "central airways": "Central",
        "trachea": "Central",
        "carina": "Central",
        "mainstem": "Central",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "right upper" in text or text == "rul":
        return "RUL"
    if "right middle" in text or text == "rml":
        return "RML"
    if "right lower" in text or text == "rll":
        return "RLL"
    if "left upper" in text or text == "lul":
        return "LUL"
    if "left lower" in text or text == "lll":
        return "LLL"
    if "central" in text or "trachea" in text or "carina" in text or "mainstem" in text:
        return "Central"

    return None


def normalize_cpt_codes(raw: Any) -> List[str] | None:
    """Normalize CPT codes to a list of string CPT code values only.

    Per specification:
    - Always returns array of strings containing only the CPT code itself
    - Example: ["31652"] not ["31652 convex probe endobronchial..."]
    - Extracts just the numeric code from longer descriptive strings
    - Modifiers if present are kept as separate entries
    """
    if raw is None:
        return None

    # CPT code pattern: 5-digit number (may have optional modifier like -26)
    cpt_pattern = re.compile(r"\b(\d{5})(?:-(\d{2}))?\b")

    def extract_cpt_codes(text: str) -> List[str]:
        """Extract CPT code(s) from a string, handling descriptive text."""
        codes = []
        # First try to extract CPT codes with pattern
        for match in cpt_pattern.finditer(text):
            code = match.group(1)
            modifier = match.group(2)
            if modifier:
                codes.append(f"{code}-{modifier}")
            else:
                codes.append(code)
        # If no pattern match but text is purely numeric (5 digits), use it
        if not codes:
            clean = text.strip()
            if re.fullmatch(r"\d{5}", clean):
                codes.append(clean)
        return codes

    normalized: List[str] = []

    if isinstance(raw, list):
        for item in raw:
            if item is None:
                continue
            item_str = str(item).strip()
            if not item_str:
                continue
            codes = extract_cpt_codes(item_str)
            for code in codes:
                if code not in normalized:
                    normalized.append(code)
    elif isinstance(raw, str):
        if not raw.strip():
            return None
        # Handle comma-separated strings
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            codes = extract_cpt_codes(part)
            for code in codes:
                if code not in normalized:
                    normalized.append(code)
    else:
        # Try to convert other types
        try:
            s = str(raw).strip()
            if s:
                codes = extract_cpt_codes(s)
                for code in codes:
                    if code not in normalized:
                        normalized.append(code)
        except Exception:
            pass

    return normalized if normalized else None


def normalize_attending_name(raw: Any) -> str | None:
    """Normalize attending physician name, removing role/specialty from the name.

    Per specification:
    - attending_name should contain only the clinician's name and credentials
    - Example: "Russell Miller MD" not "Russell Miller MD, Pulmonologist"
    - Role/specialty should go in provider_role field instead
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip()
    if not text or text.lower() in {"none", "n/a", "na", "null", ""}:
        return None

    # Common roles/specialties to strip from the name
    role_patterns = [
        r",?\s*(?:Interventional\s+)?Pulmonologist\s*$",
        r",?\s*(?:Interventional\s+)?Pulmonology\s*$",
        r",?\s*Thoracic\s+Surgeon\s*$",
        r",?\s*Thoracic\s+Surgery\s*$",
        r",?\s*Pulmonary(?:/Critical\s+Care)?\s*$",
        r",?\s*Critical\s+Care\s*$",
        r",?\s*Fellow\s*$",
        r",?\s*Attending\s*$",
        r",?\s*Physician\s*$",
        r",?\s*Surgeon\s*$",
    ]

    result = text
    for pattern in role_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()

    # Clean up any trailing commas or whitespace
    result = result.rstrip(",").strip()

    # Guardrail: ignore common header words mistakenly captured as "names"
    banned = {"PARTICIPATION", "ATTENDING", "PROCEDURE", "NOTE", "SIGNED"}
    token = result.lstrip("*").strip().upper()
    if token in banned:
        return None

    return result if result else None


def normalize_provider_role(raw: Any) -> str | None:
    """Normalize provider role/specialty to canonical form.

    Common values: Pulmonologist, Interventional Pulmonologist, Thoracic Surgeon, Fellow
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null", ""}:
        return None

    mapping = {
        "pulmonologist": "Pulmonologist",
        "pulmonology": "Pulmonologist",
        "interventional pulmonologist": "Interventional Pulmonologist",
        "interventional pulmonology": "Interventional Pulmonologist",
        "ip": "Interventional Pulmonologist",
        "thoracic surgeon": "Thoracic Surgeon",
        "thoracic surgery": "Thoracic Surgeon",
        "fellow": "Fellow",
        "pulmonary fellow": "Fellow",
        "ip fellow": "Fellow",
        "attending": "Attending",
        "attending physician": "Attending",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "interventional" in text and "pulmon" in text:
        return "Interventional Pulmonologist"
    if "pulmon" in text:
        return "Pulmonologist"
    if "thoracic" in text:
        return "Thoracic Surgeon"
    if "fellow" in text:
        return "Fellow"

    return text_raw.strip()  # Return original if no match


def normalize_immediate_complications(raw: Any) -> str | None:
    """Normalize immediate complications field to standardized vocabulary.

    Per specification:
    - Normalize variations like "No immediate complications" to "None"
    - For complications that did occur, use concise standard terms
    - Standardize vocabulary for consistency
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip()
    if not text:
        return None

    lowered = text.lower()

    # Normalize "no complications" variations to "None"
    no_complication_patterns = [
        r"^no\s+(?:immediate\s+)?complications?\.?$",
        r"^none\s+(?:noted|observed|reported)?\.?$",
        r"^no\s+(?:immediate\s+)?adverse\s+events?\.?$",
        r"^nil\.?$",
        r"^n/a\.?$",
        r"^na\.?$",
        r"^none\.?$",
    ]
    for pattern in no_complication_patterns:
        if re.match(pattern, lowered):
            return "None"

    # Standard complication mappings
    complication_mapping = {
        "bleeding": "Bleeding",
        "hemorrhage": "Bleeding",
        "pneumothorax": "Pneumothorax",
        "ptx": "Pneumothorax",
        "hypoxia": "Hypoxia",
        "hypoxemia": "Hypoxia",
        "desaturation": "Hypoxia",
        "respiratory failure": "Respiratory Failure",
        "bronchospasm": "Bronchospasm",
        "wheezing": "Bronchospasm",
        "arrhythmia": "Arrhythmia",
        "fever": "Fever",
        "infection": "Infection",
    }

    # Check for specific complications
    for key, value in complication_mapping.items():
        if key in lowered:
            return value

    # If none of the above, return cleaned original
    return text


def normalize_radiographic_findings(raw: Any) -> str | None:
    """Normalize radiographic findings, excluding non-imaging content.

    Per specification:
    - Only include actual radiographic/imaging descriptions (CT, PET, CXR findings)
    - Exclude sampling criteria (e.g., "nodes >= 5mm were sampled")
    - Exclude procedural details (needle passes, ROSE results, etc.)
    - Return null if no imaging description is present
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip()
    if not text or text.lower() in {"none", "n/a", "na", "null", ""}:
        return None

    lowered = text.lower()

    # Patterns that indicate this is NOT a radiographic finding (should be excluded)
    exclusion_patterns = [
        r"sampling\s+criteria",
        r"nodes?\s*(?:>=?|≥|greater\s+than)\s*\d+\s*mm\s+(?:were\s+)?sampled",
        r"short\s+axis\s+(?:diameter\s+)?(?:criteria|threshold)",
        r"(?:were\s+)?met\s+(?:for\s+)?sampling",
        r"needle\s+pass",
        r"rose\s+(?:result|showed|demonstrates)",
        r"specimen\s+(?:sent|obtained)",
        r"cytolog",
        r"patholog",
        r"biopsy\s+(?:result|showed)",
    ]

    for pattern in exclusion_patterns:
        if re.search(pattern, lowered):
            return None

    # Patterns that indicate valid radiographic findings
    valid_imaging_patterns = [
        r"(?:ct|pet|cxr|x-ray|chest\s+x-ray|imaging|scan)",
        r"(?:nodule|mass|lesion|opacity|consolidation)",
        r"(?:hilar|mediastinal)\s+(?:adenopathy|lymphadenopathy)",
        r"(?:suv|standardized\s+uptake)",
        r"(?:avid|hypermetabolic)",
        r"(?:effusion|collapse|atelectasis)",
        r"(?:lobe|segment|upper|lower|middle)",
    ]

    has_imaging_content = any(re.search(pat, lowered) for pat in valid_imaging_patterns)

    if has_imaging_content:
        return text

    # If no clear imaging content, return None rather than procedural text
    return None


def apply_cross_field_consistency(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply cross-field consistency checks and corrections.

    Per specification, ensures fields are consistent with each other:
    - pneumothorax_intervention is null when pneumothorax=false
    - bronch_tbbx_tool is null when bronch_num_tbbx is null/0
    - ebus_stations_detail stations match ebus_stations_sampled
    - etc.
    """
    result = dict(data)

    # Pneumothorax consistency
    if result.get("pneumothorax") is False:
        result["pneumothorax_intervention"] = None

    # Transbronchial biopsy consistency
    num_tbbx = result.get("bronch_num_tbbx")
    if num_tbbx is None or num_tbbx == 0:
        result["bronch_tbbx_tool"] = None

    # EBUS station consistency
    # Ensure ebus_stations_sampled and ebus_stations_detail are consistent
    station_detail = result.get("ebus_stations_detail") or []
    stations_sampled = result.get("ebus_stations_sampled") or []

    if station_detail:
        # Get stations from detail
        detail_stations = {d.get("station") for d in station_detail if d.get("station")}
        # Merge with sampled
        all_stations = set(stations_sampled) | detail_stations
        if all_stations:
            result["ebus_stations_sampled"] = sort_ebus_stations(all_stations)
            # Also update linear_ebus_stations to match
            result["linear_ebus_stations"] = sort_ebus_stations(all_stations)

    # Derive global ROSE result from per-station data if available
    if station_detail and not result.get("ebus_rose_result"):
        derived_rose = derive_global_ebus_rose_result(station_detail)
        if derived_rose:
            result["ebus_rose_result"] = derived_rose

    families = set(result.get("procedure_families") or [])

    # Pleural consistency - if no pleural procedure, clear pleural fields
    if not result.get("pleural_procedure_type") and not ({"PLEURAL", "THORACOSCOPY"} & families):
        pleural_fields = [
            "pleural_side", "pleural_volume_drained_ml", "pleural_fluid_appearance",
            "pleural_guidance", "pleural_intercostal_space", "pleural_catheter_type",
            "pleural_opening_pressure_measured", "pleural_opening_pressure_cmh2o",
        ]
        for field in pleural_fields:
            if field in result:
                result[field] = None

    # CAO/stent consistency - clear state if no CAO/STENT procedures this run
    if "CAO" not in families:
        for field in ["cao_location", "cao_primary_modality", "cao_tumor_location",
                      "cao_obstruction_pre_pct", "cao_obstruction_post_pct", "cao_interventions"]:
            if field in result:
                result[field] = None
    if not ({"STENT", "CAO"} & families):
        for field in ["stent_type", "stent_location", "stent_size", "stent_action", "airway_stent_removal"]:
            if field in result:
                result[field] = None

    return result


def normalize_navigation_platform(raw: Any) -> str | None:
    """Normalize navigation platform to schema enum values.

    Schema enum: ["Ion", "Monarch", "Galaxy", "superDimension", "ILLUMISITE",
                  "SPiN", "LungVision", "ARCHIMEDES", "None"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"n/a", "na", "null", ""}:
        return None
    if text == "none":
        return "None"

    mapping = {
        # Ion (Intuitive)
        "ion": "Ion",
        "ion intuitive": "Ion",
        "intuitive ion": "Ion",
        # Monarch (Ethicon/Auris)
        "monarch": "Monarch",
        "auris": "Monarch",
        "ethicon monarch": "Monarch",
        # Galaxy (Noah Medical)
        "galaxy": "Galaxy",
        "noah": "Galaxy",
        "noah medical": "Galaxy",
        # superDimension (Medtronic)
        "superdimension": "superDimension",
        "super dimension": "superDimension",
        "sd": "superDimension",
        "medtronic superdimension": "superDimension",
        # EMN (electromagnetic navigation) - generic term, typically superDimension
        "emn": "superDimension",
        "electromagnetic navigation": "superDimension",
        "electromagnetic": "superDimension",
        # ILLUMISITE (Medtronic)
        "illumisite": "ILLUMISITE",
        "illuminsite": "ILLUMISITE",  # common typo
        # SPiN (Veran)
        "spin": "SPiN",
        "veran": "SPiN",
        "veran spin": "SPiN",
        # LungVision (Body Vision)
        "lungvision": "LungVision",
        "lung vision": "LungVision",
        "body vision": "LungVision",
        "bodyvision": "LungVision",
        # ARCHIMEDES (Broncus)
        "archimedes": "ARCHIMEDES",
        "broncus": "ARCHIMEDES",
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "ion" in text and "intuitive" in text:
        return "Ion"
    if "monarch" in text or "auris" in text:
        return "Monarch"
    if "galaxy" in text or "noah" in text:
        return "Galaxy"
    if "superdimension" in text or "super dimension" in text:
        return "superDimension"
    if "illumisite" in text:
        return "ILLUMISITE"
    if "spin" in text or "veran" in text:
        return "SPiN"
    if "lungvision" in text or "lung vision" in text or "body vision" in text:
        return "LungVision"
    if "archimedes" in text or "broncus" in text:
        return "ARCHIMEDES"

    # Check if it's already a valid enum value (case-insensitive)
    allowed = {"Ion", "Monarch", "Galaxy", "superDimension", "ILLUMISITE",
               "SPiN", "LungVision", "ARCHIMEDES", "None"}
    for val in allowed:
        if text == val.lower():
            return val

    return None


def normalize_valve_type(raw: Any) -> str | None:
    """Normalize BLVR valve type to schema enum values.

    Schema enum: ["Zephyr (Pulmonx)", "Spiration (Olympus)"]
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null", ""}:
        return None

    mapping = {
        # Zephyr (Pulmonx)
        "zephyr": "Zephyr (Pulmonx)",
        "zephyr (pulmonx)": "Zephyr (Pulmonx)",
        "pulmonx": "Zephyr (Pulmonx)",
        "pulmonx zephyr": "Zephyr (Pulmonx)",
        "ebv": "Zephyr (Pulmonx)",  # Endobronchial Valve = Zephyr
        "endobronchial valve": "Zephyr (Pulmonx)",
        # Spiration (Olympus)
        "spiration": "Spiration (Olympus)",
        "spiration (olympus)": "Spiration (Olympus)",
        "olympus": "Spiration (Olympus)",
        "olympus spiration": "Spiration (Olympus)",
        "svs": "Spiration (Olympus)",  # Spiration Valve System
        "spiration valve": "Spiration (Olympus)",
        "ibv": "Spiration (Olympus)",  # Intrabronchial Valve = Spiration
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "zephyr" in text or "pulmonx" in text:
        return "Zephyr (Pulmonx)"
    if "spiration" in text or "olympus" in text:
        return "Spiration (Olympus)"

    return None


def normalize_assistant_role(raw: Any) -> str | None:
    """Normalize assistant role to schema enum values.

    Schema enum: ['RN', 'RT', 'Tech', 'Resident', 'PA', 'NP', 'Medical Student']
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null", ""}:
        return None

    mapping = {
        "rn": "RN",
        "registered nurse": "RN",
        "nurse": "RN",
        "rt": "RT",
        "respiratory therapist": "RT",
        "respiratory tech": "RT",
        "tech": "Tech",
        "technician": "Tech",
        "resident": "Resident",
        "res": "Resident",
        "pa": "PA",
        "physician assistant": "PA",
        "physician's assistant": "PA",
        "np": "NP",
        "nurse practitioner": "NP",
        "medical student": "Medical Student",
        "med student": "Medical Student",
        "student": "Medical Student",
        "fellow": "Resident",  # Map fellow to Resident as closest match
    }

    if text in mapping:
        return mapping[text]

    # Fuzzy matching
    if "fellow" in text:
        return "Resident"
    if "nurse" in text and "practitioner" in text:
        return "NP"
    if "nurse" in text:
        return "RN"
    if "respiratory" in text:
        return "RT"
    if "technician" in text or "tech" in text:
        return "Tech"
    if "resident" in text:
        return "Resident"
    if "physician assistant" in text or "pa" in text:
        return "PA"
    if "student" in text:
        return "Medical Student"

    return None


def normalize_bronchoscope_diameter(raw: Any) -> float | None:
    """Extract numeric value from bronchoscope diameter strings like '12 mm'.

    Schema expects a float number.
    """
    if raw is None:
        return None
    
    # If already a number, return it
    if isinstance(raw, (int, float)):
        return float(raw)
    
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    
    # Extract number from strings like "12 mm", "12mm", "12.5 mm"
    match = re.search(r"(\d+(?:\.\d+)?)", str(text_raw))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None


def normalize_complication_list(raw: Any) -> List[str] | None:
    """Normalize complication list to schema enum values.

    Schema enum: ['Bleeding - Mild', 'Bleeding - Moderate', 'Bleeding - Severe',
                  'Pneumothorax', 'Hypoxia', 'Respiratory failure', 'Hypotension',
                  'Arrhythmia', 'Bronchospasm', 'Laryngospasm', 'Aspiration',
                  'Infection', 'Air embolism', 'Cardiac arrest', 'Death', 'Other']
    """
    if raw is None:
        return None
    
    # Convert to list if needed
    if isinstance(raw, str):
        # Split by comma or semicolon
        items = re.split(r"[,;]", raw)
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    
    normalized = []
    for item in items:
        if item is None:
            continue
        
        text = str(item).strip().lower()
        if not text or text in {"none", "n/a", "na", "null", ""}:
            continue
        
        # Mapping for complications
        mapping = {
            "bleeding - mild": "Bleeding - Mild",
            "mild bleeding": "Bleeding - Mild",
            "bleeding - moderate": "Bleeding - Moderate",
            "moderate bleeding": "Bleeding - Moderate",
            "bleeding - severe": "Bleeding - Severe",
            "severe bleeding": "Bleeding - Severe",
            "bleeding": "Bleeding - Mild",  # Default to Mild if severity not specified
            "pneumothorax": "Pneumothorax",
            "ptx": "Pneumothorax",
            "hypoxia": "Hypoxia",
            "hypoxemia": "Hypoxia",
            "respiratory failure": "Respiratory failure",
            "respiratory distress": "Respiratory failure",
            "hypotension": "Hypotension",
            "arrhythmia": "Arrhythmia",
            "bronchospasm": "Bronchospasm",
            "laryngospasm": "Laryngospasm",
            "aspiration": "Aspiration",
            "infection": "Infection",
            "air embolism": "Air embolism",
            "cardiac arrest": "Cardiac arrest",
            "death": "Death",
            "other": "Other",
        }
        
        if text in mapping:
            normalized.append(mapping[text])
        elif "bleeding" in text:
            # Try to infer severity from context
            if "mild" in text or "minor" in text:
                normalized.append("Bleeding - Mild")
            elif "severe" in text or "major" in text or "significant" in text:
                normalized.append("Bleeding - Severe")
            elif "moderate" in text:
                normalized.append("Bleeding - Moderate")
            else:
                normalized.append("Bleeding - Mild")  # Default
        elif "pneumothorax" in text or "ptx" in text:
            normalized.append("Pneumothorax")
        elif "hypoxia" in text or "hypoxemia" in text:
            normalized.append("Hypoxia")
        elif "respiratory failure" in text or "respiratory distress" in text:
            normalized.append("Respiratory failure")
        else:
            # Try title case for other values
            title = text.title()
            allowed = [
                "Pneumothorax", "Hypoxia", "Respiratory failure", "Hypotension",
                "Arrhythmia", "Bronchospasm", "Laryngospasm", "Aspiration",
                "Infection", "Air embolism", "Cardiac arrest", "Death", "Other"
            ]
            if title in allowed and title not in normalized:
                normalized.append(title)
    
    return normalized if normalized else None


def normalize_radial_ebus_probe_position(raw: Any) -> str | None:
    """Normalize radial EBUS probe position to schema enum values.

    Schema enum: ['Concentric', 'Eccentric', 'Adjacent', 'Not visualized']
    """
    text_raw = _coerce_to_text(raw)
    if text_raw is None:
        return None
    text = text_raw.strip().lower()
    if not text or text in {"none", "n/a", "na", "null", ""}:
        return None
    
    # If it's a descriptive string that doesn't match, return None
    # Common non-matching descriptions should be filtered out
    if "aerated lung" in text or "lung" in text and "on" in text:
        return None  # This is a description, not a position
    
    mapping = {
        "concentric": "Concentric",
        "eccentric": "Eccentric",
        "adjacent": "Adjacent",
        "not visualized": "Not visualized",
        "not visualised": "Not visualized",
        "not seen": "Not visualized",
    }
    
    if text in mapping:
        return mapping[text]
    
    # Fuzzy matching
    if "concentric" in text:
        return "Concentric"
    if "eccentric" in text:
        return "Eccentric"
    if "adjacent" in text:
        return "Adjacent"
    if "not" in text and ("visual" in text or "seen" in text):
        return "Not visualized"
    
    return None


def process_granular_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process granular per-site registry data and derive aggregate fields.

    This function:
    1. Validates EBUS consistency between granular detail and sampled stations
    2. Derives aggregate fields from granular arrays when present
    3. Merges derived values into aggregate fields if not already set

    Returns:
        Updated data dict with derived aggregate fields and validation warnings
    """
    from app.registry.schema_granular import (
        EnhancedRegistryGranularData,
        validate_ebus_consistency,
        derive_aggregate_fields,
    )

    result = dict(data)
    granular_data = result.get("granular_data")

    if granular_data is None:
        return result

    # Parse granular_data if it's a dict (from JSON) rather than already a model
    if isinstance(granular_data, dict):
        try:
            granular = EnhancedRegistryGranularData(**granular_data)
        except Exception as e:
            logger.warning(f"Failed to parse granular_data: {e}")
            result["granular_validation_warnings"] = [f"Failed to parse granular_data: {str(e)}"]
            return result
    else:
        granular = granular_data

    warnings: List[str] = []

    # Validate EBUS consistency if we have both detail and sampled stations
    stations_sampled = result.get("ebus_stations_sampled") or result.get("linear_ebus_stations")
    if granular.linear_ebus_stations_detail and stations_sampled:
        consistency_errors = validate_ebus_consistency(
            granular.linear_ebus_stations_detail,
            stations_sampled
        )
        if consistency_errors:
            warnings.extend(consistency_errors)
            logger.warning(f"EBUS consistency issues: {consistency_errors}")

    # Derive aggregate fields from granular data
    derived = derive_aggregate_fields(granular)

    # Merge derived values into aggregate fields if not already set
    field_mapping = {
        "linear_ebus_stations": ["linear_ebus_stations", "ebus_stations_sampled"],
        "ebus_total_passes": ["ebus_total_passes"],
        "ebus_rose_result": ["ebus_rose_result"],
        "nav_targets_count": ["nav_targets_count"],
        "nav_til_confirmed_count": ["nav_til_confirmed_count"],
        "blvr_number_of_valves": ["blvr_number_of_valves"],
        "blvr_target_lobe": ["blvr_target_lobe"],
        "cryo_specimens_count": ["cryo_specimens_count"],
    }

    for derived_key, target_fields in field_mapping.items():
        derived_value = derived.get(derived_key)
        if derived_value is not None:
            for target_field in target_fields:
                current_value = result.get(target_field)
                # Only set if current value is None, empty list, or empty dict
                if current_value in (None, [], {}):
                    result[target_field] = derived_value

    # Store validation warnings
    if warnings:
        existing_warnings = result.get("granular_validation_warnings", [])
        result["granular_validation_warnings"] = existing_warnings + warnings

    return result


POSTPROCESSORS: Dict[str, Callable[[Any], Any]] = {
    "sedation_type": normalize_sedation_type,
    "airway_type": normalize_airway_type,
    "pleural_guidance": map_pleural_guidance,
    "pleural_procedure_type": normalize_pleural_procedure,
    "pleural_side": normalize_pleural_side,
    "pleural_intercostal_space": normalize_pleural_intercostal_space,
    "patient_mrn": postprocess_patient_mrn,
    "procedure_date": normalize_procedure_date,
    "disposition": normalize_disposition,
    "asa_class": postprocess_asa_class,
    "final_diagnosis_prelim": normalize_final_diagnosis_prelim,
    "stent_type": normalize_stent_type,
    "stent_location": normalize_stent_location,
    "stent_deployment_method": normalize_stent_deployment_method,
    "ebus_rose_result": normalize_ebus_rose_result,
    "ebus_needle_gauge": normalize_ebus_needle_gauge,
    "ebus_needle_type": normalize_ebus_needle_type,
    "ebus_stations_detail": normalize_ebus_stations_detail,
    "ebus_elastography_pattern": normalize_elastography_pattern,
    # List field normalizers - convert comma-separated strings to lists
    "anesthesia_agents": normalize_anesthesia_agents,
    "ebus_stations_sampled": normalize_ebus_stations,
    "linear_ebus_stations": normalize_linear_ebus_stations,
    "nav_sampling_tools": normalize_nav_sampling_tools,
    "nav_registration_method": normalize_nav_registration_method,
    "follow_up_plan": normalize_follow_up_plan,
    "pleural_thoracoscopy_findings": normalize_list_field,
    "fb_tool_used": normalize_list_field,
    "bronch_specimen_tests": normalize_list_field,
    "cao_location": normalize_cao_location,
    "cao_tumor_location": normalize_cao_tumor_location,
    "cpt_codes": normalize_cpt_codes,
    # Ablation modality - converts list to single value and normalizes to enum
    "ablation_modality": normalize_ablation_modality,
    # Airway device size - converts int/float to string
    "airway_device_size": normalize_airway_device_size,
    # Multi-assistant support - converts single name or comma-separated to list
    "assistant_names": normalize_assistant_names,
    # Also handle legacy field name migration
    "assistant_name": normalize_assistant_name_single,
    # New normalizers for validation consistency
    "ventilation_mode": normalize_ventilation_mode,
    "procedure_setting": normalize_procedure_setting,
    "bronch_location_lobe": normalize_bronch_location_lobe,
    # Provider and role normalization
    "attending_name": normalize_attending_name,
    "provider_role": normalize_provider_role,
    # Complication and safety field normalization
    "bronch_immediate_complications": normalize_immediate_complications,
    # Radiographic findings - exclude non-imaging content
    "radiographic_findings": normalize_radiographic_findings,
    # Navigation platform - canonical internal values (both field names)
    "navigation_platform": normalize_navigation_platform,
    "nav_platform": normalize_navigation_platform,
    # BLVR valve type - both field names for compatibility
    "valve_type": normalize_valve_type,
    "blvr_valve_type": normalize_valve_type,
    # Assistant role normalization
    "assistant_role": normalize_assistant_role,
    # Bronchoscope diameter - extract number from string
    "bronchoscope_outer_diameter_mm": normalize_bronchoscope_diameter,
    # Complication list normalization
    "complication_list": normalize_complication_list,
    "complications": normalize_complication_list,
    # Radial EBUS probe position
    "radial_ebus_probe_position": normalize_radial_ebus_probe_position,
}


_EBUS_SAMPLING_INDICATORS_RE = re.compile(
    r"\b(?:needle|tbna|fna|biops(?:y|ied|ies)|sampl(?:e|ed|es)|pass(?:es)?|aspirat(?:e|ed|ion)|core|forceps)\b",
    re.IGNORECASE,
)

_EBUS_SAMPLING_CRITERIA_RE = re.compile(r"\bsampling\s+criteria\b", re.IGNORECASE)
_EBUS_SIZE_MM_RE = re.compile(r"\b\d+(?:\.\d+)?\s*mm\b", re.IGNORECASE)

_EBUS_EXPLICIT_NEGATION_PHRASES_RE = re.compile(
    r"\b(?:"
    r"not\s+biops(?:y|ied)\b"
    r"|not\s+sampled\b"
    r"|not\s+perform(?:ed|ing)?\b"
    r"|no\s+biops(?:y|ies)\b"
    r"|no\s+sampling\b"
    r"|no\s+needle\b"
    r"|decision\s+to\s+not\b"
    r"|decid(?:ed|ing)\s+to\s+not\b"
    r"|deferred\b"
    r"|inspected\s+only\b"
    r"|sized\s+only\b"
    r"|viewed\s+only\b"
    r")",
    re.IGNORECASE,
)

_EBUS_BENIGN_ONLY_PHRASES_RE = re.compile(
    r"\b(?:"
    r"benign\s+ultrasound\b"
    r"|benign\s+characteristics\b"
    r"|ultrasound\s+characteristics\b"
    r"|sonographically\s+benign\b"
    r")",
    re.IGNORECASE,
)

_EBUS_MEASURE_ONLY_RE = re.compile(
    r"\bmeasur(?:e|ed|ement|ing)\b",
    re.IGNORECASE,
)


def _ebus_station_pattern(station: str) -> re.Pattern[str] | None:
    token = (station or "").strip().upper()
    if not token:
        return None
    if token.isdigit():
        # Station "7" is highly collision-prone; require "station 7" or a station-style line prefix.
        return re.compile(rf"(?i)(?:\bstation\s*{re.escape(token)}\b|\b{re.escape(token)}\b\s*[:\-])")
    # Substations are frequently documented as 11Rs/11Ri and should match canonical
    # event station keys (11R/11L) during reconciliation.
    token_pat = re.escape(token)
    if re.fullmatch(r"\d{1,2}[RL]", token):
        token_pat = f"{token_pat}(?:S|I)?"
    return re.compile(rf"(?i)(?:\bstation\s*{token_pat}\b|\b{token_pat}\b)")


def _station_has_sampling_negation(full_text: str, station: str) -> bool:
    pattern = _ebus_station_pattern(station)
    if pattern is None:
        return False

    text = full_text or ""

    def _station_has_sampling_positive() -> bool:
        for line in text.splitlines():
            if pattern.search(line) and _EBUS_SAMPLING_INDICATORS_RE.search(line):
                return True
        for match in pattern.finditer(text):
            end_of_line = text.find("\n", match.start())
            if end_of_line == -1:
                end_of_line = len(text)
            snippet = text[match.start():end_of_line]
            if _EBUS_SAMPLING_INDICATORS_RE.search(snippet):
                return True
        return False

    station_has_sampling_positive = _station_has_sampling_positive()

    # Prefer line-local evidence to avoid leaking negations from adjacent stations.
    for line in text.splitlines():
        if not pattern.search(line):
            continue
        if _EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(line):
            return True
        if _EBUS_MEASURE_ONLY_RE.search(line) and not _EBUS_SAMPLING_INDICATORS_RE.search(line):
            return True
        if _EBUS_BENIGN_ONLY_PHRASES_RE.search(line) and not station_has_sampling_positive:
            return True

    # Fallback: station mention context up to end-of-line.
    for match in pattern.finditer(text):
        end_of_line = text.find("\n", match.start())
        if end_of_line == -1:
            end_of_line = len(text)
        snippet = text[match.start():end_of_line]
        if _EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(snippet):
            return True
        if _EBUS_MEASURE_ONLY_RE.search(snippet) and not _EBUS_SAMPLING_INDICATORS_RE.search(snippet):
            return True
        if _EBUS_BENIGN_ONLY_PHRASES_RE.search(snippet) and not station_has_sampling_positive:
            return True

    return False


def _station_has_strong_sampling_evidence(full_text: str, station: str) -> bool:
    pattern = _ebus_station_pattern(station)
    if pattern is None:
        return False

    text = full_text or ""
    for line in text.splitlines():
        if not pattern.search(line):
            continue
        if _EBUS_SAMPLING_INDICATORS_RE.search(line) and not _EBUS_SAMPLING_CRITERIA_RE.search(line):
            return True
    return False


def _station_has_measure_only_context(full_text: str, station: str) -> bool:
    """True when a station is discussed only as measured/criteria-met (not sampled)."""
    pattern = _ebus_station_pattern(station)
    if pattern is None:
        return False

    text = full_text or ""
    for line in text.splitlines():
        if not pattern.search(line):
            continue
        if _EBUS_SAMPLING_INDICATORS_RE.search(line) and not _EBUS_SAMPLING_CRITERIA_RE.search(line):
            continue
        if _EBUS_SAMPLING_CRITERIA_RE.search(line):
            return True
        if _EBUS_MEASURE_ONLY_RE.search(line):
            return True
        if _EBUS_SIZE_MM_RE.search(line):
            return True
    return False


def sanitize_ebus_events(record: RegistryRecord, full_text: str) -> list[str]:
    """Correct EBUS node events when the note explicitly negates sampling.

    Guardrail: if node_events marks a station as needle_aspiration but the note
    near that station says "not biopsied"/"not sampled"/"no needle", force
    action="inspected_only".
    """

    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    node_events = getattr(linear, "node_events", None) if linear is not None else None
    if not isinstance(node_events, list) or not node_events:
        return warnings

    sampling_actions = {"needle_aspiration", "core_biopsy", "forceps_biopsy"}
    for event in node_events:
        action = getattr(event, "action", None)
        if action not in sampling_actions:
            continue
        station = getattr(event, "station", None)
        if not isinstance(station, str) or not station.strip():
            continue
        station_token = station.strip().upper()
        reason: str | None = None
        if _station_has_sampling_negation(full_text, station_token):
            reason = "Found negation for"
        elif (
            not _station_has_strong_sampling_evidence(full_text, station_token)
            and _station_has_measure_only_context(full_text, station_token)
        ):
            reason = "Criteria/measurement without sampling for"

        if reason:
            original_quote = getattr(event, "evidence_quote", None)
            setattr(event, "action", "inspected_only")

            # Evidence must remain verifiable (note-derived). Prefer the line that triggered
            # the correction (negation or criteria-only context) over internal debug markers.
            replacement_quote: str | None = None
            pattern = _ebus_station_pattern(station_token)
            if pattern is not None and full_text:
                for raw_line in (full_text or "").splitlines():
                    line = (raw_line or "").strip()
                    if not line:
                        continue
                    if not pattern.search(line):
                        continue
                    if reason.startswith("Found negation"):
                        if _EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(line):
                            replacement_quote = line
                            break
                        if _EBUS_MEASURE_ONLY_RE.search(line) and not _EBUS_SAMPLING_INDICATORS_RE.search(line):
                            replacement_quote = line
                            break
                        if _EBUS_BENIGN_ONLY_PHRASES_RE.search(line):
                            replacement_quote = line
                            break
                    else:
                        if _EBUS_SAMPLING_CRITERIA_RE.search(line):
                            replacement_quote = line
                            break
                        if _EBUS_MEASURE_ONLY_RE.search(line) and not _EBUS_SAMPLING_INDICATORS_RE.search(line):
                            replacement_quote = line
                            break
                        if _EBUS_SIZE_MM_RE.search(line) and not _EBUS_SAMPLING_INDICATORS_RE.search(line):
                            replacement_quote = line
                            break

            if replacement_quote:
                setattr(event, "evidence_quote", _compact_evidence_quote(replacement_quote, limit=280))
            elif isinstance(original_quote, str) and original_quote.strip():
                setattr(event, "evidence_quote", _compact_evidence_quote(original_quote, limit=280))
            else:
                setattr(event, "evidence_quote", f"Station {station_token}: inspected only.")

            if reason.startswith("Found negation"):
                warnings.append(f"AUTO_CORRECTED_EBUS_NEGATION: {station_token}")
            else:
                warnings.append(f"AUTO_CORRECTED_EBUS_CRITERIA_ONLY: {station_token}")

    # Keep legacy aggregate station lists consistent when present.
    sampled: list[str] = []
    for event in node_events:
        action = getattr(event, "action", None)
        if action not in sampling_actions:
            continue
        station = getattr(event, "station", None)
        if isinstance(station, str) and station.strip():
            sampled.append(station.strip().upper())

    granular = getattr(record, "granular_data", None)
    stations_detail = getattr(granular, "linear_ebus_stations_detail", None) if granular is not None else None
    if isinstance(stations_detail, list) and stations_detail:
        for detail in stations_detail:
            sampled_flag = getattr(detail, "sampled", None)
            if sampled_flag is False:
                continue
            station = getattr(detail, "station", None)
            if isinstance(station, str) and station.strip():
                sampled.append(station.strip().upper())
    if hasattr(linear, "stations_sampled"):
        setattr(linear, "stations_sampled", sorted(set(sampled)) if sampled else None)

    return warnings


def reconcile_ebus_sampling_from_narrative(record: RegistryRecord, full_text: str) -> list[str]:
    """Upgrade linear_ebus node_events when sampling is documented in narrative text.

    Motivation: Some extraction paths populate `linear_ebus.node_events` but leave
    `action="inspected_only"` even when the note explicitly documents TBNA sampling
    (e.g., "Sampling ... beginning with 11L, followed by 4R"). CPT derivation relies
    on node_events actions when node_events are present, so this reconciliation keeps
    codes stable without requiring schema changes.
    """

    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    if linear is None or getattr(linear, "performed", None) is not True:
        return warnings
    if not full_text:
        return warnings

    strong_sampling_re = re.compile(
        r"\b(?:needle|tbna|fna|biops(?:y|ied|ies)|sampl(?:e|ed|es|ing)|pass(?:es)?|aspirat(?:e|ed|ion)|core|forceps)\b",
        re.IGNORECASE,
    )

    # Collect stations with strong sampling evidence in the same line.
    try:
        from app.ner.entity_types import normalize_station
    except Exception:  # pragma: no cover
        normalize_station = None  # type: ignore[assignment]

    station_to_quote: dict[str, str] = {}
    for raw_line in (full_text or "").splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue
        if not _EBUS_STATION_TOKEN_RE.search(line):
            continue
        if _EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(line):
            continue

        has_strong_sampling = bool(strong_sampling_re.search(line))
        has_criteria = bool(_EBUS_SAMPLING_CRITERIA_RE.search(line))

        # Skip criteria-only mentions (e.g., "sampling criteria met ...") unless there is
        # an additional strong sampling cue (needle/TBNA/aspiration/etc) in the same line.
        if has_criteria and not has_strong_sampling:
            continue
        if not has_strong_sampling:
            continue

        for match in _EBUS_STATION_TOKEN_RE.finditer(line):
            token = match.group(1) or ""
            station = normalize_station(token) if normalize_station is not None else token.strip().upper()
            if not station:
                continue
            station = validate_station_format(str(station)) or str(station).strip().upper()
            if not station:
                continue
            station_to_quote.setdefault(station, line)

    # Site-block parsing: some notes place the station token in a "Site 1: Station 11L ..."
    # header and document sampling later ("The site was sampled ...") without repeating
    # the station token. Attribute sampling language within a site block to the station.
    site_header_re = re.compile(
        r"(?im)^\s*site\s*#?\s*(?P<num>\d{1,2})\s*(?:[:\-–.]|\)\s*)\s*(?P<rest>.*)$"
    )
    block_sampling_re = re.compile(
        r"\b(?:needle|tbna|fna|biops(?:y|ied|ies)|sampl(?:e|ed|es)|pass(?:es)?|aspirat(?:e|ed|ion)|core|forceps)\b",
        re.IGNORECASE,
    )

    site_headers = list(site_header_re.finditer(full_text or ""))
    for idx, match in enumerate(site_headers):
        start = match.start()
        end = site_headers[idx + 1].start() if idx + 1 < len(site_headers) else len(full_text)
        block = (full_text[start:end] or "").strip("\r\n")
        if not block:
            continue

        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # Station token may be in the header line or elsewhere in the block.
        station_token: str | None = None
        header_line = lines[0]
        for m_station in _EBUS_STATION_TOKEN_RE.finditer(header_line):
            token = (m_station.group(1) or "").strip()
            if token:
                station_token = token
                break
        if station_token is None:
            for ln in lines[1:]:
                for m_station in _EBUS_STATION_TOKEN_RE.finditer(ln):
                    token = (m_station.group(1) or "").strip()
                    if token:
                        station_token = token
                        break
                if station_token is not None:
                    break

        if not station_token:
            continue

        station = (
            normalize_station(station_token)
            if normalize_station is not None
            else station_token.strip().upper()
        )
        if not station:
            continue
        station = validate_station_format(str(station)) or str(station).strip().upper()
        if not station:
            continue

        sampling_quote: str | None = None
        for ln in lines:
            if _EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(ln):
                continue
            if block_sampling_re.search(ln):
                sampling_quote = ln
                break

        if sampling_quote:
            station_to_quote.setdefault(station, sampling_quote)

    if not station_to_quote:
        return warnings

    # Upgrade inspected-only events for stations we believe were sampled.
    existing = getattr(linear, "node_events", None)
    node_events = list(existing) if isinstance(existing, list) else []

    from app.registry.schema import NodeInteraction

    by_station: dict[str, NodeInteraction] = {}
    for event in node_events:
        station = getattr(event, "station", None)
        if not isinstance(station, str) or not station.strip():
            continue
        station_key = station.strip().upper()
        if normalize_station is not None:
            station_key = normalize_station(station_key) or station_key
        station_key = validate_station_format(str(station_key)) or str(station_key).strip().upper()
        station_key = station_key.strip().upper()
        if not station_key:
            continue
        # Canonicalize station text in-place to avoid duplicate keys (e.g., 11Rs + 11R).
        setattr(event, "station", station_key)
        by_station[station_key] = event

    upgraded: list[str] = []
    added: list[str] = []
    for station, quote in station_to_quote.items():
        key = station.strip().upper()
        event = by_station.get(key)
        if event is not None:
            if getattr(event, "action", None) == "inspected_only":
                setattr(event, "action", "needle_aspiration")
                setattr(event, "evidence_quote", _compact_evidence_quote(quote, limit=280))
                upgraded.append(key)
            continue

        node_events.append(
            NodeInteraction(
                station=station,
                action="needle_aspiration",
                outcome=None,
                evidence_quote=_compact_evidence_quote(quote, limit=280),
            )
        )
        added.append(key)

    if upgraded or added:
        setattr(linear, "node_events", node_events)
        if hasattr(linear, "stations_sampled"):
            stations_sampled = getattr(linear, "stations_sampled", None)
            existing_stations: list[str] = []
            if isinstance(stations_sampled, list):
                for value in stations_sampled:
                    token = str(value).strip().upper()
                    if not token:
                        continue
                    if normalize_station is not None:
                        token = normalize_station(token) or token
                    token = validate_station_format(str(token)) or str(token).strip().upper()
                    token = token.strip().upper()
                    if token:
                        existing_stations.append(token)
            merged = sorted(set(existing_stations) | set(station_to_quote.keys()))
            setattr(linear, "stations_sampled", merged if merged else None)

        if upgraded:
            warnings.append(
                f"EBUS_NARRATIVE_RECONCILE: upgraded inspected_only→needle_aspiration for {sorted(set(upgraded))}"
            )
        if added:
            warnings.append(
                f"EBUS_NARRATIVE_RECONCILE: added node_events for sampled stations {sorted(set(added))}"
            )

    return warnings


_PROCEDURE_DETAIL_SECTION_PATTERN = re.compile(
    r"(?im)^\s*(?:procedure\s+in\s+detail|description\s+of\s+procedure|procedure\s+description)\s*:?"
)
_LINEAR_EBUS_MARKER_RE = re.compile(r"(?i)\b(?:ebus|endobronchial\s+ultrasound)\b")


def cull_hollow_ebus_claims(record: RegistryRecord, full_text: str) -> list[str]:
    """Cull hallucinated linear EBUS when no station evidence exists.

    If linear_ebus.performed is true but there are no extracted node_events and no
    stations_sampled, require an explicit EBUS marker in the PROCEDURE IN DETAIL
    narrative (headers ignored). If absent, flip performed to false.
    """
    warnings: list[str] = []

    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    if linear is None or getattr(linear, "performed", None) is not True:
        return warnings

    node_events = getattr(linear, "node_events", None)
    if isinstance(node_events, list) and node_events:
        return warnings

    stations_sampled = getattr(linear, "stations_sampled", None)
    if isinstance(stations_sampled, list) and stations_sampled:
        return warnings

    # If the note doesn't have a clear procedure-detail section, do not cull;
    # the record may still be correct but the narrative is formatted differently.
    text = full_text or ""
    match = _PROCEDURE_DETAIL_SECTION_PATTERN.search(text)
    if not match:
        return warnings

    procedure_detail = text[match.end() :].lstrip("\r\n ")
    if _LINEAR_EBUS_MARKER_RE.search(procedure_detail):
        return warnings

    # No stations/events and no EBUS marker in the narrative body => treat as hallucination.
    setattr(linear, "performed", False)
    if hasattr(linear, "stations_sampled"):
        setattr(linear, "stations_sampled", None)
    if hasattr(linear, "stations_planned"):
        setattr(linear, "stations_planned", None)
    if hasattr(linear, "stations_detail"):
        setattr(linear, "stations_detail", None)
    if hasattr(linear, "passes_per_station"):
        setattr(linear, "passes_per_station", None)
    if hasattr(linear, "needle_gauge"):
        setattr(linear, "needle_gauge", None)
    if hasattr(linear, "needle_type"):
        setattr(linear, "needle_type", None)
    if hasattr(linear, "elastography_used"):
        setattr(linear, "elastography_used", None)
    if hasattr(linear, "elastography_pattern"):
        setattr(linear, "elastography_pattern", None)
    if hasattr(linear, "doppler_used"):
        setattr(linear, "doppler_used", None)
    if hasattr(linear, "node_events"):
        setattr(linear, "node_events", [])

    warnings.append(
        "AUTO_CORRECTED: Culled hollow linear_ebus claim (no stations/text evidence)."
    )
    return warnings


_EBUS_FALLBACK_STATION_RE = re.compile(
    r"\b(?:station|level)\s*(\d{1,2}[RL]?(?:s|i)?)\b",
    re.IGNORECASE,
)
_EBUS_STATION_TOKEN_RE = re.compile(
    r"\b(2R|2L|4R|4L|7|10R|10L|11R(?:S|I)?|11L(?:S|I)?)\b",
    re.IGNORECASE,
)

_EBUS_SPECIMEN_SECTION_HEADER_RE = re.compile(r"(?im)^\s*specimens?\b.*:?$")
_EBUS_SPECIMEN_TBNA_LINE_RE = re.compile(
    r"\b(?:tbna|fna|transbronchial\s+needle|needle\s+aspirat)\w*\b",
    re.IGNORECASE,
)
_EBUS_SPECIMEN_SECTION_STOP_RE = re.compile(
    r"(?im)^\s*(?:complications?|estimated\s+blood\s+loss|ebl|post[- ]procedure|disposition|recommendations?)\b"
)


def _extract_tbna_stations_from_specimen_log(full_text: str) -> dict[str, dict[str, object]]:
    if not full_text:
        return {}

    header = _EBUS_SPECIMEN_SECTION_HEADER_RE.search(full_text)
    if not header:
        return {}

    from app.ner.entity_types import normalize_station

    stations: dict[str, dict[str, object]] = {}
    tail = full_text[header.end() :]
    cursor = header.end()
    for raw_line in tail.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            if stations:
                break
            cursor += len(raw_line)
            continue
        if _EBUS_SPECIMEN_SECTION_STOP_RE.search(line) and stations:
            break
        if not _EBUS_SPECIMEN_TBNA_LINE_RE.search(line):
            cursor += len(raw_line)
            continue

        # Global offsets for the stripped line (best-effort; used for evidence_quote highlighting).
        leading_ws = len(line) - len(line.lstrip())
        trailing_len = len(line.rstrip())
        line_start = cursor + leading_ws
        line_end = cursor + trailing_len

        for match in _EBUS_STATION_TOKEN_RE.finditer(line):
            station = normalize_station(match.group(1) or "")
            if not station:
                continue

            token_start = cursor + match.start(1)
            token_end = cursor + match.end(1)
            if token_end <= token_start:
                continue

            stations.setdefault(
                station,
                {
                    "station": station,
                    "evidence_quote": stripped,
                    "token_start": int(token_start),
                    "token_end": int(token_end),
                    "line_start": int(line_start),
                    "line_end": int(line_end),
                },
            )

        cursor += len(raw_line)

    return stations


def reconcile_ebus_sampling_from_specimen_log(record: RegistryRecord, full_text: str) -> list[str]:
    """Restrict linear EBUS stations_sampled to specimen-log TBNA stations when available.

    Some notes list stations measured/surveyed in narrative, but the specimen log
    is the most reliable source of what was actually sampled/sent.
    """
    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    if linear is None or getattr(linear, "performed", False) is not True:
        return warnings
    if not full_text:
        return warnings

    station_to_meta = _extract_tbna_stations_from_specimen_log(full_text)
    if not station_to_meta:
        return warnings

    confirmed = sorted(station_to_meta.keys())
    existing = getattr(linear, "node_events", None)
    node_events = list(existing) if isinstance(existing, list) else []

    sampling_actions = {"needle_aspiration", "core_biopsy", "forceps_biopsy"}

    confirmed_meta: dict[str, dict[str, object]] = {}
    for station, meta in station_to_meta.items():
        token = str(station).strip().upper()
        if token:
            confirmed_meta[token] = {
                "station": str(station).strip(),
                "evidence_quote": str(meta.get("evidence_quote") or "").strip(),
                "token_start": meta.get("token_start"),
                "token_end": meta.get("token_end"),
                "line_start": meta.get("line_start"),
                "line_end": meta.get("line_end"),
            }

    from app.registry.schema import NodeInteraction

    seen_confirmed: set[str] = set()
    for event in node_events:
        station = getattr(event, "station", None)
        if not isinstance(station, str) or not station.strip():
            continue
        station_token = station.strip().upper()
        meta = confirmed_meta.get(station_token)
        if meta:
            seen_confirmed.add(station_token)
            setattr(event, "action", "needle_aspiration")
            existing_quote = getattr(event, "evidence_quote", None)
            if not (isinstance(existing_quote, str) and existing_quote.strip()):
                setattr(event, "evidence_quote", meta.get("evidence_quote") or existing_quote)
        elif getattr(event, "action", None) in sampling_actions:
            setattr(event, "action", "inspected_only")

    for station_token, meta in confirmed_meta.items():
        if station_token in seen_confirmed:
            continue
        node_events.append(
            NodeInteraction(
                station=meta["station"],
                action="needle_aspiration",
                outcome=None,
                evidence_quote=meta["evidence_quote"] or f"TBNA of {meta['station']} documented in specimen log.",
            )
        )

    setattr(linear, "node_events", node_events)
    if hasattr(linear, "stations_sampled"):
        setattr(linear, "stations_sampled", confirmed if confirmed else None)

    # Best-effort evidence spans for UI highlighting.
    try:
        from app.common.spans import Span

        evidence = getattr(record, "evidence", None)
        if not isinstance(evidence, dict):
            evidence = {}

        for idx, station in enumerate(confirmed):
            meta = station_to_meta.get(station) if isinstance(station, str) else None
            if not isinstance(meta, dict):
                continue
            start = meta.get("token_start")
            end = meta.get("token_end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                evidence.setdefault(
                    f"procedures_performed.linear_ebus.stations_sampled.{idx}",
                    [],
                ).append(Span(text=full_text[start:end], start=start, end=end, confidence=0.9))

        for idx, event in enumerate(node_events):
            station = getattr(event, "station", None)
            if not isinstance(station, str) or not station.strip():
                continue
            meta = confirmed_meta.get(station.strip().upper())
            if not isinstance(meta, dict):
                continue

            start = meta.get("token_start")
            end = meta.get("token_end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                evidence.setdefault(
                    f"procedures_performed.linear_ebus.node_events.{idx}.station",
                    [],
                ).append(Span(text=full_text[start:end], start=start, end=end, confidence=0.9))

            line_start = meta.get("line_start")
            line_end = meta.get("line_end")
            if isinstance(line_start, int) and isinstance(line_end, int) and line_end > line_start:
                evidence.setdefault(
                    f"procedures_performed.linear_ebus.node_events.{idx}.evidence_quote",
                    [],
                ).append(
                    Span(
                        text=full_text[line_start:line_end].strip(),
                        start=line_start,
                        end=line_end,
                        confidence=0.9,
                    )
                )

        record.evidence = evidence
    except Exception:
        pass

    warnings.append(f"EBUS_SPECIMEN_OVERRIDE: stations_sampled={confirmed} from specimen log TBNA entries.")
    return warnings


_EBUS_SITE_HEADER_RE = re.compile(r"(?im)^\s*site\s*#?\s*(?P<num>\d{1,2})\s*(?:[:\-–.]|\)\s*)\s*")
_EBUS_PASSES_RE = re.compile(
    r"\b(?P<count>\d{1,2})\b[^.\n]{0,80}\b(?:needle\s+passes?|passes?|endobronchial\s+ultrasound\s+guided\s+transbronchial\s+biops(?:y|ies))\b",
    re.IGNORECASE,
)
_EBUS_ELASTO_TYPE_RE = re.compile(r"\btype\s*(?P<num>[1-3])\b[^.\n]{0,120}\belasto", re.IGNORECASE)


def _compact_evidence_quote(text: str, *, limit: int = 420) -> str:
    quote = re.sub(r"\s+", " ", (text or "").strip())
    if len(quote) <= limit:
        return quote
    return quote[:limit].rstrip()


def enrich_ebus_node_event_sampling_details(record: RegistryRecord, full_text: str) -> list[str]:
    """Populate per-station EBUS sampling details (passes + elastography) from narrative blocks.

    Motivation: avoid "lazy evidence" where specimen logs overwrite richer narrative
    details like elastography grading and per-station sampling counts.
    """
    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    if linear is None or getattr(linear, "performed", None) is not True:
        return warnings

    existing_events = getattr(linear, "node_events", None)
    node_events = list(existing_events) if isinstance(existing_events, list) else []
    if not full_text:
        return warnings

    try:
        from app.ner.entity_types import normalize_station
    except Exception:
        normalize_station = None  # type: ignore[assignment]

    details: dict[str, dict[str, object]] = {}
    site_headers = list(_EBUS_SITE_HEADER_RE.finditer(full_text))
    for idx, match in enumerate(site_headers):
        start = match.start()
        end = site_headers[idx + 1].start() if idx + 1 < len(site_headers) else len(full_text)
        block = full_text[start:end].strip()
        if not block:
            continue

        # Station token extraction:
        # - Only trust the *site header line* (avoids matching "7/2025" dates later in the block).
        # - For digit-only stations like "7", require nearby nodal context (e.g., "(subcarinal) node").
        site_line = block.splitlines()[0].strip() if block.splitlines() else block
        station_raw = ""
        for m_station in _EBUS_STATION_TOKEN_RE.finditer(site_line):
            token = (m_station.group(1) or "").strip()
            if not token:
                continue
            if token.isdigit():
                before_char = site_line[m_station.start(1) - 1] if m_station.start(1) > 0 else ""
                after_char = site_line[m_station.end(1)] if m_station.end(1) < len(site_line) else ""
                if before_char in {"/", "-"} or after_char in {"/", "-"}:
                    continue
                lookbehind = site_line[max(0, m_station.start(1) - 24) : m_station.start(1)].lower()
                lookahead = site_line[m_station.end(1) : m_station.end(1) + 42].lower()
                if not (
                    re.search(r"\b(?:station|level)\b", lookbehind)
                    or re.search(r"\b(?:node|ln|lymph|subcarinal|paratracheal)\b|\(", lookahead)
                ):
                    continue
            station_raw = token
            break
        if not station_raw:
            continue
        station = station_raw.strip().upper()
        if normalize_station is not None:
            station = normalize_station(station) or station
        station = validate_station_format(str(station)) or str(station).strip().upper()
        station = station.strip().upper()
        if not station:
            continue

        passes_val: int | None = None
        m_passes = _EBUS_PASSES_RE.search(block)
        if m_passes:
            try:
                passes_val = int(m_passes.group("count"))
            except Exception:
                passes_val = None

        pattern_val: str | None = None
        m_type = _EBUS_ELASTO_TYPE_RE.search(block)
        if m_type:
            try:
                num = int(m_type.group("num"))
            except Exception:
                num = None
            if num in (1, 2, 3):
                pattern_val = f"Type {num}"

        evidence_quote = _compact_evidence_quote(block)
        sampled_in_block = bool(_EBUS_SAMPLING_INDICATORS_RE.search(block)) and not bool(
            _EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(block)
        )
        details[station] = {
            "passes": passes_val,
            "elastography_pattern": pattern_val,
            "evidence_quote": evidence_quote,
            "sampled": sampled_in_block,
        }

    if not details:
        return warnings

    if not node_events:
        from app.registry.schema import NodeInteraction

        created: list[str] = []
        created_sampled: list[str] = []
        for station, meta in details.items():
            has_signal = bool(meta.get("sampled")) or isinstance(meta.get("passes"), int) or isinstance(
                meta.get("elastography_pattern"), str
            )
            if not has_signal:
                continue
            action = "needle_aspiration" if bool(meta.get("sampled")) else "inspected_only"
            node_events.append(
                NodeInteraction(
                    station=station,
                    action=action,
                    outcome=None,
                    passes=meta.get("passes"),
                    elastography_pattern=meta.get("elastography_pattern"),
                    evidence_quote=meta.get("evidence_quote"),
                )
            )
            created.append(station)
            if action == "needle_aspiration":
                created_sampled.append(station)

        if created:
            setattr(linear, "node_events", node_events)
            if hasattr(linear, "stations_sampled"):
                prior = getattr(linear, "stations_sampled", None)
                prior_norm: list[str] = []
                if isinstance(prior, list):
                    for value in prior:
                        token = str(value).strip().upper()
                        if not token:
                            continue
                        if normalize_station is not None:
                            token = normalize_station(token) or token
                        token = validate_station_format(str(token)) or str(token).strip().upper()
                        token = token.strip().upper()
                        if token:
                            prior_norm.append(token)
                merged_sampled = sort_ebus_stations(set(prior_norm) | set(created_sampled))
                setattr(linear, "stations_sampled", merged_sampled if merged_sampled else None)

            if any(isinstance(details[s].get("elastography_pattern"), str) for s in created):
                setattr(linear, "elastography_used", True)

            warnings.append(
                "AUTO_EBUS_GRANULARITY: added node_events from site blocks for "
                f"{sort_ebus_stations(set(created))}"
            )

    updated: list[str] = []
    for event in node_events:
        station_raw = getattr(event, "station", None)
        if not isinstance(station_raw, str) or not station_raw.strip():
            continue
        station = station_raw.strip().upper()
        if normalize_station is not None:
            station = normalize_station(station) or station
        station = validate_station_format(str(station)) or str(station).strip().upper()
        station = station.strip().upper()
        meta = details.get(station)
        if not meta:
            continue

        existing_passes = getattr(event, "passes", None)
        if existing_passes is None and isinstance(meta.get("passes"), int):
            setattr(event, "passes", meta["passes"])

        existing_pattern = getattr(event, "elastography_pattern", None)
        if existing_pattern in (None, "") and isinstance(meta.get("elastography_pattern"), str):
            setattr(event, "elastography_pattern", meta["elastography_pattern"])
            if getattr(linear, "elastography_used", None) is not True:
                setattr(linear, "elastography_used", True)

        quote = meta.get("evidence_quote")
        if isinstance(quote, str) and quote.strip():
            existing_quote = getattr(event, "evidence_quote", None)
            existing_text = existing_quote.strip().lower() if isinstance(existing_quote, str) else ""
            # Prefer narrative excerpts when they contain granularity not present in the
            # existing quote (which is often specimen-log-derived).
            needs_upgrade = False
            if re.match(r"(?i)^site\s+\d{1,2}\s*:", quote.strip()) and not re.match(
                r"(?i)^site\s+\d{1,2}\s*:", existing_text.strip()
            ):
                needs_upgrade = True
            if not existing_text:
                needs_upgrade = True
            elif isinstance(meta.get("passes"), int) and "pass" not in existing_text and "biops" not in existing_text:
                needs_upgrade = True
            elif isinstance(meta.get("elastography_pattern"), str) and "type" not in existing_text:
                needs_upgrade = True
            if needs_upgrade:
                setattr(event, "evidence_quote", quote)

        updated.append(station)

    if node_events:
        setattr(linear, "node_events", node_events)

    if updated:
        warnings.append(
            "AUTO_EBUS_GRANULARITY: populated passes/elastography for "
            f"{sort_ebus_stations(set(updated))}"
        )

    return warnings

_EBUS_STATION_LIST_HEADER_RE = re.compile(
    r"\b(?:following\s+station\(s\)|following\s+stations?|stations?\s+(?:sampled|biopsied|aspirated)|lymph\s+node\s+stations?\s+(?:sampled|biopsied))\b",
    re.IGNORECASE,
)
_EBUS_INSPECTION_HINTS_RE = re.compile(
    r"\b(?:inspect|inspection|visualiz|view|survey|assess|measure|measurement)\b",
    re.IGNORECASE,
)
_EBUS_SAMPLED_SECTION_RE = re.compile(
    r"\b(?:EBUS\s+)?Lymph\s+Nodes\s+Sampled\b",
    re.IGNORECASE,
)


def populate_ebus_node_events_fallback(record: RegistryRecord, full_text: str) -> list[str]:
    """Populate basic EBUS node_events from station lines when missing."""
    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    if linear is None or getattr(linear, "performed", False) is not True:
        return warnings

    existing = getattr(linear, "node_events", None)
    if isinstance(existing, list) and existing:
        return warnings

    if not full_text:
        return warnings

    from app.ner.entity_types import normalize_station
    from app.registry.schema import NodeInteraction

    station_events: dict[str, NodeInteraction] = {}
    station_evidence: dict[str, dict[str, int]] = {}
    stations_sampled: list[str] = []
    in_sampled_section = False
    in_station_list = False
    station_list_default_sampling = False
    station_list_seen_station = False

    passes_re = re.compile(r"(?i)\b(\d{1,2})\s*(?:pass|passes)\b")
    rose_re = re.compile(r"(?i)\brose\s*(?:showed|revealed|:|was)?\s*([^.\\n]{1,180})")

    cursor = 0
    for raw_line in full_text.splitlines(keepends=True):
        line_offset = cursor
        cursor += len(raw_line)
        line = raw_line.rstrip("\r\n")
        if _EBUS_SAMPLED_SECTION_RE.search(line):
            in_sampled_section = True
            continue
        if _EBUS_STATION_LIST_HEADER_RE.search(line) and re.search(r"(?i)\b(?:tbna|needle|aspirat|sample|biops)\w*\b", line):
            in_station_list = True
            station_list_default_sampling = True
            station_list_seen_station = False
            continue
        if _EBUS_STATION_LIST_HEADER_RE.search(line) and _EBUS_INSPECTION_HINTS_RE.search(line):
            in_station_list = True
            station_list_default_sampling = False
            station_list_seen_station = False
            continue
        if not line.strip():
            in_sampled_section = False
            if in_station_list:
                # Many notes put a blank line between a "following station(s):" header
                # and the actual station lines. Keep the section open across blank lines.
                continue
            station_list_default_sampling = False

        if not _EBUS_FALLBACK_STATION_RE.search(line):
            if not (in_sampled_section or in_station_list or "lymph node" in line.lower()):
                continue

        has_station_token = bool(_EBUS_STATION_TOKEN_RE.search(line)) or bool(
            _EBUS_FALLBACK_STATION_RE.search(line)
        )
        if in_station_list:
            if has_station_token:
                station_list_seen_station = True
            else:
                # End the station-list section once we move past station lines.
                if station_list_seen_station and not re.search(r"(?i)\blymph\s+node\b", line):
                    if not _EBUS_SAMPLING_INDICATORS_RE.search(line) and not _EBUS_INSPECTION_HINTS_RE.search(line):
                        in_station_list = False
                        station_list_default_sampling = False
                        station_list_seen_station = False
                        # Re-evaluate this line outside the station-list context.
                        if not _EBUS_FALLBACK_STATION_RE.search(line) and "lymph node" not in line.lower():
                            continue

        sampling = bool(_EBUS_SAMPLING_INDICATORS_RE.search(line)) or in_sampled_section
        if in_station_list and not sampling:
            sampling = station_list_default_sampling
        negated = bool(_EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(line))
        inspection = bool(_EBUS_INSPECTION_HINTS_RE.search(line))
        if _EBUS_MEASURE_ONLY_RE.search(line) and not sampling:
            inspection = True

        if not sampling and not inspection:
            continue

        action = "needle_aspiration" if sampling and not negated else "inspected_only"

        passes_val: int | None = None
        rose_val: str | None = None
        passes_span: tuple[int, int] | None = None
        rose_span: tuple[int, int] | None = None

        if action != "inspected_only":
            m_passes = passes_re.search(line)
            if m_passes:
                try:
                    passes_val = int(m_passes.group(1))
                except Exception:
                    passes_val = None
                passes_span = (line_offset + m_passes.start(), line_offset + m_passes.end())

            m_rose = rose_re.search(line)
            if m_rose:
                raw = (m_rose.group(1) or "").strip()
                raw = raw.strip(" :;-\"'").strip()
                if raw:
                    rose_val = raw[:180]
                    rose_span = (line_offset + m_rose.start(), line_offset + m_rose.end())

        station_tokens: list[tuple[str, int, int]] = []
        for match in _EBUS_FALLBACK_STATION_RE.finditer(line):
            token = match.group(1) or ""
            start = line_offset + match.start(1)
            end = line_offset + match.end(1)
            station_tokens.append((token, start, end))
        if in_sampled_section or in_station_list or "lymph node" in line.lower():
            for match in _EBUS_STATION_TOKEN_RE.finditer(line):
                token = match.group(1) or ""
                if token.isdigit():
                    prefix = line[: match.start(1)]
                    if re.search(r"(?i)\b(?:site|case|patient)\s+#?\s*$", prefix):
                        continue
                    before_char = line[match.start(1) - 1] if match.start(1) > 0 else ""
                    after_char = line[match.end(1)] if match.end(1) < len(line) else ""
                    if before_char in {"/", "-"} or after_char in {"/", "-"}:
                        continue
                    lookbehind = line[max(0, match.start(1) - 24) : match.start(1)].lower()
                    lookahead = line[match.end(1) : match.end(1) + 42].lower()
                    if not (
                        re.search(r"\b(?:station|level)\b", lookbehind)
                        or re.search(r"\b(?:node|ln|lymph|subcarinal|paratracheal)\b|\(", lookahead)
                    ):
                        continue
                start = line_offset + match.start(1)
                end = line_offset + match.end(1)
                station_tokens.append((token, start, end))

        leading_ws = len(line) - len(line.lstrip())
        trailing_len = len(line.rstrip())
        quote_start = line_offset + leading_ws
        quote_end = line_offset + trailing_len

        for token, token_start, token_end in station_tokens:
            station = normalize_station(token)
            if not station:
                continue
            existing = station_events.get(station)
            if existing:
                if existing.action == "inspected_only" and action != "inspected_only":
                    existing.action = action
                    existing.evidence_quote = line.strip()
                    stations_sampled.append(station)
                    station_evidence[station] = {
                        "token_start": int(token_start),
                        "token_end": int(token_end),
                        "quote_start": int(quote_start),
                        "quote_end": int(quote_end),
                    }
                if getattr(existing, "passes", None) is None and isinstance(passes_val, int):
                    existing.passes = passes_val
                if getattr(existing, "rose_result", None) in (None, "") and isinstance(rose_val, str) and rose_val:
                    existing.rose_result = rose_val
                meta = station_evidence.get(station)
                if isinstance(meta, dict):
                    if passes_span and "passes_start" not in meta and "passes_end" not in meta:
                        meta["passes_start"] = int(passes_span[0])
                        meta["passes_end"] = int(passes_span[1])
                    if rose_span and "rose_start" not in meta and "rose_end" not in meta:
                        meta["rose_start"] = int(rose_span[0])
                        meta["rose_end"] = int(rose_span[1])
                continue
            station_events[station] = NodeInteraction(
                station=station,
                action=action,
                outcome=None,
                passes=passes_val,
                rose_result=rose_val,
                evidence_quote=line.strip(),
            )
            station_evidence[station] = {
                "token_start": int(token_start),
                "token_end": int(token_end),
                "quote_start": int(quote_start),
                "quote_end": int(quote_end),
            }
            if passes_span:
                station_evidence[station]["passes_start"] = int(passes_span[0])
                station_evidence[station]["passes_end"] = int(passes_span[1])
            if rose_span:
                station_evidence[station]["rose_start"] = int(rose_span[0])
                station_evidence[station]["rose_end"] = int(rose_span[1])
            if action != "inspected_only":
                stations_sampled.append(station)

    if not station_events:
        # Some notes clearly document EBUS-TBNA sampling but omit station tokens
        # (e.g., "Lymph node sizing and sampling were performed using EBUS-TBNA ...").
        tbna_match = re.search(r"(?is)\bebus[-\s]?tbna\b", full_text)
        if not tbna_match:
            tbna_match = re.search(
                r"(?is)\b(?:convex\s+probe|cp-?ebus)\b[^.\n]{0,240}"
                r"\b(?:tbna|transbronchial\s+needle|needle\s+aspirat|needle\s+pass|passes|fna|biops)\w*\b",
                full_text,
            )
        if tbna_match and not _EBUS_EXPLICIT_NEGATION_PHRASES_RE.search(tbna_match.group(0) or ""):
            snippet = re.sub(r"\s+", " ", (tbna_match.group(0) or "").strip())
            start = int(tbna_match.start())
            end = int(tbna_match.end())
            station_events["UNSPECIFIED"] = NodeInteraction(
                station="UNSPECIFIED",
                action="needle_aspiration",
                outcome=None,
                evidence_quote=snippet[:280] if snippet else "EBUS-TBNA sampling documented (stations not specified).",
            )
            station_evidence["UNSPECIFIED"] = {
                "token_start": start,
                "token_end": end,
                "quote_start": start,
                "quote_end": end,
            }
            setattr(linear, "node_events", list(station_events.values()))
            if hasattr(linear, "stations_sampled"):
                setattr(linear, "stations_sampled", ["UNSPECIFIED"])
            try:
                from app.common.spans import Span

                evidence = getattr(record, "evidence", None)
                if not isinstance(evidence, dict):
                    evidence = {}
                text = full_text[start:end].strip()
                evidence.setdefault("procedures_performed.linear_ebus.stations_sampled.0", []).append(
                    Span(text=text, start=start, end=end, confidence=0.9)
                )
                evidence.setdefault("procedures_performed.linear_ebus.node_events.0.station", []).append(
                    Span(text=text, start=start, end=end, confidence=0.9)
                )
                evidence.setdefault("procedures_performed.linear_ebus.node_events.0.evidence_quote", []).append(
                    Span(text=text, start=start, end=end, confidence=0.9)
                )
                record.evidence = evidence
            except Exception:
                pass
            warnings.append("EBUS_FALLBACK: sampling documented but stations missing; added placeholder node_event.")
        return warnings

    setattr(linear, "node_events", list(station_events.values()))
    if hasattr(linear, "stations_sampled"):
        setattr(linear, "stations_sampled", sorted(set(stations_sampled)) if stations_sampled else None)

    # Best-effort evidence spans for UI highlighting.
    try:
        from app.common.spans import Span

        evidence = getattr(record, "evidence", None)
        if not isinstance(evidence, dict):
            evidence = {}

        node_events = getattr(linear, "node_events", None)
        if isinstance(node_events, list):
            for idx, event in enumerate(node_events):
                station = getattr(event, "station", None)
                if not isinstance(station, str) or not station.strip():
                    continue
                meta = station_evidence.get(station)
                if not isinstance(meta, dict):
                    continue
                start = meta.get("token_start")
                end = meta.get("token_end")
                if isinstance(start, int) and isinstance(end, int) and end > start:
                    evidence.setdefault(
                        f"procedures_performed.linear_ebus.node_events.{idx}.station",
                        [],
                    ).append(Span(text=full_text[start:end], start=start, end=end, confidence=0.9))

                quote_start = meta.get("quote_start")
                quote_end = meta.get("quote_end")
                if isinstance(quote_start, int) and isinstance(quote_end, int) and quote_end > quote_start:
                    evidence.setdefault(
                        f"procedures_performed.linear_ebus.node_events.{idx}.evidence_quote",
                        [],
                    ).append(
                        Span(
                            text=full_text[quote_start:quote_end].strip(),
                            start=quote_start,
                            end=quote_end,
                            confidence=0.9,
                        )
                    )

                passes_start = meta.get("passes_start")
                passes_end = meta.get("passes_end")
                if isinstance(passes_start, int) and isinstance(passes_end, int) and passes_end > passes_start:
                    evidence.setdefault(
                        f"procedures_performed.linear_ebus.node_events.{idx}.passes",
                        [],
                    ).append(
                        Span(
                            text=full_text[passes_start:passes_end].strip(),
                            start=passes_start,
                            end=passes_end,
                            confidence=0.9,
                        )
                    )

                rose_start = meta.get("rose_start")
                rose_end = meta.get("rose_end")
                if isinstance(rose_start, int) and isinstance(rose_end, int) and rose_end > rose_start:
                    evidence.setdefault(
                        f"procedures_performed.linear_ebus.node_events.{idx}.rose_result",
                        [],
                    ).append(
                        Span(
                            text=full_text[rose_start:rose_end].strip(),
                            start=rose_start,
                            end=rose_end,
                            confidence=0.9,
                        )
                    )

        sampled = getattr(linear, "stations_sampled", None)
        if isinstance(sampled, list):
            for idx, station in enumerate(sampled):
                if not isinstance(station, str) or not station.strip():
                    continue
                meta = station_evidence.get(station)
                if not isinstance(meta, dict):
                    continue
                start = meta.get("token_start")
                end = meta.get("token_end")
                if isinstance(start, int) and isinstance(end, int) and end > start:
                    evidence.setdefault(
                        f"procedures_performed.linear_ebus.stations_sampled.{idx}",
                        [],
                    ).append(Span(text=full_text[start:end], start=start, end=end, confidence=0.9))

        record.evidence = evidence
    except Exception:
        pass

    warnings.append("EBUS_REGEX_FALLBACK: populated node_events from station lines.")
    return warnings


_EBUS_ROSE_MARKER_RE = re.compile(
    r"\b(?:rose|rapid\s+on[-\s]?site|on[-\s]?site\s+path|onsite\s+path|on[-\s]?site\s+cytolog)\b",
    re.IGNORECASE,
)
_EBUS_ROSE_BENIGN_RE = re.compile(
    r"\b(?:benign|reactive|lymphocytes?|adequate\s+lymphocytes?|granulom(?:a|as)|negative\s+for\s+malignan|no\s+malignan|did\s+not\s+identify\s+malignan)\b",
    re.IGNORECASE,
)
_EBUS_ROSE_MALIGNANT_RE = re.compile(
    r"\b(?:malignan(?:t|cy)|carcinoma|adenocarcinoma|squamous|small\s+cell|positive\s+for\s+malignan)\b",
    re.IGNORECASE,
)
_EBUS_ROSE_SUSPICIOUS_RE = re.compile(r"\bsuspici(?:ous|on)\b", re.IGNORECASE)
_EBUS_ROSE_NONDx_RE = re.compile(
    r"\b(?:non[-\s]?diagnostic|nondiagnostic|inadequate|insufficient|scant)\b",
    re.IGNORECASE,
)
_EBUS_NEGATED_MALIGNANCY_RE = re.compile(
    r"\b(?:no|not|without|negative\s+for|did\s+not\s+identify)\b[^.\n]{0,40}\bmalignan",
    re.IGNORECASE,
)


def _infer_ebus_node_outcome_from_rose_window(rose_window: str, station: str) -> str | None:
    """Infer NodeOutcomeType from ROSE/onsite-path wording near a station token."""
    if not rose_window or not station:
        return None

    station_token = station.strip().upper()
    if not station_token:
        return None

    lowered = rose_window.lower()
    station_lower = station_token.lower()

    # Require the station token to appear in the window to avoid applying global ROSE
    # statements (e.g., "ROSE negative") to every station.
    if station_lower not in lowered and f"station {station_lower}" not in lowered:
        return None

    # Check a tight neighborhood around each station mention.
    for match in re.finditer(re.escape(station_lower), lowered):
        local = lowered[max(0, match.start() - 140) : min(len(lowered), match.end() + 140)]

        if _EBUS_ROSE_NONDx_RE.search(local):
            return "nondiagnostic"
        if _EBUS_ROSE_SUSPICIOUS_RE.search(local):
            return "suspicious"
        if _EBUS_ROSE_MALIGNANT_RE.search(local) and not _EBUS_NEGATED_MALIGNANCY_RE.search(local):
            return "malignant"
        if _EBUS_ROSE_BENIGN_RE.search(local):
            return "benign"

    return None


def enrich_ebus_node_event_outcomes(record: RegistryRecord, full_text: str) -> list[str]:
    """Enrich EBUS node_events outcomes from ROSE/onsite-path wording when possible."""
    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    node_events = getattr(linear, "node_events", None) if linear is not None else None
    if not isinstance(node_events, list) or not node_events:
        return warnings
    if not full_text:
        return warnings

    # Build ROSE windows (bounded slices) for efficient station lookups.
    rose_windows: list[str] = []
    for marker in _EBUS_ROSE_MARKER_RE.finditer(full_text):
        start = max(0, marker.start() - 600)
        end = min(len(full_text), marker.end() + 600)
        rose_windows.append(full_text[start:end])

    if not rose_windows:
        return warnings

    for event in node_events:
        if getattr(event, "action", None) != "needle_aspiration":
            continue
        station = getattr(event, "station", None)
        if not isinstance(station, str) or not station.strip():
            continue

        outcome = getattr(event, "outcome", None)
        if outcome not in (None, "unknown", "deferred_to_final_path"):
            continue

        inferred: str | None = None
        for window in rose_windows:
            inferred = _infer_ebus_node_outcome_from_rose_window(window, station)
            if inferred:
                break

        if inferred:
            setattr(event, "outcome", inferred)
            warnings.append(f"AUTO_EBUS_OUTCOME: {station.strip().upper()} -> {inferred}")

    return warnings


def enrich_linear_ebus_needle_gauge(record: RegistryRecord, full_text: str) -> list[str]:
    """Populate procedures_performed.linear_ebus.needle_gauge from note text when missing."""
    warnings: list[str] = []

    procedures = getattr(record, "procedures_performed", None)
    linear = getattr(procedures, "linear_ebus", None) if procedures is not None else None
    if linear is None or getattr(linear, "performed", None) is not True:
        return warnings

    if not full_text:
        return warnings

    existing = getattr(linear, "needle_gauge", None)
    expected_gauge: int | None = None
    if isinstance(existing, str) and existing.strip():
        m = re.search(r"(?i)\b(19|21|22|25)\b", existing)
        if m:
            try:
                expected_gauge = int(m.group(1))
            except Exception:
                expected_gauge = None

    match = re.search(r"\b(19|21|22|25)\s*[-]?\s*(?:G|gauge)\b", full_text, re.IGNORECASE)
    if not match:
        match = re.search(r"\b(19|21|22|25)\s+gauge\s+needle\b", full_text, re.IGNORECASE)
    if not match:
        return warnings

    try:
        gauge = int(match.group(1))
    except Exception:
        gauge = None

    # If the field is already populated, only add evidence spans (do not overwrite).
    if existing and expected_gauge is not None and gauge is not None and gauge != expected_gauge:
        return warnings

    if not existing and gauge is not None:
        setattr(linear, "needle_gauge", f"{gauge}G")
        warnings.append("AUTO_EBUS_NEEDLE_GAUGE: parsed from note text")

    try:
        from app.common.spans import Span

        evidence = getattr(record, "evidence", None)
        if not isinstance(evidence, dict):
            evidence = {}
        key = "procedures_performed.linear_ebus.needle_gauge"
        if key not in evidence:
            evidence.setdefault(key, []).append(
                Span(text=match.group(0).strip(), start=match.start(), end=match.end(), confidence=0.9)
            )
            record.evidence = evidence
    except Exception:
        pass

    return warnings


_EUS_B_MARKER_RE = re.compile(r"(?i)\bEUS-?B\b")
_EUS_B_FINDINGS_HEADER_RE = re.compile(r"(?im)^\s*EUS-?B\s+Findings\b")
_EUS_B_SITES_HEADER_RE = re.compile(r"(?im)^\s*EUS-?B\s+Sites\s+Sampled\s*:\s*$")
_EUS_B_SITE_LINE_RE = re.compile(r"(?im)^\s*Site\s+\d{1,2}\s*:\s*(?P<rest>.+?)\s*$")
_EUS_B_NEEDLE_GAUGE_RE = re.compile(r"(?i)\b(19|21|22|25)\s*[- ]?(?:g|gauge)\b")
_EUS_B_PASSES_RE = re.compile(
    r"\b(?P<count>\d{1,2})\b[^.\n]{0,80}\b(?:needle\s+passes?|passes?|endoscopic\s+ultrasound\s+guided\s+transbronchial\s+biops(?:y|ies))\b",
    re.IGNORECASE,
)
_EUS_B_ROSE_RE = re.compile(
    r"(?i)\bOverall\s+EUS-?B\b[^.\n]{0,120}\bROSE\b[^.\n]{0,80}:\s*\"?(?P<val>[A-Za-z][^\"]{0,40})\"?"
)


def enrich_eus_b_sampling_details(record: RegistryRecord, full_text: str) -> list[str]:
    """Populate procedures_performed.eus_b.{sites_sampled,needle_gauge,passes,rose_result} when missing."""
    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    eus_b = getattr(procedures, "eus_b", None) if procedures is not None else None
    if eus_b is None or getattr(eus_b, "performed", None) is not True:
        return warnings
    if not full_text:
        return warnings

    header = _EUS_B_FINDINGS_HEADER_RE.search(full_text)
    marker = _EUS_B_MARKER_RE.search(full_text) if header is None else header
    if marker is None:
        return warnings

    section = full_text[marker.start() :]
    base_offset = marker.start()
    changed = False

    existing_sites = getattr(eus_b, "sites_sampled", None)
    if not isinstance(existing_sites, list) or not any(isinstance(s, str) and s.strip() for s in existing_sites):
        sites: list[str] = []
        header = _EUS_B_SITES_HEADER_RE.search(section)
        if header:
            tail = section[header.end() :]
            for match in _EUS_B_SITE_LINE_RE.finditer(tail):
                rest = (match.group("rest") or "").strip()
                if not rest:
                    continue
                site = None
                m_site = re.search(
                    r"(?i)\bthe\s+(?P<site>[^.\n]{3,80}?)(?:\s+was|\s+is|\s+were|\s+on\b|\s+in\b|[.,])",
                    rest,
                )
                if m_site:
                    site = (m_site.group("site") or "").strip()
                elif re.search(r"(?i)\badrenal\b", rest):
                    site = "Left adrenal mass"
                if site:
                    site = site[0].upper() + site[1:] if site else site
                    if site not in sites:
                        sites.append(site)
                if len(sites) >= 6:
                    break

        if not sites and re.search(r"(?i)\bleft\s+adrenal\b", section):
            sites = ["Left adrenal mass"]

        if sites:
            setattr(eus_b, "sites_sampled", sites)
            changed = True

    if getattr(eus_b, "needle_gauge", None) in (None, ""):
        match = _EUS_B_NEEDLE_GAUGE_RE.search(section)
        if match:
            try:
                gauge = int(match.group(1))
            except Exception:
                gauge = None
            if gauge in (19, 21, 22, 25):
                setattr(eus_b, "needle_gauge", f"{gauge}G")
                changed = True
                try:
                    from app.common.spans import Span

                    evidence = getattr(record, "evidence", None)
                    if not isinstance(evidence, dict):
                        evidence = {}
                    evidence.setdefault("procedures_performed.eus_b.needle_gauge", []).append(
                        Span(
                            text=match.group(0).strip(),
                            start=base_offset + match.start(),
                            end=base_offset + match.end(),
                            confidence=0.9,
                        )
                    )
                    record.evidence = evidence
                except Exception:
                    pass

    if getattr(eus_b, "passes", None) in (None, 0):
        match = _EUS_B_PASSES_RE.search(section)
        if match:
            try:
                count = int(match.group("count"))
            except Exception:
                count = None
            if isinstance(count, int) and 1 <= count <= 30:
                setattr(eus_b, "passes", count)
                changed = True
                try:
                    from app.common.spans import Span

                    evidence = getattr(record, "evidence", None)
                    if not isinstance(evidence, dict):
                        evidence = {}
                    evidence.setdefault("procedures_performed.eus_b.passes", []).append(
                        Span(
                            text=match.group(0).strip(),
                            start=base_offset + match.start(),
                            end=base_offset + match.end(),
                            confidence=0.9,
                        )
                    )
                    record.evidence = evidence
                except Exception:
                    pass

    if getattr(eus_b, "rose_result", None) in (None, ""):
        match = _EUS_B_ROSE_RE.search(section)
        if match:
            val = (match.group("val") or "").strip().strip("\"' ")
            if val:
                setattr(eus_b, "rose_result", val)
                changed = True
                try:
                    from app.common.spans import Span

                    evidence = getattr(record, "evidence", None)
                    if not isinstance(evidence, dict):
                        evidence = {}
                    evidence.setdefault("procedures_performed.eus_b.rose_result", []).append(
                        Span(
                            text=match.group(0).strip(),
                            start=base_offset + match.start(),
                            end=base_offset + match.end(),
                            confidence=0.9,
                        )
                    )
                    record.evidence = evidence
                except Exception:
                    pass

    if changed:
        warnings.append("AUTO_EUS_B_DETAIL: populated eus_b sampling details from note text")
    return warnings


_PLEURAL_THORACOSCOPY_BIOPSY_RE = re.compile(
    r"\bbiops(?:y|ies)\b[^.\n]{0,120}\bpleur(?:a|al|e)?\b"
    r"|\bpleur(?:a|al|e)?\b[^.\n]{0,120}\bbiops(?:y|ies)\b",
    re.IGNORECASE,
)


def enrich_medical_thoracoscopy_biopsies_taken(record: RegistryRecord, full_text: str) -> list[str]:
    """Set pleural_procedures.medical_thoracoscopy.biopsies_taken when pleural biopsies are documented."""
    warnings: list[str] = []
    pleural = getattr(record, "pleural_procedures", None)
    thor = getattr(pleural, "medical_thoracoscopy", None) if pleural is not None else None
    if thor is None or getattr(thor, "performed", None) is not True:
        return warnings

    if getattr(thor, "biopsies_taken", None) is True:
        return warnings

    if not full_text:
        return warnings

    match = _PLEURAL_THORACOSCOPY_BIOPSY_RE.search(full_text)
    if not match:
        return warnings

    setattr(thor, "biopsies_taken", True)

    try:
        from app.common.spans import Span

        evidence = getattr(record, "evidence", None)
        if not isinstance(evidence, dict):
            evidence = {}
        evidence.setdefault("pleural_procedures.medical_thoracoscopy.biopsies_taken", []).append(
            Span(
                text=(match.group(0) or "").strip(),
                start=int(match.start()),
                end=int(match.end()),
                confidence=0.9,
            )
        )
        record.evidence = evidence
    except Exception:
        # Evidence is best-effort; do not fail postprocess.
        pass

    warnings.append(
        "AUTO_THORACOSCOPY_BIOPSY: set pleural_procedures.medical_thoracoscopy.biopsies_taken=true from note text"
    )
    return warnings


_OUTCOMES_ABORT_RE = re.compile(
    r"(?i)\b(?:procedure\s+)?(?:aborted|terminated)\b[^.\n]{0,200}"
)
_OUTCOMES_FAIL_RE = re.compile(
    r"(?i)\b(?:unable\s+to|could\s+not|cannot|failed\s+to|unsuccessful|not\s+successful)\b[^.\n]{0,220}"
)
_OUTCOMES_SUBOPTIMAL_RE = re.compile(r"(?i)\bsuboptimal\b")
_OUTCOMES_CONTEXT_RE = re.compile(
    r"(?i)\b(?:navigat|localiz|registration|radial|r-?ebus|probe|view|lesion|target|specimen|sampling|biops)\w*\b"
)
_OUTCOMES_COMPLICATION_KW_RE = re.compile(
    r"(?i)\b(?:bleed|hemorrhag|hypox|desaturat|pneumothorax|air\s*leak|arrhythm|bradycard|tachycard)\w*\b"
)
_OUTCOMES_COMPLICATION_DURATION_RE = re.compile(
    r"(?i)\btotal\s+time\b[^.\n]{0,20}(?:with\s+)?(?P<dur>\d{1,3}\s*min(?:ute)?s?)\b"
)
_OUTCOMES_COMPLICATION_INTERVENTION_RE = re.compile(
    r"(?i)\b(?:treated|managed|controlled)\s+with\b\s*(?P<int>[^.\n]{3,120})"
)
_OUTCOMES_COMPLETION_RE = re.compile(
    r"(?i)\b(?:"
    r"procedure\s+(?:was\s+)?completed"
    r"|procedure\s+was\s+successfully\s+completed"
    r"|patient\s+tolerated\s+the\s+procedure\s+well"
    r"|at\s+the\s+conclusion\s+of\s+the\s+operation[^.\n]{0,80}\bstable\s+condition\b"
    r"|in\s+stable\s+condition(?:\s+at\s+the\s+conclusion)?"
    r")\b"
)


def enrich_procedure_success_status(record: RegistryRecord, full_text: str) -> list[str]:
    """Populate outcomes.procedure_success_status and outcomes.aborted_reason from explicit note language.

    Goal: capture suboptimal/failed/aborted procedural success status with evidence spans for UI highlighting.
    """
    warnings: list[str] = []
    text = full_text or ""
    if not text.strip():
        return warnings

    def _sentence_span(match_start: int, match_end: int) -> tuple[int, int]:
        # Expand match to a sentence-like span for better UI highlighting.
        start = max(
            text.rfind("\n", 0, match_start),
            text.rfind(".", 0, match_start),
            text.rfind(";", 0, match_start),
            text.rfind(":", 0, match_start),
        )
        if start == -1:
            start = 0
        else:
            start = min(len(text), start + 1)
        end_candidates = [pos for pos in (text.find("\n", match_end), text.find(".", match_end)) if pos != -1]
        end = min(end_candidates) if end_candidates else len(text)
        end = min(len(text), end + (1 if end < len(text) and text[end] == "." else 0))
        return start, end

    status: str | None = None
    reason: str | None = None
    ev_start: int | None = None
    ev_end: int | None = None
    outcomes_in = getattr(record, "outcomes", None)
    procedure_completed_flag = bool(
        getattr(outcomes_in, "procedure_completed", False) if outcomes_in is not None else False
    )
    completion_language_present = bool(_OUTCOMES_COMPLETION_RE.search(text))

    aborted_match = _OUTCOMES_ABORT_RE.search(text)
    if aborted_match:
        status = "Aborted"
        s, e = _sentence_span(aborted_match.start(), aborted_match.end())
        snippet = text[s:e].strip()
        if snippet:
            reason = snippet[:240]
            ev_start, ev_end = s, e
        reason_match = re.search(
            r"(?i)\b(?:due\s+to|because|secondary\s+to)\b\s*(?P<reason>[^.\n]{3,200})",
            aborted_match.group(0) or "",
        )
        if reason_match:
            reason = (reason_match.group("reason") or "").strip()[:240] or reason
    else:
        fail_match = _OUTCOMES_FAIL_RE.search(text)
        if fail_match:
            status = "Failed"
            s, e = _sentence_span(fail_match.start(), fail_match.end())
            snippet = text[s:e].strip()
            if snippet:
                reason = snippet[:240]
                ev_start, ev_end = s, e
            # If the note documents a failed sub-step but the overall procedure
            # is completed, classify as partial success instead of full failure.
            if procedure_completed_flag or completion_language_present:
                status = "Partial success"
        else:
            # Prefer the most specific "radial/probe view ... suboptimal" evidence when present.
            radial_suboptimal = re.search(
                r"(?i)\bradial\b[^.\n]{0,120}\bsuboptimal\b|\bsuboptimal\b[^.\n]{0,120}\bradial\b",
                text,
            )
            candidate = radial_suboptimal
            if candidate is None:
                for match in _OUTCOMES_SUBOPTIMAL_RE.finditer(text):
                    s, e = _sentence_span(match.start(), match.end())
                    window = text[s:e]
                    if _OUTCOMES_CONTEXT_RE.search(window):
                        candidate = match
                        break

            if candidate is not None:
                s, e = _sentence_span(candidate.start(), candidate.end())
                snippet = text[s:e].strip()
                if snippet:
                    status = "Partial success"
                    reason = snippet[:240]
                    ev_start, ev_end = s, e

    if status is None or ev_start is None or ev_end is None or ev_end <= ev_start:
        return warnings

    outcomes = getattr(record, "outcomes", None)
    current_status = getattr(outcomes, "procedure_success_status", None) if outcomes is not None else None
    current_reason = getattr(outcomes, "aborted_reason", None) if outcomes is not None else None
    current_old_reason = getattr(outcomes, "procedure_aborted_reason", None) if outcomes is not None else None

    changed = False

    # Ensure outcomes is a validated object when missing.
    if outcomes is None:
        record_data = record.model_dump()
        record_data["outcomes"] = {}
        record.outcomes = RegistryRecord.model_validate(record_data).outcomes
        outcomes = getattr(record, "outcomes", None)

    if outcomes is None:
        return warnings

    if current_status in (None, "", "Unknown"):
        setattr(outcomes, "procedure_success_status", status)
        changed = True

    if reason and current_reason in (None, ""):
        setattr(outcomes, "aborted_reason", reason)
        changed = True

    # Compatibility: keep legacy field aligned when we have an explicit abort.
    if status == "Aborted" and reason and current_old_reason in (None, ""):
        setattr(outcomes, "procedure_aborted_reason", reason)
        changed = True

    if current_reason in (None, "") and current_old_reason not in (None, ""):
        setattr(outcomes, "aborted_reason", current_old_reason)
        changed = True

    # Evidence spans (best-effort; do not fail postprocess).
    try:
        from app.common.spans import Span

        evidence = getattr(record, "evidence", None)
        if not isinstance(evidence, dict):
            evidence = {}
        snippet = text[ev_start:ev_end].strip()
        if snippet:
            evidence.setdefault("outcomes.procedure_success_status", []).append(
                Span(text=snippet, start=int(ev_start), end=int(ev_end), confidence=0.9)
            )
            evidence.setdefault("outcomes.aborted_reason", []).append(
                Span(text=snippet, start=int(ev_start), end=int(ev_end), confidence=0.9)
            )
            if status == "Aborted":
                evidence.setdefault("outcomes.procedure_aborted_reason", []).append(
                    Span(text=snippet, start=int(ev_start), end=int(ev_end), confidence=0.9)
                )
        record.evidence = evidence
    except Exception:
        pass

    if changed:
        warnings.append(f"AUTO_OUTCOMES_STATUS: set outcomes.procedure_success_status={status!r}")
    return warnings


def enrich_outcomes_complication_details(record: RegistryRecord, full_text: str) -> list[str]:
    """Populate outcomes.complication_duration and outcomes.complication_intervention from explicit text."""
    warnings: list[str] = []
    text = full_text or ""
    if not text.strip():
        return warnings

    outcomes = getattr(record, "outcomes", None)
    if outcomes is None:
        record_data = record.model_dump()
        record_data["outcomes"] = {}
        record.outcomes = RegistryRecord.model_validate(record_data).outcomes
        outcomes = getattr(record, "outcomes", None)
    if outcomes is None:
        return warnings

    duration_existing = getattr(outcomes, "complication_duration", None)
    intervention_existing = getattr(outcomes, "complication_intervention", None)
    changed = False

    ev_start: int | None = None
    ev_end: int | None = None

    if duration_existing in (None, ""):
        for match in _OUTCOMES_COMPLICATION_DURATION_RE.finditer(text):
            # Require a nearby complication keyword within the same sentence window.
            start = max(
                text.rfind("\n", 0, match.start()),
                text.rfind(".", 0, match.start()),
                text.rfind(";", 0, match.start()),
            )
            start = 0 if start == -1 else start + 1
            end_candidates = [pos for pos in (text.find("\n", match.end()), text.find(".", match.end())) if pos != -1]
            end = min(end_candidates) if end_candidates else len(text)
            sentence = text[start:end]
            if not _OUTCOMES_COMPLICATION_KW_RE.search(sentence):
                continue
            dur = (match.group("dur") or "").strip()
            if dur:
                setattr(outcomes, "complication_duration", dur)
                changed = True
                ev_start, ev_end = start, end
                break

    if intervention_existing in (None, ""):
        for match in _OUTCOMES_COMPLICATION_INTERVENTION_RE.finditer(text):
            start = max(
                text.rfind("\n", 0, match.start()),
                text.rfind(".", 0, match.start()),
                text.rfind(";", 0, match.start()),
            )
            start = 0 if start == -1 else start + 1
            end_candidates = [pos for pos in (text.find("\n", match.end()), text.find(".", match.end())) if pos != -1]
            end = min(end_candidates) if end_candidates else len(text)
            sentence = text[start:end]
            if not _OUTCOMES_COMPLICATION_KW_RE.search(sentence):
                continue
            val = (match.group("int") or "").strip().strip(" :;-\"'")[:180]
            if val:
                setattr(outcomes, "complication_intervention", val)
                changed = True
                if ev_start is None or ev_end is None:
                    ev_start, ev_end = start, end
                break

    if ev_start is not None and ev_end is not None and ev_end > ev_start:
        try:
            from app.common.spans import Span

            evidence = getattr(record, "evidence", None)
            if not isinstance(evidence, dict):
                evidence = {}
            snippet = text[ev_start:ev_end].strip()
            if snippet:
                evidence.setdefault("outcomes.complication_duration", []).append(
                    Span(text=snippet, start=int(ev_start), end=int(ev_end), confidence=0.9)
                )
                evidence.setdefault("outcomes.complication_intervention", []).append(
                    Span(text=snippet, start=int(ev_start), end=int(ev_end), confidence=0.9)
                )
            record.evidence = evidence
        except Exception:
            pass

    if changed:
        warnings.append("AUTO_COMPLICATION_DETAILS: populated outcomes complication duration/intervention from note text")
    return warnings


_BAL_STANDARD_LINE_RE = re.compile(
    r"(?i)\b(?:bronch(?:ial)?\s+alveolar\s+lavage|broncho[-\s]?alveolar\s+lavage|BAL)\b"
    r"[^.\n]{0,80}\b(?:was\s+)?performed\b[^.\n]{0,80}\b(?:at|in)\b\s+(?P<loc>[^.\n]{3,220})"
)
_BAL_MINI_PREFIX_RE = re.compile(r"(?i)\bmini\s+(?:bronch(?:ial)?\s+alveolar\s+lavage|bal)\b")
_BAL_INSTILLED_RE = re.compile(r"(?i)\binstilled\s+(?P<num>\d{1,4})\s*(?:cc|ml)\b")
_BAL_RETURN_RE = re.compile(
    r"(?i)\b(?:suction\s*returned(?:\s+with)?|returned\s+with|recovered)\s+(?P<num>\d{1,4})\s*(?:cc|ml)\b"
)


def enrich_bal_from_procedure_detail(record: RegistryRecord, full_text: str) -> list[str]:
    """Prefer explicit standard-BAL documentation over mini-BAL snippets when unambiguous."""
    warnings: list[str] = []
    procedures = getattr(record, "procedures_performed", None)
    bal = getattr(procedures, "bal", None) if procedures is not None else None
    if bal is None or getattr(bal, "performed", None) is not True:
        return warnings
    if not full_text:
        return warnings

    candidates: list[dict[str, object]] = []
    for match in _BAL_STANDARD_LINE_RE.finditer(full_text):
        start = match.start()
        end = match.end()
        prefix = full_text[max(0, start - 40) : start]
        if _BAL_MINI_PREFIX_RE.search(prefix):
            continue

        loc_raw = (match.group("loc") or "").strip().strip(" ,;:-")
        if not loc_raw:
            continue
        if len(loc_raw) > 180:
            loc_raw = loc_raw[:180].rsplit(" ", 1)[0].strip() or loc_raw[:180].strip()

        window = full_text[start : min(len(full_text), start + 420)]
        instilled = None
        recovered = None
        instilled_span: tuple[int, int] | None = None
        recovered_span: tuple[int, int] | None = None
        m_inst = _BAL_INSTILLED_RE.search(window)
        if m_inst:
            try:
                instilled = float(int(m_inst.group("num")))
            except Exception:
                instilled = None
            instilled_span = (start + m_inst.start(), start + m_inst.end())
        m_ret = _BAL_RETURN_RE.search(window)
        if m_ret:
            try:
                recovered = float(int(m_ret.group("num")))
            except Exception:
                recovered = None
            recovered_span = (start + m_ret.start(), start + m_ret.end())

        candidates.append(
            {
                "loc": loc_raw,
                "instilled": instilled,
                "recovered": recovered,
                "instilled_span": instilled_span,
                "recovered_span": recovered_span,
                "span": (start, end),
            }
        )

    if not candidates:
        # Even when the note lacks an explicit "BAL ... performed at <location>" line,
        # attach evidence spans for already-populated volume fields so the UI can
        # highlight the supporting text (e.g., "instilled 40 cc").
        try:
            from app.common.spans import Span

            evidence = getattr(record, "evidence", None)
            if not isinstance(evidence, dict):
                evidence = {}

            instilled_existing = getattr(bal, "volume_instilled_ml", None)
            if (
                instilled_existing not in (None, "", 0)
                and "procedures_performed.bal.volume_instilled_ml" not in evidence
            ):
                try:
                    vol_int = int(float(instilled_existing))
                except Exception:
                    vol_int = None
                if vol_int is not None:
                    m = re.search(
                        rf"(?i)\b(?:instilled|infused)\s+{vol_int}\s*(?:cc|ml)\b",
                        full_text,
                    ) or re.search(
                        rf"(?i)\b{vol_int}\s*(?:cc|ml)\b[^.\n]{{0,40}}\b(?:ns\s+)?(?:instilled|infused)\b",
                        full_text,
                    )
                    if m:
                        evidence.setdefault("procedures_performed.bal.volume_instilled_ml", []).append(
                            Span(text=m.group(0).strip(), start=m.start(), end=m.end(), confidence=0.9)
                        )

            recovered_existing = getattr(bal, "volume_recovered_ml", None)
            if (
                recovered_existing not in (None, "", 0)
                and "procedures_performed.bal.volume_recovered_ml" not in evidence
            ):
                try:
                    vol_int = int(float(recovered_existing))
                except Exception:
                    vol_int = None
                if vol_int is not None:
                    m = re.search(
                        rf"(?i)\b(?:returned\s+with|suction\s*returned(?:\s+with)?|recovered)\s+{vol_int}\s*(?:cc|ml)\b",
                        full_text,
                    ) or re.search(
                        rf"(?i)\b{vol_int}\s*(?:cc|ml)\b\s*(?:return(?:ed)?|recovered)\b",
                        full_text,
                    )
                    if m:
                        evidence.setdefault("procedures_performed.bal.volume_recovered_ml", []).append(
                            Span(text=m.group(0).strip(), start=m.start(), end=m.end(), confidence=0.9)
                        )

            if evidence:
                record.evidence = evidence
        except Exception:
            pass

        return warnings

    # Allow multi-site BAL when volumes are consistent (common templated notes):
    # - multiple "performed at <location>" statements
    # - shared instilled/recovered volumes
    locs_in_order: list[str] = []
    for c in candidates:
        loc_val = str(c.get("loc") or "").strip()
        if loc_val and loc_val not in locs_in_order:
            locs_in_order.append(loc_val)

    instilled_values = {c.get("instilled") for c in candidates if isinstance(c.get("instilled"), float)}
    recovered_values = {c.get("recovered") for c in candidates if isinstance(c.get("recovered"), float)}

    if len(locs_in_order) > 1 and (len(instilled_values) > 1 or len(recovered_values) > 1):
        warnings.append(
            "AMBIGUOUS_BAL_DETAIL: multiple standard BAL candidates in note; not overriding bal fields"
        )
        return warnings

    loc = "; ".join(locs_in_order).strip()
    if len(loc) > 180:
        loc = loc[:180].rsplit(" ", 1)[0].strip() or loc[:180].strip()

    instilled_val = next(iter(instilled_values), None) if instilled_values else None
    recovered_val = next(iter(recovered_values), None) if recovered_values else None

    changed = False
    existing_loc = getattr(bal, "location", None)
    if isinstance(existing_loc, str):
        existing_loc_norm = existing_loc.strip()
    else:
        existing_loc_norm = ""

    if loc and loc != existing_loc_norm:
        setattr(bal, "location", loc)
        changed = True

    if isinstance(instilled_val, float) and instilled_val > 0:
        existing_instilled = getattr(bal, "volume_instilled_ml", None)
        if existing_instilled is None or float(existing_instilled) != instilled_val:
            setattr(bal, "volume_instilled_ml", instilled_val)
            changed = True

    if isinstance(recovered_val, float) and recovered_val >= 0:
        existing_recovered = getattr(bal, "volume_recovered_ml", None)
        if existing_recovered is None or float(existing_recovered) != recovered_val:
            setattr(bal, "volume_recovered_ml", recovered_val)
            changed = True

    try:
        from app.common.spans import Span

        evidence = getattr(record, "evidence", None)
        if not isinstance(evidence, dict):
            evidence = {}

        for cand in candidates:
            start, end = cand.get("span") or (None, None)
            if isinstance(start, int) and isinstance(end, int) and end > start:
                evidence.setdefault("procedures_performed.bal.location", []).append(
                    Span(text=full_text[start:end].strip(), start=start, end=end, confidence=0.9)
                )

        instilled_key = "procedures_performed.bal.volume_instilled_ml"
        if instilled_key not in evidence:
            for cand in candidates:
                span = cand.get("instilled_span")
                if not (isinstance(span, tuple) and len(span) == 2):
                    continue
                start, end = span
                if isinstance(start, int) and isinstance(end, int) and end > start:
                    evidence.setdefault(instilled_key, []).append(
                        Span(text=full_text[start:end].strip(), start=start, end=end, confidence=0.9)
                    )
                    break

        recovered_key = "procedures_performed.bal.volume_recovered_ml"
        if recovered_key not in evidence:
            for cand in candidates:
                span = cand.get("recovered_span")
                if not (isinstance(span, tuple) and len(span) == 2):
                    continue
                start, end = span
                if isinstance(start, int) and isinstance(end, int) and end > start:
                    evidence.setdefault(recovered_key, []).append(
                        Span(text=full_text[start:end].strip(), start=start, end=end, confidence=0.9)
                    )
                    break

        # Fallback: when we have numeric values but no per-candidate volume span,
        # search the full text for a matching "instilled/returned" phrase.
        instilled_existing = getattr(bal, "volume_instilled_ml", None)
        if instilled_existing not in (None, "", 0) and instilled_key not in evidence:
            try:
                vol_int = int(float(instilled_existing))
            except Exception:
                vol_int = None
            if vol_int is not None:
                m = re.search(
                    rf"(?i)\b(?:instilled|infused)\s+{vol_int}\s*(?:cc|ml)\b",
                    full_text,
                ) or re.search(
                    rf"(?i)\b{vol_int}\s*(?:cc|ml)\b[^.\n]{{0,40}}\b(?:ns\s+)?(?:instilled|infused)\b",
                    full_text,
                )
                if m:
                    evidence.setdefault(instilled_key, []).append(
                        Span(text=m.group(0).strip(), start=m.start(), end=m.end(), confidence=0.9)
                    )

        recovered_existing = getattr(bal, "volume_recovered_ml", None)
        if recovered_existing not in (None, "", 0) and recovered_key not in evidence:
            try:
                vol_int = int(float(recovered_existing))
            except Exception:
                vol_int = None
            if vol_int is not None:
                m = re.search(
                    rf"(?i)\b(?:returned\s+with|suction\s*returned(?:\s+with)?|recovered)\s+{vol_int}\s*(?:cc|ml)\b",
                    full_text,
                ) or re.search(
                    rf"(?i)\b{vol_int}\s*(?:cc|ml)\b\s*(?:return(?:ed)?|recovered)\b",
                    full_text,
                )
                if m:
                    evidence.setdefault(recovered_key, []).append(
                        Span(text=m.group(0).strip(), start=m.start(), end=m.end(), confidence=0.9)
                    )

        record.evidence = evidence
    except Exception:
        pass

    if not changed:
        return warnings

    warnings.append("AUTO_BAL_DETAIL: set BAL location/volumes from explicit 'performed at' statement")
    return warnings
