"""Entity type definitions and categorization for granular NER."""

from enum import Enum
from typing import Dict, Set


class EntityCategory(str, Enum):
    """Top-level entity categories."""
    ANATOMY = "ANAT"
    DEVICE = "DEV"
    PROCEDURE = "PROC"
    MEASUREMENT = "MEAS"
    OBSERVATION = "OBS"
    OUTCOME = "OUTCOME"
    CONTEXT = "CTX"
    OTHER = "OTHER"


# Entity type to category mapping
ENTITY_CATEGORIES: Dict[str, EntityCategory] = {
    # Anatomy
    "ANAT_AIRWAY": EntityCategory.ANATOMY,
    "ANAT_LN_STATION": EntityCategory.ANATOMY,
    "ANAT_LUNG_LOC": EntityCategory.ANATOMY,
    "ANAT_PLEURA": EntityCategory.ANATOMY,

    # Devices
    "DEV_INSTRUMENT": EntityCategory.DEVICE,
    "DEV_STENT": EntityCategory.DEVICE,
    "DEV_CATHETER": EntityCategory.DEVICE,
    "DEV_NEEDLE": EntityCategory.DEVICE,
    "DEV_STENT_MATERIAL": EntityCategory.DEVICE,
    "DEV_STENT_SIZE": EntityCategory.DEVICE,
    "DEV_CATHETER_SIZE": EntityCategory.DEVICE,
    "DEV_VALVE": EntityCategory.DEVICE,
    "DEV_DEVICE": EntityCategory.DEVICE,

    # Procedures
    "PROC_METHOD": EntityCategory.PROCEDURE,
    "PROC_ACTION": EntityCategory.PROCEDURE,
    "PROC_MEDICATION": EntityCategory.PROCEDURE,

    # Measurements
    "MEAS_SIZE": EntityCategory.MEASUREMENT,
    "MEAS_VOL": EntityCategory.MEASUREMENT,
    "MEAS_COUNT": EntityCategory.MEASUREMENT,
    "MEAS_AIRWAY_DIAM": EntityCategory.MEASUREMENT,
    "MEAS_TIME": EntityCategory.MEASUREMENT,
    "MEAS_PLEURAL_DRAIN": EntityCategory.MEASUREMENT,
    "MEAS_PRESS": EntityCategory.MEASUREMENT,
    "MEAS_OTHER": EntityCategory.MEASUREMENT,
    "MEAS_ENERGY": EntityCategory.MEASUREMENT,
    "MEAS_TEMP": EntityCategory.MEASUREMENT,

    # Observations
    "OBS_LESION": EntityCategory.OBSERVATION,
    "OBS_FINDING": EntityCategory.OBSERVATION,
    "OBS_ROSE": EntityCategory.OBSERVATION,

    # Outcomes
    "OUTCOME_COMPLICATION": EntityCategory.OUTCOME,
    "OUTCOME_AIRWAY_LUMEN_PRE": EntityCategory.OUTCOME,
    "OUTCOME_AIRWAY_LUMEN_POST": EntityCategory.OUTCOME,
    "OUTCOME_SYMPTOMS": EntityCategory.OUTCOME,
    "OUTCOME_PLEURAL": EntityCategory.OUTCOME,

    # Context
    "CTX_TIME": EntityCategory.CONTEXT,
    "CTX_HISTORICAL": EntityCategory.CONTEXT,
    "CTX_INDICATION": EntityCategory.CONTEXT,
    "CTX_STENT_PRESENT": EntityCategory.CONTEXT,
    "NEG_STENT": EntityCategory.CONTEXT,

    # Other
    "MEDICATION": EntityCategory.OTHER,
    "LATERALITY": EntityCategory.OTHER,
    "SPECIMEN": EntityCategory.OTHER,
}


# Entity types that are directly relevant for CPT code derivation
CPT_RELEVANT_ENTITIES: Set[str] = {
    # Critical for EBUS station counting (31652 vs 31653)
    "ANAT_LN_STATION",

    # Critical for procedure detection
    "PROC_METHOD",
    "PROC_ACTION",

    # Critical for lobe counting (TBBx codes)
    "ANAT_LUNG_LOC",

    # For ROSE outcomes
    "OBS_ROSE",

    # For valve counting (BLVR codes)
    "DEV_VALVE",
    "MEAS_COUNT",

    # For stent codes
    "DEV_STENT",
    "CTX_STENT_PRESENT",
    "NEG_STENT",

    # For complications
    "OUTCOME_COMPLICATION",
}


# Action keywords that indicate sampling (vs inspection)
SAMPLING_ACTION_KEYWORDS: Set[str] = {
    "aspiration", "aspirated", "aspirate",
    "fna", "tbna",
    "biopsy", "biopsied", "biopsies",
    "sampled", "sample", "sampling",
    "needle", "needles",
    "passes", "pass",
    "fnb",
    "core",
    "forceps",
}


# Action keywords that indicate inspection only
INSPECTION_ACTION_KEYWORDS: Set[str] = {
    "viewed", "visualized", "seen", "observed",
    "inspected", "inspection",
    "patent", "normal",
    "surveyed",
    "examined",
}


# Valid EBUS station names (canonical set)
VALID_LN_STATIONS: Set[str] = {
    "2R", "2L",
    "4R", "4L",
    "5",
    "7",
    "8",
    "9",
    "10R", "10L",
    "11R", "11L", "11Rs", "11Ls", "11Ri", "11Li",
}


# Lobe codes for transbronchial biopsy
LOBE_CODES: Set[str] = {
    "RUL", "RML", "RLL",
    "LUL", "LLL",
    "Lingula",
}


def get_entity_category(entity_type: str) -> EntityCategory:
    """Get the category for an entity type."""
    return ENTITY_CATEGORIES.get(entity_type, EntityCategory.OTHER)


def is_cpt_relevant(entity_type: str) -> bool:
    """Check if an entity type is relevant for CPT derivation."""
    return entity_type in CPT_RELEVANT_ENTITIES


def normalize_station(text: str) -> str | None:
    """
    Normalize a lymph node station string to canonical format.

    Returns None if not a valid station.
    """
    # Remove common prefixes/suffixes
    text = text.strip().upper()

    # Handle "station X" or "level X" patterns
    for prefix in ["STATION ", "LEVEL ", "LN "]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    # Map common variations
    variations = {
        "2 RIGHT": "2R", "2 LEFT": "2L",
        "4 RIGHT": "4R", "4 LEFT": "4L",
        "10 RIGHT": "10R", "10 LEFT": "10L",
        "11 RIGHT": "11R", "11 LEFT": "11L",
        "11RS": "11Rs", "11LS": "11Ls",
        "11RI": "11Ri", "11LI": "11Li",
        "SUBCARINAL": "7",
        "PARA-AORTIC": "5",
        "AP WINDOW": "5",
    }

    text = variations.get(text, text)

    # Validate against known stations
    if text in VALID_LN_STATIONS:
        return text

    # Try case-insensitive match
    for station in VALID_LN_STATIONS:
        if text.upper() == station.upper():
            return station

    return None


def normalize_lobe(text: str) -> str | None:
    """
    Normalize a lung lobe string to canonical format.

    Returns None if not a valid lobe.
    """
    text = text.strip().upper()

    # Map full names to codes
    full_names = {
        "RIGHT UPPER LOBE": "RUL",
        "RIGHT MIDDLE LOBE": "RML",
        "RIGHT LOWER LOBE": "RLL",
        "LEFT UPPER LOBE": "LUL",
        "LEFT LOWER LOBE": "LLL",
        "LINGULA": "Lingula",
        "RUL": "RUL",
        "RML": "RML",
        "RLL": "RLL",
        "LUL": "LUL",
        "LLL": "LLL",
    }

    return full_names.get(text)
