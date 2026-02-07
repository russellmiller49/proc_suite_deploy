"""Registry Enum Normalization Layer.

This module provides normalization functions to convert noisy/variant enum values
into the canonical values expected by Pydantic registry models. This runs BEFORE
Pydantic validation to prevent validation errors from common variations.

Normalization targets identified from v2.8 validation runs:
- Gender: 'M'/'F' -> 'Male'/'Female'
- Bronchus sign: True/False -> 'Positive'/'Negative'
- Nav imaging verification: 'Cone Beam CT' -> 'CBCT'
- Pleurodesis agent: contains 'talc' -> 'Talc Slurry'
- IPC action: 'insert', 'placement' -> 'Insertion'
- Thoracoscopy forceps: 'Biopsy forceps' -> 'Rigid forceps'
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# GENDER NORMALIZATION
# =============================================================================

GENDER_MAP: Dict[str, str] = {
    "m": "Male",
    "male": "Male",
    "man": "Male",
    "f": "Female",
    "female": "Female",
    "woman": "Female",
}


def normalize_gender(value: Any) -> Optional[str]:
    """Normalize gender value to canonical enum.

    Args:
        value: Raw gender value (e.g., 'M', 'male', 'Female ')

    Returns:
        'Male', 'Female', 'Unknown', or None if input is None/empty
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        return GENDER_MAP.get(normalized, "Unknown")
    return "Unknown"


# =============================================================================
# BRONCHUS SIGN NORMALIZATION
# =============================================================================

def normalize_bronchus_sign(value: Any) -> Optional[str]:
    """Normalize bronchus sign value to 'Positive'/'Negative'.

    Args:
        value: Raw value (True, False, 'true', 'false', 'positive', etc.)

    Returns:
        'Positive', 'Negative', or None
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return "Positive" if value else "Negative"

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "positive", "present", "1"):
            return "Positive"
        if normalized in ("false", "no", "negative", "absent", "0"):
            return "Negative"
        if not normalized:
            return None

    return None


# =============================================================================
# NAV IMAGING VERIFICATION NORMALIZATION
# =============================================================================

NAV_IMAGING_MAP: Dict[str, str] = {
    "cone beam ct": "CBCT",
    "cone-beam ct": "CBCT",
    "cone beam": "CBCT",
    "conebeam ct": "CBCT",
    "conebeam": "CBCT",
    "cbct": "CBCT",
    "fluoro": "Fluoroscopy",
    "fluoroscopy": "Fluoroscopy",
    "fluoroscopic": "Fluoroscopy",
    "c-arm": "Fluoroscopy",
    "radial ebus": "Radial EBUS",
    "radial probe": "Radial EBUS",
    "augmented fluoroscopy": "Augmented fluoroscopy",
    "augmented fluoro": "Augmented fluoroscopy",
    "none": "None",
}


def normalize_nav_imaging_verification(value: Any) -> Optional[str]:
    """Normalize nav imaging verification to canonical enum.

    Args:
        value: Raw value (e.g., 'Cone Beam CT', 'CBCT', 'fluoro')

    Returns:
        Canonical value ('CBCT', 'Fluoroscopy', 'None') or original if no match
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        return NAV_IMAGING_MAP.get(normalized, value.strip())
    return str(value)


# =============================================================================
# PLEURODESIS AGENT NORMALIZATION
# =============================================================================

def normalize_pleurodesis_agent(value: Any) -> Optional[str]:
    """Normalize pleurodesis agent to canonical value.

    Args:
        value: Raw value (e.g., 'Talc', 'talc slurry', 'Talc Slurry')

    Returns:
        'Talc Slurry', 'Doxycycline', 'Bleomycin', or original if no match
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if "talc" in normalized:
            return "Talc Slurry"
        if "doxycycline" in normalized:
            return "Doxycycline"
        if "bleomycin" in normalized:
            return "Bleomycin"
        return value.strip()
    return str(value)


# =============================================================================
# IPC ACTION NORMALIZATION
# =============================================================================

def normalize_ipc_action(value: Any) -> Optional[str]:
    """Normalize IPC (tunneled catheter) action to canonical enum.

    Args:
        value: Raw value (e.g., 'insert', 'placement', 'remove', 'tPA')

    Returns:
        'Insertion', 'Removal', 'Fibrinolytic instillation', or original
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if any(term in normalized for term in ["insert", "placement", "placed", "tunneled catheter"]):
            return "Insertion"
        if any(term in normalized for term in ["remove", "removal", "pulled"]):
            return "Removal"
        if any(term in normalized for term in ["tpa", "fibrinolytic", "alteplase"]):
            return "Fibrinolytic instillation"
        return value.strip()
    return str(value)


# =============================================================================
# FORCEPS TYPE NORMALIZATION (THORACOSCOPY)
# =============================================================================

def normalize_forceps_type(value: Any, context: str = "") -> Optional[str]:
    """Normalize forceps type for thoracoscopy/bronchoscopy.

    Args:
        value: Raw value (e.g., 'Biopsy forceps', 'rigid forceps')
        context: Context hint ('thoracoscopy', 'bronchoscopy')

    Returns:
        'Rigid forceps', 'Standard', 'Cryoprobe', or original
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None

        # Thoracoscopy context - rigid forceps
        if context.lower() == "thoracoscopy" or "rigid" in normalized:
            if "biopsy" in normalized or "forceps" in normalized:
                return "Rigid forceps"

        if "cryo" in normalized:
            return "Cryoprobe"

        if "standard" in normalized or "conventional" in normalized:
            return "Standard"

        return value.strip()
    return str(value)


# =============================================================================
# SEDATION TYPE NORMALIZATION
# =============================================================================

SEDATION_MAP: Dict[str, str] = {
    "general": "General",
    "general anesthesia": "General",
    "ga": "General",
    "moderate": "Moderate",
    "moderate sedation": "Moderate",
    "conscious sedation": "Moderate",
    "mac": "MAC",
    "monitored anesthesia care": "MAC",
    "local": "Local Only",
    "local only": "Local Only",
    "local anesthesia": "Local Only",
    "topical": "Local Only",
    "none": "None",
    "awake": "None",
}


def normalize_sedation_type(value: Any) -> Optional[str]:
    """Normalize sedation type to canonical enum.

    Args:
        value: Raw value (e.g., 'General anesthesia', 'MAC', 'local')

    Returns:
        'General', 'Moderate', 'MAC', 'Local Only', 'None', or original
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        return SEDATION_MAP.get(normalized, value.strip())
    return str(value)


# =============================================================================
# AIRWAY TYPE NORMALIZATION
# =============================================================================

AIRWAY_MAP: Dict[str, str] = {
    "ett": "ETT",
    "endotracheal tube": "ETT",
    "endotracheal": "ETT",
    "intubated": "ETT",
    "lma": "LMA",
    "laryngeal mask": "LMA",
    "rigid": "Rigid",
    "rigid bronchoscope": "Rigid",
    "trach": "Tracheostomy",
    "tracheostomy": "Tracheostomy",
    "native": "Native",
    "natural": "Native",
    "none": "Native",
    "nasal": "Native",
    "oral": "Native",
}


def normalize_airway_type(value: Any) -> Optional[str]:
    """Normalize airway type to canonical enum.

    Args:
        value: Raw value (e.g., 'ETT', 'endotracheal tube', 'LMA')

    Returns:
        'ETT', 'LMA', 'Rigid', 'Tracheostomy', 'Native', or original
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        return AIRWAY_MAP.get(normalized, value.strip())
    return str(value)


# =============================================================================
# ASA CLASS NORMALIZATION
# =============================================================================

def normalize_asa_class(value: Any) -> Optional[int]:
    """Normalize ASA class to integer 1-6.

    Args:
        value: Raw value (e.g., 'II', '3', 'ASA 4', 'III-E')

    Returns:
        Integer 1-6 or None
    """
    if value is None:
        return None

    # Already an int
    if isinstance(value, int):
        if 1 <= value <= 6:
            return value
        return None

    if isinstance(value, str):
        normalized = value.strip().upper()
        if not normalized:
            return None

        # Remove 'ASA' prefix and emergency suffix
        normalized = normalized.replace("ASA", "").replace("-E", "").replace("E", "").strip()

        # Roman numeral mapping
        roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
        if normalized in roman_map:
            return roman_map[normalized]

        # Try direct integer parse
        try:
            val = int(normalized)
            if 1 <= val <= 6:
                return val
        except ValueError:
            pass

    return None


# =============================================================================
# CAO ETIOLOGY NORMALIZATION
# =============================================================================

def normalize_cao_etiology(value: Any) -> Optional[str]:
    """Normalize CAO etiology to canonical enum.

    Args:
        value: Raw value (e.g., 'Benign - other', 'Infectious', 'malignancy')

    Returns:
        'Malignant', 'Benign', 'Other', or original
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if any(term in normalized for term in ["malignant", "malignancy", "cancer", "tumor", "carcinoma"]):
            return "Malignant"
        if "benign" in normalized:
            if "other" in normalized:
                return "Other"
            return "Benign"
        if "infectious" in normalized:
            return "Other"
        if "stricture" in normalized or "stenosis" in normalized:
            return "Benign"
        return value.strip()
    return str(value)


# =============================================================================
# PLEURAL BIOPSY LOCATION NORMALIZATION
# =============================================================================

def normalize_pleural_biopsy_location(value: Any) -> Optional[str]:
    """Normalize pleural biopsy location to canonical enum.

    Args:
        value: Raw value (e.g., 'Pleural space', 'parietal pleura')

    Returns:
        'Parietal pleura - chest wall', 'Visceral pleura', or original
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if "parietal" in normalized or "chest wall" in normalized:
            return "Parietal pleura - chest wall"
        if "visceral" in normalized:
            return "Visceral pleura"
        if "diaphragm" in normalized:
            return "Parietal pleura - diaphragm"
        if "mediastinal" in normalized:
            return "Parietal pleura - mediastinal"
        if normalized in ("pleural space", "pleura", "pleural"):
            return "Parietal pleura - chest wall"  # Default for unspecified
        return value.strip()
    return str(value)


# =============================================================================
# BLEEDING SEVERITY NORMALIZATION
# =============================================================================

def normalize_bleeding_severity(value: Any) -> Optional[str]:
    """Normalize bleeding severity to canonical enum.

    Args:
        value: Raw value (e.g., 'mild', 'Mild (<50mL)', 'no significant bleeding')

    Returns:
        'None', 'Mild', 'Mild (<50mL)', 'Moderate', 'Severe', or original
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if "no" in normalized and ("bleeding" in normalized or "significant" in normalized):
            return "None"
        if "none" in normalized or "minimal" in normalized:
            return "None"
        if "mild" in normalized:
            if "<50" in normalized or "50ml" in normalized.replace(" ", ""):
                return "Mild (<50mL)"
            return "Mild"
        if "moderate" in normalized:
            return "Moderate"
        if "severe" in normalized or "massive" in normalized:
            return "Severe"
        return value.strip()
    return str(value)


# =============================================================================
# MAIN NORMALIZATION FUNCTION
# =============================================================================

def normalize_registry_enums(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize free-text/synonym values into constrained enum vocab.

    This function should be called BEFORE Pydantic validation to convert
    common variations into canonical enum values.

    Args:
        raw: Raw registry dict from LLM extraction

    Returns:
        Normalized registry dict with canonical enum values
    """
    # Make a copy to avoid modifying the original
    data = dict(raw)

    # Top-level field normalization
    if "gender" in data:
        data["gender"] = normalize_gender(data["gender"])

    if "patient_gender" in data:
        data["patient_gender"] = normalize_gender(data["patient_gender"])

    if "bronchus_sign" in data:
        data["bronchus_sign"] = normalize_bronchus_sign(data["bronchus_sign"])

    if "bronchus_sign_present" in data:
        data["bronchus_sign_present"] = normalize_bronchus_sign(data["bronchus_sign_present"])

    if "nav_imaging_verification" in data:
        data["nav_imaging_verification"] = normalize_nav_imaging_verification(
            data["nav_imaging_verification"]
        )

    if "pleurodesis_agent" in data:
        data["pleurodesis_agent"] = normalize_pleurodesis_agent(data["pleurodesis_agent"])

    if "sedation_type" in data:
        data["sedation_type"] = normalize_sedation_type(data["sedation_type"])

    if "airway_type" in data:
        data["airway_type"] = normalize_airway_type(data["airway_type"])

    if "asa_class" in data:
        data["asa_class"] = normalize_asa_class(data["asa_class"])

    if "bleeding_severity" in data:
        data["bleeding_severity"] = normalize_bleeding_severity(data["bleeding_severity"])

    # Nested field normalization - pleural_procedures
    if "pleural_procedures" in data and isinstance(data["pleural_procedures"], dict):
        pp = data["pleural_procedures"]

        # IPC action
        if "ipc" in pp and isinstance(pp["ipc"], dict):
            if "action" in pp["ipc"]:
                pp["ipc"]["action"] = normalize_ipc_action(pp["ipc"]["action"])

        # Pleurodesis agent
        if "pleurodesis" in pp and isinstance(pp["pleurodesis"], dict):
            if "agent" in pp["pleurodesis"]:
                pp["pleurodesis"]["agent"] = normalize_pleurodesis_agent(pp["pleurodesis"]["agent"])

    # Nested field normalization - cao_interventions_detail
    if "cao_interventions_detail" in data and isinstance(data["cao_interventions_detail"], dict):
        cao = data["cao_interventions_detail"]
        if "etiology" in cao:
            cao["etiology"] = normalize_cao_etiology(cao["etiology"])

    # Nested field normalization - granular_data.navigation_targets
    if "granular_data" in data and isinstance(data["granular_data"], dict):
        gd = data["granular_data"]
        if "navigation_targets" in gd and isinstance(gd["navigation_targets"], list):
            for target in gd["navigation_targets"]:
                if isinstance(target, dict) and "confirmation_method" in target:
                    target["confirmation_method"] = normalize_nav_imaging_verification(
                        target["confirmation_method"]
                    )

    return data


__all__ = [
    "normalize_registry_enums",
    "normalize_gender",
    "normalize_bronchus_sign",
    "normalize_nav_imaging_verification",
    "normalize_pleurodesis_agent",
    "normalize_ipc_action",
    "normalize_forceps_type",
    "normalize_sedation_type",
    "normalize_airway_type",
    "normalize_asa_class",
    "normalize_cao_etiology",
    "normalize_pleural_biopsy_location",
    "normalize_bleeding_severity",
]
