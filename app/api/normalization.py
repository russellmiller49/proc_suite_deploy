"""Normalization layer for registry payloads.

This module provides functions to normalize noisy incoming registry payloads
so they conform to the strict Pydantic schemas in proc_registry / proc_schemas.
This normalization does NOT loosen schema rules; it only reshapes inputs.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


def _strip_unit_suffix(value: Any, suffix: str) -> Any:
    """Strip a unit suffix from a string value and convert to float.

    Args:
        value: The value to process (may be string, number, or other)
        suffix: The unit suffix to strip (e.g., "mm", "cm")

    Returns:
        A float if conversion succeeds, otherwise the original value
    """
    if isinstance(value, str):
        # Case-insensitive suffix matching
        pattern = re.compile(re.escape(suffix) + r"\s*$", re.IGNORECASE)
        if pattern.search(value):
            try:
                cleaned = pattern.sub("", value).strip()
                return float(cleaned)
            except ValueError:
                return value
    return value


def _normalize_numeric_with_unit(value: Any, unit_patterns: list[str]) -> Any:
    """Normalize a value that might have various unit suffixes.

    Args:
        value: The value to normalize
        unit_patterns: List of unit patterns to try stripping

    Returns:
        Normalized numeric value or original value
    """
    if value is None:
        return None
    for pattern in unit_patterns:
        result = _strip_unit_suffix(value, pattern)
        if result != value:
            return result
    return value


def simplify_billing_cpt_codes(payload: dict[str, Any]) -> None:
    """Add a simplified CPT code list when the detailed objects carry no metadata.

    For UI-friendly JSON output, callers may prefer a simple list of CPT code
    strings when the detailed objects contain only `code` (and otherwise empty
    metadata like description/modifiers).

    This function is safe to call on either pre- or post-validation payloads.
    """
    billing = payload.get("billing")
    if not isinstance(billing, dict) or not billing:
        return

    cpt_data = billing.get("cpt_codes")
    if not isinstance(cpt_data, list) or not cpt_data:
        return

    code_dicts: list[dict[str, Any]] = [
        c
        for c in cpt_data
        if isinstance(c, dict)
        and isinstance(c.get("code"), str)
        and c.get("code").strip()
    ]
    if not code_dicts:
        return

    def _has_details(item: dict[str, Any]) -> bool:
        if item.get("description"):
            return True
        if item.get("modifier"):
            return True
        modifiers = item.get("modifiers")
        if isinstance(modifiers, list) and any(isinstance(m, str) and m.strip() for m in modifiers):
            return True
        units = item.get("units")
        return units not in (None, 1)

    if any(_has_details(item) for item in code_dicts):
        return

    simplified: list[str] = []
    seen: set[str] = set()
    for item in code_dicts:
        code = str(item.get("code")).strip()
        if code and code not in seen:
            simplified.append(code)
            seen.add(code)

    if simplified:
        billing["cpt_codes_simple"] = simplified


# Role mapping: common variations -> canonical enum values
# From IP_Registry.json: ["RN", "RT", "Tech", "Resident", "PA", "NP", "Medical Student", null]
ASSISTANT_ROLE_MAP: dict[str, str] = {
    # Fellow variations -> Resident (fellows are considered residents in training)
    "fellow": "Resident",
    "pulm fellow": "Resident",
    "pulmonary fellow": "Resident",
    "ip fellow": "Resident",
    "interventional pulmonology fellow": "Resident",
    "pgy4": "Resident",
    "pgy5": "Resident",
    "pgy6": "Resident",
    "pgy7": "Resident",
    "pgy8": "Resident",
    # Resident variations
    "resident": "Resident",
    "intern": "Resident",
    "pgy1": "Resident",
    "pgy2": "Resident",
    "pgy3": "Resident",
    # RN variations
    "rn": "RN",
    "nurse": "RN",
    "registered nurse": "RN",
    # RT variations
    "rt": "RT",
    "respiratory therapist": "RT",
    "resp therapist": "RT",
    # Tech variations
    "tech": "Tech",
    "technician": "Tech",
    "technologist": "Tech",
    "bronch tech": "Tech",
    # PA variations
    "pa": "PA",
    "pa-c": "PA",
    "physician assistant": "PA",
    # NP variations
    "np": "NP",
    "nurse practitioner": "NP",
    "aprn": "NP",
    # Medical student variations
    "medical student": "Medical Student",
    "med student": "Medical Student",
    "ms3": "Medical Student",
    "ms4": "Medical Student",
    "student": "Medical Student",
}

# Forceps type mapping: LLM outputs -> canonical enum values
# From IP_Registry.json: ["Standard", "Cryoprobe", null]
FORCEPS_TYPE_MAP: dict[str, str] = {
    # Standard variations
    "standard": "Standard",
    "standard forceps": "Standard",
    "forceps": "Standard",
    "biopsy forceps": "Standard",
    "conventional": "Standard",
    "regular": "Standard",
    # Cryoprobe variations
    "cryoprobe": "Cryoprobe",
    "cryo": "Cryoprobe",
    "cryobiopsy": "Cryoprobe",
    "cryo probe": "Cryoprobe",
    # Mixed values - if cryoprobe is mentioned, use Cryoprobe
    "needle, cryoprobe": "Cryoprobe",
    "cryoprobe, needle": "Cryoprobe",
    "forceps, cryoprobe": "Cryoprobe",
    "standard, cryoprobe": "Cryoprobe",
    # Needle alone -> Standard (TBNA uses needles, not forceps for TBBx)
    "needle": "Standard",
}

# Probe position mapping: descriptive text -> canonical enum values
# From IP_Registry.json: ["Concentric", "Eccentric", "Adjacent", "Not visualized", null]
PROBE_POSITION_MAP: dict[str, str] = {
    # Not visualized variations
    "not visualized": "Not visualized",
    "aerated lung on radial ebus": "Not visualized",
    "aerated lung": "Not visualized",
    "no lesion seen": "Not visualized",
    "not seen": "Not visualized",
    "no target identified": "Not visualized",
    "negative": "Not visualized",
    # Concentric variations
    "concentric": "Concentric",
    "central": "Concentric",
    "within lesion": "Concentric",
    "lesion visualized concentrically": "Concentric",
    # Eccentric variations
    "eccentric": "Eccentric",
    "off-center": "Eccentric",
    "peripheral": "Eccentric",
    "lesion visualized eccentrically": "Eccentric",
    # Adjacent variations
    "adjacent": "Adjacent",
    "beside lesion": "Adjacent",
    "near lesion": "Adjacent",
}

# Stent type mapping: LLM outputs -> canonical enum values
# From IP_Registry.json: ["Silicone - Dumon", "Silicone - Hood", "Silicone - Novatech",
#                         "SEMS - Uncovered", "SEMS - Covered", "SEMS - Partially covered",
#                         "Hybrid", "Y-Stent", "Other", null]
STENT_TYPE_MAP: dict[str, str] = {
    # Y-Stent variations (LLM often combines silicone + y-stent)
    "y-stent": "Y-Stent",
    "y stent": "Y-Stent",
    "ystent": "Y-Stent",
    "silicone-y-stent": "Y-Stent",
    "silicone y-stent": "Y-Stent",
    "silicone y stent": "Y-Stent",
    "dumon y-stent": "Y-Stent",
    "dumon y stent": "Y-Stent",
    # Silicone - Dumon variations
    "silicone - dumon": "Silicone - Dumon",
    "silicone-dumon": "Silicone - Dumon",
    "silicone dumon": "Silicone - Dumon",
    "dumon": "Silicone - Dumon",
    "dumon stent": "Silicone - Dumon",
    # Silicone - Hood variations
    "silicone - hood": "Silicone - Hood",
    "silicone-hood": "Silicone - Hood",
    "silicone hood": "Silicone - Hood",
    "hood": "Silicone - Hood",
    "hood stent": "Silicone - Hood",
    # Silicone - Novatech variations
    "silicone - novatech": "Silicone - Novatech",
    "silicone-novatech": "Silicone - Novatech",
    "silicone novatech": "Silicone - Novatech",
    "novatech": "Silicone - Novatech",
    "novatech stent": "Silicone - Novatech",
    # Generic silicone -> Dumon (most common)
    "silicone": "Silicone - Dumon",
    "silicone stent": "Silicone - Dumon",
    # SEMS variations
    "sems - uncovered": "SEMS - Uncovered",
    "sems-uncovered": "SEMS - Uncovered",
    "sems uncovered": "SEMS - Uncovered",
    "uncovered sems": "SEMS - Uncovered",
    "uncovered metal stent": "SEMS - Uncovered",
    "sems - covered": "SEMS - Covered",
    "sems-covered": "SEMS - Covered",
    "sems covered": "SEMS - Covered",
    "covered sems": "SEMS - Covered",
    "covered metal stent": "SEMS - Covered",
    "sems - partially covered": "SEMS - Partially covered",
    "sems-partially covered": "SEMS - Partially covered",
    "sems partially covered": "SEMS - Partially covered",
    "partially covered sems": "SEMS - Partially covered",
    "partially covered metal stent": "SEMS - Partially covered",
    # Generic SEMS -> Uncovered (most common)
    "sems": "SEMS - Uncovered",
    "metal stent": "SEMS - Uncovered",
    # Hybrid
    "hybrid": "Hybrid",
    "hybrid stent": "Hybrid",
    # Other
    "other": "Other",
}


def normalize_registry_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize noisy incoming registry payloads.

    This function normalizes data to conform to the strict Pydantic schemas
    in proc_registry / proc_schemas. It does NOT loosen schema rules; it only
    reshapes inputs to match expected formats.

    Args:
        raw: The raw incoming payload (dict-like)

    Returns:
        A normalized dict that should validate against registry schemas
    """
    # Make a shallow copy to avoid mutating the original
    payload: dict[str, Any] = {k: v for k, v in raw.items()}

    # Normalize providers.assistant_role
    providers = payload.get("providers")
    if isinstance(providers, dict):
        role = providers.get("assistant_role")
        if isinstance(role, str):
            normalized_role = role.strip().lower()
            providers["assistant_role"] = ASSISTANT_ROLE_MAP.get(normalized_role, role)

    # Normalize equipment.bronchoscope_outer_diameter_mm: "12 mm" -> 12.0
    equipment = payload.get("equipment")
    if isinstance(equipment, dict):
        diameter = equipment.get("bronchoscope_outer_diameter_mm")
        if diameter is not None:
            equipment["bronchoscope_outer_diameter_mm"] = _normalize_numeric_with_unit(
                diameter, ["mm", "millimeters", "millimeter"]
            )

        # Also normalize fluoroscopy_time_seconds and fluoroscopy_dose_mgy
        fluoro_time = equipment.get("fluoroscopy_time_seconds")
        if fluoro_time is not None:
            equipment["fluoroscopy_time_seconds"] = _normalize_numeric_with_unit(
                fluoro_time, ["s", "sec", "seconds", "second"]
            )

        fluoro_dose = equipment.get("fluoroscopy_dose_mgy")
        if fluoro_dose is not None:
            equipment["fluoroscopy_dose_mgy"] = _normalize_numeric_with_unit(
                fluoro_dose, ["mgy", "mGy"]
            )

    # Normalize procedures_performed fields
    procedures = payload.get("procedures_performed")
    if isinstance(procedures, dict):
        # Normalize radial_ebus.probe_position
        radial_ebus = procedures.get("radial_ebus")
        if isinstance(radial_ebus, dict):
            probe_position = radial_ebus.get("probe_position")
            if isinstance(probe_position, str):
                text = probe_position.strip().lower()
                radial_ebus["probe_position"] = PROBE_POSITION_MAP.get(text, probe_position)

        # Normalize navigational_bronchoscopy fields
        nav_bronch = procedures.get("navigational_bronchoscopy")
        if isinstance(nav_bronch, dict):
            divergence = nav_bronch.get("divergence_mm")
            if divergence is not None:
                nav_bronch["divergence_mm"] = _normalize_numeric_with_unit(
                    divergence, ["mm", "millimeters", "millimeter"]
                )

            # Normalize sampling_tools_used list
            # Enum: ["Needle", "Forceps", "Brush", "Cryoprobe", "NeedleInNeedle"]
            tools = nav_bronch.get("sampling_tools_used")
            if isinstance(tools, list):
                tool_map = {
                    "needle": "Needle",
                    "tbna needle": "Needle",
                    "21g needle": "Needle",
                    "22g needle": "Needle",
                    "ion needle": "Needle",
                    "forceps": "Forceps",
                    "biopsy forceps": "Forceps",
                    "standard forceps": "Forceps",
                    "brush": "Brush",
                    "bronchial brush": "Brush",
                    "cryoprobe": "Cryoprobe",
                    "cryo": "Cryoprobe",
                    "cryobiopsy": "Cryoprobe",
                    "needleinneedle": "NeedleInNeedle",
                    "needle in needle": "NeedleInNeedle",
                }
                valid_tools = ("Needle", "Forceps", "Brush", "Cryoprobe", "NeedleInNeedle")
                normalized_tools = []
                for tool in tools:
                    if isinstance(tool, str):
                        normalized = tool_map.get(tool.strip().lower(), tool)
                        # Only add if it's a valid enum value
                        if normalized in valid_tools and normalized not in normalized_tools:
                            normalized_tools.append(normalized)
                nav_bronch["sampling_tools_used"] = normalized_tools

        # Normalize transbronchial_biopsy.forceps_type
        tbbx = procedures.get("transbronchial_biopsy")
        if isinstance(tbbx, dict):
            forceps_type = tbbx.get("forceps_type")
            if isinstance(forceps_type, str):
                text = forceps_type.strip().lower()
                tbbx["forceps_type"] = FORCEPS_TYPE_MAP.get(text, forceps_type)
                # If still not valid after mapping, check if cryoprobe is mentioned
                if tbbx["forceps_type"] not in ("Standard", "Cryoprobe", None):
                    if "cryo" in text:
                        tbbx["forceps_type"] = "Cryoprobe"
                    else:
                        # Default to Standard for unrecognized values
                        tbbx["forceps_type"] = "Standard"

        # Normalize transbronchial_cryobiopsy.forceps_type (same enum)
        cryo_bx = procedures.get("transbronchial_cryobiopsy")
        if isinstance(cryo_bx, dict):
            forceps_type = cryo_bx.get("forceps_type")
            if isinstance(forceps_type, str):
                text = forceps_type.strip().lower()
                cryo_bx["forceps_type"] = FORCEPS_TYPE_MAP.get(text, forceps_type)
                if cryo_bx["forceps_type"] not in ("Standard", "Cryoprobe", None):
                    if "cryo" in text:
                        cryo_bx["forceps_type"] = "Cryoprobe"
                    else:
                        cryo_bx["forceps_type"] = "Cryoprobe"  # Default for cryobiopsy

        # Normalize airway_stent.stent_type
        airway_stent = procedures.get("airway_stent")
        if isinstance(airway_stent, dict):
            stent_type = airway_stent.get("stent_type")
            if isinstance(stent_type, str):
                text = stent_type.strip().lower()
                airway_stent["stent_type"] = STENT_TYPE_MAP.get(text, stent_type)
                # If still not valid, try to infer from keywords
                valid_stent_types = (
                    "Silicone - Dumon", "Silicone - Hood", "Silicone - Novatech",
                    "SEMS - Uncovered", "SEMS - Covered", "SEMS - Partially covered",
                    "Hybrid", "Y-Stent", "Other"
                )
                if airway_stent["stent_type"] not in valid_stent_types:
                    # Check for Y-stent pattern (most specific first)
                    if "y-stent" in text or "y stent" in text or "ystent" in text:
                        airway_stent["stent_type"] = "Y-Stent"
                    elif "sems" in text or "metal" in text:
                        if "covered" in text and "uncovered" not in text:
                            if "partial" in text:
                                airway_stent["stent_type"] = "SEMS - Partially covered"
                            else:
                                airway_stent["stent_type"] = "SEMS - Covered"
                        else:
                            airway_stent["stent_type"] = "SEMS - Uncovered"
                    elif "silicone" in text:
                        if "dumon" in text:
                            airway_stent["stent_type"] = "Silicone - Dumon"
                        elif "hood" in text:
                            airway_stent["stent_type"] = "Silicone - Hood"
                        elif "novatech" in text:
                            airway_stent["stent_type"] = "Silicone - Novatech"
                        else:
                            airway_stent["stent_type"] = "Silicone - Dumon"  # Default silicone
                    elif "hybrid" in text:
                        airway_stent["stent_type"] = "Hybrid"
                    else:
                        airway_stent["stent_type"] = "Other"

    # Normalize clinical_context.lesion_size_mm
    clinical_context = payload.get("clinical_context")
    if isinstance(clinical_context, dict):
        lesion_size = clinical_context.get("lesion_size_mm")
        if lesion_size is not None:
            clinical_context["lesion_size_mm"] = _normalize_numeric_with_unit(
                lesion_size, ["mm", "millimeters", "millimeter", "cm"]
            )
            # Handle cm -> mm conversion
            if isinstance(lesion_size, str) and "cm" in lesion_size.lower():
                try:
                    cm_val = float(re.sub(r"[^\d.]", "", lesion_size))
                    clinical_context["lesion_size_mm"] = cm_val * 10
                except ValueError:
                    pass

    # Normalize patient_demographics numeric fields
    demographics = payload.get("patient_demographics")
    if isinstance(demographics, dict):
        height = demographics.get("height_cm")
        if height is not None:
            demographics["height_cm"] = _normalize_numeric_with_unit(
                height, ["cm", "centimeters", "centimeter"]
            )

        weight = demographics.get("weight_kg")
        if weight is not None:
            demographics["weight_kg"] = _normalize_numeric_with_unit(
                weight, ["kg", "kilograms", "kilogram", "kgs"]
            )

    # Normalize pleural_procedures.thoracentesis.volume_removed_ml
    pleural = payload.get("pleural_procedures")
    if isinstance(pleural, dict):
        thoracentesis = pleural.get("thoracentesis")
        if isinstance(thoracentesis, dict):
            volume = thoracentesis.get("volume_removed_ml")
            if volume is not None:
                thoracentesis["volume_removed_ml"] = _normalize_numeric_with_unit(
                    volume, ["ml", "mL", "cc", "milliliters"]
                )

    # Normalize pathology_results: if it's a string, convert to proper dict structure
    pathology = payload.get("pathology_results")
    if isinstance(pathology, str):
        # LLM returned a string like "ROSE malignant" - convert to proper structure
        pathology_text = pathology.strip()
        payload["pathology_results"] = {
            "final_diagnosis": pathology_text if pathology_text else None,
            "final_staging": None,
            "histology": None,
            "molecular_markers": None,
            "adequacy": None,
            "rose_result": pathology_text if "rose" in pathology_text.lower() else None,
        }

    # Normalize granular_data fields
    granular_data = payload.get("granular_data")
    if isinstance(granular_data, dict):
        # Normalize navigation_targets[].rebus_view to schema enum values
        # Schema enum: ["Concentric", "Eccentric", "Adjacent", "Not visualized"]
        nav_targets = granular_data.get("navigation_targets")
        if isinstance(nav_targets, list):
            for target in nav_targets:
                if isinstance(target, dict):
                    rebus_view = target.get("rebus_view")
                    if isinstance(rebus_view, str):
                        text = rebus_view.strip().lower()
                        # Use existing PROBE_POSITION_MAP or fuzzy match
                        normalized = PROBE_POSITION_MAP.get(text)
                        if normalized is None:
                            # Fuzzy matching for descriptive text
                            if "concentric" in text:
                                normalized = "Concentric"
                            elif "eccentric" in text:
                                normalized = "Eccentric"
                            elif "adjacent" in text:
                                normalized = "Adjacent"
                            elif "not" in text and ("visual" in text or "seen" in text):
                                normalized = "Not visualized"
                        if normalized:
                            target["rebus_view"] = normalized
                        else:
                            # Invalid value - set to None to let schema handle it
                            target["rebus_view"] = None

        # Normalize specimens_collected[].source_location - handle null values
        specimens = granular_data.get("specimens_collected")
        if isinstance(specimens, list):
            for specimen in specimens:
                if isinstance(specimen, dict):
                    source_loc = specimen.get("source_location")
                    # source_location is required - provide default if null
                    if source_loc is None or (
                        isinstance(source_loc, str) and not source_loc.strip()
                    ):
                        # Try to derive from source_procedure if available
                        source_proc = specimen.get("source_procedure", "")
                        specimen["source_location"] = source_proc if source_proc else "Unknown"

    # ==========================================================================
    # Derive procedures_performed fields from granular_data
    # ==========================================================================
    # This ensures granular data is reflected in top-level procedure fields
    granular_data = payload.get("granular_data")
    if granular_data:
        from app.registry.schema_granular import derive_procedures_from_granular

        existing_procedures = payload.get("procedures_performed") or {}
        updated_procedures, derivation_warnings = derive_procedures_from_granular(
            granular_data, existing_procedures
        )
        payload["procedures_performed"] = updated_procedures

        # Add derivation warnings to granular_validation_warnings
        if derivation_warnings:
            existing_warnings = payload.get("granular_validation_warnings", [])
            if existing_warnings is None:
                existing_warnings = []
            payload["granular_validation_warnings"] = existing_warnings + derivation_warnings

    # ==========================================================================
    # Fix therapeutic_aspiration vs transbronchial_biopsy misclassification
    # ==========================================================================
    # Therapeutic aspiration involves airways (Trachea, RMS, LMS, Carina, etc.)
    # Transbronchial biopsy involves lung parenchyma (lobes/segments)
    procedures = payload.get("procedures_performed")
    if not isinstance(procedures, dict):
        procedures = {}
    tbbx = procedures.get("transbronchial_biopsy")
    if tbbx and tbbx.get("performed"):
        tbbx_locations = tbbx.get("locations", []) or []

        # Definite airway locations that indicate aspiration, not biopsy
        airway_patterns = {
            "trachea", "rms", "lms", "bi", "bronchus intermedius",
            "carina", "mainstem", "rc1", "rc2", "lc1", "lc2",
            "right mainstem", "left mainstem",
        }

        # Lobar/segmental patterns indicate parenchymal biopsy
        parenchymal_patterns = {
            "posterior", "anterior", "lateral", "medial", "basal",
            "apical", "superior", "inferior", "segment", "lb", "rb",
        }

        # Pure lobe names without segment info are ambiguous
        # They're considered airways if they're in a list of mostly airways
        pure_lobe_names = {"rul", "rml", "rll", "lul", "lll", "lingula"}

        # Check if locations are airways vs parenchymal
        airway_locs = []
        parenchymal_locs = []
        ambiguous_locs = []  # Could be airways (carinas) or parenchyma

        for loc in tbbx_locations:
            loc_lower = loc.lower().strip()

            # Check for definite airway patterns
            is_definite_airway = any(pattern in loc_lower for pattern in airway_patterns)

            # Check for parenchymal patterns (segments)
            is_parenchymal = any(seg in loc_lower for seg in parenchymal_patterns)

            # Check if it's just a pure lobe name
            is_pure_lobe = loc_lower in pure_lobe_names

            if is_definite_airway and not is_parenchymal:
                airway_locs.append(loc)
            elif is_parenchymal:
                parenchymal_locs.append(loc)
            elif is_pure_lobe:
                ambiguous_locs.append(loc)
            else:
                # Unknown - treat as parenchymal to be safe
                parenchymal_locs.append(loc)

        # If we have definite airways and ambiguous lobes but NO parenchymal segments,
        # this is likely therapeutic aspiration (e.g., "RMS, LMS, BI, RUL carina, RML carina")
        if airway_locs and not parenchymal_locs:
            # Ambiguous lobe names in context of airways are likely carinas
            airway_locs.extend(ambiguous_locs)

            # Move to therapeutic_aspiration
            therapeutic = procedures.get("therapeutic_aspiration") or {}
            if not therapeutic.get("performed"):
                therapeutic["performed"] = True
                therapeutic["material"] = "Mucus plug"  # Default
                therapeutic["location"] = ", ".join(airway_locs[:3])
                procedures["therapeutic_aspiration"] = therapeutic

            # Clear incorrect transbronchial_biopsy
            tbbx["performed"] = False
            tbbx["locations"] = None
            procedures["transbronchial_biopsy"] = tbbx

        # If we have parenchymal locations, keep TBBx with those
        elif parenchymal_locs:
            tbbx["locations"] = parenchymal_locs + ambiguous_locs

    # Only set procedures_performed if it was already present or is non-empty
    if procedures or "procedures_performed" in raw:
        payload["procedures_performed"] = procedures

    # ==========================================================================
    # Derive outcomes from note content
    # ==========================================================================
    # Check for procedure completion and complications based on common phrases
    outcomes = payload.get("outcomes") or {}
    if outcomes.get("procedure_completed") is None:
        # Look for indicators in the note that procedure was completed
        # This is a fallback - ideally the LLM sets this
        pass  # Will be handled by LLM or explicit derivation

    simplify_billing_cpt_codes(payload)
    return payload


__all__ = ["normalize_registry_payload", "simplify_billing_cpt_codes", "FORCEPS_TYPE_MAP"]
