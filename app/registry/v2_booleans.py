"""V2 Registry Entry to Boolean Procedure Flags Extractor.

This module provides the canonical mapping from V2 "Golden Extraction" registry
entries (raw dict format) to flattened boolean procedure flags matching the
IP_Registry.json schema's `procedures_performed` and `pleural_procedures` sections.

IMPORTANT: This is the authoritative source of truth for V2 -> V3-style boolean
mapping. All consumers (ML data prep, registry service, etc.) should import from
this module to ensure consistent semantics.

NOTE: If the registry schema changes (e.g., new procedures added),
update PROCEDURE_BOOLEAN_FIELDS and the mapping logic in extract_v2_booleans().
"""

from __future__ import annotations

from typing import Any, Dict, List

# Canonical list of procedure boolean fields from IP_Registry.json schema.
# These correspond to the `performed` flag in each procedure object under:
# - procedures_performed.<procedure_name>.performed
# - pleural_procedures.<procedure_name>.performed
#
# Ordered by: bronchoscopy procedures first, then pleural procedures.
# This ordering is the canonical ML label column order for multi-hot encoding.
PROCEDURE_BOOLEAN_FIELDS: List[str] = [
    # Bronchoscopy / diagnostic procedures
    "diagnostic_bronchoscopy",
    "bal",
    "bronchial_wash",
    "brushings",
    "endobronchial_biopsy",
    "tbna_conventional",
    "linear_ebus",
    "radial_ebus",
    "navigational_bronchoscopy",
    "transbronchial_biopsy",
    "transbronchial_cryobiopsy",
    "therapeutic_aspiration",
    "foreign_body_removal",
    "airway_dilation",
    "airway_stent",
    "thermal_ablation",
    "tumor_debulking_non_thermal",
    "cryotherapy",
    "blvr",
    "peripheral_ablation",
    "bronchial_thermoplasty",
    "whole_lung_lavage",
    "rigid_bronchoscopy",
    # Pleural procedures
    "thoracentesis",
    "chest_tube",
    "ipc",
    "medical_thoracoscopy",
    "pleurodesis",
    "pleural_biopsy",
    "fibrinolytic_therapy",
    # Other interventions (not part of procedures_performed/pleural_procedures in IP_Registry.json)
    "percutaneous_tracheostomy",
    "peg_insertion",
]


def extract_v2_booleans(entry: Dict[str, Any]) -> Dict[str, int]:
    """Extract procedure boolean flags from a V2 or V3 Golden Extraction registry entry.

    This function maps registry fields to a flat dict of 0/1 flags for each
    procedure in PROCEDURE_BOOLEAN_FIELDS. It handles both:
    - V2-style flat fields (e.g., `pleural_procedure_type`, `linear_ebus_stations`)
    - V3-style nested fields (e.g., `procedures_performed.endobronchial_biopsy.performed`)

    The mapping rules encode domain knowledge about how registry entries
    indicate procedure presence. This is the canonical source of truth for
    translating registry semantics to boolean flags.

    Args:
        entry: A registry entry dict (typically from `registry_entry` in
               Golden Extraction JSON files).

    Returns:
        Dict mapping each field in PROCEDURE_BOOLEAN_FIELDS to 0 or 1.
        All fields are guaranteed to be present with integer values.

    Example:
        >>> entry = {"pleural_procedure_type": "Thoracentesis", "pleurodesis_performed": True}
        >>> flags = extract_v2_booleans(entry)
        >>> flags["thoracentesis"]
        1
        >>> flags["pleurodesis"]
        1
        >>> flags["chest_tube"]
        0
    """
    flags: Dict[str, int] = {name: 0 for name in PROCEDURE_BOOLEAN_FIELDS}

    # =========================================================================
    # V3 NESTED FORMAT DETECTION
    # =========================================================================
    # Check for V3-style nested procedures_performed structure
    procedures_performed = entry.get("procedures_performed") or {}
    pleural_procedures = entry.get("pleural_procedures") or {}

    if procedures_performed or pleural_procedures:
        # V3 format: extract from nested structure
        for proc_name in PROCEDURE_BOOLEAN_FIELDS:
            # Check procedures_performed.<proc_name>.performed
            proc_key = proc_name
            # Backward compatibility: the schema uses `mechanical_debulking` but the ML
            # schema uses `tumor_debulking_non_thermal`.
            if proc_name == "tumor_debulking_non_thermal" and "mechanical_debulking" in procedures_performed:
                proc_key = "mechanical_debulking"

            proc_obj = procedures_performed.get(proc_key, {})
            if isinstance(proc_obj, dict) and proc_obj.get("performed") is True:
                flags[proc_name] = 1

            # Check pleural_procedures.<proc_name>.performed
            pleural_obj = pleural_procedures.get(proc_name, {})
            if isinstance(pleural_obj, dict) and pleural_obj.get("performed") is True:
                flags[proc_name] = 1

        # Set diagnostic_bronchoscopy if any bronch procedure was performed
        bronch_procs = [
            k
            for k in PROCEDURE_BOOLEAN_FIELDS
            if k
            not in [
                "thoracentesis",
                "chest_tube",
                "ipc",
                "medical_thoracoscopy",
                "pleurodesis",
                "pleural_biopsy",
                "fibrinolytic_therapy",
                "diagnostic_bronchoscopy",
                "percutaneous_tracheostomy",
                "peg_insertion",
            ]
        ]
        if any(flags[p] for p in bronch_procs):
            flags["diagnostic_bronchoscopy"] = 1

        # If V3 format found anything, return early (skip V2 logic)
        if any(flags.values()):
            return flags

    # =========================================================================
    # V2 FLAT FORMAT (fallback)
    # =========================================================================
    # Helper to safely get values
    def _get(key: str, default=None):
        return entry.get(key, default)

    # Helper to check if a list/array field is non-empty
    def _has_items(key: str) -> bool:
        val = _get(key)
        if isinstance(val, list):
            return len(val) > 0
        if isinstance(val, str) and val:
            return True
        return False

    # =========================================================================
    # PLEURAL PROCEDURES (from V2 pleural_procedure_type enum)
    # =========================================================================
    pleural_type = _get("pleural_procedure_type")
    if pleural_type:
        pleural_lower = pleural_type.lower()
        if "thoracentesis" in pleural_lower:
            flags["thoracentesis"] = 1
        if "chest tube" in pleural_lower or "tube thoracostomy" in pleural_lower:
            flags["chest_tube"] = 1
        if "tunneled" in pleural_lower or "catheter" in pleural_lower or "ipc" in pleural_lower:
            flags["ipc"] = 1
        if "thoracoscopy" in pleural_lower and "medical" in pleural_lower:
            flags["medical_thoracoscopy"] = 1

    # V2: pleurodesis_performed boolean
    if _get("pleurodesis_performed") is True:
        flags["pleurodesis"] = 1

    # =========================================================================
    # LINEAR EBUS
    # =========================================================================
    # V2: linear_ebus_stations or ebus_stations_sampled (non-empty list)
    if _has_items("linear_ebus_stations") or _has_items("ebus_stations_sampled"):
        flags["linear_ebus"] = 1

    # =========================================================================
    # RADIAL EBUS
    # =========================================================================
    # V2: nav_rebus_used boolean
    if _get("nav_rebus_used") is True:
        flags["radial_ebus"] = 1

    # =========================================================================
    # NAVIGATIONAL BRONCHOSCOPY
    # =========================================================================
    # V2: nav_platform (non-null)
    if _get("nav_platform"):
        flags["navigational_bronchoscopy"] = 1

    # =========================================================================
    # PERIPHERAL ABLATION
    # =========================================================================
    # V2: ablation_peripheral_performed boolean
    if _get("ablation_peripheral_performed") is True:
        flags["peripheral_ablation"] = 1

    # =========================================================================
    # BLVR
    # =========================================================================
    # V2: blvr_number_of_valves or blvr_target_lobe indicates BLVR performed
    if _get("blvr_number_of_valves") or _get("blvr_target_lobe"):
        flags["blvr"] = 1

    # =========================================================================
    # TRANSBRONCHIAL CRYOBIOPSY
    # =========================================================================
    # V2: nav_cryobiopsy_for_nodule boolean
    if _get("nav_cryobiopsy_for_nodule") is True:
        flags["transbronchial_cryobiopsy"] = 1

    # =========================================================================
    # AIRWAY STENT
    # =========================================================================
    # V2: stent_type (non-null)
    if _get("stent_type"):
        flags["airway_stent"] = 1

    # =========================================================================
    # CAO PROCEDURES (thermal ablation, cryotherapy, dilation)
    # =========================================================================
    cao_modality = _get("cao_primary_modality")
    if cao_modality:
        cao_lower = cao_modality.lower()
        if any(term in cao_lower for term in ["thermal", "electrocautery", "argon", "laser", "apc"]):
            flags["thermal_ablation"] = 1
        if any(term in cao_lower for term in ["mechanical debulking", "rigid coring", "microdebrider"]):
            flags["tumor_debulking_non_thermal"] = 1
        if "cryo" in cao_lower:
            flags["cryotherapy"] = 1
        if "dilation" in cao_lower or "balloon" in cao_lower:
            flags["airway_dilation"] = 1

    # =========================================================================
    # WHOLE LUNG LAVAGE
    # =========================================================================
    # V2: wll_volume_instilled_l or wll_dlt_used
    if _get("wll_volume_instilled_l") or _get("wll_dlt_used"):
        flags["whole_lung_lavage"] = 1

    # =========================================================================
    # FOREIGN BODY REMOVAL
    # =========================================================================
    # V2: fb_removal_success or fb_object_type
    if _get("fb_removal_success") is True or _get("fb_object_type"):
        flags["foreign_body_removal"] = 1

    # =========================================================================
    # BRONCHIAL THERMOPLASTY
    # =========================================================================
    # V2: bt_lobe_treated or bt_activation_count
    if _get("bt_lobe_treated") or _get("bt_activation_count"):
        flags["bronchial_thermoplasty"] = 1

    # =========================================================================
    # TRANSBRONCHIAL BIOPSY
    # =========================================================================
    # V2: bronch_num_tbbx > 0
    tbbx_count = _get("bronch_num_tbbx")
    if tbbx_count and tbbx_count > 0:
        flags["transbronchial_biopsy"] = 1

    # V2: bronch_tbbx_tool contains "cryo"
    tbbx_tool = _get("bronch_tbbx_tool")
    if tbbx_tool and "cryo" in str(tbbx_tool).lower():
        flags["transbronchial_cryobiopsy"] = 1

    # =========================================================================
    # SAMPLING TOOLS (BAL, brushings, TBNA, etc.)
    # =========================================================================
    # V2: nav_sampling_tools array
    sampling_tools = _get("nav_sampling_tools") or []
    if isinstance(sampling_tools, list):
        for tool in sampling_tools:
            tool_lower = str(tool).lower()
            if "brush" in tool_lower:
                flags["brushings"] = 1
            if "cryo" in tool_lower:
                flags["transbronchial_cryobiopsy"] = 1
            if "forceps" in tool_lower or "biopsy" in tool_lower:
                flags["transbronchial_biopsy"] = 1
            if "needle" in tool_lower or "tbna" in tool_lower:
                flags["tbna_conventional"] = 1

    # V2: bronch_specimen_tests or explicit BAL indicators
    specimen_tests = _get("bronch_specimen_tests") or []
    if isinstance(specimen_tests, list):
        for test in specimen_tests:
            test_lower = str(test).lower()
            if "bal" in test_lower or "lavage" in test_lower:
                flags["bal"] = 1
            if "wash" in test_lower:
                flags["bronchial_wash"] = 1
            if "brush" in test_lower or "cytology" in test_lower:
                flags["brushings"] = 1

    # =========================================================================
    # ENDOBRONCHIAL BIOPSY
    # =========================================================================
    # V2: ebus_intranodal_forceps_used
    if _get("ebus_intranodal_forceps_used") is True:
        flags["endobronchial_biopsy"] = 1

    # =========================================================================
    # RIGID BRONCHOSCOPY
    # =========================================================================
    # V2: procedure_setting or airway_type indicates rigid
    procedure_setting = _get("procedure_setting")
    if procedure_setting and "rigid" in str(procedure_setting).lower():
        flags["rigid_bronchoscopy"] = 1

    airway_type = _get("airway_type")
    if airway_type and "rigid" in str(airway_type).lower():
        flags["rigid_bronchoscopy"] = 1

    # =========================================================================
    # DIAGNOSTIC BRONCHOSCOPY (derived flag)
    # =========================================================================
    # If any bronchoscopic procedure is detected, set diagnostic_bronchoscopy
    # This is a base procedure that's almost always present with any bronch
    bronch_indicators = [
        flags["linear_ebus"],
        flags["radial_ebus"],
        flags["navigational_bronchoscopy"],
        flags["transbronchial_biopsy"],
        flags["transbronchial_cryobiopsy"],
        flags["blvr"],
        flags["peripheral_ablation"],
        flags["thermal_ablation"],
        flags["cryotherapy"],
        flags["airway_dilation"],
        flags["airway_stent"],
        flags["foreign_body_removal"],
        flags["bronchial_thermoplasty"],
        flags["whole_lung_lavage"],
        flags["rigid_bronchoscopy"],
        flags["bal"],
        flags["bronchial_wash"],
        flags["brushings"],
        flags["endobronchial_biopsy"],
        flags["tbna_conventional"],
    ]
    if any(bronch_indicators):
        flags["diagnostic_bronchoscopy"] = 1

    return flags


__all__ = ["PROCEDURE_BOOLEAN_FIELDS", "extract_v2_booleans"]
