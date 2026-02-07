"""Registry tagging constants.

This module centralizes the procedure-family tags used to:
- gate schema fields and validation rules
- drive prompt field filtering (optional)
"""

from __future__ import annotations

# Procedure family tags used to gate schema fields and validation rules
PROCEDURE_FAMILIES: frozenset[str] = frozenset(
    {
        "EBUS",  # Linear endobronchial ultrasound
        "NAVIGATION",  # Electromagnetic or robotic navigation bronchoscopy
        "CAO",  # Central airway obstruction / debulking
        "PLEURAL",  # Thoracentesis, chest tube, pleuroscopy, pleurodesis
        "BLVR",  # Bronchoscopic lung volume reduction (valves)
        "STENT",  # Airway stent placement/removal
        "BIOPSY",  # Tissue sampling (transbronchial, endobronchial)
        "BAL",  # Bronchoalveolar lavage
        "CRYO_BIOPSY",  # Transbronchial cryobiopsy
        "THERMOPLASTY",  # Bronchial thermoplasty
        "FOREIGN_BODY",  # Foreign body removal
        "HEMOPTYSIS",  # Bronchoscopy for hemoptysis management
        "DIAGNOSTIC",  # Diagnostic bronchoscopy (inspection only)
        "THORACOSCOPY",  # Medical thoracoscopy / pleuroscopy
    }
)

# Field applicability by procedure family.
# Fields not listed here are considered universal.
FIELD_APPLICABLE_TAGS: dict[str, set[str]] = {
    # EBUS-specific fields
    "ebus_scope_brand": {"EBUS"},
    "ebus_stations_sampled": {"EBUS"},
    "ebus_stations_detail": {"EBUS"},
    "ebus_needle_gauge": {"EBUS"},
    "ebus_needle_type": {"EBUS"},
    "ebus_systematic_staging": {"EBUS"},
    "ebus_rose_available": {"EBUS"},
    "ebus_rose_result": {"EBUS"},
    "ebus_intranodal_forceps_used": {"EBUS"},
    "ebus_photodocumentation_complete": {"EBUS"},
    "ebus_elastography_used": {"EBUS"},
    "ebus_elastography_pattern": {"EBUS"},
    "linear_ebus_stations": {"EBUS"},

    # Navigation-specific fields
    "nav_platform": {"NAVIGATION"},
    "nav_target_location": {"NAVIGATION"},
    "nav_imaging_verification": {"NAVIGATION"},
    "nav_rebus_used": {"NAVIGATION"},
    "nav_cone_beam_ct": {"NAVIGATION"},
    "nav_divergence": {"NAVIGATION"},
    "nav_target_size": {"NAVIGATION"},

    # CAO-specific fields
    "cao_location": {"CAO"},
    "cao_primary_modality": {"CAO"},
    "cao_tumor_location": {"CAO"},
    "cao_obstruction_pre_pct": {"CAO"},
    "cao_obstruction_post_pct": {"CAO"},
    "cao_interventions": {"CAO"},  # Multi-site CAO intervention array

    # STENT-specific fields
    "stent_type": {"STENT", "CAO"},
    "stent_location": {"STENT", "CAO"},
    "stent_size": {"STENT", "CAO"},
    "stent_action": {"STENT", "CAO"},
    "airway_stent_removal": {"STENT", "CAO"},

    # Pleural-specific fields
    "pleural_procedure_type": {"PLEURAL", "THORACOSCOPY"},
    "pleural_side": {"PLEURAL", "THORACOSCOPY"},
    "pleural_fluid_volume": {"PLEURAL", "THORACOSCOPY"},
    "pleural_volume_drained_ml": {"PLEURAL", "THORACOSCOPY"},
    "pleural_fluid_appearance": {"PLEURAL", "THORACOSCOPY"},
    "pleural_guidance": {"PLEURAL"},
    "pleural_intercostal_space": {"PLEURAL", "THORACOSCOPY"},
    "pleural_catheter_type": {"PLEURAL"},
    "pleural_pleurodesis_agent": {"PLEURAL", "THORACOSCOPY"},
    "pleural_opening_pressure_measured": {"PLEURAL", "THORACOSCOPY"},
    "pleural_opening_pressure_cmh2o": {"PLEURAL", "THORACOSCOPY"},
    "pleural_thoracoscopy_findings": {"PLEURAL", "THORACOSCOPY"},

    # BLVR-specific fields
    "blvr_valve_type": {"BLVR"},
    "blvr_target_lobe": {"BLVR"},
    "blvr_valve_count": {"BLVR"},
    "blvr_chartis_result": {"BLVR"},

    # Thermoplasty-specific fields
    "thermoplasty_activations": {"THERMOPLASTY"},
    "thermoplasty_lobes_treated": {"THERMOPLASTY"},

    # BAL-specific fields
    "bal_location": {"BAL"},
    "bal_volume_instilled": {"BAL"},
    "bal_volume_returned": {"BAL"},

    # Cryobiopsy-specific fields
    "cryo_probe_size": {"CRYO_BIOPSY"},
    "cryo_freeze_time": {"CRYO_BIOPSY"},
    "cryo_specimens_count": {"CRYO_BIOPSY"},
}

