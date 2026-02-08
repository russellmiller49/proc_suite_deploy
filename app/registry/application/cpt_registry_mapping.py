"""CPT code to registry field mapping.

Maps CPT codes to IP Registry schema fields for automatic population
of boolean flags and structured fields during registry export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegistryFieldMapping:
    """Mapping from a CPT code to registry fields.

    Attributes:
        fields: Dict of field names to values to set (e.g., {"ebus_performed": True})
        hints: Dict of hints for fields that need additional context
               (e.g., {"station_count_hint": "1-2"} for 31652)
        v3_only_fields: Fields only applicable to v3 schema
    """
    fields: dict[str, Any] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)
    v3_only_fields: dict[str, Any] = field(default_factory=dict)


# CPT code mappings to IP Registry fields
# Based on common interventional pulmonology procedures
CPT_TO_REGISTRY_MAPPING: dict[str, RegistryFieldMapping] = {
    # Diagnostic bronchoscopy base code
    "31622": RegistryFieldMapping(
        fields={},  # Base bronchoscopy, no specific registry flags
        hints={"procedure_type": "diagnostic_bronchoscopy"},
    ),

    # BAL
    "31624": RegistryFieldMapping(
        fields={"bal_performed": True},
        hints={"procedure_type": "bronchoalveolar_lavage"},
    ),

    # Navigation bronchoscopy
    "31627": RegistryFieldMapping(
        fields={"navigation_performed": True},
        hints={"navigation_type": "electromagnetic"},
    ),

    # Transbronchial lung biopsy
    "31628": RegistryFieldMapping(
        fields={"tblb_performed": True},
        hints={"biopsy_technique": "forceps"},
    ),

    # TBNA (legacy schema does not distinguish; mapped to tblb_performed)
    "31629": RegistryFieldMapping(
        fields={"tblb_performed": True},
        hints={"biopsy_technique": "tbna"},
    ),

    # EBUS-TBNA 1-2 stations
    "31652": RegistryFieldMapping(
        fields={"ebus_performed": True},
        hints={"station_count_hint": "1-2", "procedure_type": "ebus_tbna"},
    ),

    # EBUS-TBNA 3+ stations
    "31653": RegistryFieldMapping(
        fields={"ebus_performed": True},
        hints={"station_count_hint": "3+", "procedure_type": "ebus_tbna"},
    ),

    # Bronchial brush biopsy
    "31623": RegistryFieldMapping(
        fields={},
        hints={"biopsy_technique": "brush"},
    ),

    # Bronchial alveolar lavage protected
    "31625": RegistryFieldMapping(
        fields={"bal_performed": True},
        hints={"bal_type": "protected"},
    ),

    # Bronchial stent placement
    "31636": RegistryFieldMapping(
        fields={"stent_placed": True},
        hints={"stent_type": "initial_placement"},
    ),

    # Bronchial stent revision
    "31637": RegistryFieldMapping(
        fields={"stent_placed": True},
        hints={"stent_type": "revision"},
    ),

    # Dilation bronchoscopy
    "31630": RegistryFieldMapping(
        fields={"dilation_performed": True},
        hints={"dilation_technique": "balloon"},
    ),

    # Rigid bronchoscopy (therapeutic)
    "31641": RegistryFieldMapping(
        fields={},
        hints={"scope_type": "rigid", "procedure_type": "therapeutic"},
    ),

    # Bronchial thermoplasty
    "31660": RegistryFieldMapping(
        fields={"ablation_performed": True},
        hints={"ablation_technique": "thermoplasty"},
        v3_only_fields={"ablation_technique": "thermoplasty"},
    ),
    "31661": RegistryFieldMapping(
        fields={"ablation_performed": True},
        hints={"ablation_technique": "thermoplasty"},
        v3_only_fields={"ablation_technique": "thermoplasty"},
    ),

    # Endobronchial valve / BLVR family
    # - 31647: valve insertion (initial lobe)
    # - 31651: valve insertion (each additional lobe)
    # - 31648: valve removal (initial lobe)
    # - 31649: valve removal (each additional lobe)
    # - 31634: Chartis / balloon occlusion assessment (pre-BLVR assessment)
    "31647": RegistryFieldMapping(
        fields={"blvr_performed": True},
        hints={"procedure_type": "blvr_valve_initial"},
    ),
    "31651": RegistryFieldMapping(
        fields={"blvr_performed": True},
        hints={"procedure_type": "blvr_valve_additional"},
    ),
    "31649": RegistryFieldMapping(
        fields={"blvr_performed": True},
        hints={"procedure_type": "blvr_valve_removal_additional"},
    ),
    "31648": RegistryFieldMapping(
        fields={"blvr_performed": True},
        hints={"procedure_type": "blvr_valve_removal_initial"},
    ),
    "31634": RegistryFieldMapping(
        fields={"blvr_performed": True},
        hints={"procedure_type": "blvr_valve_assessment_chartis"},
    ),

    # Thoracentesis (not directly in IP registry but may indicate pleural involvement)
    "32555": RegistryFieldMapping(
        fields={},  # Not a registry field directly
        hints={"pleural_procedure": "thoracentesis_imaging"},
    ),
    "32556": RegistryFieldMapping(
        fields={},
        hints={"pleural_procedure": "thoracentesis_with_catheter"},
    ),
    "32557": RegistryFieldMapping(
        fields={},
        hints={"pleural_procedure": "chest_tube_imaging"},
    ),

    # Pleuroscopy / VATS
    "32601": RegistryFieldMapping(
        fields={},
        hints={"pleural_procedure": "pleuroscopy_diagnostic"},
    ),
    "32609": RegistryFieldMapping(
        fields={},
        hints={"pleural_procedure": "pleuroscopy_pleural_biopsy"},
    ),
    "32650": RegistryFieldMapping(
        fields={},
        hints={"pleural_procedure": "pleurodesis"},
    ),

    # Cryotherapy / ablation
    "31641": RegistryFieldMapping(
        fields={"ablation_performed": True},
        hints={"ablation_technique": "cryotherapy"},
        v3_only_fields={"ablation_technique": "cryotherapy"},
    ),

    # Radial EBUS
    "31654": RegistryFieldMapping(
        fields={"radial_ebus_performed": True},
        hints={"procedure_type": "radial_ebus"},
    ),

    # EUS-B (endoscopic ultrasound via bronchoscope)
    "43237": RegistryFieldMapping(
        fields={"eus_b_performed": True},
        hints={"procedure_type": "eus_b"},
    ),
    "43238": RegistryFieldMapping(
        fields={"eus_b_performed": True},
        hints={"procedure_type": "eus_b"},
    ),
}


def get_registry_fields_for_code(
    code: str,
    version: str = "v2",
) -> dict[str, Any]:
    """Get registry fields to set based on a CPT code.

    Args:
        code: CPT code (e.g., "31652")
        version: Registry schema version ("v2" or "v3")

    Returns:
        Dict of field names to values for the registry entry.
        Returns empty dict if code is not mapped.
    """
    mapping = CPT_TO_REGISTRY_MAPPING.get(code)
    if not mapping:
        return {}

    result = dict(mapping.fields)

    # Add v3-only fields if applicable
    if version == "v3":
        result.update(mapping.v3_only_fields)

    return result


def get_registry_hints_for_code(code: str) -> dict[str, str]:
    """Get hints for a CPT code mapping.

    These hints provide context for more detailed field population
    that may require additional processing (e.g., parsing station count).

    Args:
        code: CPT code

    Returns:
        Dict of hint keys to hint values.
    """
    mapping = CPT_TO_REGISTRY_MAPPING.get(code)
    if not mapping:
        return {}
    return dict(mapping.hints)


def aggregate_registry_fields(
    codes: list[str],
    version: str = "v2",
) -> dict[str, Any]:
    """Aggregate registry fields from multiple CPT codes (NESTED structure).

    Returns a nested dict matching the RegistryRecord schema with detailed
    fields derived from CPT code families:
    {
        "procedures_performed": {
            "linear_ebus": {"performed": True, "station_count_bucket": "3+"},
            "bal": {"performed": True},
            "blvr": {"performed": True, "procedure_type": "Valve placement"},
        },
        "pleural_procedures": {
            "thoracentesis": {"performed": True, "guidance": "Ultrasound", "indication": "Diagnostic"},
            "chest_tube": {"performed": True, "action": "Insertion"},
        }
    }

    This mapper is deliberately conservative: only sets performed/high-level fields
    derivable from CPT alone; lets the LLM fill in detailed fields (locations, side,
    segments) from the note.

    Args:
        codes: List of CPT codes
        version: Registry schema version

    Returns:
        Nested dict of registry fields matching schema structure.
    """
    code_set = set(str(c) for c in codes)
    procedures: dict[str, dict[str, Any]] = {}
    pleural: dict[str, dict[str, Any]] = {}

    # --- BRONCHOSCOPY PROCEDURES ---

    # Linear EBUS-TBNA: 31652 (1-2 stations), 31653 (3+ stations)
    if code_set & {"31652", "31653"}:
        linear = {"performed": True}
        # Station count bucket derivable from CPT code
        if "31653" in code_set:
            linear["station_count_bucket"] = "3+"
        elif "31652" in code_set:
            linear["station_count_bucket"] = "1-2"
        procedures["linear_ebus"] = linear

    # Radial EBUS: 31654
    if "31654" in code_set:
        procedures["radial_ebus"] = {"performed": True}

    # BAL: 31624 (single lobe), 31625 (each additional lobe)
    if code_set & {"31624", "31625"}:
        procedures["bal"] = {"performed": True}

    # Transbronchial lung biopsy (forceps/cryo): 31628 (single lobe), 31632 (additional lobes)
    if "31628" in code_set or "31632" in code_set:
        procedures["transbronchial_biopsy"] = {"performed": True}

    # Peripheral/lung TBNA (non-nodal): 31629 (single lobe), 31633 (additional lobes)
    if "31629" in code_set or "31633" in code_set:
        procedures["peripheral_tbna"] = {"performed": True}

    # Navigation bronchoscopy: 31627
    if "31627" in code_set:
        procedures["navigational_bronchoscopy"] = {"performed": True}

    # Airway stent: 31636 (initial placement), 31637 (each additional), 31638 (removal)
    if code_set & {"31636", "31637", "31638"}:
        stent = {"performed": True}
        if "31638" in code_set:
            stent["action"] = "Removal"
            stent["airway_stent_removal"] = True
        procedures["airway_stent"] = stent

    # Airway dilation: 31630 (balloon), 31631 (each additional)
    if code_set & {"31630", "31631"}:
        dilation = {"performed": True}
        dilation["technique"] = "Balloon"  # 31630/31631 are balloon dilation codes
        procedures["airway_dilation"] = dilation

    # BLVR / endobronchial valves:
    # - 31647: valve insertion initial lobe
    # - 31651: valve insertion additional lobes
    # - 31648/31649: valve removal (initial/additional lobes)
    # - 31634: Chartis / balloon occlusion assessment
    blvr_codes = {"31634", "31647", "31648", "31649", "31651"}
    if code_set & blvr_codes:
        blvr = {"performed": True}
        has_insertion = bool(code_set & {"31647", "31651"})
        has_removal = bool(code_set & {"31648", "31649"}) and not has_insertion
        if has_insertion:
            blvr["procedure_type"] = "Valve placement"
        elif has_removal:
            blvr["procedure_type"] = "Valve removal"
        else:
            blvr["procedure_type"] = "Valve assessment"
        procedures["blvr"] = blvr

    # Bronchial thermoplasty: 31660 (initial lobe), 31661 (additional lobes)
    if code_set & {"31660", "31661"}:
        procedures["bronchial_thermoplasty"] = {"performed": True}

    # Rigid bronchoscopy: 31641
    if "31641" in code_set:
        procedures["rigid_bronchoscopy"] = {"performed": True}

    # Diagnostic bronchoscopy: 31622
    if "31622" in code_set:
        procedures["diagnostic_bronchoscopy"] = {"performed": True}

    # Bronchial brushings: 31623
    if "31623" in code_set:
        procedures["brushings"] = {"performed": True}

    # --- OTHER INTERVENTIONS ---

    # Percutaneous tracheostomy (new trach creation): 31600, 31601
    # Note: 31612 is percutaneous tracheal puncture / transtracheal access and is
    # not a tracheostomy creation; do not map it to percutaneous_tracheostomy.
    if code_set & {"31600", "31601"}:
        procedures["percutaneous_tracheostomy"] = {"performed": True}

    # EUS-B (endoscopic ultrasound via bronchoscope)
    if code_set & {"43237", "43238"}:
        procedures["eus_b"] = {"performed": True}

    # PEG insertion: 43246 (endoscopic PEG), 49440 (percutaneous gastrostomy)
    if code_set & {"43246", "49440"}:
        procedures["peg_insertion"] = {"performed": True}

    # --- PLEURAL PROCEDURES ---

    # Thoracentesis CPT codes:
    # 32554 - Thoracentesis, diagnostic, without imaging guidance
    # 32555 - Thoracentesis, diagnostic, with imaging guidance
    # 32556 - Thoracentesis, therapeutic, with insertion of indwelling catheter, without imaging
    # 32557 - Thoracentesis, therapeutic, with insertion of indwelling catheter, with imaging
    thoracentesis_codes = {"32554", "32555", "32556", "32557"}
    if thoracentesis_codes & code_set:
        thora: dict[str, Any] = {"performed": True}

        # Imaging guidance (US vs landmark) from code family
        if code_set & {"32555", "32557"}:
            thora["guidance"] = "Ultrasound"
        elif code_set & {"32554", "32556"}:
            thora["guidance"] = "None/Landmark"

        # Indication (diagnostic vs therapeutic vs both)
        has_dx = bool(code_set & {"32554", "32555"})
        has_tx = bool(code_set & {"32556", "32557"})
        if has_dx and has_tx:
            thora["indication"] = "Both"
        elif has_tx:
            thora["indication"] = "Therapeutic"
        elif has_dx:
            thora["indication"] = "Diagnostic"

        pleural["thoracentesis"] = thora

    # Chest tube / tube thoracostomy: 32551
    if "32551" in code_set:
        tube = {"performed": True, "action": "Insertion"}
        pleural["chest_tube"] = tube

    # Medical thoracoscopy / pleuroscopy: 32601 (diagnostic) vs 32609 (pleural biopsy)
    if code_set & {"32601", "32609"}:
        thor = {"performed": True}
        if "32609" in code_set:
            thor["biopsies_taken"] = True
        pleural["medical_thoracoscopy"] = thor

    # Pleurodesis: 32560 (instillation), 32650 (chemical via thoracoscopy)
    if code_set & {"32560", "32650"}:
        pleurodesis = {"performed": True}
        if "32650" in code_set:
            pleurodesis["technique"] = "Thoracoscopic"
        elif "32560" in code_set:
            pleurodesis["technique"] = "Instillation"
        pleural["pleurodesis"] = pleurodesis

    # Build result with only non-empty sections
    result: dict[str, Any] = {}
    if procedures:
        result["procedures_performed"] = procedures
    if pleural:
        result["pleural_procedures"] = pleural

    return result


def aggregate_registry_fields_flat(
    codes: list[str],
    version: str = "v2",
) -> dict[str, Any]:
    """Aggregate registry fields from multiple CPT codes (FLAT structure - legacy).

    Combines fields from multiple codes, with later codes overwriting
    earlier ones for the same field. Boolean fields use OR semantics.

    Args:
        codes: List of CPT codes
        version: Registry schema version

    Returns:
        Flat dict of registry fields (legacy format).
    """
    result: dict[str, Any] = {}

    for code in codes:
        fields = get_registry_fields_for_code(code, version)
        for field_name, value in fields.items():
            if field_name in result and isinstance(value, bool):
                # For boolean fields, use OR semantics
                result[field_name] = result[field_name] or value
            else:
                result[field_name] = value

    return result


def aggregate_registry_hints(codes: list[str]) -> dict[str, list[str]]:
    """Aggregate hints from multiple CPT codes.

    Collects all hints into lists, allowing multiple values per key.

    Args:
        codes: List of CPT codes

    Returns:
        Dict mapping hint keys to lists of hint values.
    """
    result: dict[str, list[str]] = {}

    for code in codes:
        hints = get_registry_hints_for_code(code)
        for key, value in hints.items():
            if key not in result:
                result[key] = []
            if value not in result[key]:
                result[key].append(value)

    return result
