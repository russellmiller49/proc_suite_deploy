"""Helpers to map flat extraction output into the nested registry schema."""

from __future__ import annotations

from typing import Any

from app.registry.postprocess import (
    normalize_radial_ebus_probe_position,
    normalize_bronchoscope_diameter,
    normalize_complication_list,
)

def _normalize_transbronchial_forceps_type(value: Any) -> str | None:
    """Normalize transbronchial biopsy forceps type to the schema literals.

    Schema expects: "Standard" | "Cryoprobe" | None.
    LLM/slot extractors may return lists or unrelated tool types (e.g. "Needle").
    """
    if value is None:
        return None

    if isinstance(value, list):
        for item in value:
            normalized = _normalize_transbronchial_forceps_type(item)
            if normalized is not None:
                return normalized
        return None

    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    lowered = raw.lower()
    if lowered in {"standard", "forceps", "standard forceps", "biopsy forceps"}:
        return "Standard"
    if lowered in {"cryoprobe", "cryo", "cryo probe", "cryo-probe"}:
        return "Cryoprobe"
    if lowered in {"needle", "tbna needle", "ebus needle"}:
        return None

    # If the string already matches expected casing, pass it through safely.
    if raw in {"Standard", "Cryoprobe"}:
        return raw

    return None


def _format_ebus_needle_gauge(value: Any) -> str | None:
    """Format EBUS needle gauge for the nested schema (e.g., '22G')."""
    if value is None:
        return None

    if isinstance(value, int):
        return f"{value}G"

    if isinstance(value, str):
        text = value.strip().upper()
        if not text:
            return None
        if text in {"19G", "21G", "22G", "25G"}:
            return text
        import re

        match = re.search(r"\b(19|21|22|25)\b", text)
        if match:
            return f"{match.group(1)}G"

    return None


def build_nested_registry_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the flat registry payload with nested sections populated."""
    payload = dict(data)
    families = {fam for fam in data.get("procedure_families") or []}

    # Remove string-typed values for fields that should be nested dicts
    # LLM sometimes returns these as narrative strings instead of structured data
    nested_fields = [
        "patient_demographics", "providers", "clinical_context", "sedation",
        "procedure_setting", "equipment", "procedures_performed", "pleural_procedures",
        "specimens", "complications", "outcomes", "billing", "metadata",
    ]
    for field in nested_fields:
        if field in payload and isinstance(payload[field], str):
            del payload[field]

    providers = _build_providers(data)
    if providers:
        payload["providers"] = providers

    demographics = _build_patient_demographics(data)
    if demographics:
        payload["patient_demographics"] = demographics

    clinical = _build_clinical_context(data)
    if clinical:
        payload["clinical_context"] = clinical

    sedation = _build_sedation(data)
    if sedation:
        payload["sedation"] = sedation

    procedure_setting = _build_procedure_setting(data)
    if procedure_setting:
        payload["procedure_setting"] = procedure_setting

    equipment = _build_equipment(data)
    if equipment:
        payload["equipment"] = equipment

    procedures = _build_procedures_performed(data, families)
    if procedures:
        existing_procs = payload.get("procedures_performed")
        if isinstance(existing_procs, dict) and existing_procs:
            merged_procs: dict[str, Any] = dict(existing_procs)
            for name, proc_payload in procedures.items():
                if name not in merged_procs or merged_procs[name] in (None, "", [], {}):
                    merged_procs[name] = proc_payload
                    continue

                if isinstance(merged_procs.get(name), dict) and isinstance(proc_payload, dict):
                    current = dict(merged_procs[name])
                    for k, v in proc_payload.items():
                        if current.get(k) in (None, "", [], {}):
                            current[k] = v
                    merged_procs[name] = current
            payload["procedures_performed"] = merged_procs
        else:
            payload["procedures_performed"] = procedures

    pleural = _build_pleural_procedures(data)
    if pleural:
        payload["pleural_procedures"] = pleural

    specimens = _build_specimens(data)
    if specimens:
        payload["specimens"] = specimens

    # Always set complications to ensure it's a dict (not the list from slot extractor)
    complications = _build_complications(data)
    if complications:
        payload["complications"] = complications
    elif "complications" in payload and isinstance(payload["complications"], list):
        # Remove list-typed complications from slot extractor; schema expects dict
        del payload["complications"]

    outcomes = _build_outcomes(data)
    if outcomes:
        payload["outcomes"] = outcomes

    billing = _build_billing(data)
    if billing:
        payload["billing"] = billing

    metadata = _build_metadata(data)
    if metadata:
        payload["metadata"] = metadata

    return payload


def _build_providers(data: dict[str, Any]) -> dict[str, Any]:
    providers: dict[str, Any] = {}

    attending = data.get("attending_name")
    if attending:
        providers["attending_name"] = attending

    if data.get("attending_npi") is not None:
        providers["attending_npi"] = data.get("attending_npi")
    if data.get("fellow_name") is not None:
        providers["fellow_name"] = data.get("fellow_name")

    assistant = data.get("assistant_name") or (data.get("assistant_names") or [None])[0]
    if assistant is not None:
        providers["assistant_name"] = assistant
    if data.get("assistant_role") is not None:
        providers["assistant_role"] = data.get("assistant_role")
    if data.get("trainee_present") is not None:
        providers["trainee_present"] = data.get("trainee_present")

    if data.get("ebus_rose_available") is not None:
        providers["rose_present"] = data.get("ebus_rose_available")

    return providers


def _build_patient_demographics(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if data.get("patient_age") is not None:
        result["age_years"] = data.get("patient_age")
    gender = data.get("gender")
    if gender:
        mapping = {"M": "Male", "F": "Female"}
        result["gender"] = mapping.get(gender, gender)
    if not result:
        return {}
    return result


def _build_clinical_context(data: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if data.get("asa_class") is not None:
        context["asa_class"] = data.get("asa_class")
    if data.get("primary_indication"):
        context["primary_indication"] = data.get("primary_indication")
    if data.get("radiographic_findings"):
        context["radiographic_findings"] = data.get("radiographic_findings")
    if data.get("lesion_size_mm") is not None:
        context["lesion_size_mm"] = data.get("lesion_size_mm")
    if data.get("lesion_location"):
        context["lesion_location"] = data.get("lesion_location")
    if data.get("pet_avid") is not None:
        context["pet_avidity"] = data.get("pet_avid")
    if data.get("pet_suv_max") is not None:
        context["suv_max"] = data.get("pet_suv_max")
    if data.get("bronchus_sign_present") is not None:
        context["bronchus_sign"] = data.get("bronchus_sign_present")
    if not context:
        return {}
    return context


def _build_sedation(data: dict[str, Any]) -> dict[str, Any]:
    sedation: dict[str, Any] = {}
    sed_type = data.get("sedation_type") or data.get("anesthesia_type")
    if isinstance(sed_type, str):
        sedation["type"] = sed_type
    if data.get("anesthesia_agents"):
        sedation["agents_used"] = data.get("anesthesia_agents")
    if data.get("sedation_paralytic_used") is not None:
        sedation["paralytic_used"] = data.get("sedation_paralytic_used")
    if data.get("sedation_reversal_given") is not None:
        sedation["reversal_given"] = data.get("sedation_reversal_given")
    if data.get("sedation_reversal_agent"):
        sedation["reversal_agent"] = data.get("sedation_reversal_agent")
    if data.get("sedation_start"):
        sedation["start_time"] = data.get("sedation_start")
    if data.get("sedation_stop"):
        sedation["end_time"] = data.get("sedation_stop")
    if data.get("sedation_intraservice_minutes") is not None:
        sedation["intraservice_minutes"] = data.get("sedation_intraservice_minutes")
    return sedation


def _build_procedure_setting(data: dict[str, Any]) -> dict[str, Any]:
    """Build procedure_setting section with airway_type and other setting fields."""
    setting: dict[str, Any] = {}
    if data.get("procedure_location"):
        setting["location"] = data.get("procedure_location")
    if data.get("patient_position"):
        setting["patient_position"] = data.get("patient_position")
    if data.get("airway_type"):
        setting["airway_type"] = data.get("airway_type")
    if data.get("ett_size") is not None:
        setting["ett_size"] = data.get("ett_size")
    return setting


def _build_equipment(data: dict[str, Any]) -> dict[str, Any]:
    equipment: dict[str, Any] = {}
    # bronchoscope_type is a separate field from airway_type
    # bronchoscope_type: Diagnostic, Therapeutic, Ultrathin, EBUS, Single-use
    # airway_type: Native, ETT, Tracheostomy, LMA, iGel (goes in procedure_setting)
    if data.get("bronchoscope_type"):
        equipment["bronchoscope_type"] = data.get("bronchoscope_type")
    # Use normalized bronchoscope_outer_diameter_mm if available, otherwise airway_device_size
    # Apply normalizer to handle string values like "12 mm"
    bronch_diameter = data.get("bronchoscope_outer_diameter_mm") or data.get("airway_device_size")
    if bronch_diameter is not None:
        bronch_diameter = normalize_bronchoscope_diameter(bronch_diameter)
    if bronch_diameter is not None:
        equipment["bronchoscope_outer_diameter_mm"] = bronch_diameter
    if data.get("nav_platform"):
        equipment["navigation_platform"] = data.get("nav_platform")
    if data.get("fluoro_time_min") is not None:
        equipment["fluoroscopy_time_seconds"] = float(data["fluoro_time_min"]) * 60
    if data.get("cbct_used") is not None:
        equipment["cbct_used"] = data.get("cbct_used")
    if data.get("augmented_fluoroscopy") is not None:
        equipment["augmented_fluoroscopy"] = data.get("augmented_fluoroscopy")
    return equipment


def _build_procedures_performed(data: dict[str, Any], families: set[str]) -> dict[str, Any]:
    procedures: dict[str, Any] = {}

    if "EBUS" in families or data.get("ebus_stations_sampled"):
        linear: dict[str, Any] = {"performed": True}
        if data.get("ebus_stations_sampled"):
            linear["stations_sampled"] = data.get("ebus_stations_sampled")
        gauge = _format_ebus_needle_gauge(data.get("ebus_needle_gauge"))
        if gauge:
            linear["needle_gauge"] = gauge
        if data.get("ebus_needle_type"):
            linear["needle_type"] = data.get("ebus_needle_type")
        if data.get("ebus_elastography_used") is not None:
            linear["elastography_used"] = data.get("ebus_elastography_used")
        if data.get("ebus_elastography_pattern"):
            linear["elastography_pattern"] = data.get("ebus_elastography_pattern")
        if data.get("ebus_photodocumentation_complete") is not None:
            linear["photodocumentation_complete"] = data.get("ebus_photodocumentation_complete")
        if data.get("ebus_stations_detail"):
            linear["stations_detail"] = data.get("ebus_stations_detail")
        if data.get("linear_ebus_stations"):
            linear["stations_planned"] = data.get("linear_ebus_stations")
        procedures["linear_ebus"] = linear

    if data.get("nav_rebus_used") is not None or data.get("nav_rebus_view") or data.get("radial_ebus_probe_position"):
        radial: dict[str, Any] = {}
        if data.get("nav_rebus_used") is not None:
            radial["performed"] = bool(data.get("nav_rebus_used"))
        # Use radial_ebus_probe_position if available (normalized), otherwise nav_rebus_view
        # Apply normalizer to handle fallback values like "Aerated lung on radial EBUS"
        probe_pos = data.get("radial_ebus_probe_position") or data.get("nav_rebus_view")
        if probe_pos:
            probe_pos = normalize_radial_ebus_probe_position(probe_pos)
        if probe_pos:
            radial["probe_position"] = probe_pos
        procedures["radial_ebus"] = radial

    if "NAVIGATION" in families or data.get("nav_tool_in_lesion"):
        nav: dict[str, Any] = {"performed": True}
        if data.get("nav_tool_in_lesion") is not None:
            nav["tool_in_lesion_confirmed"] = data.get("nav_tool_in_lesion")
        if data.get("nav_sampling_tools"):
            nav["sampling_tools_used"] = data.get("nav_sampling_tools")
        if data.get("nav_divergence"):
            nav["divergence_mm"] = data.get("nav_divergence")
        if data.get("nav_target_size"):
            nav["target_size_mm"] = data.get("nav_target_size")
        if data.get("nav_target_location"):
            nav["target_location"] = data.get("nav_target_location")
        if data.get("nav_imaging_verification"):
            nav["imaging_verification"] = data.get("nav_imaging_verification")
        procedures["navigational_bronchoscopy"] = nav

    if data.get("bronch_num_tbbx") or data.get("bronch_biopsy_sites"):
        tblb: dict[str, Any] = {"performed": True}
        if data.get("bronch_num_tbbx") is not None:
            tblb["number_of_samples"] = data.get("bronch_num_tbbx")
        if data.get("bronch_biopsy_sites"):
            # Extract location strings from dict objects if needed
            sites = data.get("bronch_biopsy_sites")
            if isinstance(sites, list):
                locations = []
                for site in sites:
                    if isinstance(site, dict):
                        # Extract location from dict
                        loc = site.get("location") or site.get("lobe")
                        if loc:
                            locations.append(str(loc))
                    elif isinstance(site, str):
                        locations.append(site)
                if locations:
                    tblb["locations"] = locations
            elif isinstance(sites, str):
                tblb["locations"] = [sites]
        if "bronch_tbbx_tool" in data:
            normalized = _normalize_transbronchial_forceps_type(data.get("bronch_tbbx_tool"))
            if normalized is not None:
                tblb["forceps_type"] = normalized
        procedures["transbronchial_biopsy"] = tblb

    # Endobronchial biopsy (airway mucosa/lesion) - distinct from transbronchial biopsy
    ebx_data = data.get("endobronchial_biopsy")
    if isinstance(ebx_data, dict) and ebx_data.get("performed"):
        ebx: dict[str, Any] = {"performed": True}
        if ebx_data.get("locations"):
            ebx["locations"] = ebx_data.get("locations")
        if ebx_data.get("number_of_samples") is not None:
            ebx["number_of_samples"] = ebx_data.get("number_of_samples")
        if ebx_data.get("forceps_type"):
            ebx["forceps_type"] = ebx_data.get("forceps_type")
        procedures["endobronchial_biopsy"] = ebx
    elif ebx_data is True:
        procedures["endobronchial_biopsy"] = {"performed": True}

    if data.get("cryo_probe_size") or data.get("cryo_specimens_count"):
        cryo: dict[str, Any] = {"performed": True}
        cryo["cryoprobe_size_mm"] = data.get("cryo_probe_size")
        cryo["number_of_samples"] = data.get("cryo_specimens_count")
        cryo["freeze_time_seconds"] = data.get("cryo_freeze_time")
        procedures["transbronchial_cryobiopsy"] = cryo

    # BAL - handle both detailed volume data and simple performed flag
    bal_data = data.get("bal")
    if data.get("bal_volume_instilled") or data.get("bal_volume_returned"):
        bal: dict[str, Any] = {"performed": True}
        bal["volume_instilled_ml"] = data.get("bal_volume_instilled")
        bal["volume_returned_ml"] = data.get("bal_volume_returned")
        bal["location"] = data.get("bal_location")
        procedures["bal"] = bal
    elif isinstance(bal_data, dict) and bal_data.get("performed"):
        procedures["bal"] = {"performed": True}
    elif bal_data is True:
        procedures["bal"] = {"performed": True}

    # Therapeutic Aspiration
    ta_data = data.get("therapeutic_aspiration")
    if isinstance(ta_data, dict) and ta_data.get("performed"):
        ta: dict[str, Any] = {"performed": True}
        if ta_data.get("material"):
            ta["material"] = ta_data.get("material")
        procedures["therapeutic_aspiration"] = ta
    elif ta_data is True:
        procedures["therapeutic_aspiration"] = {"performed": True}

    stent_action = data.get("stent_action")
    stent_removal = data.get("airway_stent_removal")
    if data.get("stent_type") or stent_action or stent_removal is True:
        stent: dict[str, Any] = {"performed": True}
        stent["stent_type"] = data.get("stent_type")
        stent["action"] = stent_action
        stent["location"] = data.get("stent_location")
        stent["size"] = data.get("stent_size")
        if stent_removal is True:
            stent["airway_stent_removal"] = True
            if not stent["action"]:
                stent["action"] = "Removal"
        elif isinstance(stent_action, str) and "remov" in stent_action.lower():
            stent["airway_stent_removal"] = True
        procedures["airway_stent"] = stent

    if data.get("blvr_target_lobe") or data.get("blvr_valve_type"):
        blvr: dict[str, Any] = {"performed": True}
        blvr["target_lobe"] = data.get("blvr_target_lobe")
        blvr["valve_type"] = data.get("blvr_valve_type")
        blvr["number_of_valves"] = data.get("blvr_valve_count")
        blvr["segments_treated"] = data.get("blvr_segments_treated")
        blvr["collateral_ventilation_assessment"] = data.get("blvr_cv_assessment_method")
        procedures["blvr"] = blvr

    # Other interventions (often appear outside bronchoscopy section)
    trach_data = data.get("percutaneous_tracheostomy")
    if isinstance(trach_data, dict) and trach_data.get("performed"):
        trach: dict[str, Any] = {"performed": True}
        if trach_data.get("method"):
            trach["method"] = trach_data.get("method")
        if trach_data.get("device_name"):
            trach["device_name"] = trach_data.get("device_name")
        if trach_data.get("size"):
            trach["size"] = trach_data.get("size")
        procedures["percutaneous_tracheostomy"] = trach
    elif trach_data is True:
        procedures["percutaneous_tracheostomy"] = {"performed": True}

    neck_us_data = data.get("neck_ultrasound")
    if isinstance(neck_us_data, dict) and neck_us_data.get("performed"):
        us: dict[str, Any] = {"performed": True}
        if neck_us_data.get("vessels_visualized") is not None:
            us["vessels_visualized"] = neck_us_data.get("vessels_visualized")
        if neck_us_data.get("findings"):
            us["findings"] = neck_us_data.get("findings")
        procedures["neck_ultrasound"] = us
    elif neck_us_data is True:
        procedures["neck_ultrasound"] = {"performed": True}

    return procedures


def _build_pleural_procedures(data: dict[str, Any]) -> dict[str, Any]:
    pleural_type = data.get("pleural_procedure_type")
    if not pleural_type:
        return {}
    pleural: dict[str, Any] = {}

    def _base_fields() -> dict[str, Any]:
        return {
            "performed": True,
            "side": data.get("pleural_side"),
            "guidance": data.get("pleural_guidance"),
            "intercostal_space": data.get("pleural_intercostal_space"),
            "volume_drained_ml": data.get("pleural_volume_drained_ml"),
            "fluid_character": data.get("pleural_fluid_appearance"),
            "opening_pressure_cmh2o": data.get("pleural_opening_pressure_cmh2o"),
            "opening_pressure_measured": data.get("pleural_opening_pressure_measured"),
        }

    if pleural_type == "Thoracentesis":
        pleural["thoracentesis"] = _base_fields()
    elif pleural_type in {"Chest Tube", "Chest Tube Removal"}:
        chest = _base_fields()
        chest["action"] = pleural_type
        pleural["chest_tube"] = chest
    elif pleural_type.startswith("Tunneled"):
        ipc = _base_fields()
        ipc["action"] = pleural_type
        pleural["ipc"] = ipc
    elif pleural_type == "Medical Thoracoscopy":
        thor = _base_fields()
        thor["findings"] = data.get("pleural_thoracoscopy_findings")
        pleural["medical_thoracoscopy"] = thor
    elif pleural_type == "Chemical Pleurodesis":
        pleuro = _base_fields()
        pleuro["agent"] = data.get("pleurodesis_agent")
        pleural["pleurodesis"] = pleuro
    return pleural


def _build_complications(data: dict[str, Any]) -> dict[str, Any]:
    """Build nested complications structure per schema.

    Schema expects:
    - complication_list: list of Literal enum values
    - bleeding: {occurred: bool, severity: str, intervention_required: list}
    - pneumothorax: {occurred: bool, size: str, intervention: list}
    - respiratory: {hypoxia_occurred: bool, lowest_spo2: int, ...}
    """
    complications: dict[str, Any] = {}

    # Build complication_list - use normalized complication_list if available
    comp_list = data.get("complication_list")
    if not comp_list:
        # Fallback to bronch_immediate_complications
        raw_comp = data.get("bronch_immediate_complications")
        if raw_comp:
            # Handle both list and string inputs
            if isinstance(raw_comp, list):
                comp_list = []
                for c in raw_comp:
                    if c and c not in ("None", "none", None):
                        comp_list.append(c)
            elif isinstance(raw_comp, str) and raw_comp not in ("None", "none", ""):
                comp_list = [raw_comp]

    # Apply normalizer to convert raw values like "Bleeding" to enum values like "Bleeding - Mild"
    if comp_list:
        comp_list = normalize_complication_list(comp_list)

    if comp_list:
        complications["complication_list"] = comp_list
        complications["any_complication"] = True

    # Build bleeding substructure
    bleeding_severity = data.get("bleeding_severity")
    interventions_raw = data.get("bleeding_intervention_required")

    intervention_list: list[str] = []
    if isinstance(interventions_raw, list):
        intervention_list = [str(x).strip() for x in interventions_raw if x is not None and str(x).strip()]
    elif isinstance(interventions_raw, str) and interventions_raw.strip():
        intervention_list = [interventions_raw.strip()]

    # Evidence hard-gating for bleeding complications:
    # Only mark bleeding occurred when an intervention to control bleeding is documented.
    if intervention_list:
        # De-dupe while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for item in intervention_list:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)

        bleeding_dict: dict[str, Any] = {"occurred": True, "intervention_required": deduped}

        # Map severity to schema enum if provided
        if isinstance(bleeding_severity, str) and bleeding_severity not in ("None", "none", "None/Scant"):
            severity_mapping = {
                "Mild": "Mild (<50mL)",
                "Moderate": "Moderate (50-200mL)",
                "Severe": "Severe (>200mL)",
            }
            if bleeding_severity in severity_mapping:
                bleeding_dict["severity"] = severity_mapping[bleeding_severity]
            elif bleeding_severity in severity_mapping.values():
                bleeding_dict["severity"] = bleeding_severity

        complications["bleeding"] = bleeding_dict

    # Build pneumothorax substructure
    pneumothorax_val = data.get("pneumothorax")
    pneumothorax_intervention = data.get("pneumothorax_intervention")
    if pneumothorax_val is not None:
        # Handle bool, string, or dict input
        if isinstance(pneumothorax_val, bool):
            if pneumothorax_val:
                pneumothorax_dict: dict[str, Any] = {"occurred": True}
                if pneumothorax_intervention:
                    if isinstance(pneumothorax_intervention, list):
                        pneumothorax_dict["intervention"] = pneumothorax_intervention
                    elif isinstance(pneumothorax_intervention, str):
                        pneumothorax_dict["intervention"] = [pneumothorax_intervention]
                complications["pneumothorax"] = pneumothorax_dict
            # else: pneumothorax=False, don't add to complications
        elif isinstance(pneumothorax_val, dict):
            complications["pneumothorax"] = pneumothorax_val

    # Build respiratory substructure
    hypoxia_val = data.get("hypoxia_respiratory_failure")
    if hypoxia_val:
        # Skip if value is "None" string
        if hypoxia_val not in ("None", "none", False):
            if isinstance(hypoxia_val, dict):
                complications["respiratory"] = hypoxia_val
            elif isinstance(hypoxia_val, bool) and hypoxia_val:
                complications["respiratory"] = {"hypoxia_occurred": True}
            elif isinstance(hypoxia_val, str):
                # String value - interpret as hypoxia occurred
                complications["respiratory"] = {"hypoxia_occurred": True}

    if data.get("other_complication_details"):
        complications["other_complication_details"] = data.get("other_complication_details")

    return complications


def _build_outcomes(data: dict[str, Any]) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    if data.get("disposition"):
        outcomes["disposition"] = data.get("disposition")
    if data.get("procedure_completed") is not None:
        outcomes["procedure_completed"] = data.get("procedure_completed")
    if data.get("procedure_aborted_reason"):
        outcomes["procedure_aborted_reason"] = data.get("procedure_aborted_reason")
    return outcomes


def _build_billing(data: dict[str, Any]) -> dict[str, Any]:
    billing: dict[str, Any] = {}
    if data.get("cpt_codes"):
        entries = []
        for code in data["cpt_codes"]:
            if not code:
                continue
            entries.append({"code": str(code)})
        if entries:
            billing["cpt_codes"] = entries
    return billing


def _build_metadata(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("note_id") or data.get("source_system"):
        return {
            "note_id": data.get("note_id"),
            "source_system": data.get("source_system"),
        }
    return {}


def _build_specimens(data: dict[str, Any]) -> dict[str, Any]:
    specimens: dict[str, Any] = {}
    if data.get("ebus_rose_result"):
        specimens["rose_result"] = data.get("ebus_rose_result")
    if data.get("specimens_collected"):
        specimens["specimens_collected"] = data.get("specimens_collected")
    return specimens
