"""IP Procedure Report Macro Engine.

This module provides the primary template system for generating IP procedural reports.
It uses Jinja2 macros organized by procedure category, with the addon templates
serving as a secondary snippet library for rare events and supplementary text.

The macro system is the canonical structured layer for core procedures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from jinja2 import Environment, FileSystemLoader, Template


# Paths
_MACRO_ROOT = Path(__file__).parent / "templates" / "macros"
_SCHEMA_PATH = _MACRO_ROOT / "template_schema.json"

# Module-level cache
_MACRO_ENV: Environment | None = None
_TEMPLATE_SCHEMA: dict[str, Any] | None = None
_MACRO_REGISTRY: dict[str, dict[str, Any]] = {}


def _load_schema() -> dict[str, Any]:
    """Load the template schema JSON."""
    global _TEMPLATE_SCHEMA
    if _TEMPLATE_SCHEMA is None:
        if _SCHEMA_PATH.exists():
            _TEMPLATE_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        else:
            _TEMPLATE_SCHEMA = {"categories": {}}
    return _TEMPLATE_SCHEMA


def _build_macro_env() -> Environment:
    """Build the Jinja2 environment for macro templates."""
    global _MACRO_ENV
    if _MACRO_ENV is None:
        _MACRO_ENV = Environment(
            loader=FileSystemLoader(str(_MACRO_ROOT)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
    return _MACRO_ENV


def _build_macro_registry() -> dict[str, dict[str, Any]]:
    """Build the registry mapping macro names to their metadata and template app."""
    global _MACRO_REGISTRY
    if _MACRO_REGISTRY:
        return _MACRO_REGISTRY

    schema = _load_schema()
    env = _build_macro_env()

    # Map category files to their template modules
    category_files = {
        "01_minor_trach_laryngoscopy": "01_minor_trach_laryngoscopy.j2",
        "02_core_bronchoscopy": "02_core_bronchoscopy.j2",
        "03_navigation_robotic_ebus": "03_navigation_robotic_ebus.j2",
        "04_blvr_cryo": "04_blvr_cryo.j2",
        "05_pleural": "05_pleural.j2",
        "06_other_interventions": "06_other_interventions.j2",
        "07_clinical_assessment": "07_clinical_assessment.j2",
    }

    for category_key, category_data in schema.get("categories", {}).items():
        template_file = category_files.get(category_key)
        if not template_file:
            continue

        try:
            template = env.get_template(template_file)
            module = template.module
        except Exception:
            continue

        macros = category_data.get("macros", {})
        for macro_name, macro_meta in macros.items():
            if hasattr(module, macro_name):
                _MACRO_REGISTRY[macro_name] = {
                    "category": category_key,
                    "description": category_data.get("description", ""),
                    "cpt": macro_meta.get("cpt"),
                    "params": macro_meta.get("params", []),
                    "defaults": macro_meta.get("defaults", {}),
                    "required": macro_meta.get("required", False),
                    "essential": macro_meta.get("essential", []),
                    "essential_labels": macro_meta.get("essential_labels", {}),
                    "note": macro_meta.get("note"),
                    "template_file": template_file,
                    "callable": getattr(module, macro_name),
                }

    return _MACRO_REGISTRY


# Initialize on import
_build_macro_registry()


def get_macro(name: str) -> Callable[..., str] | None:
    """Get a macro function by name.

    Args:
        name: The macro name (e.g., "thoracentesis", "linear_ebus_tbna")

    Returns:
        The callable macro function, or None if not found
    """
    registry = _build_macro_registry()
    entry = registry.get(name)
    return entry["callable"] if entry else None


def get_macro_metadata(name: str) -> dict[str, Any] | None:
    """Get metadata for a macro by name.

    Args:
        name: The macro name

    Returns:
        Dict with category, cpt, params, defaults, required, note; or None
    """
    registry = _build_macro_registry()
    entry = registry.get(name)
    if entry:
        return {k: v for k, v in entry.items() if k != "callable"}
    return None


def list_macros() -> list[str]:
    """List all available macro names.

    Returns:
        List of macro names
    """
    return list(_build_macro_registry().keys())


def list_macros_by_category(category: str) -> list[str]:
    """List macros in a specific category.

    Args:
        category: The category key (e.g., "05_pleural")

    Returns:
        List of macro names in that category
    """
    registry = _build_macro_registry()
    return [name for name, meta in registry.items() if meta.get("category") == category]


def list_categories() -> list[str]:
    """List all available macro categories.

    Returns:
        List of category keys
    """
    schema = _load_schema()
    return list(schema.get("categories", {}).keys())


def get_category_description(category: str) -> str | None:
    """Get the description for a category.

    Args:
        category: The category key

    Returns:
        Description string or None
    """
    schema = _load_schema()
    cat_data = schema.get("categories", {}).get(category)
    return cat_data.get("description") if cat_data else None


def render_macro(name: str, **kwargs) -> str | None:
    """Render a macro with the given parameters.

    Args:
        name: The macro name
        **kwargs: Parameters to pass to the macro

    Returns:
        The rendered text, or None if macro not found
    """
    macro_fn = get_macro(name)
    if macro_fn is None:
        return None

    # Apply defaults from schema
    meta = get_macro_metadata(name)
    if meta:
        defaults = meta.get("defaults", {})
        for key, default_val in defaults.items():
            if key not in kwargs:
                kwargs[key] = default_val

    try:
        return macro_fn(**kwargs)
    except Exception as e:
        return f"[Error rendering {name}: {e}]"


def find_macros_by_cpt(cpt_code: str) -> list[dict[str, Any]]:
    """Find macros that match a specific CPT code.

    Args:
        cpt_code: The CPT code to search for

    Returns:
        List of {name, metadata} dicts
    """
    registry = _build_macro_registry()
    results = []
    cpt_str = str(cpt_code)

    for name, meta in registry.items():
        cpt = meta.get("cpt", "")
        if cpt and cpt_str in str(cpt):
            results.append({"name": name, **{k: v for k, v in meta.items() if k != "callable"}})

    return results


def get_base_utilities() -> dict[str, Callable]:
    """Get the base utility macros (specimen_list, ventilation_parameters, etc.).

    Returns:
        Dict mapping utility name to callable
    """
    env = _build_macro_env()
    try:
        base_template = env.get_template("base.j2")
        module = base_template.module
        return {
            "specimen_list": getattr(module, "specimen_list", None),
            "chest_ultrasound_findings": getattr(module, "chest_ultrasound_findings", None),
            "ventilation_parameters": getattr(module, "ventilation_parameters", None),
            "procedure_tolerance": getattr(module, "procedure_tolerance", None),
            "anesthesia_block": getattr(module, "anesthesia_block", None),
            "cxr_ordered": getattr(module, "cxr_ordered", None),
            "staff_present": getattr(module, "staff_present", None),
            "consent_block": getattr(module, "consent_block", None),
            "timeout_block": getattr(module, "timeout_block", None),
            "rose_result": getattr(module, "rose_result", None),
        }
    except Exception:
        return {}


def validate_essential_fields(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate procedures for essential fields and populate acknowledged_omissions.

    For each procedure, checks if essential fields (per schema) are present.
    Missing essential fields are added to bundle["acknowledged_omissions"][proc_id]
    with human-readable labels.

    Args:
        bundle: The procedure bundle (modified in place)

    Returns:
        The modified bundle with acknowledged_omissions populated
    """
    registry = _build_macro_registry()

    # Initialize acknowledged_omissions if not present
    if "acknowledged_omissions" not in bundle:
        bundle["acknowledged_omissions"] = {}

    procedures = bundle.get("procedures", [])

    for proc in procedures:
        proc_type = proc.get("proc_type")
        proc_id = proc.get("proc_id") or proc_type or "unknown"

        # Get params - support both "params" and "data" keys
        params = proc.get("params") or proc.get("data") or {}

        # Look up macro metadata
        meta = registry.get(proc_type, {})
        essential_fields = meta.get("essential", [])
        essential_labels = meta.get("essential_labels", {})

        missing = []
        for field in essential_fields:
            value = params.get(field)
            # Treat None, empty string, empty list as missing
            if value is None or value == "" or value == []:
                # Use human-readable label if available
                label = essential_labels.get(field, field.replace("_", " ").title())
                missing.append(label)

        if missing:
            if proc_id not in bundle["acknowledged_omissions"]:
                bundle["acknowledged_omissions"][proc_id] = []
            bundle["acknowledged_omissions"][proc_id].extend(missing)

    return bundle


def get_essential_fields(proc_type: str) -> tuple[list[str], dict[str, str]]:
    """Get essential fields and their human-readable labels for a procedure type.

    Args:
        proc_type: The procedure type/macro name

    Returns:
        Tuple of (list of essential field names, dict of field->label mappings)
    """
    registry = _build_macro_registry()
    meta = registry.get(proc_type, {})
    return meta.get("essential", []), meta.get("essential_labels", {})


def render_procedure_bundle(bundle: dict[str, Any], addon_getter: callable = None) -> str:
    """Render a complete procedure report from a bundle.

    The bundle format:
    {
        "patient": {...},
        "encounter": {...},
        "procedures": [
            {"proc_type": "thoracentesis", "params": {...}},
            {"proc_type": "linear_ebus_tbna", "params": {...}},
            ...
        ],
        "addons": ["ion_partial_registration", "cbct_spin_adjustment_1"],
        "acknowledged_omissions": {...},
        "free_text_hint": "..."
    }

    Args:
        bundle: The procedure bundle
        addon_getter: Optional function to get addon body by slug. If None,
                     attempts to import from ip_addons module.

    Returns:
        The complete rendered report
    """
    # Validate essential fields and populate acknowledged_omissions
    bundle = validate_essential_fields(bundle)

    # Get the addon body function
    if addon_getter is None:
        try:
            from app.reporting.ip_addons import get_addon_body
            addon_getter = get_addon_body
        except ImportError:
            # Fall back to loading directly
            import json
            from config.settings import KnowledgeSettings

            _addons_path = KnowledgeSettings().addon_templates_path
            if _addons_path.exists():
                _data = json.loads(_addons_path.read_text(encoding="utf-8"))
                _by_slug = {t["slug"]: t["body"] for t in _data.get("templates", [])}
                addon_getter = lambda slug: _by_slug.get(slug)
            else:
                addon_getter = lambda slug: None

    sections = []

    # Rule 2: Respect chronological ordering from source text
    # Sort by sequence if present, otherwise preserve original order (do NOT sort by CPT or type)
    procedures = bundle.get("procedures", [])
    sorted_procs = sorted(
        procedures,
        key=lambda p: (p.get("sequence") or float("inf"), procedures.index(p))
    )

    # Render each procedure using macros in chronological order
    for proc in sorted_procs:
        proc_type = proc.get("proc_type")
        # Support both "params" (new format) and "data" (schema format)
        params = proc.get("params") or proc.get("data") or {}

        if proc_type:
            rendered = render_macro(proc_type, **params)
            if rendered:
                sections.append(rendered)

    # Render addons section if present
    addons = bundle.get("addons", [])
    if addons:
        addon_texts = []
        for slug in addons:
            body = addon_getter(slug)
            if body:
                addon_texts.append(f"- {body}")

        if addon_texts:
            sections.append("\n## Additional Procedures / Events\n" + "\n".join(addon_texts))

    # Add free text hint if provided
    free_text = bundle.get("free_text_hint", "")
    if free_text:
        sections.append(f"\n## Additional Notes\n{free_text}")

    # Rule 1: Render acknowledged omissions section
    acknowledged_omissions = bundle.get("acknowledged_omissions", {})
    if acknowledged_omissions:
        omission_lines = ["## Missing Details (not present in original dictation)"]
        for key, items in acknowledged_omissions.items():
            if items:
                omission_lines.append(f"- **{key}**: {'; '.join(items)}")
        if len(omission_lines) > 1:  # More than just the header
            sections.append("\n".join(omission_lines))

    return "\n\n".join(sections)


def merge_procedure_params(
    existing: dict[str, Any],
    updates: dict[str, Any],
    allow_override: bool = False
) -> dict[str, Any]:
    """Merge updates into existing procedure params.

    Only fills in null/missing values by default. Does not change existing values
    unless allow_override is True.

    Args:
        existing: The existing params dict
        updates: New values to merge in
        allow_override: If True, updates can overwrite existing non-null values

    Returns:
        Merged params dict
    """
    merged = dict(existing)

    for key, new_value in updates.items():
        if new_value is None:
            continue

        old_value = merged.get(key)

        # Only fill if existing is null/empty or override is allowed
        if old_value is None or old_value == "" or old_value == []:
            merged[key] = new_value
        elif allow_override:
            merged[key] = new_value
        # Otherwise keep existing value

    return merged


def update_bundle(
    existing_bundle: dict[str, Any],
    updates: dict[str, Any],
    allow_override: bool = False,
    allow_new_procedures: bool = False
) -> dict[str, Any]:
    """Update an existing bundle with new/clarified information.

    This implements Phase 2 of the two-phase workflow:
    - Phase 1: Initial sketch → bundle + note + missing field list
    - Phase 2: Clarification text → patched bundle (this function)

    Rules:
    - Only fill fields explicitly supported by the new text
    - Don't reorder procedures (preserve sequence)
    - Don't change counts unless allow_override is True
    - Don't add new procedures unless allow_new_procedures is True

    Args:
        existing_bundle: The current bundle from Phase 1
        updates: Dict with same structure containing new values to merge
        allow_override: If True, updates can overwrite existing non-null values
        allow_new_procedures: If True, new procedures in updates are appended

    Returns:
        Updated bundle (creates a new dict, does not modify existing)

    Example updates dict:
    {
        "procedures": [
            {
                "proc_id": "robotic_ion_1",  # Match by proc_id
                "params": {
                    "lesion_location": "RB1",
                    "vent_params": {"mode": "VC", "respiratory_rate": 14}
                }
            }
        ],
        "acknowledged_omissions": {
            "robotic_ion_1": []  # Clear these omissions since we filled them
        }
    }
    """
    import copy
    result = copy.deepcopy(existing_bundle)

    # Update patient/encounter info if provided
    if "patient" in updates:
        result["patient"] = merge_procedure_params(
            result.get("patient", {}),
            updates["patient"],
            allow_override
        )

    if "encounter" in updates:
        result["encounter"] = merge_procedure_params(
            result.get("encounter", {}),
            updates["encounter"],
            allow_override
        )

    # Update procedures
    existing_procs = result.get("procedures", [])
    update_procs = updates.get("procedures", [])

    # Build lookup by proc_id for existing procedures
    proc_by_id = {}
    for i, proc in enumerate(existing_procs):
        pid = proc.get("proc_id") or proc.get("proc_type")
        if pid:
            proc_by_id[pid] = i

    new_procs_to_add = []

    for update_proc in update_procs:
        update_id = update_proc.get("proc_id") or update_proc.get("proc_type")

        if update_id and update_id in proc_by_id:
            # Update existing procedure
            idx = proc_by_id[update_id]
            existing_proc = existing_procs[idx]

            # Merge params
            existing_params = existing_proc.get("params") or existing_proc.get("data") or {}
            update_params = update_proc.get("params") or update_proc.get("data") or {}

            merged_params = merge_procedure_params(
                existing_params,
                update_params,
                allow_override
            )

            # Handle nested dicts like vent_params
            for key in update_params:
                if isinstance(update_params.get(key), dict) and isinstance(existing_params.get(key), dict):
                    merged_params[key] = merge_procedure_params(
                        existing_params[key],
                        update_params[key],
                        allow_override
                    )

            # Update the procedure (preserve original key - params or data)
            if "params" in existing_proc:
                existing_procs[idx]["params"] = merged_params
            else:
                existing_procs[idx]["data"] = merged_params

        elif allow_new_procedures:
            # This is a new procedure
            new_procs_to_add.append(update_proc)

    # Append new procedures at the end (with sequence numbers)
    if new_procs_to_add:
        max_seq = max(
            (p.get("sequence") or 0 for p in existing_procs),
            default=0
        )
        for proc in new_procs_to_add:
            if "sequence" not in proc:
                max_seq += 1
                proc["sequence"] = max_seq
            existing_procs.append(proc)

    result["procedures"] = existing_procs

    # Update acknowledged_omissions
    # If an update specifies an empty list for a proc_id, clear those omissions
    if "acknowledged_omissions" in updates:
        if "acknowledged_omissions" not in result:
            result["acknowledged_omissions"] = {}

        for proc_id, items in updates["acknowledged_omissions"].items():
            if items == [] or items is None:
                # Clear omissions for this proc_id
                if proc_id in result["acknowledged_omissions"]:
                    del result["acknowledged_omissions"][proc_id]
            elif isinstance(items, list):
                # Replace or set
                result["acknowledged_omissions"][proc_id] = items

    # Update addons if provided
    if "addons" in updates:
        existing_addons = set(result.get("addons", []))
        new_addons = updates["addons"]
        if isinstance(new_addons, list):
            existing_addons.update(new_addons)
        result["addons"] = list(existing_addons)

    # Update free_text_hint if provided
    if "free_text_hint" in updates:
        existing_hint = result.get("free_text_hint", "")
        new_hint = updates["free_text_hint"]
        if new_hint:
            if existing_hint:
                result["free_text_hint"] = f"{existing_hint}\n{new_hint}"
            else:
                result["free_text_hint"] = new_hint

    return result


def get_missing_fields_summary(bundle: dict[str, Any]) -> str:
    """Generate a human-readable summary of missing fields for UI display.

    This is used in Phase 1 to show the user what's missing and prompt for
    clarification.

    Args:
        bundle: The procedure bundle (should have acknowledged_omissions populated)

    Returns:
        Formatted string for display, e.g.:
        "Missing / incomplete details:
        - robotic_ion_1: Lesion segment; Ventilation parameters
        - global: Referring physician"
    """
    # First validate to ensure acknowledged_omissions is populated
    validated = validate_essential_fields(bundle)
    omissions = validated.get("acknowledged_omissions", {})

    if not omissions:
        return ""

    lines = ["**Missing / incomplete structured details from this dictation:**", ""]

    for proc_id, items in omissions.items():
        if items:
            items_str = "; ".join(items)
            lines.append(f"- **{proc_id}**: {items_str}")

    if len(lines) <= 2:
        return ""

    lines.append("")
    lines.append("_To fill any of these, provide additional natural language. Example:_")
    lines.append("_\"Segment RB1. Vent VC 14/450/5/40%. Specimens to histology, cultures.\"_")

    return "\n".join(lines)


def render_bundle_with_summary(bundle: dict[str, Any], addon_getter: callable = None) -> tuple[str, str]:
    """Render a bundle and return both the report and missing fields summary.

    This is the main entry point for Phase 1 of the two-phase workflow.

    Args:
        bundle: The procedure bundle
        addon_getter: Optional function to get addon body by slug

    Returns:
        Tuple of (rendered_report, missing_fields_summary)
    """
    report = render_procedure_bundle(bundle, addon_getter)
    summary = get_missing_fields_summary(bundle)

    return report, summary


# Category mapping for convenience
CATEGORY_MACROS = {
    "minor_trach_laryngoscopy": [
        "minor_trach_bleeding",
        "chemical_cauterization",
        "trach_tube_change",
        "percutaneous_trach",
        "trach_revision",
        "tracheobronchoscopy_via_trach",
        "trach_decannulation",
        "trach_downsize",
        "granulation_debridement",
        "flexible_laryngoscopy",
    ],
    "core_bronchoscopy": [
        "bronchial_washing",
        "bronchial_brushings",
        "bronchoalveolar_lavage",
        "endobronchial_biopsy",
        "transbronchial_lung_biopsy",
        "transbronchial_needle_aspiration",
        "balloon_dilation",
        "airway_stent_placement",
        "endobronchial_tumor_destruction",
        "tumor_destruction_multimodal",
        "therapeutic_aspiration",
        "endobronchial_tumor_excision",
        "rigid_bronchoscopy",
        "foreign_body_removal",
        "endobronchial_hemostasis",
        "endobronchial_blocker",
        "awake_foi",
        "dlt_confirmation",
        "stent_surveillance",
    ],
    "navigation_robotic_ebus": [
        "emn_bronchoscopy",
        "fiducial_marker_placement",
        "radial_ebus_survey",
        "robotic_bronchoscopy_ion",
        "ion_registration_complete",
        "ion_registration_partial",
        "ion_registration_drift",
        "linear_ebus_tbna",
        "rebus_guide_sheath_sampling",
        "cbct_assisted_bronchoscopy",
        "transbronchial_dye_marking",
        "robotic_bronchoscopy_monarch",
        "ebus_intranodal_forceps_biopsy",
        "ebus_19g_fnb",
    ],
    "blvr_cryo": [
        "endobronchial_valve_placement",
        "endobronchial_valve_removal_exchange",
        "post_blvr_protocol",
        "transbronchial_cryobiopsy",
        "endobronchial_cryoablation",
        "cryoablation_alternative",
    ],
    "pleural": [
        "chest_tube_placement",
        "thoracentesis",
        "thoracentesis_manometry",
        "tpc_placement",
        "tpc_removal",
        "intrapleural_fibrinolysis",
        "medical_thoracoscopy",
        "pigtail_catheter_placement",
        "transthoracic_needle_biopsy",
        "thoravent_placement",
        "chemical_pleurodesis_chest_tube",
        "chemical_pleurodesis_ipc",
        "ipc_exchange",
        "chest_tube_exchange",
        "chest_tube_removal",
        "us_guided_pleural_biopsy",
        "focused_thoracic_ultrasound",
    ],
    "other_interventions": [
        "whole_lung_lavage",
        "eus_b",
        "paracentesis",
        "peg_placement",
        "peg_removal_exchange",
        "bpf_localization",
        "ebv_for_air_leak",
        "bpf_sealant_application",
    ],
    "clinical_assessment": [
        "pre_anesthesia_assessment",
        "general_bronchoscopy_note",
        "ip_operative_report",
        "tpc_discharge_instructions",
        "blvr_discharge_instructions",
        "chest_tube_discharge_instructions",
        "peg_discharge_instructions",
    ],
}


__all__ = [
    "get_macro",
    "get_macro_metadata",
    "list_macros",
    "list_macros_by_category",
    "list_categories",
    "get_category_description",
    "render_macro",
    "find_macros_by_cpt",
    "get_base_utilities",
    "validate_essential_fields",
    "get_essential_fields",
    "render_procedure_bundle",
    "merge_procedure_params",
    "update_bundle",
    "get_missing_fields_summary",
    "render_bundle_with_summary",
    "CATEGORY_MACROS",
]
