"""Registry data structures built from the configured JSON schema.

Implementation note: this module holds the dynamic RegistryRecord builder and related
type overrides. The stable public import surface remains `app.registry.schema`.
"""

from __future__ import annotations

import json
import re
from uuid import UUID
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_serializer,
    field_validator,
    model_validator,
)

from config.settings import KnowledgeSettings
from app.common.spans import Span

from app.registry.schema.ebus_events import NodeActionType, NodeInteraction, NodeOutcomeType
from app.registry.schema_granular import (
    AirwayDeviceActionProcedure,
    AirwayStentProcedure,
    ClinicalContext,
    EnhancedRegistryGranularData,
    IPCProcedure,
    PatientDemographics,
    derive_aggregate_fields,
    validate_ebus_consistency,
)

_SCHEMA_PATH = KnowledgeSettings().registry_schema_path


class LinearEBUSProcedure(BaseModel):
    """Custom type override for procedures_performed.linear_ebus.

    Adds `node_events` for granular per-station actions while remaining
    backward-compatible with the JSON schema fields.
    """

    model_config = ConfigDict(extra="ignore")

    performed: bool | None = None
    stations_sampled: list[str] | None = None
    targets_sampled: list[str] | None = None
    stations_planned: list[str] | None = None
    stations_detail: list[dict[str, Any]] | None = None
    passes_per_station: int | None = None
    needle_gauge: str | None = None
    needle_type: str | None = None
    photodocumentation_complete: bool | None = None
    elastography_used: bool | None = None
    elastography_pattern: str | None = None
    doppler_used: bool | None = None

    node_events: list[NodeInteraction] = Field(
        default_factory=list,
        description="Per-station interactions, including inspected-only vs sampled actions.",
    )


ThermalAblationModality = Literal[
    "APC",
    "Electrocautery",
    "Laser (Nd:YAG)",
    "Laser (CO2)",
    "Laser (Diode)",
]


class ThermalAblationProcedure(BaseModel):
    """Custom type override for procedures_performed.thermal_ablation.

    The JSON schema currently models a single modality, but real-world cases can
    include multiple modalities (e.g., APC + electrocautery). This override
    normalizes incoming values into a list of enum modalities.
    """

    model_config = ConfigDict(extra="ignore")

    performed: bool | None = None
    modality: list[ThermalAblationModality] | None = None
    location: str | None = None
    indication: Literal[
        "Tumor debulking",
        "Hemostasis",
        "Granulation tissue",
        "Web/stenosis",
        "Other",
    ] | None = None
    power_setting_watts: float | None = Field(default=None, ge=0)
    apc_flow_rate: float | None = Field(default=None, ge=0)

    @field_validator("modality", mode="before")
    @classmethod
    def normalize_thermal_ablation_modality(cls, v):
        if v is None:
            return None

        def _normalize_one(raw: str) -> ThermalAblationModality | None:
            s = str(raw or "").strip()
            if not s:
                return None
            lowered = s.lower()
            if lowered in {"apc", "argon plasma", "argon plasma coagulation"}:
                return "APC"
            if "argon plasma" in lowered or lowered == "apc":
                return "APC"
            if "electrocaut" in lowered or re.search(r"\bcauter(y|iz)", lowered):
                return "Electrocautery"
            if "nd" in lowered and "yag" in lowered or "nd:yag" in lowered:
                return "Laser (Nd:YAG)"
            if "co2" in lowered:
                return "Laser (CO2)"
            if "diode" in lowered:
                return "Laser (Diode)"
            # Allow already-normalized canonical values.
            canonical = {
                "apc": "APC",
                "electrocautery": "Electrocautery",
                "laser (nd:yag)": "Laser (Nd:YAG)",
                "laser (co2)": "Laser (CO2)",
                "laser (diode)": "Laser (Diode)",
            }.get(lowered)
            return canonical  # type: ignore[return-value]

        if isinstance(v, str):
            split_re = re.compile(r"\s*(?:;|,|/|\band\b)\s*", re.IGNORECASE)
            tokens = [tok for tok in split_re.split(v) if tok and tok.strip()]
            out: list[ThermalAblationModality] = []
            for tok in tokens:
                norm = _normalize_one(tok)
                if norm and norm not in out:
                    out.append(norm)
            return out or None

        if isinstance(v, list):
            out: list[ThermalAblationModality] = []
            for item in v:
                norm = _normalize_one(str(item))
                if norm and norm not in out:
                    out.append(norm)
            return out or None

        # Unsupported type; leave as-is for downstream validation to handle.
        return v


class PeripheralTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_key: str
    laterality: Literal["R", "L"] | None = None
    lobe: str | None = None
    segment: str | None = None
    size_mm_long: int | None = None
    size_mm_short: int | None = None
    size_mm_cc: int | None = None
    density: Literal["Solid", "Part-solid", "GGO", "Cavitary", "Other"] | None = None
    pet_avid: bool | None = None
    pet_suvmax: float | None = None
    pet_delayed_suvmax: float | None = None
    comparative_change: Literal["Increased", "Decreased", "Stable", "New", "Resolved"] | None = None
    source_event_id: UUID | None = None
    source_relative_day_offset: int | None = None
    path_result: Literal["Positive", "Negative", "Non-diagnostic", "Atypical", "Suspicious"] | None = None
    path_diagnosis_text: str | None = None
    path_source_event_id: UUID | None = None
    path_relative_day_offset: int | None = None


class MediastinalTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    station: str | None = None
    location_text: str | None = None
    short_axis_mm: int | None = None
    pet_avid: bool | None = None
    pet_suvmax: float | None = None
    pet_delayed_suvmax: float | None = None
    comparative_change: Literal["Increased", "Decreased", "Stable", "New", "Resolved"] | None = None
    source_event_id: UUID | None = None
    source_relative_day_offset: int | None = None


class CaseTargets(BaseModel):
    model_config = ConfigDict(extra="ignore")

    peripheral_targets: list[PeripheralTarget] = Field(default_factory=list)
    mediastinal_targets: list[MediastinalTarget] = Field(default_factory=list)


class ImagingSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relative_day_offset: int
    modality: Literal["ct", "pet_ct", "cta", "mri", "other"]
    subtype: Literal["preop", "followup", "restaging", "surveillance"] | None = None
    event_title: str | None = None
    response: Literal["Progression", "Stable", "Response", "Mixed", "Indeterminate"] | None = None
    overall_impression_text: str | None = None
    qa_flags: list[str] = Field(default_factory=list)


class ImagingSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    baseline: ImagingSnapshot | None = None
    followups: list[ImagingSnapshot] = Field(default_factory=list)


class ClinicalUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relative_day_offset: int
    update_type: Literal["clinical_update", "treatment_update", "complication", "tumor_board", "other"]
    event_title: str | None = None
    performance_status_text: str | None = None
    symptom_change: Literal["Better", "Worse", "Stable"] | None = None
    treatment_change_text: str | None = None
    complication_text: str | None = None
    summary_text: str | None = None
    hospital_admission: bool | None = None
    icu_admission: bool | None = None
    deceased: bool | None = None
    disease_status: Literal["Progression", "Stable", "Response", "Mixed", "Indeterminate"] | None = None
    source_event_id: UUID | None = None
    qa_flags: list[str] = Field(default_factory=list)


class ClinicalState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relative_day_offset: int | None = None
    performance_status_text: str | None = None
    symptom_change: Literal["Better", "Worse", "Stable"] | None = None
    treatment_change_text: str | None = None
    complication_text: str | None = None
    summary_text: str | None = None
    hospital_admission: bool | None = None
    icu_admission: bool | None = None
    deceased: bool | None = None
    disease_status: Literal["Progression", "Stable", "Response", "Mixed", "Indeterminate"] | None = None
    source_event_id: UUID | None = None


class ClinicalCourse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    updates: list[ClinicalUpdate] = Field(default_factory=list)
    current_state: ClinicalState | None = None

# Optional overrides for individual fields (identified via dotted paths).
CUSTOM_FIELD_TYPES: dict[tuple[str, ...], Any] = {}
CUSTOM_FIELD_TYPES[("RegistryRecord", "pleural_procedures", "ipc")] = IPCProcedure
CUSTOM_FIELD_TYPES[("RegistryRecord", "clinical_context")] = ClinicalContext
CUSTOM_FIELD_TYPES[("RegistryRecord", "patient_demographics")] = PatientDemographics
CUSTOM_FIELD_TYPES[("RegistryRecord", "procedures_performed", "airway_device_action")] = AirwayDeviceActionProcedure
CUSTOM_FIELD_TYPES[("RegistryRecord", "procedures_performed", "airway_stent")] = AirwayStentProcedure
CUSTOM_FIELD_TYPES[("RegistryRecord", "procedures_performed", "airway_stent_revision")] = AirwayStentProcedure
CUSTOM_FIELD_TYPES[("RegistryRecord", "procedures_performed", "linear_ebus")] = LinearEBUSProcedure
CUSTOM_FIELD_TYPES[("RegistryRecord", "procedures_performed", "linear_ebus", "stations_detail")] = list[dict[str, Any]]
CUSTOM_FIELD_TYPES[("RegistryRecord", "procedures_performed", "thermal_ablation")] = ThermalAblationProcedure
CUSTOM_FIELD_TYPES[("RegistryRecord", "targets")] = CaseTargets
CUSTOM_FIELD_TYPES[("RegistryRecord", "imaging_summary")] = ImagingSummary
CUSTOM_FIELD_TYPES[("RegistryRecord", "clinical_course")] = ClinicalCourse
_MODEL_CACHE: dict[tuple[str, ...], type[BaseModel]] = {}


def _load_schema() -> dict[str, Any]:
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Registry schema not found at {_SCHEMA_PATH}")
    return json.loads(_SCHEMA_PATH.read_text())


def _pascal_case(parts: list[str]) -> str:
    tokens = []
    for part in parts:
        tokens.extend(re.split(r"[^0-9A-Za-z]", part))
    return "".join(token.capitalize() for token in tokens if token)


def _schema_type(prop: dict[str, Any], path: tuple[str, ...]) -> Any:
    override = CUSTOM_FIELD_TYPES.get(path)
    if override:
        return override

    enum = prop.get("enum")
    if enum:
        values = tuple(v for v in enum if v is not None)
        if values:
            return Literal[values]  # type: ignore[arg-type]

    typ = prop.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)

    if typ == "string":
        return str
    if typ == "number":
        return float
    if typ == "integer":
        return int
    if typ == "boolean":
        return bool
    if typ == "array":
        items = prop.get("items") or {}
        item_type = _schema_type(items, path + ("item",))
        return list[item_type]  # type: ignore[index]
    if typ == "object" or prop.get("properties"):
        properties = prop.get("properties") or {}
        additional = prop.get("additionalProperties")
        if not properties:
            if additional is True:
                return dict[str, Any]
            if isinstance(additional, dict):
                value_type = _schema_type(additional, path + ("value",))
                return dict[str, value_type]  # type: ignore[index]
        return _build_submodel(path, prop)
    return Any


def _build_submodel(path: tuple[str, ...], schema: dict[str, Any]) -> type[BaseModel]:
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]

    properties = schema.get("properties", {})
    field_defs: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        field_type = _schema_type(prop, path + (name,))
        field_defs[name] = (field_type | None, Field(default=None))  # type: ignore[operator]

    model_name = _pascal_case(list(path)) or "RegistrySubModel"
    model = create_model(
        model_name,
        __config__=ConfigDict(extra="ignore"),
        **field_defs,  # type: ignore[arg-type]
    )
    _MODEL_CACHE[path] = model
    return model


def _build_registry_model() -> type[BaseModel]:
    schema = _load_schema()
    base_model = _build_submodel(("RegistryRecord",), schema)

    class RegistryRecord(base_model):  # type: ignore[misc,valid-type]
        """Concrete registry record model with evidence fields.

        Includes optional granular per-site data for research/QI.
        When granular_data is present, aggregate fields can be derived automatically
        using derive_aggregate_fields() for backward compatibility.
        """

        model_config = ConfigDict(extra="ignore")

        evidence: dict[str, list[Span]] = Field(default_factory=dict)
        version: str | None = None
        procedure_families: list[str] = Field(default_factory=list)
        established_tracheostomy_route: bool = Field(
            default=False,
            description="True when bronchoscopy/tracheoscopy is performed through an established tracheostomy route.",
        )
        linear_ebus_stations: list[str] | None = Field(default=None)
        ebus_stations_sampled: list[str] | None = Field(default=None)
        ebus_stations_detail: list[dict[str, Any]] | None = Field(default=None)
        follow_up_plan: str | None = Field(default=None)
        ebus_systematic_staging: bool | None = Field(default=None, exclude=True)
        ebus_scope_brand: str | None = Field(default=None, exclude=True)
        pleural_procedure_type: str | None = Field(default=None, exclude=True)
        pleural_side: str | None = Field(default=None, exclude=True)
        pleural_fluid_volume: str | float | None = Field(default=None, exclude=True)
        pleural_volume_drained_ml: float | None = Field(default=None, exclude=True)
        pleural_fluid_appearance: str | None = Field(default=None, exclude=True)
        pleural_guidance: str | None = Field(default=None, exclude=True)
        pleural_intercostal_space: str | None = Field(default=None, exclude=True)
        pleural_catheter_type: str | None = Field(default=None, exclude=True)
        pleural_pleurodesis_agent: str | None = Field(default=None, exclude=True)
        pleural_opening_pressure_measured: bool | None = Field(default=None, exclude=True)
        pleural_opening_pressure_cmh2o: float | None = Field(default=None, exclude=True)
        pleural_thoracoscopy_findings: str | None = Field(default=None, exclude=True)
        bronch_num_tbbx: int | None = Field(default=None, exclude=True)
        bronch_tbbx_tool: str | None = Field(default=None, exclude=True)

        # Granular per-site data (EBUS stations, navigation targets, CAO sites, etc.)
        granular_data: EnhancedRegistryGranularData | None = Field(
            default=None,
            description="Optional granular per-site registry data for research/QI. "
                        "Complements existing aggregate fields for backward compatibility."
        )

        # Validation warnings from granular data consistency checks
        granular_validation_warnings: list[str] = Field(default_factory=list)

        @field_serializer("procedure_setting")
        @classmethod
        def serialize_procedure_setting(cls, value):
            """Serialize nested procedure_setting (required by the UI)."""
            if value is None:
                return None
            if isinstance(value, BaseModel):
                return value.model_dump()
            return value

        @model_validator(mode="before")
        @classmethod
        def hoist_granular_arrays(cls, values: Any):
            """Ensure legacy flat granular arrays get nested under granular_data."""
            if not isinstance(values, dict):
                return values

            granular = values.get("granular_data")
            if granular is None:
                granular_dict: dict[str, Any] = {}
            elif isinstance(granular, BaseModel):
                granular_dict = granular.model_dump()
            elif isinstance(granular, dict):
                granular_dict = dict(granular)
            else:
                # Unsupported type (e.g., list) - leave values untouched
                return values

            moved = False
            for field in GRANULAR_ARRAY_FIELDS:
                if field in values and values[field] is not None:
                    granular_dict.setdefault(field, values.pop(field))
                    moved = True

            if moved or granular_dict:
                values["granular_data"] = granular_dict
            else:
                values.pop("granular_data", None)
            return values

        @model_validator(mode="before")
        @classmethod
        def map_pleural_legacy_fields(cls, values: Any):
            """Map legacy pleural flat fields <-> pleural_procedures for compatibility."""
            if not isinstance(values, dict):
                return values

            pleural_type = values.get("pleural_procedure_type")

            pleural_raw = values.get("pleural_procedures")
            pleural_dict: dict[str, Any] | None = None
            if isinstance(pleural_raw, BaseModel):
                pleural_dict = pleural_raw.model_dump()
            elif isinstance(pleural_raw, dict):
                pleural_dict = dict(pleural_raw)

            def _map_guidance_to_schema(guidance: Any) -> str | None:
                if guidance is None:
                    return None
                if not isinstance(guidance, str):
                    return None
                normalized = guidance.strip()
                if not normalized:
                    return None
                if normalized.lower() == "blind":
                    return "None/Landmark"
                return normalized

            def _infer_legacy_from_nested(pleural: dict[str, Any]) -> dict[str, Any]:
                # Prefer thoracentesis over other pleural procedures for the legacy flat field.
                if isinstance(pleural.get("thoracentesis"), dict) and pleural["thoracentesis"].get("performed"):
                    thora = pleural["thoracentesis"]
                    legacy: dict[str, Any] = {"pleural_procedure_type": "Thoracentesis"}
                    legacy["pleural_side"] = thora.get("side")
                    guidance = thora.get("guidance")
                    if guidance == "None/Landmark":
                        legacy["pleural_guidance"] = "Blind"
                    else:
                        legacy["pleural_guidance"] = guidance
                    legacy["pleural_opening_pressure_cmh2o"] = thora.get("opening_pressure_cmh2o")
                    if thora.get("opening_pressure_cmh2o") is not None:
                        legacy["pleural_opening_pressure_measured"] = True
                    return legacy

                if isinstance(pleural.get("chest_tube"), dict) and pleural["chest_tube"].get("performed"):
                    tube = pleural["chest_tube"]
                    action = tube.get("action")
                    legacy_type = "Chest Tube Removal" if action == "Removal" else "Chest Tube"
                    legacy = {"pleural_procedure_type": legacy_type}
                    legacy["pleural_side"] = tube.get("side")
                    guidance = tube.get("guidance")
                    if guidance == "None":
                        legacy["pleural_guidance"] = "Blind"
                    else:
                        legacy["pleural_guidance"] = guidance
                    return legacy

                if isinstance(pleural.get("ipc"), dict) and pleural["ipc"].get("performed"):
                    ipc = pleural["ipc"]
                    action = ipc.get("action")
                    if action == "Removal":
                        legacy_type = "Tunneled Catheter Removal"
                    else:
                        legacy_type = "Tunneled Catheter"
                    legacy = {"pleural_procedure_type": legacy_type}
                    legacy["pleural_side"] = ipc.get("side")
                    return legacy

                if isinstance(pleural.get("medical_thoracoscopy"), dict) and pleural["medical_thoracoscopy"].get("performed"):
                    thor = pleural["medical_thoracoscopy"]
                    legacy = {"pleural_procedure_type": "Medical Thoracoscopy"}
                    legacy["pleural_side"] = thor.get("side")
                    return legacy

                if isinstance(pleural.get("pleurodesis"), dict) and pleural["pleurodesis"].get("performed"):
                    pleuro = pleural["pleurodesis"]
                    legacy = {"pleural_procedure_type": "Chemical Pleurodesis"}
                    legacy["pleural_side"] = pleuro.get("side")
                    return legacy

                return {}

            def _build_nested_from_legacy() -> dict[str, Any]:
                base: dict[str, Any] = {"performed": True}
                if values.get("pleural_side") is not None:
                    base["side"] = values.get("pleural_side")

                guidance = _map_guidance_to_schema(values.get("pleural_guidance"))
                if guidance is not None:
                    base["guidance"] = guidance

                if pleural_type == "Thoracentesis":
                    thora = dict(base)
                    if values.get("pleural_volume_drained_ml") is not None:
                        thora["volume_removed_ml"] = values.get("pleural_volume_drained_ml")
                    if values.get("pleural_opening_pressure_cmh2o") is not None:
                        thora["opening_pressure_cmh2o"] = values.get("pleural_opening_pressure_cmh2o")
                        thora["manometry_performed"] = True
                    return {"thoracentesis": thora}

                if pleural_type in {"Chest Tube", "Chest Tube Removal"}:
                    tube = dict(base)
                    tube["action"] = "Removal" if pleural_type == "Chest Tube Removal" else "Insertion"
                    return {"chest_tube": tube}

                if isinstance(pleural_type, str) and pleural_type.startswith("Tunneled"):
                    ipc = {"performed": True}
                    if values.get("pleural_side") is not None:
                        ipc["side"] = values.get("pleural_side")
                    ipc["action"] = "Insertion"
                    return {"ipc": ipc}

                if pleural_type == "Medical Thoracoscopy":
                    thor = {"performed": True}
                    if values.get("pleural_side") is not None:
                        thor["side"] = values.get("pleural_side")
                    return {"medical_thoracoscopy": thor}

                if pleural_type == "Chemical Pleurodesis":
                    pleuro = {"performed": True}
                    if values.get("pleural_side") is not None:
                        pleuro["side"] = values.get("pleural_side")
                    if values.get("pleural_pleurodesis_agent") is not None:
                        pleuro["agent"] = values.get("pleural_pleurodesis_agent")
                    return {"pleurodesis": pleuro}

                return {}

            # Legacy -> nested (only if nested missing)
            if pleural_type and not pleural_dict:
                built = _build_nested_from_legacy()
                if built:
                    values["pleural_procedures"] = built

            # Nested -> legacy (only fill missing flat fields)
            if pleural_dict and not pleural_type:
                inferred = _infer_legacy_from_nested(pleural_dict)
                for key, val in inferred.items():
                    if values.get(key) is None and val is not None:
                        values[key] = val

            return values

        @model_validator(mode="before")
        @classmethod
        def canonicalize_literal_values(cls, values: Any):
            """Normalize common literal mismatches before field validation.

            This is a choke point to prevent Pydantic Literal crashes when upstream
            extractors or self-correction patches use near-miss values.
            """
            if not isinstance(values, dict):
                return values

            procedures_raw = values.get("procedures_performed")
            if procedures_raw is None:
                return values

            if isinstance(procedures_raw, BaseModel):
                procedures: dict[str, Any] = procedures_raw.model_dump()
            elif isinstance(procedures_raw, dict):
                procedures = dict(procedures_raw)
            else:
                return values

            updated = False

            aspiration_raw = procedures.get("therapeutic_aspiration")
            if isinstance(aspiration_raw, BaseModel):
                aspiration: dict[str, Any] = aspiration_raw.model_dump()
            elif isinstance(aspiration_raw, dict):
                aspiration = dict(aspiration_raw)
            else:
                aspiration = {}

            material = aspiration.get("material")
            if isinstance(material, str):
                material_norm = material.strip()
                material_key = material_norm.lower()
                if material_key in {"blood clot", "bloodclot", "blood_clot"}:
                    aspiration["material"] = "Blood/clot"
                elif material_key in {"blood/clot", "blood / clot"}:
                    aspiration["material"] = "Blood/clot"
                elif material_key in {"mucus plug", "mucusplug", "mucus plugging"}:
                    aspiration["material"] = "Mucus plug"
                elif material_key in {"secretions/mucus", "mucus/secretions", "mucus"}:
                    aspiration["material"] = "Mucus"
                else:
                    # Keyword-based normalization (avoid inferring "Purulent" from "thick" alone).
                    if (
                        "purulent" in material_key
                        or "pus" in material_key
                        or "infect" in material_key
                        or ("white-yellow" in material_key)
                        or ("yellow-white" in material_key)
                        or ("white" in material_key and "yellow" in material_key)
                    ):
                        aspiration["material"] = "Purulent secretions"
                    elif "blood" in material_key or "clot" in material_key:
                        aspiration["material"] = "Blood/clot"
                    elif "plug" in material_key and ("mucus" in material_key or "mucous" in material_key):
                        aspiration["material"] = "Mucus plug"
                    elif "mucus" in material_key or "mucous" in material_key:
                        aspiration["material"] = "Mucus"
                    elif "secretions" in material_key:
                        aspiration["material"] = "Mucus"

            if aspiration and aspiration != aspiration_raw:
                procedures["therapeutic_aspiration"] = aspiration
                updated = True

            blvr_raw = procedures.get("blvr")
            if isinstance(blvr_raw, BaseModel):
                blvr: dict[str, Any] = blvr_raw.model_dump()
            elif isinstance(blvr_raw, dict):
                blvr = dict(blvr_raw)
            else:
                blvr = {}

            procedure_type = blvr.get("procedure_type")
            if isinstance(procedure_type, str):
                cleaned = procedure_type.strip()
                if cleaned:
                    normalized_key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
                    mapping: dict[str, str] = {
                        "valve placement": "Valve placement",
                        "placement": "Valve placement",
                        "insert": "Valve placement",
                        "insertion": "Valve placement",
                        "valve insertion": "Valve placement",
                        "valve removal": "Valve removal",
                        "removal": "Valve removal",
                        "remove": "Valve removal",
                        "extraction": "Valve removal",
                        "valve assessment": "Valve assessment",
                        "assessment": "Valve assessment",
                        "chartis": "Valve assessment",
                        "coil placement": "Coil placement",
                        "coil": "Coil placement",
                    }
                    normalized = mapping.get(normalized_key)
                    if normalized and normalized != cleaned:
                        blvr["procedure_type"] = normalized
                        procedures["blvr"] = blvr
                        updated = True

            if updated:
                values["procedures_performed"] = procedures
            return values

        @model_validator(mode="before")
        @classmethod
        def migrate_providers_team(cls, values: Any):
            """Populate providers_team from legacy providers (backward compatible)."""
            if not isinstance(values, dict):
                return values
            if values.get("providers_team") is not None:
                return values

            providers = values.get("providers")
            if providers is None:
                return values
            if isinstance(providers, BaseModel):
                providers_dict = providers.model_dump()
            elif isinstance(providers, dict):
                providers_dict = dict(providers)
            else:
                return values

            team: list[dict[str, Any]] = []

            attending_name = providers_dict.get("attending_name")
            if attending_name:
                team.append(
                    {
                        "role": "attending",
                        "name": attending_name,
                        "npi": providers_dict.get("attending_npi"),
                    }
                )

            fellow_name = providers_dict.get("fellow_name")
            if fellow_name:
                team.append(
                    {
                        "role": "fellow",
                        "name": fellow_name,
                        "fellow_pgy_level": providers_dict.get("fellow_pgy_level"),
                    }
                )

            assistant_name = providers_dict.get("assistant_name")
            if assistant_name:
                team.append(
                    {
                        "role": "assistant",
                        "name": assistant_name,
                        "assistant_role": providers_dict.get("assistant_role"),
                    }
                )

            if team:
                values["providers_team"] = team
            return values

    RegistryRecord.__name__ = "RegistryRecord"
    return RegistryRecord


GRANULAR_ARRAY_FIELDS = (
    "linear_ebus_stations_detail",
    "navigation_targets",
    "cao_interventions_detail",
    "blvr_valve_placements",
    "blvr_chartis_measurements",
    "cryobiopsy_sites",
    "thoracoscopy_findings_detail",
    "specimens_collected",
)


RegistryRecord = _build_registry_model()

class BLVRData(BaseModel):
    lobes: list[str]
    valve_count: int | None
    valve_details: list[dict[str, Any]]
    manufacturer: str | None
    chartis: dict[str, str]

class DestructionEvent(BaseModel):
    modality: str
    site: str

class EnhancedDilationEvent(BaseModel):
    site: str
    balloon_size: str | None
    inflation_pressure: str | None

class AspirationEvent(BaseModel):
    site: str
    volume: str | None
    character: str | None


class CaoIntervention(BaseModel):
    """Structured data for a single CAO intervention site.

    Supports multi-site CAO procedures where each airway segment may have
    different pre/post obstruction levels and treatment modalities.
    """

    location: str  # e.g., "RML", "RLL", "BI", "LMS", "distal_trachea"
    pre_obstruction_pct: int | None = None  # 0-100, approximate pre-procedure obstruction
    post_obstruction_pct: int | None = None  # 0-100, approximate post-procedure obstruction
    modalities: list[str] = Field(default_factory=list)  # e.g., ["APC", "cryo", "mechanical"]
    notes: str | None = None  # Additional context (e.g., "Post-obstructive pus drained")


class BiopsySite(BaseModel):
    """Structured data for a biopsy location.

    Supports multiple biopsy sites beyond just lobar locations.
    """

    location: str  # e.g., "distal_trachea", "RLL", "carina", "LMS"
    lobe: str | None = None  # If applicable: "RUL", "RML", "RLL", "LUL", "LLL"
    segment: str | None = None  # If applicable: segment name
    specimens_count: int | None = None  # Number of specimens from this site


__all__ = [
    "RegistryRecord",
    "BLVRData",
    "DestructionEvent",
    "EnhancedDilationEvent",
    "AspirationEvent",
    "CaoIntervention",
    "BiopsySite",
    # Granular data exports
    "EnhancedRegistryGranularData",
    "validate_ebus_consistency",
    "derive_aggregate_fields",
]
