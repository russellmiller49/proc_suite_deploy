from __future__ import annotations

from typing import Any

from proc_schemas.clinical import pleural as pleural_schemas

from .base import DictPayloadAdapter, ExtractionAdapter


def _pleural_side(source: dict[str, Any]) -> str | None:
    return source.get("pleural_side") or source.get("laterality") or source.get("side")


def _uses_manometry(source: dict[str, Any]) -> bool:
    return bool(source.get("pleural_opening_pressure_measured")) or source.get("pleural_opening_pressure_cmh2o") is not None


class ThoracentesisAdapter(ExtractionAdapter):
    proc_type = "thoracentesis_detailed"
    schema_model = pleural_schemas.ThoracentesisDetailed
    schema_id = "thoracentesis_detailed_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        pleural_type = (source.get("pleural_procedure_type") or "").lower()
        return pleural_type == "thoracentesis" and not _uses_manometry(source)

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        ultrasound_feasible = source.get("pleural_ultrasound_feasible")
        if ultrasound_feasible is None:
            ultrasound_feasible = source.get("ultrasound_feasible")
        return {
            "side": _pleural_side(source) or "unspecified",
            # Do not infer feasibility from guidance presence; only set when explicitly provided.
            "ultrasound_feasible": ultrasound_feasible,
            "intercostal_space": source.get("intercostal_space")
            or source.get("pleural_intercostal_space")
            or "unspecified",
            "entry_location": source.get("entry_location", "mid-axillary"),
            "volume_removed_ml": source.get("pleural_volume_drained_ml"),
            "fluid_appearance": source.get("pleural_fluid_appearance"),
            "drainage_device": source.get("drainage_device"),
            "suction_cmh2o": source.get("suction"),
            "specimen_tests": source.get("specimen_tests") or source.get("specimens"),
            "cxr_ordered": source.get("cxr_ordered"),
            "effusion_volume": source.get("pleural_effusion_volume"),
            "effusion_echogenicity": source.get("pleural_echogenicity"),
            "loculations": source.get("pleural_loculations"),
            "diaphragm_motion": source.get("pleural_diaphragm_motion"),
            "lung_sliding_pre": source.get("pleural_lung_sliding_pre"),
            "lung_sliding_post": source.get("pleural_lung_sliding_post"),
            "lung_consolidation": source.get("pleural_lung_consolidation"),
            "pleura_description": source.get("pleural_description"),
            "pleural_guidance": source.get("pleural_guidance"),
        }


class ThoracentesisManometryAdapter(ExtractionAdapter):
    proc_type = "thoracentesis_manometry"
    schema_model = pleural_schemas.ThoracentesisManometry
    schema_id = "thoracentesis_manometry_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        pleural_type = (source.get("pleural_procedure_type") or "").lower()
        return pleural_type == "thoracentesis" and _uses_manometry(source)

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "side": _pleural_side(source) or "unspecified",
            "guidance": source.get("pleural_guidance"),
            "intercostal_space": source.get("intercostal_space")
            or source.get("pleural_intercostal_space")
            or "unspecified",
            "entry_location": source.get("entry_location", "mid-axillary"),
            "fluid_appearance": source.get("pleural_fluid_appearance"),
            "specimen_tests": source.get("specimen_tests") or source.get("specimens"),
            "effusion_size": source.get("pleural_effusion_volume"),
            "effusion_echogenicity": source.get("pleural_echogenicity"),
            "loculations": source.get("pleural_loculations"),
            "diaphragm_motion": source.get("pleural_diaphragm_motion"),
            "lung_sliding_pre": source.get("pleural_lung_sliding_pre"),
            "lung_sliding_post": source.get("pleural_lung_sliding_post"),
            "lung_consolidation": source.get("pleural_lung_consolidation"),
            "pleura_description": source.get("pleural_description"),
            "opening_pressure_cmh2o": source.get("pleural_opening_pressure_cmh2o"),
            "pressure_readings": source.get("pleural_pressure_readings"),
            "stopping_criteria": source.get("pleural_stopping_criteria"),
            "post_procedure_imaging": source.get("post_procedure_imaging"),
            "total_removed_ml": source.get("pleural_volume_drained_ml"),
        }


class ChestTubeAdapter(ExtractionAdapter):
    proc_type = "chest_tube"
    schema_model = pleural_schemas.ChestTube
    schema_id = "chest_tube_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        pleural_type = (source.get("pleural_procedure_type") or "").lower()
        return pleural_type == "chest tube"

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "side": _pleural_side(source) or "unspecified",
            "intercostal_space": source.get("intercostal_space")
            or source.get("pleural_intercostal_space")
            or "unspecified",
            "entry_line": source.get("entry_location", "mid-axillary"),
            "guidance": source.get("pleural_guidance"),
            "fluid_removed_ml": source.get("pleural_volume_drained_ml"),
            "fluid_appearance": source.get("pleural_fluid_appearance"),
            "specimen_tests": source.get("specimen_tests") or source.get("specimens"),
            "cxr_ordered": source.get("cxr_ordered"),
            "effusion_volume": source.get("pleural_effusion_volume"),
            "effusion_echogenicity": source.get("pleural_echogenicity"),
            "loculations": source.get("pleural_loculations"),
            "diaphragm_motion": source.get("pleural_diaphragm_motion"),
            "lung_sliding_pre": source.get("pleural_lung_sliding_pre"),
            "lung_sliding_post": source.get("pleural_lung_sliding_post"),
            "lung_consolidation": source.get("pleural_lung_consolidation"),
            "pleura_description": source.get("pleural_description"),
        }


class TunneledPleuralCatheterInsertAdapter(ExtractionAdapter):
    proc_type = "tunneled_pleural_catheter_insert"
    schema_model = pleural_schemas.TunneledPleuralCatheterInsert
    schema_id = "tunneled_pleural_catheter_insert_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        pleural_type = (source.get("pleural_procedure_type") or "").lower()
        return pleural_type == "tunneled catheter"

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "side": _pleural_side(source),
            "intercostal_space": source.get("pleural_intercostal_space") or source.get("intercostal_space"),
            "entry_location": source.get("entry_location"),
            "tunnel_length_cm": source.get("tunnel_length_cm"),
            "exit_site": source.get("exit_site"),
            "anesthesia_lidocaine_ml": source.get("anesthesia_lidocaine_ml"),
            "fluid_removed_ml": source.get("pleural_volume_drained_ml"),
            "fluid_appearance": source.get("pleural_fluid_appearance"),
            "pleural_pressures": source.get("pleural_pressures"),
            "drainage_device": source.get("drainage_device"),
            "suction": source.get("suction"),
            "specimen_tests": source.get("specimen_tests") or source.get("specimens"),
            "cxr_ordered": source.get("cxr_ordered"),
            "pleural_guidance": source.get("pleural_guidance"),
        }


class PigtailCatheterAdapter(ExtractionAdapter):
    proc_type = "pigtail_catheter"
    schema_model = pleural_schemas.PigtailCatheter
    schema_id = "pigtail_catheter_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        pleural_type = (source.get("pleural_procedure_type") or "").lower()
        return pleural_type == "pigtail catheter"

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "side": _pleural_side(source) or "unspecified",
            "intercostal_space": source.get("intercostal_space")
            or source.get("pleural_intercostal_space")
            or "unspecified",
            "entry_location": source.get("entry_location", "mid-axillary"),
            "size_fr": source.get("size_fr", "unspecified"),
            "anesthesia_lidocaine_ml": source.get("anesthesia_lidocaine_ml"),
            "fluid_removed_ml": source.get("pleural_volume_drained_ml"),
            "fluid_appearance": source.get("pleural_fluid_appearance"),
            "specimen_tests": source.get("specimen_tests") or source.get("specimens"),
            "cxr_ordered": source.get("cxr_ordered"),
        }


class ParacentesisAdapter(ExtractionAdapter):
    proc_type = "paracentesis"
    schema_model = pleural_schemas.Paracentesis
    schema_id = "paracentesis_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return bool(source.get("paracentesis_performed"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "volume_removed_ml": source.get("paracentesis_volume_ml") or 0,
            "site_description": source.get("paracentesis_site"),
            "fluid_character": source.get("paracentesis_fluid_character"),
            "tests": source.get("paracentesis_tests"),
            "imaging_guidance": source.get("paracentesis_guidance"),
        }


class PegPlacementAdapter(ExtractionAdapter):
    proc_type = "peg_placement"
    schema_model = pleural_schemas.PEGPlacement
    schema_id = "peg_placement_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return bool(source.get("peg_placed"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "incision_location": source.get("peg_incision_location"),
            "tube_size_fr": source.get("peg_size_fr"),
            "bumper_depth_cm": source.get("peg_bumper_depth_cm"),
            "procedural_time_min": source.get("peg_time_minutes"),
            "complications": source.get("peg_complications"),
        }


class PegExchangeAdapter(ExtractionAdapter):
    proc_type = "peg_exchange"
    schema_model = pleural_schemas.PEGExchange
    schema_id = "peg_exchange_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return bool(source.get("peg_exchanged"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "new_tube_size_fr": source.get("peg_size_fr"),
            "bumper_depth_cm": source.get("peg_bumper_depth_cm"),
            "complications": source.get("peg_complications"),
        }


class TunneledPleuralCatheterRemoveAdapter(DictPayloadAdapter):
    proc_type = "tunneled_pleural_catheter_remove"
    schema_model = pleural_schemas.TunneledPleuralCatheterRemove
    schema_id = "tunneled_pleural_catheter_remove_v1"
    source_key = "tunneled_pleural_catheter_remove"


class TransthoracicNeedleBiopsyAdapter(DictPayloadAdapter):
    proc_type = "transthoracic_needle_biopsy"
    schema_model = pleural_schemas.TransthoracicNeedleBiopsy
    schema_id = "transthoracic_needle_biopsy_v1"
    source_key = "transthoracic_needle_biopsy"


__all__ = [
    "ChestTubeAdapter",
    "ParacentesisAdapter",
    "PegExchangeAdapter",
    "PegPlacementAdapter",
    "PigtailCatheterAdapter",
    "ThoracentesisAdapter",
    "ThoracentesisManometryAdapter",
    "TransthoracicNeedleBiopsyAdapter",
    "TunneledPleuralCatheterInsertAdapter",
    "TunneledPleuralCatheterRemoveAdapter",
]
