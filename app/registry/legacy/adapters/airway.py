from __future__ import annotations

from typing import Any

from proc_schemas.clinical import airway as airway_schemas

from app.reporting.partial_schemas import (
    AirwayDilationPartial,
    AirwayStentPlacementPartial,
    EndobronchialTumorDestructionPartial,
)

from .base import DictPayloadAdapter, ExtractionAdapter


def _nav_platform(source: dict[str, Any]) -> str:
    raw = str(source.get("nav_platform") or "").strip().lower()
    if not raw:
        return ""
    # Normalize common EMN platform names so robotic adapters don't misclassify them.
    # Example: "superDimension" contains the substring "ion" and can be incorrectly
    # treated as an Ion robotic case if we rely on naive substring checks.
    if "superdimension" in raw or "super-dimension" in raw or "super dimension" in raw:
        return "emn"
    return raw


def _coerce_size_mm(value: Any) -> float | None:
    """Try to coerce a size in mm, handling cm inputs."""
    if value in (None, "", []):
        return None
    try:
        num = float(value)
    except Exception:
        return None
    return round(num, 2)


def _station_size_mm(source: dict[str, Any]) -> float | None:
    """Pick a best-effort station size from common extraction keys."""
    # Prefer explicit EBUS station sizing if present
    for key in ("ebus_station_size_mm", "station_size_mm", "lesion_size_mm", "nav_lesion_size_mm"):
        size = _coerce_size_mm(source.get(key))
        if size:
            return size
    # Handle cm inputs if available
    for key in ("ebus_station_size_cm", "station_size_cm", "lesion_size_cm", "nav_lesion_size_cm"):
        size_cm = _coerce_size_mm(source.get(key))
        if size_cm:
            return round(size_cm * 10, 2)
    return None


class BronchoscopyShellAdapter(ExtractionAdapter):
    proc_type = "bronchoscopy_core"
    schema_model = airway_schemas.BronchoscopyShell
    schema_id = "bronchoscopy_shell_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return bool(source.get("airway_overview"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "airway_overview": source.get("airway_overview"),
            "right_lung_overview": source.get("right_lung_overview"),
            "left_lung_overview": source.get("left_lung_overview"),
            "mucosa_overview": source.get("mucosa_overview"),
            "secretions_overview": source.get("secretions_overview"),
        }


class EMNAdapter(ExtractionAdapter):
    proc_type = "emn_bronchoscopy"
    schema_model = airway_schemas.EMNBronchoscopy
    schema_id = "emn_bronchoscopy_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        platform = _nav_platform(source)
        if platform != "emn":
            return False
        if source.get("nav_registration_method") or source.get("nav_rebus_used"):
            return True
        procs = source.get("procedures_performed") or {}
        nav = procs.get("navigational_bronchoscopy") if isinstance(procs, dict) else None
        return isinstance(nav, dict) and nav.get("performed") is True

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "navigation_system": source.get("nav_platform", "EMN"),
            "target_lung_segment": source.get("nav_target_segment", "target lesion"),
            "registration_method": source.get("nav_registration_method"),
            "adjunct_imaging": [img for img in [source.get("nav_imaging_verification")] if img],
            "notes": source.get("nav_notes"),
        }
        if source.get("nav_lesion_size_cm") is not None:
            payload["lesion_size_cm"] = source.get("nav_lesion_size_cm")
        if source.get("nav_tool_to_target_distance_cm") is not None:
            payload["tool_to_target_distance_cm"] = source.get("nav_tool_to_target_distance_cm")
        return payload


class RoboticIonBronchoscopyAdapter(ExtractionAdapter):
    proc_type = "robotic_ion_bronchoscopy"
    schema_model = airway_schemas.RoboticIonBronchoscopy
    schema_id = "robotic_ion_bronchoscopy_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return _nav_platform(source) == "ion" and bool(source.get("ventilation_mode"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        # Rule 1: Do NOT hallucinate vent values - only use if explicitly provided
        return {
            "navigation_plan_source": "pre-procedure CT" if source.get("nav_registration_method") else None,
            "vent_mode": source.get("ventilation_mode"),
            "vent_rr": source.get("vent_rr"),  # No default - leave null if not stated
            "vent_tv_ml": source.get("vent_tv_ml"),  # No default - leave null if not stated
            "vent_peep_cm_h2o": source.get("vent_peep_cm_h2o"),  # No default - leave null if not stated
            "vent_fio2_pct": source.get("vent_fio2_pct"),  # No default - leave null if not stated
            "vent_flow_rate": source.get("vent_flow_rate"),
            "vent_pmean_cm_h2o": source.get("vent_pmean_cm_h2o"),
            "cbct_performed": source.get("nav_imaging_verification") == "Cone Beam CT",
            "radial_pattern": source.get("nav_rebus_view"),
        }


class IonRegistrationCompleteAdapter(ExtractionAdapter):
    proc_type = "ion_registration_complete"
    schema_model = airway_schemas.IonRegistrationComplete
    schema_id = "ion_registration_complete_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return _nav_platform(source) == "ion" and bool(source.get("nav_registration_method"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "method": source.get("nav_registration_method"),
            "fiducial_error_mm": source.get("nav_registration_error_mm"),
            "alignment_quality": source.get("nav_registration_alignment"),
        }


class RoboticNavigationAdapter(ExtractionAdapter):
    proc_type = "robotic_navigation"
    schema_model = airway_schemas.RoboticNavigation
    schema_id = "robotic_navigation_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        platform = _nav_platform(source)
        platform_match = (
            platform in ("ion", "monarch", "auris", "robotic", "galaxy")
            or "ion" in platform
            or "monarch" in platform
            or "robot" in platform
            or "galaxy" in platform
        )
        has_nav_details = bool(
            source.get("nav_registration_method")
            or source.get("nav_registration_error_mm") is not None
            or source.get("nav_imaging_verification")
            or source.get("nav_rebus_used")
            or source.get("nav_notes")
        )
        if not platform_match:
            return False
        if has_nav_details:
            return True
        procs = source.get("procedures_performed") or {}
        nav = procs.get("navigational_bronchoscopy") if isinstance(procs, dict) else None
        return isinstance(nav, dict) and nav.get("performed") is True

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        platform_raw = source.get("nav_platform")
        platform = None
        if platform_raw:
            lowered = str(platform_raw).lower()
            if "ion" in lowered:
                platform = "Ion"
            elif "monarch" in lowered or "auris" in lowered:
                platform = "Monarch"
            elif "robot" in lowered:
                platform = platform_raw
            else:
                platform = platform_raw
        return {
            "platform": platform,
            "lesion_location": source.get("lesion_location") or source.get("nav_target_segment"),
            "registration_method": source.get("nav_registration_method"),
            "registration_error_mm": source.get("nav_registration_error_mm"),
            "notes": source.get("nav_notes"),
        }


class CBCTFusionAdapter(ExtractionAdapter):
    proc_type = "cbct_cact_fusion"
    schema_model = airway_schemas.CBCTFusion
    schema_id = "cbct_cact_fusion_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        if _nav_platform(source) != "ion":
            return False
        return source.get("nav_imaging_verification") in ("Cone Beam CT", "O-Arm")

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "overlay_result": source.get("nav_imaging_verification"),
        }


class RoboticMonarchBronchoscopyAdapter(ExtractionAdapter):
    proc_type = "robotic_monarch_bronchoscopy"
    schema_model = airway_schemas.RoboticMonarchBronchoscopy
    schema_id = "robotic_monarch_bronchoscopy_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return _nav_platform(source) in ("monarch", "auris")

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "radial_pattern": source.get("nav_rebus_view"),
            "cbct_used": source.get("nav_imaging_verification") in ("Cone Beam CT", "O-Arm"),
            "vent_mode": source.get("ventilation_mode"),
            "vent_rr": source.get("vent_rr"),
            "vent_tv_ml": source.get("vent_tv_ml"),
            "vent_peep_cm_h2o": source.get("vent_peep_cm_h2o"),
            "vent_fio2_pct": source.get("vent_fio2_pct"),
            "vent_flow_rate": source.get("vent_flow_rate"),
            "vent_pmean_cm_h2o": source.get("vent_pmean_cm_h2o"),
        }


class RadialEBUSSurveyAdapter(ExtractionAdapter):
    proc_type = "radial_ebus_survey"
    schema_model = airway_schemas.RadialEBUSSurvey
    schema_id = "radial_ebus_survey_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return bool(source.get("nav_rebus_used") or source.get("nav_rebus_view"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "location": source.get("nav_target_segment", "target lesion"),
            "rebus_features": source.get("nav_rebus_view"),
        }


class RadialEBUSSamplingAdapter(ExtractionAdapter):
    proc_type = "radial_ebus_sampling"
    schema_model = airway_schemas.RadialEBUSSampling
    schema_id = "radial_ebus_sampling_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        radial_evidence = bool(source.get("nav_rebus_used") or source.get("nav_rebus_view"))
        return radial_evidence and bool(source.get("nav_sampling_tools"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "ultrasound_pattern": source.get("nav_rebus_view"),
            "sampling_tools": source.get("nav_sampling_tools") or [],
            "lesion_size_mm": source.get("nav_lesion_size_mm"),
        }


class ToolInLesionConfirmationAdapter(ExtractionAdapter):
    proc_type = "tool_in_lesion_confirmation"
    schema_model = airway_schemas.ToolInLesionConfirmation
    schema_id = "tool_in_lesion_confirmation_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return bool(source.get("nav_tool_in_lesion"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "confirmation_method": source.get("nav_imaging_verification") or "imaging confirmation",
            "rebus_pattern": source.get("nav_rebus_view"),
        }


class EBUSTBNAAdapter(ExtractionAdapter):
    proc_type = "ebus_tbna"
    schema_model = airway_schemas.EBUSTBNA
    schema_id = "ebus_tbna_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        stations = source.get("linear_ebus_stations") or source.get("ebus_stations_sampled") or []
        return isinstance(stations, list) and bool(stations)

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        stations_raw = source.get("linear_ebus_stations") or source.get("ebus_stations_sampled") or []
        stations = stations_raw if isinstance(stations_raw, list) else []
        station_size = _station_size_mm(source) if len(stations) == 1 else None
        use_forceps = bool(source.get("ebus_intranodal_forceps_used"))
        detail_map = {}
        for item in source.get("ebus_stations_detail") or []:
            name = (item.get("station") or "").upper() if isinstance(item, dict) else None
            if name:
                detail_map[name] = item

        station_entries = []
        passes_global = source.get("ebus_passes")
        try:
            passes_global = int(passes_global) if passes_global not in (None, "", []) else None
        except Exception:
            passes_global = None

        for station in stations:
            tools = ["TBNA"]
            if use_forceps:
                tools.append("Forceps")
            detail = detail_map.get(station, {})
            has_detail = isinstance(detail, dict) and bool(detail)
            station_entries.append(
                {
                    "station_name": station,
                    "size_mm": detail.get("size_mm") if has_detail else station_size,
                    "passes": detail.get("passes") if has_detail else passes_global,
                    "echo_features": source.get("ebus_echo_features") or source.get("ebus_elastography_pattern"),
                    "biopsy_tools": tools,
                    "rose_result": detail.get("rose_result") if has_detail else source.get("ebus_rose_result"),
                }
            )
        return {
            "needle_gauge": source.get("ebus_needle_gauge"),
            "stations": station_entries,
            "elastography_used": source.get("ebus_elastography_used") or source.get("ebus_elastography"),
            "elastography_pattern": source.get("ebus_elastography_pattern"),
            "rose_available": source.get("ebus_rose_available"),
            "overall_rose_diagnosis": source.get("ebus_rose_result"),
        }


class WholeLungLavageAdapter(ExtractionAdapter):
    proc_type = "whole_lung_lavage"
    schema_model = airway_schemas.WholeLungLavage
    schema_id = "whole_lung_lavage_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return source.get("wll_volume_instilled_l") is not None

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "side": source.get("wll_side", "right"),
            "dlt_size_fr": source.get("wll_dlt_used_size"),
            "position": source.get("wll_position"),
            "total_volume_l": source.get("wll_volume_instilled_l"),
            "max_volume_l": source.get("wll_volume_instilled_l"),
            "aliquot_volume_l": source.get("wll_aliquot_volume_l"),
            "dwell_time_min": source.get("wll_dwell_time_min"),
            "num_cycles": source.get("wll_num_cycles"),
        }


class BLVRValvePlacementAdapter(ExtractionAdapter):
    proc_type = "blvr_valve_placement"
    schema_model = airway_schemas.BLVRValvePlacement
    schema_id = "blvr_valve_placement_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        return bool(source.get("blvr_valve_type") or source.get("blvr_number_of_valves"))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        target_lobe = source.get("blvr_target_lobe") or "target lobe"
        valves: list[dict[str, Any]] = []
        valve_count = source.get("blvr_number_of_valves") or 1
        for _ in range(max(1, int(valve_count))):
            valves.append({"valve_type": source.get("blvr_valve_type") or "Valve", "lobe": target_lobe})
        return {
            "lobes_treated": [target_lobe],
            "valves": valves,
        }


class BPFLocalizationAdapter(DictPayloadAdapter):
    proc_type = "bpf_localization_occlusion"
    schema_model = airway_schemas.BPFLocalizationOcclusion
    schema_id = "bpf_localization_occlusion_v1"
    source_key = "bpf_localization"


class BPFValvePlacementAdapter(DictPayloadAdapter):
    proc_type = "bpf_valve_air_leak"
    schema_model = airway_schemas.BPFValvePlacement
    schema_id = "bpf_valve_air_leak_v1"
    source_key = "bpf_valve_placement"


class BPFSealantApplicationAdapter(DictPayloadAdapter):
    proc_type = "bpf_endobronchial_sealant"
    schema_model = airway_schemas.BPFSealantApplication
    schema_id = "bpf_endobronchial_sealant_v1"
    source_key = "bpf_sealant_application"


class BronchialWashingAdapter(DictPayloadAdapter):
    proc_type = "bronchial_washing"
    schema_model = airway_schemas.BronchialWashing
    schema_id = "bronchial_washing_v1"
    source_key = "bronchial_washing"


class BronchialBrushingAdapter(DictPayloadAdapter):
    proc_type = "bronchial_brushings"
    schema_model = airway_schemas.BronchialBrushing
    schema_id = "bronchial_brushings_v1"
    source_key = "bronchial_brushings"


class BALAdapter(DictPayloadAdapter):
    proc_type = "bal"
    schema_model = airway_schemas.BAL
    schema_id = "bal_v1"
    source_key = "bal"


class FiducialMarkerPlacementAdapter(DictPayloadAdapter):
    proc_type = "fiducial_marker_placement"
    schema_model = airway_schemas.FiducialMarkerPlacement
    schema_id = "fiducial_marker_placement_v1"
    source_key = "fiducial_marker_placement"


class BALVariantAdapter(DictPayloadAdapter):
    proc_type = "bal_variant"
    schema_model = airway_schemas.BronchoalveolarLavageAlt
    schema_id = "bal_alt_v1"
    source_key = "bal_variant"


class EndobronchialBiopsyAdapter(DictPayloadAdapter):
    proc_type = "endobronchial_biopsy"
    schema_model = airway_schemas.EndobronchialBiopsy
    schema_id = "endobronchial_biopsy_v1"
    source_key = "endobronchial_biopsy"


class TransbronchialLungBiopsyAdapter(DictPayloadAdapter):
    proc_type = "transbronchial_lung_biopsy"
    schema_model = airway_schemas.TransbronchialLungBiopsy
    schema_id = "transbronchial_lung_biopsy_v1"
    source_key = "transbronchial_lung_biopsy"


class TransbronchialBiopsyAdapter(ExtractionAdapter):
    proc_type = "transbronchial_biopsy"
    schema_model = airway_schemas.TransbronchialBiopsyBasic
    schema_id = "transbronchial_biopsy_v1"

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        num = source.get("bronch_num_tbbx")
        try:
            if num is not None and int(num) > 0:
                return True
        except Exception:
            pass
        procs = source.get("procedures_performed") or {}
        tbbx = procs.get("transbronchial_biopsy") if isinstance(procs, dict) else None
        return isinstance(tbbx, dict) and tbbx.get("performed") is True

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any]:
        num_raw = source.get("bronch_num_tbbx")
        try:
            num_bx = int(num_raw) if num_raw is not None else None
        except Exception:
            num_bx = None
        return {
            "lobe": source.get("bronch_location_lobe") or "target lobe",
            "segment": source.get("bronch_location_segment"),
            "guidance": source.get("bronch_guidance") or "Fluoroscopy",
            "tool": source.get("bronch_tbbx_tool") or "Forceps",
            "number_of_biopsies": num_bx or 0,
            "specimen_tests": source.get("bronch_specimen_tests"),
            "complications": source.get("bronch_immediate_complications"),
            "notes": source.get("bronch_indication"),
        }


class TransbronchialNeedleAspirationAdapter(DictPayloadAdapter):
    proc_type = "transbronchial_needle_aspiration"
    schema_model = airway_schemas.TransbronchialNeedleAspiration
    schema_id = "transbronchial_needle_aspiration_v1"
    source_key = "transbronchial_needle_aspiration"


class TherapeuticAspirationAdapter(DictPayloadAdapter):
    proc_type = "therapeutic_aspiration"
    schema_model = airway_schemas.TherapeuticAspiration
    schema_id = "therapeutic_aspiration_v1"
    source_key = "therapeutic_aspiration"


class AirwayDilationAdapter(DictPayloadAdapter):
    proc_type = "airway_dilation"
    schema_model = AirwayDilationPartial
    schema_id = "airway_dilation_v1"
    source_key = "airway_dilation"


class EndobronchialTumorDestructionAdapter(DictPayloadAdapter):
    proc_type = "endobronchial_tumor_destruction"
    schema_model = EndobronchialTumorDestructionPartial
    schema_id = "endobronchial_tumor_destruction_v1"
    source_key = "endobronchial_tumor_destruction"


class AirwayStentPlacementAdapter(DictPayloadAdapter):
    proc_type = "airway_stent_placement"
    schema_model = AirwayStentPlacementPartial
    schema_id = "airway_stent_placement_v1"
    source_key = "airway_stent_placement"


class RigidBronchoscopyAdapter(DictPayloadAdapter):
    proc_type = "rigid_bronchoscopy"
    schema_model = airway_schemas.RigidBronchoscopy
    schema_id = "rigid_bronchoscopy_v1"
    source_key = "rigid_bronchoscopy"


class TransbronchialCryobiopsyAdapter(DictPayloadAdapter):
    proc_type = "transbronchial_cryobiopsy"
    schema_model = airway_schemas.TransbronchialCryobiopsy
    schema_id = "transbronchial_cryobiopsy_v1"
    source_key = "transbronchial_cryobiopsy"


class EndobronchialCryoablationAdapter(DictPayloadAdapter):
    proc_type = "endobronchial_cryoablation"
    schema_model = airway_schemas.EndobronchialCryoablation
    schema_id = "endobronchial_cryoablation_v1"
    source_key = "endobronchial_cryoablation"


class CryoExtractionMucusAdapter(DictPayloadAdapter):
    proc_type = "cryo_extraction_mucus"
    schema_model = airway_schemas.CryoExtractionMucus
    schema_id = "cryo_extraction_mucus_v1"
    source_key = "cryo_extraction_mucus"


class EndobronchialHemostasisAdapter(DictPayloadAdapter):
    proc_type = "endobronchial_hemostasis"
    schema_model = airway_schemas.EndobronchialHemostasis
    schema_id = "endobronchial_hemostasis_v1"
    source_key = "endobronchial_hemostasis"


class EndobronchialBlockerAdapter(DictPayloadAdapter):
    proc_type = "endobronchial_blocker"
    schema_model = airway_schemas.EndobronchialBlockerPlacement
    schema_id = "endobronchial_blocker_v1"
    source_key = "endobronchial_blocker"


class PDTLightAdapter(DictPayloadAdapter):
    proc_type = "pdt_light"
    schema_model = airway_schemas.PhotodynamicTherapyLight
    schema_id = "pdt_light_v1"
    source_key = "pdt_light"


class PDTDebridementAdapter(DictPayloadAdapter):
    proc_type = "pdt_debridement"
    schema_model = airway_schemas.PhotodynamicTherapyDebridement
    schema_id = "pdt_debridement_v1"
    source_key = "pdt_debridement"


class ForeignBodyRemovalAdapter(DictPayloadAdapter):
    proc_type = "foreign_body_removal"
    schema_model = airway_schemas.ForeignBodyRemoval
    schema_id = "foreign_body_removal_v1"
    source_key = "foreign_body_removal"


class AwakeFOIAdapter(DictPayloadAdapter):
    proc_type = "awake_foi"
    schema_model = airway_schemas.AwakeFiberopticIntubation
    schema_id = "awake_foi_v1"
    source_key = "awake_foi"


class DLTPlacementAdapter(DictPayloadAdapter):
    proc_type = "dlt_placement"
    schema_model = airway_schemas.DoubleLumenTubePlacement
    schema_id = "dlt_placement_v1"
    source_key = "dlt_placement"


class StentSurveillanceAdapter(DictPayloadAdapter):
    proc_type = "stent_surveillance"
    schema_model = airway_schemas.AirwayStentSurveillance
    schema_id = "stent_surveillance_v1"
    source_key = "stent_surveillance"


__all__ = [
    "AwakeFOIAdapter",
    "AirwayDilationAdapter",
    "AirwayStentPlacementAdapter",
    "BALAdapter",
    "BALVariantAdapter",
    "BLVRValvePlacementAdapter",
    "BPFLocalizationAdapter",
    "BPFSealantApplicationAdapter",
    "BPFValvePlacementAdapter",
    "BronchoscopyShellAdapter",
    "BronchialBrushingAdapter",
    "BronchialWashingAdapter",
    "CBCTFusionAdapter",
    "CryoExtractionMucusAdapter",
    "DLTPlacementAdapter",
    "EBUSTBNAAdapter",
    "EMNAdapter",
    "EndobronchialTumorDestructionAdapter",
    "EndobronchialBiopsyAdapter",
    "EndobronchialBlockerAdapter",
    "EndobronchialCryoablationAdapter",
    "EndobronchialHemostasisAdapter",
    "ForeignBodyRemovalAdapter",
    "IonRegistrationCompleteAdapter",
    "RoboticNavigationAdapter",
    "PDTDebridementAdapter",
    "PDTLightAdapter",
    "RadialEBUSSamplingAdapter",
    "RadialEBUSSurveyAdapter",
    "RigidBronchoscopyAdapter",
    "RoboticIonBronchoscopyAdapter",
    "RoboticMonarchBronchoscopyAdapter",
    "StentSurveillanceAdapter",
    "TherapeuticAspirationAdapter",
    "ToolInLesionConfirmationAdapter",
    "TransbronchialCryobiopsyAdapter",
    "TransbronchialLungBiopsyAdapter",
    "TransbronchialBiopsyAdapter",
    "TransbronchialNeedleAspirationAdapter",
    "WholeLungLavageAdapter",
]
