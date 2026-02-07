from __future__ import annotations

from typing import Any

from .cpt_utils import CPTProcessingResult
from .logging_utils import IssueLogger

GENERIC_INDICATIONS = {"mass", "nodule", "ild", "lesion"}
IMAGING_DRIVEN_CODES = {"31652", "31653", "+31627", "31628", "32408", "+31654"}
EBUS_CODES = {"31652", "31653"}
NAV_CODES = {"+31627"}
STENT_CODES = {"31631", "31636", "+31637"}
STENT_DETAIL_FIELDS = [
    "stent_type",
    "stent_diameter_mm",
    "stent_length_mm",
    "stent_location",
]
PLEURAL_CODES = {"32554", "32555", "32556", "32557", "32550", "32551", "32650", "32651", "32652", "32653", "32654"}
PLEURAL_DETAIL_FIELDS = [
    ("pleural_side", "side"),
    ("pleural_guidance", "guidance"),
    ("pleural_fluid_appearance", "appearance"),
    ("pleural_volume_drained_ml", "volume"),
]
BLVR_CODES = {"31647", "31648", "+31649", "+31651"}
ABLATION_CODES = {"32408", "31641"}
ABLATION_FIELDS = [
    "ablation_modality",
    "ablation_device_name",
    "ablation_duration_seconds",
    "ablation_margin_assessed",
]


class ClinicalQCChecker:
    """Flag-only completeness checks that require human review."""

    def check(
        self,
        entry: dict[str, Any],
        cpt_context: CPTProcessingResult | None,
        entry_id: str,
        logger: IssueLogger,
    ) -> None:
        active_codes = set(cpt_context.active_codes()) if cpt_context else _codes_from_entry(entry)
        self._check_indication(entry, entry_id, logger)
        self._check_imaging_details(entry, active_codes, entry_id, logger)
        self._check_procedure_details(entry, active_codes, entry_id, logger)
        self._check_complication_completeness(entry, entry_id, logger)
        self._check_follow_up(entry, entry_id, logger)

    def _check_indication(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        primary = _clean(entry.get("primary_indication"))
        if not primary or len(primary) <= 5 or primary.lower() in GENERIC_INDICATIONS:
            logger.log(
                entry_id=entry_id,
                issue_type="indication_too_vague",
                severity="warn",
                action="flagged_for_manual",
                field="primary_indication",
                details={"value": primary},
            )

    def _check_imaging_details(
        self,
        entry: dict[str, Any],
        active_codes: set[str],
        entry_id: str,
        logger: IssueLogger,
    ) -> None:
        if active_codes & IMAGING_DRIVEN_CODES:
            findings = _clean(entry.get("radiographic_findings"))
            if not findings:
                logger.log(
                    entry_id=entry_id,
                    issue_type="radiographic_findings_missing",
                    severity="warn",
                    action="flagged_for_manual",
                    field="radiographic_findings",
                    details="Imaging-driven procedure without radiographic findings",
                )

    def _check_procedure_details(
        self,
        entry: dict[str, Any],
        active_codes: set[str],
        entry_id: str,
        logger: IssueLogger,
    ) -> None:
        if active_codes & EBUS_CODES:
            stations = entry.get("ebus_stations_sampled") or entry.get("linear_ebus_stations")
            if not _has_content(stations):
                logger.log(
                    entry_id=entry_id,
                    issue_type="ebus_details_missing",
                    severity="warn",
                    action="flagged_for_manual",
                    field="ebus_stations_sampled",
                    details="EBUS codes present without sampled stations",
                )
        if active_codes & NAV_CODES:
            if not _has_content(entry.get("nav_platform")):
                logger.log(
                    entry_id=entry_id,
                    issue_type="nav_platform_missing",
                    severity="warn",
                    action="flagged_for_manual",
                    field="nav_platform",
                    details="Navigation add-on without nav_platform",
                )
        if active_codes & STENT_CODES:
            missing = [field for field in STENT_DETAIL_FIELDS if not _has_content(entry.get(field))]
            if missing:
                logger.log(
                    entry_id=entry_id,
                    issue_type="stent_details_missing",
                    severity="warn",
                    action="flagged_for_manual",
                    field="stent_details",
                    details={"missing_fields": missing},
                )
        if active_codes & PLEURAL_CODES:
            for field, label in PLEURAL_DETAIL_FIELDS:
                if not _has_content(entry.get(field)):
                    logger.log(
                        entry_id=entry_id,
                        issue_type=f"pleural_detail_missing_{label}",
                        severity="warn",
                        action="flagged_for_manual",
                        field=field,
                        details="Pleural detail missing",
                    )
        if active_codes & BLVR_CODES:
            blvr_missing = [
                field
                for field in ("blvr_target_lobe", "blvr_valve_type", "blvr_number_of_valves")
                if not _has_content(entry.get(field))
            ]
            if blvr_missing:
                logger.log(
                    entry_id=entry_id,
                    issue_type="blvr_details_missing",
                    severity="warn",
                    action="flagged_for_manual",
                    field="blvr_details",
                    details={"missing_fields": blvr_missing},
                )
        ablation_flag = bool(active_codes & ABLATION_CODES) or bool(entry.get("ablation_peripheral_performed"))
        if ablation_flag:
            ablation_missing = [field for field in ABLATION_FIELDS if not _has_content(entry.get(field))]
            if ablation_missing:
                logger.log(
                    entry_id=entry_id,
                    issue_type="ablation_detail_missing",
                    severity="warn",
                    action="flagged_for_manual",
                    field="ablation_details",
                    details={"missing_fields": ablation_missing},
                )

    def _check_complication_completeness(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        status = _clean(entry.get("data_entry_status"))
        if status.lower() != "complete":
            return
        required = ["bleeding_severity", "pneumothorax", "hypoxia_respiratory_failure", "disposition"]
        for field in required:
            value = entry.get(field)
            if not _has_content(value):
                logger.log(
                    entry_id=entry_id,
                    issue_type="complication_field_missing",
                    severity="warn",
                    action="flagged_for_manual",
                    field=field,
                    details="Required complication field missing when status=Complete",
                )

    def _check_follow_up(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        if not _has_content(entry.get("follow_up_plan")):
            logger.log(
                entry_id=entry_id,
                issue_type="follow_up_plan_missing",
                severity="warn",
                action="flagged_for_manual",
                field="follow_up_plan",
                details="Follow-up plan missing",
            )


def _codes_from_entry(entry: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    raw = entry.get("cpt_codes") or []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                base = item.split("-")[0]
                codes.add(base)
    elif isinstance(raw, str):
        codes.add(raw.split("-")[0])
    return codes


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return any(item not in (None, "") for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return True
