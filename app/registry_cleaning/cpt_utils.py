from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .logging_utils import IssueLogger

CODE_PATTERN = re.compile(r"(\+?\d{4,5})")
MODIFIER_SPLIT = re.compile(r"[-\s,()]+")
KNOWN_MODIFIERS = {"25", "57", "59", "XS", "XU", "XE", "XP", "LT", "RT", "76", "79"}
SEDATION_CODES = {"99151", "99152", "+99153", "99153", "99154", "99155", "99156", "+99157", "99157"}
SEDATION_BLOCKING_TYPES = {"General", "Monitored Anesthesia Care"}


def _expand_range(start: int, end: int) -> set[str]:
    return {f"{code:05d}" for code in range(start, end + 1)}


E_AND_M_CODES = (
    _expand_range(99202, 99205)
    | _expand_range(99211, 99215)
    | _expand_range(99217, 99223)
    | _expand_range(99231, 99239)
    | {"99291", "+99292"}
)


@dataclass
class CPTCode:
    code: str
    modifiers: list[str]
    raw: str
    index: int
    active: bool = True

    def has_modifier(self, modifier_values: Sequence[str]) -> bool:
        lookup = {value.upper() for value in modifier_values}
        return any(mod.upper() in lookup for mod in self.modifiers)


@dataclass
class CPTProcessingResult:
    codes: list[CPTCode]

    def active_codes(self) -> list[str]:
        return [code.code for code in self.codes if code.active]

    def has_code(self, target: str) -> bool:
        return any(code.active and code.code == target for code in self.codes)


class CPTProcessor:
    """Normalize CPT codes and enforce billing logic from the knowledge base."""

    def __init__(self, knowledge_path: str | Path) -> None:
        path = Path(knowledge_path)
        if not path.exists():
            raise FileNotFoundError(f"Coding knowledge base not found at {path}")
        self.kb = json.loads(path.read_text())
        self.known_codes = self._build_known_codes()
        self.add_on_requirements = self._build_add_on_requirements()
        self.bundling_rules = self.kb.get("bundling_rules", {})
        self.ncci_pairs = self.kb.get("ncci_pairs", [])
        self.em_minor_codes = set(self.kb.get("bundling_rules", {}).get("em_same_day_minor_procedure", {}).get("procedure_examples", []))
        self.em_major_codes = set(self.kb.get("bundling_rules", {}).get("em_global_period_thoracoscopy", {}).get("global_90_day_codes_family", []))

    def process_entry(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> CPTProcessingResult:
        codes = self._normalize_codes(entry, entry_id, logger)
        if not codes:
            entry["cpt_codes"] = []
            entry["calculated_total_rvu"] = 0.0
            return CPTProcessingResult(codes=codes)

        self._flag_unknown_codes(codes, entry_id, logger)
        self._enforce_add_on_rules(codes, entry_id, logger)
        self._apply_auto_drop_rules(codes, entry_id, logger)
        self._apply_flag_only_rules(codes, entry_id, logger)
        self._apply_ncci_pairs(codes, entry_id, logger)
        self._enforce_sedation_rules(entry, codes, entry_id, logger)
        self._enforce_em_rules(codes, entry, entry_id, logger)

        entry["cpt_codes"] = self._rebuild_codes(codes)
        entry["calculated_total_rvu"] = self._calculate_total_rvu(codes)
        return CPTProcessingResult(codes=codes)

    # ---- normalization helpers ----

    def _normalize_codes(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> list[CPTCode]:
        raw_codes = entry.get("cpt_codes")
        if raw_codes is None:
            entry["cpt_codes"] = []
            logger.log(
                entry_id=entry_id,
                issue_type="cpt_codes_missing",
                severity="warn",
                action="flagged_for_manual",
                field="cpt_codes",
                details={"old": None, "new": []},
            )
            return []
        if isinstance(raw_codes, str):
            raw_codes = [raw_codes]
        if not isinstance(raw_codes, list):
            logger.log(
                entry_id=entry_id,
                issue_type="cpt_codes_invalid",
                severity="error",
                action="flagged_for_manual",
                field="cpt_codes",
                details={"message": f"Unable to parse CPT codes from {raw_codes!r}"},
            )
            return []

        normalized: list[CPTCode] = []
        for idx, value in enumerate(raw_codes):
            if value is None:
                continue
            code_obj = self._parse_code(value)
            if not code_obj:
                logger.log(
                    entry_id=entry_id,
                    issue_type="cpt_parse_failed",
                    severity="warn",
                    action="flagged_for_manual",
                    field="cpt_codes",
                    details={"raw": value},
                )
                continue
            code_obj.index = idx
            normalized.append(code_obj)
        return normalized

    def _parse_code(self, value: Any) -> CPTCode | None:
        if isinstance(value, dict):
            code_raw = value.get("code") or value.get("cpt")
            raw_string = str(value.get("raw") or code_raw or "")
        else:
            raw_string = str(value)
        match = CODE_PATTERN.search(raw_string)
        if not match:
            return None
        code = match.group(1)
        modifiers = self._extract_modifiers(raw_string[match.end() :])
        return CPTCode(code=code, modifiers=modifiers, raw=raw_string, index=0)

    def _extract_modifiers(self, remainder: str) -> list[str]:
        modifiers: list[str] = []
        if not remainder:
            return modifiers
        for token in MODIFIER_SPLIT.split(remainder):
            token = token.strip().upper()
            if not token:
                continue
            if token in KNOWN_MODIFIERS:
                modifiers.append(token)
        return modifiers

    def _rebuild_codes(self, codes: list[CPTCode]) -> list[str]:
        rendered: list[str] = []
        for code in sorted(codes, key=lambda item: item.index):
            if not code.active:
                continue
            if code.modifiers:
                rendered.append(f"{code.code}-{'-'.join(code.modifiers)}")
            else:
                rendered.append(code.code)
        return rendered

    # ---- rule enforcement ----

    def _flag_unknown_codes(self, codes: list[CPTCode], entry_id: str, logger: IssueLogger) -> None:
        for code in self._active_code_set(codes):
            if code not in self.known_codes:
                logger.log(
                    entry_id=entry_id,
                    issue_type="unknown_cpt",
                    severity="warn",
                    action="flagged_for_manual",
                    field="cpt_codes",
                    details={"code": code},
                )

    def _enforce_add_on_rules(self, codes: list[CPTCode], entry_id: str, logger: IssueLogger) -> None:
        active_codes = self._active_code_set(codes)
        for add_on, base_codes in self.add_on_requirements.items():
            if add_on not in active_codes:
                continue
            if active_codes.isdisjoint(base_codes):
                logger.log(
                    entry_id=entry_id,
                    issue_type="add_on_without_primary",
                    severity="warn",
                    action="flagged_for_manual",
                    field="cpt_codes",
                    details={"add_on": add_on, "expected_primary": sorted(base_codes)},
                )

    def _apply_auto_drop_rules(self, codes: list[CPTCode], entry_id: str, logger: IssueLogger) -> None:
        rule_map = {
            "diagnostic_with_surgical": ("therapeutic_codes", "drop_codes"),
            "therapeutic_airway_bundles_diagnostic": ("dominant_codes", "drop_codes"),
            "pdt_bundles_diagnostic_bronch": ("pdt_codes", "drop_codes"),
            "tblbx_bundles_tbna_brush": ("dominant_codes", "drop_codes"),
            "thoracoscopy_diagnostic_with_surgical": ("therapeutic_codes", "drop_codes"),
            "pleurodesis_with_thoracoscopy": ("thoracoscopy_codes", "pleurodesis_codes"),
            "lung_biopsy_with_resection": ("therapeutic_codes", "drop_codes"),
            "chest_tube_with_open_thoracotomy": ("therapeutic_codes", "drop_codes"),
            "pleural_post_procedure_imaging": ("pleural_codes", "bundled_imaging"),
            "pleural_chest_xray_bundled": ("pleural_codes", "radiology_codes"),
            "pleural_post_procedure_cxr": ("primary_codes", "drop_codes"),
        }
        for rule_name, (trigger_key, drop_key) in rule_map.items():
            data = self.bundling_rules.get(rule_name)
            if not data:
                continue
            trigger_codes = set(data.get(trigger_key, []))
            drop_codes = set(data.get(drop_key, []))
            if not trigger_codes or not drop_codes:
                continue
            self._drop_when_triggered(
                codes=codes,
                entry_id=entry_id,
                logger=logger,
                rule_name=rule_name,
                trigger_codes=trigger_codes,
                drop_codes=drop_codes,
                reason=data.get("description", ""),
            )

    def _apply_flag_only_rules(self, codes: list[CPTCode], entry_id: str, logger: IssueLogger) -> None:
        active_codes = self._active_code_set(codes)
        stent_rule = self.bundling_rules.get("stent_dilation_same_segment", {})
        stent_codes = set(stent_rule.get("stent_codes", []))
        dilation_codes = set(stent_rule.get("dilation_codes", []))
        if active_codes & stent_codes and active_codes & dilation_codes:
            logger.log(
                entry_id=entry_id,
                issue_type="possible_stent_dilation_bundle",
                severity="warn",
                action="flagged_for_manual",
                field="cpt_codes",
                details={"rule": "stent_dilation_same_segment"},
            )

        same_site = self.bundling_rules.get("31640_vs_31641_same_site", {})
        exclusive_codes = set(same_site.get("mutually_exclusive_codes", []))
        if {"31640", "31641"}.issubset(active_codes) or len(active_codes & exclusive_codes) > 1:
            logger.log(
                entry_id=entry_id,
                issue_type="31640_31641_same_session",
                severity="warn",
                action="flagged_for_manual",
                field="cpt_codes",
                details={"rule": "31640_vs_31641_same_site"},
            )

        radial_linear = self.bundling_rules.get("radial_linear_exclusive", {})
        exclusive = set(radial_linear.get("mutually_exclusive_codes", []))
        if len(active_codes & exclusive) > 1:
            logger.log(
                entry_id=entry_id,
                issue_type="radial_linear_overlap",
                severity="warn",
                action="flagged_for_manual",
                field="cpt_codes",
                details={"rule": "radial_linear_exclusive"},
            )

        chartis = self.bundling_rules.get("chartis_bundling", {})
        if chartis:
            if chartis.get("chartis_code") in active_codes and active_codes & set(chartis.get("valve_codes", [])):
                logger.log(
                    entry_id=entry_id,
                    issue_type="chartis_with_blvr_same_session",
                    severity="warn",
                    action="flagged_for_manual",
                    field="cpt_codes",
                    details={"rule": "chartis_bundling"},
                )

        pleural_vs_thora = self.bundling_rules.get("pleural_drainage_vs_thoracoscopy", {})
        if pleural_vs_thora:
            chest_tube = pleural_vs_thora.get("open_chest_tube")
            thorac_codes = set(pleural_vs_thora.get("thoracoscopy_therapeutic", []))
            if chest_tube in active_codes and active_codes & thorac_codes:
                logger.log(
                    entry_id=entry_id,
                    issue_type="possible_pleural_thoracoscopy_bundle",
                    severity="warn",
                    action="flagged_for_manual",
                    field="cpt_codes",
                    details={"rule": "pleural_drainage_vs_thoracoscopy"},
                )

    def _apply_ncci_pairs(self, codes: list[CPTCode], entry_id: str, logger: IssueLogger) -> None:
        for pair in self.ncci_pairs:
            primary = str(pair.get("primary"))
            secondary = str(pair.get("secondary"))
            allow_modifier = bool(pair.get("modifier_allowed"))
            if not primary or not secondary:
                continue
            if not self._has_active_code(codes, primary) or not self._has_active_code(codes, secondary):
                continue
            if not allow_modifier:
                self._deactivate_code(
                    codes,
                    target=secondary,
                    entry_id=entry_id,
                    logger=logger,
                    issue_type="ncci_unconditional_drop",
                    details={"primary": primary, "old": secondary, "new": None},
                )
                continue
            secondary_obj = self._find_active_code(codes, secondary)
            if secondary_obj and secondary_obj.has_modifier(["59", "XS"]):
                logger.log(
                    entry_id=entry_id,
                    issue_type="ncci_with_modifier",
                    severity="info",
                    action="flagged_for_manual",
                    field="cpt_codes",
                    details={"primary": primary, "secondary": secondary},
                )
            else:
                logger.log(
                    entry_id=entry_id,
                    issue_type="ncci_conflict_no_modifier",
                    severity="warn",
                    action="flagged_for_manual",
                    field="cpt_codes",
                    details={"primary": primary, "secondary": secondary},
                )

    def _enforce_sedation_rules(
        self,
        entry: dict[str, Any],
        codes: list[CPTCode],
        entry_id: str,
        logger: IssueLogger,
    ) -> None:
        sedation_type = entry.get("sedation_type")
        if sedation_type in SEDATION_BLOCKING_TYPES:
            for code in list(SEDATION_CODES):
                if self._has_active_code(codes, code):
                    self._deactivate_code(
                        codes,
                        target=code,
                        entry_id=entry_id,
                        logger=logger,
                        issue_type="sedation_incompatible_with_anesthesia",
                        details={"sedation_type": sedation_type, "old": code, "new": None},
                    )

    def _enforce_em_rules(self, codes: list[CPTCode], entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        active_codes = self._active_code_set(codes)
        em_codes_present = active_codes & E_AND_M_CODES
        if not em_codes_present:
            return
        minor_overlap = active_codes & self.em_minor_codes
        if minor_overlap:
            for em_code in em_codes_present:
                obj = self._find_active_code(codes, em_code)
                if obj and not obj.has_modifier(["25"]):
                    logger.log(
                        entry_id=entry_id,
                        issue_type="em_without_25_modifier",
                        severity="warn",
                        action="flagged_for_manual",
                        field="cpt_codes",
                        details={"em_code": em_code, "procedures": sorted(minor_overlap)},
                    )
        major_overlap = active_codes & self.em_major_codes
        if major_overlap:
            for em_code in em_codes_present:
                obj = self._find_active_code(codes, em_code)
                if obj and not obj.has_modifier(["57"]):
                    logger.log(
                        entry_id=entry_id,
                        issue_type="major_proc_em_without_57",
                        severity="warn",
                        action="flagged_for_manual",
                        field="cpt_codes",
                        details={"em_code": em_code, "procedures": sorted(major_overlap)},
                    )

    # ---- lower-level helpers ----

    def _drop_when_triggered(
        self,
        *,
        codes: list[CPTCode],
        entry_id: str,
        logger: IssueLogger,
        rule_name: str,
        trigger_codes: set[str],
        drop_codes: set[str],
        reason: str,
    ) -> None:
        active_codes = self._active_code_set(codes)
        if active_codes.isdisjoint(trigger_codes):
            return
        affected = active_codes & drop_codes
        for target in affected:
            self._deactivate_code(
                codes,
                target=target,
                entry_id=entry_id,
                logger=logger,
                issue_type="bundling_auto_drop",
                details={"rule": rule_name, "reason": reason, "old": target, "new": None},
            )

    def _deactivate_code(
        self,
        codes: list[CPTCode],
        *,
        target: str,
        entry_id: str,
        logger: IssueLogger,
        issue_type: str,
        details: dict[str, Any],
        field: str = "cpt_codes",
    ) -> None:
        removed = False
        for code in codes:
            if code.active and code.code == target:
                code.active = False
                removed = True
        if removed:
            logger.log(
                entry_id=entry_id,
                issue_type=issue_type,
                severity="info",
                action="auto_fixed",
                field=field,
                details=details,
            )

    def _has_active_code(self, codes: list[CPTCode], target: str) -> bool:
        return any(code.active and code.code == target for code in codes)

    def _find_active_code(self, codes: list[CPTCode], target: str) -> CPTCode | None:
        for code in codes:
            if code.active and code.code == target:
                return code
        return None

    def _active_code_set(self, codes: list[CPTCode]) -> set[str]:
        return {code.code for code in codes if code.active}

    def _calculate_total_rvu(self, codes: list[CPTCode]) -> float:
        rvus = self.kb.get("rvus", {})
        rvus_additional = self.kb.get("rvus_additional", {})
        total = 0.0
        for code in self._active_code_set(codes):
            metrics = rvus.get(code) or rvus_additional.get(code)
            if not isinstance(metrics, dict):
                continue
            work = float(metrics.get("work") or 0.0)
            pe = float(metrics.get("pe") or 0.0)
            mp = float(metrics.get("mp") or 0.0)
            total += work + pe + mp
        return round(total, 3)

    def _build_known_codes(self) -> set[str]:
        codes: set[str] = set()
        for container_key in ("rvus", "rvus_additional", "hcpcs"):
            container = self.kb.get(container_key, {})
            if isinstance(container, dict):
                codes.update(str(key) for key in container.keys())
        code_lists = self.kb.get("code_lists", {}) or {}
        for values in code_lists.values():
            for value in values or []:
                codes.add(str(value))
        pleural = self.kb.get("pleural", {}) or {}
        cpt_map = pleural.get("cpt_map", {}) or {}
        for key in cpt_map.keys():
            codes.add(str(key))
        codes.update(str(code) for code in self.kb.get("add_on_codes", []) or [])
        return codes

    def _build_add_on_requirements(self) -> dict[str, set[str]]:
        return {
            "+31627": {"31624", "31625", "31626", "31628", "31629", "31630", "31632", "31633", "31652", "31653"},
            "+31632": {"31628"},
            "+31633": {"31629"},
            "+99153": {"99152", "99151"},
            "+99157": {"99156"},
            "+99292": {"99291"},
        }
