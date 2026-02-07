"""Knowledge Base adapter for JSON/CSV files.

Loads the IP coding and billing knowledge base from JSON files
and implements the KnowledgeBaseRepository interface.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Set

from app.domain.knowledge_base.models import ProcedureInfo, NCCIPair
from app.domain.knowledge_base.repository import KnowledgeBaseRepository
from app.common.exceptions import KnowledgeBaseError


_KB_ALLOW_VERSION_MISMATCH_ENV_VAR = "PSUITE_KNOWLEDGE_ALLOW_VERSION_MISMATCH"


def _allow_version_mismatch() -> bool:
    value = os.environ.get(_KB_ALLOW_VERSION_MISMATCH_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes"}


def _extract_semver_from_filename(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_v(\d+)[._](\d+)\.json$", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_semver_from_kb_version(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().lstrip("v").split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _validate_filename_semver_matches_version(kb_version: object, kb_path: Path) -> None:
    if _allow_version_mismatch():
        return
    file_semver = _extract_semver_from_filename(kb_path.resolve())
    kb_semver = _extract_semver_from_kb_version(kb_version)
    if not file_semver or not kb_semver:
        return
    if file_semver != kb_semver:
        raise KnowledgeBaseError(
            f"KB filename semver v{file_semver[0]}_{file_semver[1]} does not match internal version "
            f"{kb_version!r} ({kb_path}). Set {_KB_ALLOW_VERSION_MISMATCH_ENV_VAR}=1 to override."
        )


class JsonKnowledgeBaseAdapter(KnowledgeBaseRepository):
    """Adapter that loads KB from the ip_coding_billing JSON format."""

    def __init__(self, data_path: str | Path, *, raw_data: dict | None = None):
        self._data_path = Path(data_path)
        self._raw_data: dict = {}
        self._raw_data_provided = raw_data is not None
        self._procedures: dict[str, ProcedureInfo] = {}
        self._ncci_pairs: dict[str, list[NCCIPair]] = {}
        self._mer_groups: dict[str, str] = {}
        self._addon_codes: set[str] = set()
        self._all_codes: set[str] = set()
        self._version: str = ""

        if raw_data is not None:
            if not isinstance(raw_data, dict):
                raise KnowledgeBaseError("KB raw_data must be a dict")
            self._raw_data = raw_data

        self._load_data()

    @property
    def version(self) -> str:
        return self._version

    @staticmethod
    def _normalize_code(code: str) -> str:
        return str(code).strip().upper().lstrip("+")

    def _load_data(self) -> None:
        """Load and parse the knowledge base JSON file."""
        if not self._raw_data_provided:
            if not self._data_path.is_file():
                raise KnowledgeBaseError(f"KB file not found: {self._data_path}")

            try:
                with self._data_path.open() as f:
                    self._raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise KnowledgeBaseError(f"Invalid JSON in KB file: {e}")

        self._version = self._raw_data.get("version", "unknown")
        _validate_filename_semver_matches_version(self._version, self._data_path)

        # Load NCCI pairs
        self._load_ncci_pairs()

        # Load MER groups from bundling_rules
        self._load_mer_groups()

        # Load addon codes from code_lists
        self._load_addon_codes()

        # Load procedure code metadata / RVUs (needs add-on metadata)
        self._load_procedures()

    def _load_procedures(self) -> None:
        """Load procedure information from master_code_index when present."""
        master_index = self._raw_data.get("master_code_index")
        if isinstance(master_index, dict) and master_index:
            self._load_procedures_from_master_index(master_index)
            return

        # Legacy fallback: fee_schedules (non-authoritative once master_code_index exists)
        self._load_procedures_from_fee_schedules()

    @staticmethod
    def _to_float(value: object) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _pick_latest_cms_financials(self, entry: dict) -> tuple[dict, int] | tuple[None, None]:
        financials = entry.get("financials")
        if not isinstance(financials, dict):
            return None, None

        candidates: list[tuple[int, str]] = []
        for key in financials:
            if not isinstance(key, str):
                continue
            match = re.match(r"^cms_pfs_(\d{4})$", key)
            if not match:
                continue
            candidates.append((int(match.group(1)), key))

        if not candidates:
            return None, None

        year, key = max(candidates, key=lambda item: item[0])
        payload = financials.get(key)
        if not isinstance(payload, dict):
            return None, None
        return payload, year

    def _lookup_fee_schedule_description(self, code: str, entry: dict) -> str | None:
        fee_schedules = self._raw_data.get("fee_schedules")
        if not isinstance(fee_schedules, dict) or not fee_schedules:
            return None

        sources = entry.get("fee_schedule_sources")
        source_names: list[str] = []
        if isinstance(sources, str) and sources.strip():
            source_names = [sources.strip()]
        elif isinstance(sources, list):
            source_names = [s.strip() for s in sources if isinstance(s, str) and s.strip()]

        def _desc_from_schedule(schedule_name: str) -> str | None:
            schedule = fee_schedules.get(schedule_name)
            if not isinstance(schedule, dict):
                return None
            codes = schedule.get("codes")
            if not isinstance(codes, dict):
                return None
            item = codes.get(code)
            if not isinstance(item, dict):
                return None
            desc = item.get("description")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
            return None

        # Prefer the entry's stated fee_schedule_sources (if present).
        for schedule_name in source_names:
            desc = _desc_from_schedule(schedule_name)
            if desc:
                return desc

        # Fallback: if sources are missing/misconfigured, try any schedule.
        if not source_names:
            for schedule_name in fee_schedules:
                desc = _desc_from_schedule(schedule_name)
                if desc:
                    return desc

        return None

    def _load_procedures_from_master_index(self, master_index: dict) -> None:
        """Load ProcedureInfo from KB master_code_index (authoritative)."""
        for code, entry in master_index.items():
            if not isinstance(code, str):
                continue
            if not isinstance(entry, dict):
                continue

            normalized = self._normalize_code(code)
            if not normalized or normalized in self._procedures:
                continue

            attributes = entry.get("attributes")
            attrs = attributes if isinstance(attributes, dict) else {}
            if str(entry.get("type") or "") != "reference" and attrs.get("status") != "deleted":
                self._all_codes.add(normalized)
            is_addon = bool(attrs.get("is_add_on")) or normalized in self._addon_codes

            short_descriptor = str(entry.get("descriptor") or entry.get("description") or "")
            professional_desc = self._lookup_fee_schedule_description(normalized, entry) or ""
            description = professional_desc or short_descriptor
            category = str(entry.get("family") or "general")

            cms_financials, cms_year = self._pick_latest_cms_financials(entry)
            simplified = entry.get("rvu_simplified") if isinstance(entry.get("rvu_simplified"), dict) else {}

            work_rvu = self._to_float(
                (cms_financials or {}).get("work_rvu") if isinstance(cms_financials, dict) else None
            ) or self._to_float(simplified.get("work"))
            facility_pe_rvu = self._to_float(
                (cms_financials or {}).get("facility_pe_rvu") if isinstance(cms_financials, dict) else None
            ) or self._to_float(simplified.get("pe"))
            malpractice_rvu = self._to_float(
                (cms_financials or {}).get("mp_rvu") if isinstance(cms_financials, dict) else None
            ) or self._to_float(simplified.get("mp"))
            total_facility_rvu = self._to_float(
                (cms_financials or {}).get("total_facility_rvu") if isinstance(cms_financials, dict) else None
            )
            if total_facility_rvu <= 0.0:
                total_facility_rvu = work_rvu + facility_pe_rvu + malpractice_rvu

            raw_data = dict(entry)
            raw_data["code"] = normalized
            raw_data["description"] = description
            raw_data["descriptor_short"] = short_descriptor
            if professional_desc:
                raw_data["description_professional"] = professional_desc
            raw_data["category"] = category
            raw_data["is_add_on"] = is_addon
            raw_data["global_days"] = attrs.get("global_days")
            raw_data["status_code"] = attrs.get("status_code")
            if isinstance(cms_financials, dict):
                raw_data["cms_pfs_year"] = cms_year
                raw_data["cms_pfs"] = dict(cms_financials)
                raw_data["total_nonfacility_rvu"] = cms_financials.get("total_nonfacility_rvu")

            self._procedures[normalized] = ProcedureInfo(
                code=normalized,
                description=description,
                category=category,
                work_rvu=work_rvu,
                facility_pe_rvu=facility_pe_rvu,
                malpractice_rvu=malpractice_rvu,
                total_facility_rvu=total_facility_rvu,
                is_addon=is_addon,
                parent_codes=[],
                bundled_with=[],
                mer_group=self._mer_groups.get(normalized),
                modifiers=[],
                notes=None,
                raw_data=raw_data,
            )

    def _load_procedures_from_fee_schedules(self) -> None:
        """Load procedure information from fee_schedules section (legacy)."""
        fee_schedules = self._raw_data.get("fee_schedules", {})

        for schedule_name, schedule_data in fee_schedules.items():
            codes_section = schedule_data.get("codes", {})

            for code, code_data in codes_section.items():
                normalized = self._normalize_code(str(code))
                if not normalized or normalized in self._procedures:
                    # Already loaded from another schedule
                    continue

                self._all_codes.add(normalized)

                # Determine category from schedule name
                category = self._extract_category(schedule_name)

                # Check if addon
                is_addon = str(code).startswith("+") or normalized in self._addon_codes

                proc_info = ProcedureInfo(
                    code=normalized,
                    description=code_data.get("description", ""),
                    category=category,
                    work_rvu=float(code_data.get("work_rvu") or 0),
                    facility_pe_rvu=float(code_data.get("facility_pe_rvu") or 0),
                    malpractice_rvu=float(code_data.get("malpractice_rvu") or 0),
                    total_facility_rvu=float(code_data.get("total_facility_rvu") or 0),
                    is_addon=is_addon,
                    parent_codes=[],
                    bundled_with=[],
                    mer_group=self._mer_groups.get(normalized),
                    modifiers=code_data.get("modifiers", []),
                    notes=code_data.get("notes"),
                    raw_data=code_data,
                )
                self._procedures[normalized] = proc_info

    def _load_ncci_pairs(self) -> None:
        """Load NCCI edit pairs from the ncci_pairs section."""
        ncci_list = self._raw_data.get("ncci_pairs", [])

        for pair_data in ncci_list:
            primary_raw = pair_data.get("primary")
            secondary_raw = pair_data.get("secondary")
            if not isinstance(primary_raw, str) or not isinstance(secondary_raw, str):
                continue
            primary = self._normalize_code(primary_raw)
            secondary = self._normalize_code(secondary_raw)
            modifier_allowed = pair_data.get("modifier_allowed", False)
            reason = pair_data.get("reason", "")

            if not primary or not secondary:
                continue

            pair = NCCIPair(
                primary=primary,
                secondary=secondary,
                modifier_allowed=modifier_allowed,
                reason=reason,
            )

            # Index by primary code
            if primary not in self._ncci_pairs:
                self._ncci_pairs[primary] = []
            self._ncci_pairs[primary].append(pair)

            # Also index by secondary for reverse lookups
            if secondary not in self._ncci_pairs:
                self._ncci_pairs[secondary] = []
            self._ncci_pairs[secondary].append(pair)

    def _load_mer_groups(self) -> None:
        """Load MER groups from bundling_rules section."""
        bundling_rules = self._raw_data.get("bundling_rules", {})

        for rule_name, rule_data in bundling_rules.items():
            if not isinstance(rule_data, dict):
                continue

            # Look for MER-related rules
            if "mer_group" in rule_data or rule_data.get("rule_type") == "mer":
                mer_group_id = rule_data.get("mer_group", rule_name)
                codes = rule_data.get("codes", [])
                for code in codes:
                    if not isinstance(code, str):
                        continue
                    normalized = self._normalize_code(code)
                    if not normalized:
                        continue
                    self._mer_groups[normalized] = mer_group_id

    def _load_addon_codes(self) -> None:
        """Load addon code list from add_on_codes/code_lists sections."""
        for code in self._raw_data.get("add_on_codes", []) or []:
            if not isinstance(code, str):
                continue
            normalized = self._normalize_code(code)
            if not normalized:
                continue
            self._addon_codes.add(normalized)
            self._addon_codes.add(f"+{normalized}")

        code_lists = self._raw_data.get("code_lists", {})

        # Any code explicitly prefixed with '+' in code_lists is an add-on signal,
        # even if the underlying fee schedule uses the plain numeric code.
        for codes in code_lists.values():
            if not isinstance(codes, list):
                continue
            for code in codes:
                if not isinstance(code, str):
                    continue
                if code.startswith("+"):
                    self._addon_codes.add(code)
                    self._addon_codes.add(code.lstrip("+"))

        # Preserve older behavior: treat lists named "*addon*" as add-on sources.
        for list_name, codes in code_lists.items():
            if "addon" not in str(list_name).lower() or not isinstance(codes, list):
                continue
            for code in codes:
                if not isinstance(code, str):
                    continue
                self._addon_codes.add(code)
                if code.startswith("+"):
                    self._addon_codes.add(code.lstrip("+"))

        # Also treat master_code_index entries with attributes.is_add_on as add-ons.
        master_index = self._raw_data.get("master_code_index")
        if isinstance(master_index, dict):
            for code, entry in master_index.items():
                if not isinstance(code, str) or not isinstance(entry, dict):
                    continue
                attributes = entry.get("attributes")
                if not isinstance(attributes, dict) or not attributes.get("is_add_on"):
                    continue
                normalized = self._normalize_code(code)
                if not normalized:
                    continue
                self._addon_codes.add(normalized)
                self._addon_codes.add(f"+{normalized}")

    def _extract_category(self, schedule_name: str) -> str:
        """Extract category from schedule name."""
        # e.g., "physician_2025_airway" -> "airway"
        parts = schedule_name.split("_")
        if len(parts) >= 3:
            return parts[-1]
        return "general"

    def get_procedure_info(self, code: str) -> Optional[ProcedureInfo]:
        """Get procedure information for a CPT code."""
        normalized = self._normalize_code(code)
        if not normalized:
            return None

        proc = self._procedures.get(normalized)
        if proc is None and os.getenv("PSUITE_KB_STRICT", "0").strip() == "1":
            raise ValueError(
                "KB_STRICT violation: Code "
                f"'{normalized}' was requested by application logic but does not exist in the "
                f"Knowledge Base source ({self._data_path})."
            )
        return proc

    def get_mer_group(self, code: str) -> Optional[str]:
        """Get the MER group ID for a code, if any."""
        normalized = self._normalize_code(code)
        return self._mer_groups.get(normalized)

    def get_ncci_pairs(self, code: str) -> list[NCCIPair]:
        """Get all NCCI pairs where this code is involved."""
        normalized = self._normalize_code(code)
        return self._ncci_pairs.get(normalized, [])

    def is_addon_code(self, code: str) -> bool:
        """Check if a code is an add-on code."""
        if str(code).startswith("+"):
            return True
        normalized = self._normalize_code(code)
        return normalized in self._addon_codes or f"+{normalized}" in self._addon_codes

    def get_all_codes(self) -> Set[str]:
        """Get all valid CPT codes in the knowledge base."""
        return self._all_codes.copy()

    def get_parent_codes(self, addon_code: str) -> list[str]:
        """Get valid parent codes for an add-on code."""
        proc = self.get_procedure_info(addon_code)
        if proc:
            return proc.parent_codes
        return []

    def get_bundled_codes(self, code: str) -> list[str]:
        """Get codes that are bundled with the given code."""
        proc = self.get_procedure_info(code)
        if proc:
            return proc.bundled_with
        return []


# Alias for backwards compatibility with starter scripts
CsvKnowledgeBaseAdapter = JsonKnowledgeBaseAdapter
