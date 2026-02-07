from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set
import json
import re

from app.autocode.ip_kb import canonical_rules


@dataclass
class CmsRvuInfo:
    code: str
    category: str
    description: Optional[str]
    work_rvu: Optional[float]
    facility_pe_rvu: Optional[float]
    nonfacility_pe_rvu: Optional[float]
    mp_rvu: Optional[float]
    total_facility_rvu: Optional[float]
    total_nonfacility_rvu: Optional[float]
    mpfs_facility_payment: Optional[float]
    mpfs_nonfacility_payment: Optional[float]
    global_days: Optional[str]
    mult_proc_indicator: Optional[str]
    status_code: Optional[str]
    is_add_on: bool


@dataclass
class CPTInfo:
    code: str
    description: Optional[str]
    groups: List[str]
    is_add_on: bool
    rvus: Optional[dict]
    fee_schedule: Optional[dict]
    cms_rvu: Optional[CmsRvuInfo]


class IPCodingKnowledgeBase:
    """
    Knowledge base for IP CPT coding.

    Primary source of truth is the `ip_coding_billing` knowledge base JSON:
    - `code_lists` / add-on metadata
    - `synonyms` / `terminology_mappings` phrase lists (so adding a new synonym is a one-file edit)

    Bundling logic remains derived from canonical rules (see `canonical_rules.py`).
    """

    # Map internal group names to v2.7 code_lists keys for compatibility
    # v2.6 flattened the code_lists structure; this mapping bridges the gap
    GROUP_ALIAS_MAP: Dict[str, List[str]] = {
        # Bronchoscopy mappings
        "bronchoscopy_airway_dilation": ["bronchoscopy_therapeutic_dilation"],
        "bronchoscopy_airway_stenosis": ["bronchoscopy_therapeutic_dilation", "bronchoscopy_therapeutic_airway_tumor"],
        "bronchoscopy_bal": ["bronchoscopy_diagnostic"],  # BAL is part of diagnostic (31624)
        "bronchoscopy_biopsy_parenchymal_additional": ["bronchoscopy_biopsy_parenchymal"],  # +31632 is in parenchymal
        "bronchoscopy_ebus_linear_additional": ["bronchoscopy_ebus_linear"],  # 31653 is in linear
        "bronchoscopy_stent_revision": ["bronchoscopy_therapeutic_stent"],  # 31638 is in stent
        "bronchoscopy_therapeutic_aspiration": ["bronchoscopy_therapeutic_airway_tumor"],  # aspiration codes
        # Thoracoscopy mappings (v2.7 consolidates site-specific biopsies into thoracoscopy_biopsy)
        "thoracoscopy_diagnostic_only": ["thoracoscopy_diagnostic"],
        "thoracoscopy_lung_biopsy": ["thoracoscopy_biopsy"],
        "thoracoscopy_mediastinal_biopsy": ["thoracoscopy_biopsy"],
        "thoracoscopy_pericardial_biopsy": ["thoracoscopy_biopsy"],
        "thoracoscopy_pleural_biopsy": ["thoracoscopy_biopsy"],
        "thoracoscopy_surgical_pleurodesis_decortication": ["thoracoscopy_surgical"],
        # Pleural mappings
        "pleural_pleurodesis": ["pleural_intrapleural_treatment"],
        "pleural_drainage": ["pleural_drainage_catheter"],
        # PDT is not in v2.7 code_lists - need to handle separately or add
        "pdt_endobronchial": [],  # No mapping available; would need KB update
    }

    def __init__(self, json_source: Path | str | Mapping[str, Any]):
        self.json_path: Path | None
        if isinstance(json_source, (str, Path)):
            self.json_path = Path(json_source)
            with self.json_path.open(encoding="utf-8") as handle:
                self.raw = json.load(handle)
        else:
            self.json_path = None
            self.raw = dict(json_source)

        self.metadata: Dict[str, Any] = self.raw.get("metadata", {})
        self.code_to_groups: Dict[str, Set[str]] = {}
        self.add_on_codes: Set[str] = set()
        self.cpt_rvus: Dict[str, dict] = self.raw.get("rvus", {})
        self.last_group_evidence: Dict[str, dict] = {}
        self.cms_rvu_meta: Dict[str, Any] = {}
        self.cms_rvus: Dict[str, CmsRvuInfo] = {}
        self._fee_schedules: Dict[str, Dict[str, Any]] = {}

        self._build_indexes()
        self._build_cms_rvu_table()
        self._build_fee_schedule_index()

    def _normalize_code(self, code: str) -> str:
        return code.lstrip("+").strip()

    def _contains_any(self, text: str, phrases: Iterable[str]) -> bool:
        return any(p.lower() in text for p in phrases)

    def _synonym_terms(self, key: str) -> List[str]:
        synonyms = self.raw.get("synonyms", {})
        if not isinstance(synonyms, Mapping):
            return []
        values = synonyms.get(key, [])
        if isinstance(values, list):
            return [v for v in values if isinstance(v, str)]
        return []

    def _build_indexes(self) -> None:
        # 1) Code lists
        for group, codes in self.raw.get("code_lists", {}).items():
            for code in codes:
                c = self._normalize_code(code)
                self.code_to_groups.setdefault(c, set()).add(group)

        # 2) Add-on codes
        for code in self.raw.get("add_on_codes", []):
            self.add_on_codes.add(self._normalize_code(code))

        # 3) Pleural / thoracoscopy maps
        pleural = self.raw.get("pleural", {})
        for submap_key in ("cpt_map", "thoracoscopy_cpt_map"):
            for code in pleural.get(submap_key, {}):
                c = self._normalize_code(code)
                self.code_to_groups.setdefault(c, set()).add(submap_key)

        # 4) Explicit lists in pleural section
        pleural_lists = [
            "thoracentesis_codes",
            "pleural_drainage_codes",
            "chest_tube_insertion_codes",
            "chest_tube_removal_codes",
            "ipc_codes",
            "thoracoscopy_diagnostic_codes",
            "thoracoscopy_surgical_codes",
            "ttna_codes",
        ]
        for list_name in pleural_lists:
            for code in pleural.get(list_name, []):
                c = self._normalize_code(code)
                self.code_to_groups.setdefault(c, set()).add(list_name)

        # 5) HCPCS devices that declare primary CPTs (e.g. C1601)
        for _hcpcs, info in self.raw.get("hcpcs", {}).items():
            for group_name, codes in info.get("primary_cpt_eligible", {}).items():
                for code in codes:
                    c = self._normalize_code(code)
                    self.code_to_groups.setdefault(c, set()).add(group_name)

    def _build_cms_rvu_table(self) -> None:
        cms_rvus = self.raw.get("cms_rvus", {}) or {}
        for key, section in cms_rvus.items():
            if key.startswith("_"):
                self.cms_rvu_meta[key] = section
                continue
            if not isinstance(section, Mapping):
                continue
            for code, payload in section.items():
                if not isinstance(payload, Mapping):
                    continue
                normalized = self._normalize_code(code)
                info = CmsRvuInfo(
                    code=normalized,
                    category=key,
                    description=payload.get("description"),
                    work_rvu=self._to_float(payload.get("work_rvu")),
                    facility_pe_rvu=self._to_float(payload.get("facility_pe_rvu")),
                    nonfacility_pe_rvu=self._to_float(payload.get("nonfacility_pe_rvu")),
                    mp_rvu=self._to_float(payload.get("mp_rvu")),
                    total_facility_rvu=self._to_float(payload.get("total_facility_rvu")),
                    total_nonfacility_rvu=self._to_float(payload.get("total_nonfacility_rvu")),
                    mpfs_facility_payment=self._to_float(payload.get("mpfs_facility_payment")),
                    mpfs_nonfacility_payment=self._to_float(payload.get("mpfs_nonfacility_payment")),
                    global_days=payload.get("global_days"),
                    mult_proc_indicator=str(payload.get("mult_proc_indicator")) if payload.get("mult_proc_indicator") is not None else None,
                    status_code=payload.get("status_code"),
                    is_add_on=bool(payload.get("is_add_on") or normalized in self.add_on_codes),
                )
                self.cms_rvus[normalized] = info

    def _build_fee_schedule_index(self) -> None:
        for name, schedule in (self.raw.get("fee_schedules") or {}).items():
            if not isinstance(schedule, Mapping):
                continue
            codes = schedule.get("codes", {})
            normalized_codes: Dict[str, Dict[str, Any]] = {}
            if isinstance(codes, Mapping):
                for code, info in codes.items():
                    if isinstance(info, Mapping):
                        normalized_codes[self._normalize_code(code)] = dict(info)
            self._fee_schedules[name] = {
                "metadata": dict(schedule.get("metadata", {})),
                "codes": normalized_codes,
            }

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ---------- Public API ----------

    def all_relevant_cpt_codes(self) -> Set[str]:
        """
        Union of everything referenced in the JSON.
        """
        codes: Set[str] = set()

        # code_lists
        for codes_list in self.raw.get("code_lists", {}).values():
            for code in codes_list:
                codes.add(self._normalize_code(code))

        # rvus (skip metadata keys starting with _)
        codes.update(k for k in self.raw.get("rvus", {}).keys() if not k.startswith("_"))

        # rvus_additional
        codes.update(self.raw.get("rvus_additional", {}).keys())

        # pleural
        pleural = self.raw.get("pleural", {})
        for submap_key in ("cpt_map", "thoracoscopy_cpt_map"):
            codes.update(map(self._normalize_code, pleural.get(submap_key, {}).keys()))

        pleural_lists = [
            "thoracentesis_codes",
            "pleural_drainage_codes",
            "chest_tube_insertion_codes",
            "chest_tube_removal_codes",
            "ipc_codes",
            "thoracoscopy_diagnostic_codes",
            "thoracoscopy_surgical_codes",
            "ttna_codes",
        ]
        for list_name in pleural_lists:
            codes.update(map(self._normalize_code, pleural.get(list_name, [])))

        # HCPCS primary_cpt_eligible
        for _hcpcs, info in self.raw.get("hcpcs", {}).items():
            for _group, cpt_list in info.get("primary_cpt_eligible", {}).items():
                codes.update(self._normalize_code(c) for c in cpt_list)

        if self.cms_rvus:
            codes.update(self.cms_rvus.keys())

        return codes

    def get_cms_rvu(self, code: str) -> Optional[CmsRvuInfo]:
        """
        Return CMS RVU metadata for a code. Accepts codes with or without the leading '+'.
        """
        return self.cms_rvus.get(self._normalize_code(code))

    def get_conversion_factor(self) -> Optional[float]:
        cf = self.cms_rvu_meta.get("_conversion_factor")
        if cf is None:
            return None
        try:
            return float(cf)
        except (TypeError, ValueError):
            return None

    def get_fee_schedule_entry(self, code: str, schedule_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Return a vendor fee schedule entry for the CPT.
        If schedule_name is omitted, the first schedule containing the code is returned.
        """
        norm = self._normalize_code(code)
        if schedule_name:
            schedule = self._fee_schedules.get(schedule_name)
            if not schedule:
                return None
            entry = schedule["codes"].get(norm)
            return dict(entry) if entry else None

        for schedule in self._fee_schedules.values():
            entry = schedule["codes"].get(norm)
            if entry:
                return dict(entry)
        return None

    def get_groups_for_code(self, code: str) -> List[str]:
        return sorted(self.code_to_groups.get(self._normalize_code(code), []))

    def is_add_on(self, code: str) -> bool:
        return self._normalize_code(code) in self.add_on_codes

    def get_cpt_info(self, code: str) -> CPTInfo:
        n = self._normalize_code(code)

        # Description from various places
        desc = None

        # 1) pleural->cpt_map / thoracoscopy_cpt_map
        pleural = self.raw.get("pleural", {})
        if n in pleural.get("cpt_map", {}):
            desc = pleural["cpt_map"][n]["description"]
        elif n in pleural.get("thoracoscopy_cpt_map", {}):
            desc = pleural["thoracoscopy_cpt_map"][n]["description"]

        # 2) fee_schedules (airway/Noah/etc.)
        fee_sched_info = self.get_fee_schedule_entry(n)
        if fee_sched_info and not desc:
            desc = fee_sched_info.get("description")

        rvus = self.raw.get("rvus", {}).get(n) or self.raw.get("rvus_additional", {}).get(n)
        cms_rvu = self.get_cms_rvu(n)
        if cms_rvu and not desc:
            desc = cms_rvu.description

        return CPTInfo(
            code=n,
            description=desc,
            groups=self.get_groups_for_code(n),
            is_add_on=self.is_add_on(n),
            rvus=rvus,
            fee_schedule=fee_sched_info,
            cms_rvu=cms_rvu,
        )

    # ---- bundling helpers (using canonical rules) ----

    def apply_bundling(self, codes: Iterable[str], return_decisions: bool = False):
        """
        Apply bundling rules derived from canonical sources:
        - ip_golden_knowledge_v2_2.json global rules
        - synthetic_CPT_corrected.json excluded_or_bundled_codes

        This is deterministic bundling based on code combinations.

        Args:
            codes: Iterable of CPT codes to process
            return_decisions: If True, returns tuple of (codes, bundling_decisions)
                            where bundling_decisions explains what was bundled

        Returns:
            List[str] if return_decisions is False
            Tuple[List[str], List[dict]] if return_decisions is True
        """
        original_codes = list(codes)
        norm_to_original: Dict[str, List[str]] = {}
        norm_codes: Set[str] = set()
        bundling_decisions: List[dict] = []

        for c in original_codes:
            n = self._normalize_code(c)
            norm_codes.add(n)
            norm_to_original.setdefault(n, []).append(c)

        def record_bundle(bundled_code: str, dominant_code: str, rule_name: str, reason: str):
            """Helper to record a bundling decision."""
            if bundled_code in norm_codes:
                info = self.get_cpt_info(bundled_code)
                dominant_info = self.get_cpt_info(dominant_code)
                bundling_decisions.append({
                    "bundled_cpt": bundled_code,
                    "bundled_description": info.description if info else f"CPT {bundled_code}",
                    "dominant_cpt": dominant_code,
                    "dominant_description": dominant_info.description if dominant_info else f"CPT {dominant_code}",
                    "rule": rule_name,
                    "reason": reason,
                })

        # Rule 1: Radial EBUS (+31654) is often a false positive when linear EBUS-TBNA (31652/31653)
        # is present WITHOUT any peripheral-lesion workflow evidence (e.g., navigation/biopsy).
        #
        # In combined cases (linear staging + nav/peripheral biopsy), +31654 can be appropriate and
        # should be kept.
        linear_present = norm_codes & canonical_rules.RADIAL_LINEAR_EBUS_EXCLUSIVE["linear_ebus_codes"]
        radial_codes = canonical_rules.RADIAL_LINEAR_EBUS_EXCLUSIVE["radial_ebus_codes"]
        peripheral_context_present = bool(norm_codes & {"31627", "31628", "31629", "31626"})
        if linear_present and not peripheral_context_present:
            linear_code = list(linear_present)[0]
            for rc in radial_codes & norm_codes:
                record_bundle(rc, linear_code, "RADIAL_LINEAR_EBUS_EXCLUSIVE",
                             "Radial EBUS cannot be billed with linear EBUS-TBNA on the same date")
            norm_codes -= radial_codes

        # Rule 2: TBLB (31628) bundles brush (31623), TBNA (31629), EBB (31625) in same lobe
        if "31628" in norm_codes:
            for bc in canonical_rules.TBLB_BUNDLES_BRUSH["bundled_codes"] & norm_codes:
                record_bundle(bc, "31628", "TBLB_BUNDLES_BRUSH",
                             "Brushing is bundled into transbronchial lung biopsy when performed in the same lobe")
            for bc in canonical_rules.TBLB_BUNDLES_TBNA["bundled_codes"] & norm_codes:
                record_bundle(bc, "31628", "TBLB_BUNDLES_TBNA",
                             "Transbronchial needle aspiration is bundled into TBLB when performed in the same lobe")
            for bc in canonical_rules.TBLB_BUNDLES_EBB["bundled_codes"] & norm_codes:
                record_bundle(bc, "31628", "TBLB_BUNDLES_EBB",
                             "Endobronchial biopsy is bundled into TBLB when performed in the same lobe")
            norm_codes -= canonical_rules.TBLB_BUNDLES_BRUSH["bundled_codes"]
            norm_codes -= canonical_rules.TBLB_BUNDLES_TBNA["bundled_codes"]
            norm_codes -= canonical_rules.TBLB_BUNDLES_EBB["bundled_codes"]

        # Rule 3: IPC (32550) bundles US guidance (76942)
        if "32550" in norm_codes:
            for bc in canonical_rules.IPC_BUNDLES_US_GUIDANCE["bundled_codes"] & norm_codes:
                record_bundle(bc, "32550", "IPC_BUNDLES_US_GUIDANCE",
                             "Ultrasound guidance is bundled into tunneled pleural catheter placement")
            norm_codes -= canonical_rules.IPC_BUNDLES_US_GUIDANCE["bundled_codes"]

        # Rule 4: Surgical thoracoscopy (32650) bundles diagnostic thoracoscopy and pleurodesis
        if "32650" in norm_codes:
            for bc in canonical_rules.THORACOSCOPY_SURGICAL_BUNDLES_DIAGNOSTIC["bundled_codes"] & norm_codes:
                record_bundle(bc, "32650", "THORACOSCOPY_SURGICAL_BUNDLES_DIAGNOSTIC",
                             "Diagnostic thoracoscopy is bundled into surgical thoracoscopy with pleurodesis")
            for bc in canonical_rules.THORACOSCOPY_BUNDLES_PLEURODESIS["bundled_codes"] & norm_codes:
                record_bundle(bc, "32650", "THORACOSCOPY_BUNDLES_PLEURODESIS",
                             "Chest tube pleurodesis is bundled into surgical thoracoscopy with pleurodesis")
            norm_codes -= canonical_rules.THORACOSCOPY_SURGICAL_BUNDLES_DIAGNOSTIC["bundled_codes"]
            norm_codes -= canonical_rules.THORACOSCOPY_BUNDLES_PLEURODESIS["bundled_codes"]

        # Rule 5: PDT (96570) bundles diagnostic bronchoscopy (31622)
        if "96570" in norm_codes:
            for bc in canonical_rules.PDT_BUNDLES_DIAGNOSTIC["bundled_codes"] & norm_codes:
                record_bundle(bc, "96570", "PDT_BUNDLES_DIAGNOSTIC",
                             "Diagnostic bronchoscopy is bundled into photodynamic therapy")
            norm_codes -= canonical_rules.PDT_BUNDLES_DIAGNOSTIC["bundled_codes"]

        # Rule 6: Stent placement bundles dilation and ablation same site
        stent_present = norm_codes & canonical_rules.STENT_BUNDLES_DILATION["stent_codes"]
        if stent_present:
            stent_code = list(stent_present)[0]
            for bc in canonical_rules.STENT_BUNDLES_DILATION["dilation_codes"] & norm_codes:
                record_bundle(bc, stent_code, "STENT_BUNDLES_DILATION",
                             "Balloon dilation is bundled into stent placement at the same site")
            for bc in canonical_rules.STENT_BUNDLES_ABLATION["ablation_codes"] & norm_codes:
                record_bundle(bc, stent_code, "STENT_BUNDLES_ABLATION",
                             "Tumor ablation/debulking is bundled into stent placement at the same site")
            norm_codes -= canonical_rules.STENT_BUNDLES_DILATION["dilation_codes"]
            norm_codes -= canonical_rules.STENT_BUNDLES_ABLATION["ablation_codes"]

        # Rule 6b: Ablation/destruction (31641) bundles dilation (31630) same site
        ablation_present = norm_codes & canonical_rules.ABLATION_BUNDLES_DILATION["ablation_codes"]
        if ablation_present:
            ablation_code = list(ablation_present)[0]
            for bc in canonical_rules.ABLATION_BUNDLES_DILATION["dilation_codes"] & norm_codes:
                record_bundle(bc, ablation_code, "ABLATION_BUNDLES_DILATION",
                             "Balloon dilation is bundled into tumor ablation/destruction at the same site")
            norm_codes -= canonical_rules.ABLATION_BUNDLES_DILATION["dilation_codes"]

        # Rule 7: Therapeutic aspiration (31645) bundles balloon (31634) for hemorrhage
        if "31645" in norm_codes:
            for bc in canonical_rules.HEMORRHAGE_BUNDLES_BALLOON["bundled_codes"] & norm_codes:
                record_bundle(bc, "31645", "HEMORRHAGE_BUNDLES_BALLOON",
                             "Balloon tamponade is bundled into therapeutic aspiration for hemorrhage control")
            norm_codes -= canonical_rules.HEMORRHAGE_BUNDLES_BALLOON["bundled_codes"]

        # Rule 8: Diagnostic bronchoscopy (31622) bundled with any surgical bronchoscopy
        # From golden knowledge: "surgical_includes_diagnostic_endoscopy"
        surgical_bronch_codes = {
            "31623", "31624", "31625", "31626", "31628", "31629", "31630",
            "31631", "31635", "31636", "31640", "31641", "31645", "31652",
            "31653", "96570"
        }
        surgical_present = norm_codes & surgical_bronch_codes
        if surgical_present and "31622" in norm_codes:
            # Pick the first surgical code as the dominant one
            dominant = sorted(surgical_present)[0]
            record_bundle("31622", dominant, "SURGICAL_INCLUDES_DIAGNOSTIC",
                         "Diagnostic bronchoscopy is bundled into surgical/therapeutic bronchoscopy procedures")
            norm_codes.discard("31622")

        # Rule 9: EBUS station upgrade: 31652 -> 31653 when 3+ stations
        # This is handled at detection time, but ensure no duplicate
        if "31653" in norm_codes and "31652" in norm_codes:
            record_bundle("31652", "31653", "EBUS_STATION_UPGRADE",
                         "EBUS 1-2 stations code replaced by 3+ stations code when 3 or more stations sampled")
            norm_codes.discard("31652")

        # Re-expand to original strings, preserving ordering
        result: List[str] = []
        for c in original_codes:
            n = self._normalize_code(c)
            if n in norm_codes:
                result.append(c)

        if return_decisions:
            return result, bundling_decisions
        return result

    def groups_from_text(self, note_text: str) -> Set[str]:
        """
        Synonym matcher using canonical rules from:
        - ip_golden_knowledge_v2_2.json
        - synthetic_CPT_corrected.json

        Returns matched procedure groups for CPT code selection.

        CONSERVATIVE CODING PRINCIPLES:
        1. High-value codes require STRONG positive evidence in the procedure note body
        2. Indications, history, and boilerplate text do NOT count as procedure evidence
        3. When ambiguous, prefer NOT billing (precision over recall for high-RVU codes)
        """
        text = note_text.lower()
        matched_groups: Set[str] = set()
        evidence: Dict[str, dict] = {}

        # Helper to check for procedure ACTION verbs (not just presence of a word)
        def has_action_verb(text: str, verb_patterns: list[str]) -> bool:
            """Check if any action verb pattern exists in text."""
            return any(v.lower() in text for v in verb_patterns)

        # ========== Navigation Detection ==========
        # Avoid short-token substring false positives (e.g., "ion" in "insertion").
        navigation_terms = self._synonym_terms("navigation_terms") or canonical_rules.NAVIGATION_SYNONYMS
        short_tokens = {"ion", "enb", "emn"}
        nav_platform = any(
            bool(re.search(rf"\b{re.escape(term)}\b", text)) if term in short_tokens else term in text
            for term in (t.lower() for t in navigation_terms)
        )

        nav_concept_terms = self._synonym_terms("navigation_concept_terms") or [
            "navigation",
            "navigational",
            "pathway",
            "registration",
            "ct-based",
            "ct based",
            "virtual",
            "3d",
            "guidance",
        ]
        nav_concept = any(term in text for term in (t.lower() for t in nav_concept_terms))

        # Any platform term is considered "direct" navigation evidence.
        # This ensures adding a new platform synonym is a one-file KB edit.
        nav_direct = nav_platform

        # Navigation FAILURE/ABORT detection - these indicate nav was NOT successfully performed
        nav_failure_patterns = [
            "navigation aborted", "navigation failed", "aborted due to",
            "unable to navigate", "navigation was unsuccessful",
            "could not navigate", "mis-registration", "misregistration",
            "navigation not performed", "navigation not completed",
            "catheter not advanced to target", "not advanced to target",
            "converted to conventional", "navigation unsuccessful"
        ]
        nav_aborted = any(f in text for f in nav_failure_patterns)

        evidence["bronchoscopy_navigation"] = {
            "platform": nav_platform, "concept": nav_concept, "direct": nav_direct,
            "aborted": nav_aborted
        }
        # Only add navigation group if NOT aborted/failed
        if nav_platform and (nav_concept or nav_direct) and not nav_aborted:
            matched_groups.add("bronchoscopy_navigation")

        # ========== EBUS Detection ==========
        # Radial EBUS
        radial_terms = self._synonym_terms("radial_ebus_terms") or canonical_rules.RADIAL_EBUS_SYNONYMS
        radial_hit = self._contains_any(text, radial_terms)
        radial_negated = False
        if radial_hit:
            radial_negated = bool(
                re.search(
                    r"\bradial\s+ebus\b[^\n]{0,120}\b(?:not|no|without|never|aborted|unable|failed|mentioned)\b",
                    text,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\b(?:not|no|without|never|aborted|unable|failed)\b[^\n]{0,120}\bradial\s+ebus\b",
                    text,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\b(?:probe|tool|catheter)\b[^\n]{0,80}\b(?:not|never)\s+advanced\b[^\n]{0,40}\b(?:lesion|target)\b",
                    text,
                    re.IGNORECASE,
                )
            )

        radial_hit = radial_hit and not radial_negated
        evidence["bronchoscopy_ebus_radial"] = {"radial": radial_hit, "negated": radial_negated}
        if radial_hit:
            matched_groups.add("bronchoscopy_ebus_radial")

        # Linear EBUS - detect stations
        linear_terms = self._synonym_terms("linear_ebus_terms") or canonical_rules.LINEAR_EBUS_SYNONYMS
        station_terms = self._synonym_terms("ebus_station_terms") or canonical_rules.EBUS_STATION_SYNONYMS
        linear_hit = self._contains_any(text, linear_terms)
        station_hit = self._contains_any(text, station_terms)

        # Count stations for 31652 vs 31653
        # Pattern 1: "stations 4R, 7, 10R" or "station 4R"
        station_pattern = re.compile(r"stations?\s+(\d{1,2}[a-z]?(?:\s*,?\s*(?:and\s+)?\d{1,2}[a-z]?)*)", re.IGNORECASE)
        station_matches = station_pattern.findall(text)
        station_count = 0
        if station_matches:
            for match in station_matches:
                stations = re.findall(r"\d{1,2}[a-z]?", match, re.IGNORECASE)
                station_count = max(station_count, len(stations))

        # Pattern 2: Look for explicit station lists like "4R, 7, 11R" near EBUS context
        # Only count numbers that look like station designations (not measurements)
        if ("ebus" in text or "tbna" in text) and ("station" in text or "nodal" in text or "node" in text):
            # Find numbers followed by R/L (station designations)
            station_nums = set(re.findall(r"\b(\d{1,2}[rl])\b", text, re.IGNORECASE))
            # Also find standalone station numbers in station context
            if "station" in text:
                station_standalone = set(re.findall(r"station\s+(\d{1,2})", text, re.IGNORECASE))
                station_nums.update(station_standalone)
            if len(station_nums) > station_count:
                station_count = len(station_nums)

        # Also look for explicit station counts in words
        if "five" in text or "5 stations" in text:
            station_count = max(station_count, 5)
        elif "four" in text or "4 stations" in text:
            station_count = max(station_count, 4)
        elif "three" in text or "3 or more" in text or "3 stations" in text or "multiple stations" in text:
            station_count = max(station_count, 3)
        elif "two" in text or "2 stations" in text:
            station_count = max(station_count, 2)
        elif station_count == 0 and ("station" in text or station_hit):
            station_count = 1

        # "multiple mediastinal and hilar stations" implies 3+ stations
        if "multiple" in text and ("mediastinal" in text or "hilar" in text) and "station" in text:
            station_count = max(station_count, 3)

        # EBUS detection: differentiate linear (nodal) vs radial (peripheral)
        # Linear EBUS triggers: "EBUS nodal", "EBUS staging", "EBUS-TBNA", stations
        linear_ebus_triggers = (
            "ebus nodal" in text or
            "ebus staging" in text or
            "ebus-tbna" in text or
            "ebus tbna" in text or
            "linear ebus" in text or
            "mediastinal staging" in text or
            station_hit or
            linear_hit
        )
        # Station context requires actual nodal stations
        has_station_context = station_hit or station_count >= 1 or "mediastinal" in text or "hilar" in text or "tbna" in text
        # Set ebus_mentioned only if linear EBUS evidence exists
        ebus_mentioned = linear_ebus_triggers

        evidence["bronchoscopy_ebus_linear"] = {
            "ebus": ebus_mentioned,
            "station_count": station_count,
            "station_context": has_station_context,
        }

        if ebus_mentioned and has_station_context:
            matched_groups.add("bronchoscopy_ebus_linear")
            if station_count >= 3:
                matched_groups.add("bronchoscopy_ebus_linear_additional")

        # ========== Biopsy Detection ==========
        # TBLB (transbronchial lung biopsy)
        tblb_terms = self._synonym_terms("tblb_terms") or canonical_rules.TBLB_SYNONYMS
        tblb_hit = self._contains_any(text, tblb_terms)
        if tblb_hit:
            matched_groups.add("bronchoscopy_biopsy_parenchymal")

        # TBNA of peripheral lesion (31629) - distinguish from EBUS-TBNA nodal staging
        tbna_hit = any(
            term in text
            for term in [
                "tbna",
                "transbronchial needle aspiration",
                "transbronchial needle biopsy",
                "needle aspiration",
            ]
        )
        tbna_station_context = bool(
            re.search(r"tbna[^\n]{0,80}station|station[^\n]{0,80}tbna", text, re.IGNORECASE)
        )
        tbna_lesion_context = bool(
            re.search(
                r"tbna[^\n]{0,120}\b(?:lesion|nodule|target|segment|lobe|rul|rml|rll|lul|lll)\b",
                text,
                re.IGNORECASE,
            )
        )
        evidence["bronchoscopy_tbna"] = {
            "tbna": tbna_hit,
            "lesion_context": tbna_lesion_context,
            "station_context": tbna_station_context,
        }
        if tbna_hit and tbna_lesion_context and not tbna_station_context:
            matched_groups.add("bronchoscopy_tbna")

        # Navigation with biopsy triggers TBLB + radial EBUS (common pattern)
        if "bronchoscopy_navigation" in matched_groups and "biop" in text:
            matched_groups.add("bronchoscopy_biopsy_parenchymal")
            # Navigation to peripheral lesion typically uses radial EBUS
            if "radial" in text or "peripheral" in text or "nodule" in text:
                matched_groups.add("bronchoscopy_ebus_radial")

        # ========== ADDITIONAL LOBE TBLB (+31632) - CONSERVATIVE DETECTION ==========
        # Per coding rules: Only emit +31632 when:
        # 1. Biopsies were obtained from TWO or MORE DISTINCT lobes
        # 2. The note NAMES both lobes in direct biopsy context
        # 3. Not just mentioning lobes in general (e.g., "right lung" doesn't count)

        # Define all lobes with their aliases
        lobe_definitions = {
            "rul": ["rul", "right upper lobe", "rml", "right middle lobe"],  # Count RUL/RML as distinct
            "rml": ["rml", "right middle lobe"],
            "rll": ["rll", "right lower lobe"],
            "lul": ["lul", "left upper lobe", "lingula"],
            "lll": ["lll", "left lower lobe"],
        }

        # Look for lobes mentioned in BIOPSY context
        # Must have biopsy-related words near lobe mentions
        biopsy_terms = [
            "biopsy", "biopsies", "biopsied", "sampled", "sampling", "tblb",
            "transbronchial", "cryobiopsy", "forceps", "samples obtained"
        ]

        lobes_biopsied: Set[str] = set()
        for lobe_key, aliases in lobe_definitions.items():
            for alias in aliases:
                # Check if lobe is mentioned near biopsy context
                lobe_pattern = re.compile(
                    rf"\b{re.escape(alias)}\b.*?(?:{'|'.join(biopsy_terms)})|"
                    rf"(?:{'|'.join(biopsy_terms)}).*?\b{re.escape(alias)}\b",
                    re.IGNORECASE
                )
                if lobe_pattern.search(text):
                    lobes_biopsied.add(lobe_key)
                    break

        # Also check for explicit multi-lobe patterns
        explicit_multilobe = any(
            m in text for m in [
                "bilateral", "both lobes", "multiple lobes", "two lobes",
                "rul and lll", "rll and lul", "rll and lll", "rul and rll"
            ]
        )

        evidence["bronchoscopy_biopsy_additional_lobe"] = {
            "lobes_biopsied": list(lobes_biopsied),
            "lobe_count": len(lobes_biopsied),
            "explicit_multilobe": explicit_multilobe,
        }

        # Only add additional lobe code if we have strong evidence of 2+ distinct lobes biopsied
        if (len(lobes_biopsied) >= 2 or explicit_multilobe) and "bronchoscopy_biopsy_parenchymal" in matched_groups:
            matched_groups.add("bronchoscopy_biopsy_parenchymal_additional")

        # ========== Therapeutic Procedures ==========
        # Ablation/destruction (31641)
        ablation_terms = self._synonym_terms("ablation_terms") or canonical_rules.ABLATION_SYNONYMS
        ablation_hit = self._contains_any(text, ablation_terms)
        if ablation_hit:
            matched_groups.add("bronchoscopy_airway_stenosis")

        # ========== STENT PLACEMENT - CONSERVATIVE DETECTION ==========
        # Per coding rules: Only emit stent codes when:
        # 1. Explicit stent language in PROCEDURE BODY (not just indications/history)
        # 2. Action verb indicating actual deployment (placed, deployed, inserted, anchored)
        # 3. For 31636 vs 31631: anatomic location must be specified
        # 4. For +31637: multiple separate bronchi must be documented
        # 5. For 31638 (revision): must have pre-existing stent + revision action
        # 6. NEGATIVE check: "no stent" or "stent not placed" means NO STENT

        # First check for NEGATIVE stent statements
        stent_negation_patterns = [
            "no stent was placed", "no stent placed", "stent not placed",
            "without stent", "no stent at this time", "possible future stent",
            "evaluate for possible", "possible stent placement", "plan for stent",
            "schedule for", "return for possible stent", "possible future"
        ]
        stent_negated = any(n in text for n in stent_negation_patterns)

        stent_type_terms = [
            "silicone stent", "dumon stent", "dumon", "metallic stent", "sems",
            "y-stent", "y stent", "covered stent", "aero stent", "aero",
            "self-expanding", "bronchial stent", "tracheal stent"
        ]
        stent_generic = "stent" in text
        stent_specific = any(t in text for t in stent_type_terms)
        stent_word = stent_generic or stent_specific

        # Action verbs for PLACEMENT (not revision) - must be specific
        stent_placement_actions = [
            "stent placed", "stent was placed", "stent deployed", "stent was deployed",
            "stent inserted", "stent was inserted", "stent positioned", "stent anchored",
            "deployed a stent", "deployed the stent", "placed a stent",
            "placement of stent", "stent deployment", "deploying the stent"
        ]
        stent_placement_action = any(a in text for a in stent_placement_actions) or bool(
            re.search(
                r"\bstent\b[^\n]{0,120}\b(?:deployed|placed|inserted|positioned|anchored)\b",
                text,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:deployed|placed|inserted|positioned|anchored)\b[^\n]{0,120}\bstent\b",
                text,
                re.IGNORECASE,
            )
        )

        # Revision verbs for 31638
        stent_revision_actions = [
            "stent revised", "stent repositioned", "stent exchanged", "stent removed and replaced",
            "stent replacement", "stent revision", "repositioned the stent", "exchanged the stent",
            "revision of stent", "revised stent", "existing stent", "pre-existing stent"
        ]
        stent_revision_action = any(a in text for a in stent_revision_actions)
        has_preexisting_stent = any(
            p in text for p in [
                "prior stent", "existing stent", "pre-existing stent", "indwelling stent",
                "stent surveillance", "previous stent", "old stent"
            ]
        )

        # Anatomic location for 31631 (tracheal) vs 31636 (bronchial)
        tracheal_location = any(loc in text for loc in ["trachea", "tracheal", "carina", "carinal"])
        bronchial_location = any(
            loc in text for loc in [
                "mainstem", "main stem", "bronchus", "bronchi", "lms", "rms",
                "left mainstem", "right mainstem", "bronchus intermedius", "lobar"
            ]
        )

        # Multiple bronchi for +31637 (requires evidence of >1 bronchial stent)
        multiple_stent_evidence = any(
            m in text for m in [
                "two stents", "2 stents", "bilateral stents", "multiple stents",
                "second stent", "another stent", "additional stent",
                "both mainstem", "both bronchi", "right and left"
            ]
        )

        # Build evidence dict for stent detection
        evidence["bronchoscopy_therapeutic_stent"] = {
            "stent_word": stent_word,
            "stent_specific": stent_specific,
            "placement_action": stent_placement_action,
            "revision_action": stent_revision_action,
            "has_preexisting": has_preexisting_stent,
            "tracheal_location": tracheal_location,
            "bronchial_location": bronchial_location,
            "multiple_stents": multiple_stent_evidence,
            "stent_negated": stent_negated,
        }

        # Only add stent group if we have STRONG evidence AND stent is NOT negated
        # Stent placement: stent word + placement action + (tracheal or bronchial location) + NOT negated
        if stent_word and stent_placement_action and (tracheal_location or bronchial_location) and not stent_negated:
            matched_groups.add("bronchoscopy_therapeutic_stent")

        # Stent revision: explicit revision action + pre-existing stent + NOT negated
        if stent_word and stent_revision_action and has_preexisting_stent and not stent_negated:
            matched_groups.add("bronchoscopy_stent_revision")

        # Dilation
        dilation_terms = self._synonym_terms("dilation_terms") or canonical_rules.DILATION_SYNONYMS
        dilation_hit = self._contains_any(text, dilation_terms)
        if dilation_hit:
            matched_groups.add("bronchoscopy_airway_dilation")

        # Therapeutic aspiration
        aspiration_terms = self._synonym_terms("aspiration_terms") or canonical_rules.THERAPEUTIC_ASPIRATION_SYNONYMS
        aspiration_hit = self._contains_any(text, aspiration_terms)
        if aspiration_hit:
            matched_groups.add("bronchoscopy_therapeutic_aspiration")

        # Foreign body removal (including valve retrieval)
        foreign_body_terms = self._synonym_terms("foreign_body_terms") or canonical_rules.FOREIGN_BODY_SYNONYMS
        fb_hit = self._contains_any(text, foreign_body_terms)
        if fb_hit:
            matched_groups.add("bronchoscopy_foreign_body")

        # ========== Pleural Procedures ==========
        # ========== IPC (32550 insertion / 32552 removal) - CONSERVATIVE DETECTION ==========
        # Per coding rules:
        # For 32550 (insertion): Must see tunneled catheter placement language
        #   - "PleurX", "tunneled pleural catheter", "cuff positioned", "tunnel created"
        # For 32552 (removal): Must see explicit removal verbs
        #   - "removed", "taken out", "catheter explanted", "IPC discontinued"
        # Rule: Never bill both 32550 and 32552 for same catheter in same session

        ipc_terms = [
            "pleurx", "tunneled pleural catheter", "indwelling pleural catheter",
            "ipc ", "ipc,", "ipc.", "tunneled catheter"
        ]
        ipc_mentioned = any(t in text for t in ipc_terms)

        # IPC INSERTION evidence - specific to insertion NOT removal
        ipc_insertion_actions = [
            "catheter placed", "catheter was placed", "pleurx placed", "pleurx was placed",
            "ipc placed", "ipc was placed", "ipc placement", "catheter placement",
            "tunneling device", "cuff positioned", "tunnel created", "tunneled into",
            "inserted a pleurx", "insertion of pleurx", "catheter inserted",
            "placed a pleurx", "pleurx catheter placed", "pleurx catheter was placed"
        ]
        ipc_insertion_action = any(a in text for a in ipc_insertion_actions)

        # IPC REMOVAL evidence - MUST be very specific to avoid false positives
        # Words like "drained" should NOT trigger removal
        ipc_removal_actions = [
            "pleurx removed", "pleurx was removed",
            "ipc removed", "ipc was removed", "catheter explanted", "ipc discontinued",
            "removal of ipc", "removal of pleurx",
            "removed the ipc", "removed the pleurx", "pleurx catheter removed"
        ]
        # Also check for absence of negative patterns
        ipc_not_removed = any(
            n in text for n in [
                "drained", "fluid was drained", "fluid drained", "successful", "functioning"
            ]
        ) and not any(a in text for a in ipc_removal_actions)

        ipc_removal_action = any(a in text for a in ipc_removal_actions)

        evidence["tunneled_pleural_catheter"] = {
            "ipc_mentioned": ipc_mentioned,
            "insertion_action": ipc_insertion_action,
            "removal_action": ipc_removal_action,
        }

        # Conservative: Only add if clear action verb present for insertion
        # For removal, ONLY detect explicit removal language
        if ipc_mentioned and ipc_insertion_action:
            matched_groups.add("tunneled_pleural_catheter_placement")
        if ipc_mentioned and ipc_removal_action:
            matched_groups.add("tunneled_pleural_catheter_removal")

        # Thoracentesis
        thoracentesis_terms = self._synonym_terms("thoracentesis_terms") or canonical_rules.THORACENTESIS_SYNONYMS
        thoracentesis_hit = self._contains_any(text, thoracentesis_terms)
        if thoracentesis_hit:
            matched_groups.add("thoracentesis")

        # ========== PLEURAL DRAINAGE (32556/32557) - CHEST TUBE DETECTION ==========
        # Per coding rules: Detect chest tube placement/insertion
        # 32556: Pleural drainage, percutaneous, without imaging guidance
        # 32557: Pleural drainage, percutaneous, with imaging guidance
        chest_tube_terms = [
            "chest tube", "thoracostomy tube", "pigtail catheter", "pigtail",
            "percutaneous drainage", "pleural drainage", "tube thoracostomy",
            "intercostal drain", "drain placed", "drain inserted"
        ]
        chest_tube_mentioned = any(t in text for t in chest_tube_terms)

        # Action verbs for chest tube placement
        chest_tube_actions = [
            "tube placed", "tube was placed", "drain placed", "drain was placed",
            "chest tube placed", "chest tube inserted", "tube inserted",
            "thoracostomy performed", "drainage catheter placed", "pigtail placed",
            "pigtail was placed", "pigtail catheter placed", "tube thoracostomy",
            "placed a chest tube", "insertion of chest tube", "placed for pneumothorax",
            "placed for effusion", "placed for drainage"
        ]
        chest_tube_action = any(a in text for a in chest_tube_actions)

        # Context for pneumothorax or pleural effusion that might need drainage
        drainage_context = any(
            c in text for c in [
                "pneumothorax", "pleural effusion", "hemothorax", "empyema",
                "fluid collection", "air leak", "drainage of"
            ]
        )

        evidence["pleural_drainage"] = {
            "chest_tube_mentioned": chest_tube_mentioned,
            "action": chest_tube_action,
            "drainage_context": drainage_context,
        }

        # Add pleural drainage group if chest tube action detected
        if chest_tube_mentioned and (chest_tube_action or drainage_context):
            matched_groups.add("pleural_drainage")

        # Pleurodesis
        pleurodesis_terms = self._synonym_terms("pleurodesis_terms") or canonical_rules.PLEURODESIS_SYNONYMS
        pleurodesis_hit = self._contains_any(text, pleurodesis_terms)
        if pleurodesis_hit:
            matched_groups.add("pleural_pleurodesis")

        # ========== THORACOSCOPY - CONSERVATIVE ANATOMIC DETECTION ==========
        # Per coding rules (ip_golden_knowledge_v2_2.json):
        # - 32601: Diagnostic thoracoscopy, NO biopsy (only when no biopsy performed)
        # - 32604: Pericardial sac with biopsy
        # - 32606: Mediastinal space with biopsy
        # - 32609: Pleura with biopsy
        # - 32602/32607/32608: Lung parenchyma
        #
        # RULES:
        # 1. Only ONE thoracoscopy code per hemithorax per session
        # 2. Biopsy codes trump diagnostic-only (32601)
        # 3. Select code based on anatomic site biopsied
        # 4. Must have explicit anatomic + biopsy language

        thoracoscopy_terms = self._synonym_terms("thoracoscopy_terms") or canonical_rules.THORACOSCOPY_SYNONYMS
        thoracoscopy_hit = self._contains_any(text, thoracoscopy_terms)

        # Detect biopsy language
        has_biopsy = any(
            b in text for b in [
                "biopsy", "biopsies", "biopsied", "sampled", "sampling",
                "forceps", "specimens", "tissue obtained", "tissue sent"
            ]
        )

        # Detect anatomic sites
        pleural_site_terms = self._synonym_terms("thoracoscopy_pleural_site_terms") or canonical_rules.THORACOSCOPY_PLEURAL_SYNONYMS
        pericardial_site_terms = self._synonym_terms("thoracoscopy_pericardial_site_terms") or canonical_rules.THORACOSCOPY_PERICARDIAL_SYNONYMS
        mediastinal_site_terms = self._synonym_terms("thoracoscopy_mediastinal_site_terms") or canonical_rules.THORACOSCOPY_MEDIASTINAL_SYNONYMS
        lung_site_terms = self._synonym_terms("thoracoscopy_lung_site_terms") or canonical_rules.THORACOSCOPY_LUNG_SYNONYMS

        pleural_site = self._contains_any(text, pleural_site_terms)
        pericardial_site = self._contains_any(text, pericardial_site_terms)
        mediastinal_site = self._contains_any(text, mediastinal_site_terms)
        lung_site = self._contains_any(text, lung_site_terms)

        # Detect drain handling (for bundling rule)
        bundled_drain_terms = self._synonym_terms("thoracoscopy_bundled_drain_terms") or canonical_rules.THORACOSCOPY_BUNDLED_DRAIN_TERMS
        separate_drain_terms = self._synonym_terms("thoracoscopy_separate_drain_terms") or canonical_rules.THORACOSCOPY_SEPARATE_DRAIN_TERMS
        temporary_drain = self._contains_any(text, bundled_drain_terms)
        permanent_drain = self._contains_any(text, separate_drain_terms)

        # Build evidence dict
        evidence["thoracoscopy"] = {
            "thoracoscopy_present": thoracoscopy_hit,
            "has_biopsy": has_biopsy,
            "pleural_site": pleural_site,
            "pericardial_site": pericardial_site,
            "mediastinal_site": mediastinal_site,
            "lung_site": lung_site,
            "temporary_drain_bundled": temporary_drain and not permanent_drain,
            "permanent_drain_separate": permanent_drain,
        }

        if thoracoscopy_hit:
            # If pleurodesis mentioned, use surgical thoracoscopy codes instead
            if pleurodesis_hit:
                matched_groups.add("thoracoscopy_surgical_pleurodesis_decortication")
            else:
                # SITE PRIORITY SELECTION - Only ONE thoracoscopy code per session
                # Per coding guidelines, when multiple sites are examined, select based on priority:
                # 1. Pericardial (32604) - highest RVU, most complex
                # 2. Mediastinal (32606) - next highest
                # 3. Lung (32607/32608) - parenchymal biopsies
                # 4. Pleural (32609) - most common
                # 5. Diagnostic only (32601) - no biopsy
                #
                # This ensures only ONE group is added, preventing multiple thoracoscopy codes

                selected_group = None

                if has_biopsy:
                    # Apply site priority - first match wins
                    if pericardial_site:
                        selected_group = "thoracoscopy_pericardial_biopsy"
                    elif mediastinal_site:
                        selected_group = "thoracoscopy_mediastinal_biopsy"
                    elif lung_site:
                        selected_group = "thoracoscopy_lung_biopsy"
                    elif pleural_site:
                        selected_group = "thoracoscopy_pleural_biopsy"
                    else:
                        # Biopsy mentioned but no specific site - default to pleural
                        selected_group = "thoracoscopy_pleural_biopsy"
                else:
                    # No biopsy - diagnostic only
                    selected_group = "thoracoscopy_diagnostic_only"

                if selected_group:
                    matched_groups.add(selected_group)

        # ========== BAL (31624) - CONSERVATIVE DETECTION ==========
        # Per coding rules: Only emit 31624 when:
        # 1. Explicit "bronchoalveolar lavage" or "BAL" mention
        # 2. Documentation of instilled and recovered volumes (e.g., "instilled 60cc, recovered 30cc")
        # 3. Context clearly refers to airway/lung (not pleural aspiration)
        # Generic "aspirated fluid/air" or "suctioned secretions" must NOT trigger 31624
        bal_explicit = any(
            term in text for term in [
                "bronchoalveolar lavage", "bal ", "bal,", "bal.", "bal\n",
                "alveolar lavage", "bal was performed", "bal performed"
            ]
        )
        # Check for volume documentation (strong evidence)
        bal_volume_pattern = re.search(
            r"(?:instilled|injected|delivered)\s*\d+\s*(?:ml|cc|milliliter)",
            text, re.IGNORECASE
        )
        bal_recovered_pattern = re.search(
            r"(?:recovered|returned|aspirated|retrieved)\s*\d+\s*(?:ml|cc|milliliter)",
            text, re.IGNORECASE
        )
        bal_has_volumes = bool(bal_volume_pattern or bal_recovered_pattern)

        # Make sure it's NOT just pleural aspiration
        pleural_context = any(
            p in text for p in ["pleural", "thoracentesis", "chest tube", "effusion"]
        )

        evidence["bronchoscopy_bal"] = {
            "bal_explicit": bal_explicit,
            "has_volumes": bal_has_volumes,
            "pleural_context": pleural_context,
        }

        # BAL: require explicit mention AND (volumes OR no pleural context)
        # Also check if it's bundled with TBLB in same lobe
        if bal_explicit and not pleural_context:
            # If BAL is in a different lobe from biopsy, it can be billed separately
            # Look for patterns like "BAL in RUL" or "lavage from RML"
            bal_lobe_match = re.search(
                r"bal[^\n]{0,80}(?:\b(?:rul|rml|rll|lul|lll)\b|"
                r"right\s+upper\s+lobe|right\s+middle\s+lobe|right\s+lower\s+lobe|"
                r"left\s+upper\s+lobe|left\s+lower\s+lobe|lingula)",
                text,
                re.IGNORECASE,
            )
            if bal_lobe_match or "bronchoscopy_biopsy_parenchymal" not in matched_groups:
                matched_groups.add("bronchoscopy_bal")

        # PDT
        pdt_terms = self._synonym_terms("pdt_terms") or canonical_rules.PDT_SYNONYMS
        pdt_hit = self._contains_any(text, pdt_terms)
        if pdt_hit:
            matched_groups.add("pdt_endobronchial")

        # Fallback: diagnostic bronchoscopy if bronchoscopy mentioned but no therapeutic
        if "bronchoscopy" in text and not matched_groups:
            matched_groups.add("bronchoscopy_diagnostic")

        self.last_group_evidence = evidence
        return matched_groups

    # Direct mapping for site-specific thoracoscopy groups to their single CPT code
    # This ensures site priority is respected even when going through codes_for_groups
    SITE_SPECIFIC_CODE_MAP: Dict[str, str] = {
        "thoracoscopy_diagnostic_only": "32601",
        "thoracoscopy_pleural_biopsy": "32609",
        "thoracoscopy_pericardial_biopsy": "32604",
        "thoracoscopy_mediastinal_biopsy": "32606",
        "thoracoscopy_lung_biopsy": "32607",
        "thoracoscopy_surgical_pleurodesis_decortication": "32650",
    }

    def codes_for_groups(self, groups: Iterable[str]) -> Set[str]:
        """
        Return CPT codes for the provided groups, preserving '+' prefixes for add-ons.
        Uses GROUP_ALIAS_MAP to translate internal group names to v2.7 code_lists keys.

        Site-specific thoracoscopy groups return ONLY their specific code to enforce
        the "one thoracoscopy code per session" rule.
        """
        codes: Set[str] = set()
        lists = self.raw.get("code_lists", {})
        for g in groups:
            # First check if it's a site-specific thoracoscopy group
            if g in self.SITE_SPECIFIC_CODE_MAP:
                codes.add(self.SITE_SPECIFIC_CODE_MAP[g])
            # Then try direct lookup in code_lists
            elif g in lists:
                for code in lists[g]:
                    if code.startswith("+"):
                        codes.add(code)
                    else:
                        codes.add(self._normalize_code(code))
            # Finally try alias mapping
            elif g in self.GROUP_ALIAS_MAP:
                for alias in self.GROUP_ALIAS_MAP[g]:
                    for code in lists.get(alias, []):
                        if code.startswith("+"):
                            codes.add(code)
                        else:
                            codes.add(self._normalize_code(code))
        return codes

    def codes_from_text(self, note_text: str) -> Set[str]:
        """
        Return candidate CPT codes based on synonym matches and code_lists.
        Preserves the '+' prefix for add-on codes.
        """
        groups = self.groups_from_text(note_text)
        return self.codes_for_groups(groups)
