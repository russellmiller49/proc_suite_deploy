"""Rule-based coding engine.

This engine applies deterministic rules to identify CPT codes from procedure notes.
It wraps the CodingRulesEngine to provide the RuleResult format expected by
the HybridPolicy.

Architecture:
- Uses IPCodingKnowledgeBase for group/evidence extraction from text
- Delegates rule application to CodingRulesEngine (R001-R014)
- Supports python/json/shadow modes via CODING_RULES_MODE env var
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from app.domain.knowledge_base.repository import KnowledgeBaseRepository
from app.domain.coding_rules.coding_rules_engine import CodingRulesEngine
from app.domain.coding_rules.evidence_context import EvidenceContext
from observability.logging_config import get_logger

logger = get_logger("rule_engine")

if TYPE_CHECKING:  # pragma: no cover
    from app.autocode.ip_kb.ip_kb import IPCodingKnowledgeBase


@dataclass
class RuleCandidate:
    """A candidate code from rule-based detection."""

    code: str
    confidence: float
    rationale: str
    rule_path: str
    evidence_text: str = ""


@dataclass
class RuleEngineResult:
    """Result from the rule-based coding engine."""

    candidates: list[RuleCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def codes(self) -> list[str]:
        """Get list of codes."""
        return [c.code for c in self.candidates]

    @property
    def confidence(self) -> dict[str, float]:
        """Get code -> confidence mapping."""
        return {c.code: c.confidence for c in self.candidates}


class RuleEngine:
    """Rule-based CPT code detection engine.

    This engine:
    1. Uses IPCodingKnowledgeBase to extract groups and evidence from text
    2. Delegates rule application to CodingRulesEngine (R001-R014)
    3. Returns RuleEngineResult for the HybridPolicy

    The sophisticated rules (stent 4-gate, EBUS logic, navigation evidence, etc.)
    are all handled by CodingRulesEngine, not this class.
    """

    VERSION = "rule_engine_v2"

    def __init__(
        self,
        kb_repo: KnowledgeBaseRepository,
        rules_mode: Optional[str] = None,
        *,
        code_families_config: dict[str, object] | None = None,
        ncci_data: dict[str, object] | None = None,
        ip_kb: "IPCodingKnowledgeBase | None" = None,
    ):
        """Initialize the rule engine.

        Args:
            kb_repo: Knowledge base repository for CPT code info
            rules_mode: Rule engine mode ("python" | "json" | "shadow")
                       Defaults to CODING_RULES_MODE env var or "python"
        """
        self.kb_repo = kb_repo
        self.rules_mode = rules_mode or os.getenv("CODING_RULES_MODE", "python")
        self.code_families_config = code_families_config
        self.ncci_data = ncci_data

        # Initialize the IP KB for group/evidence extraction
        # This is the clinical knowledge base that detects procedures from text
        try:
            from app.autocode.ip_kb.ip_kb import IPCodingKnowledgeBase

            if ip_kb is not None and isinstance(ip_kb, IPCodingKnowledgeBase):
                self._ip_kb = ip_kb
            else:
                from config.settings import KnowledgeSettings

                self._ip_kb = IPCodingKnowledgeBase(KnowledgeSettings().kb_path)
            self._ip_kb_available = True
            logger.info(
                "IP Knowledge Base initialized",
                extra={"kb_version": getattr(self._ip_kb, "version", "unknown")},
            )
        except (ImportError, Exception) as e:
            logger.warning(f"IPCodingKnowledgeBase not available: {e}")
            self._ip_kb = None
            self._ip_kb_available = False

        # Initialize CodingRulesEngine for rule application (R001-R014)
        self._coding_rules_engine = CodingRulesEngine(mode=self.rules_mode)
        logger.info(
            "CodingRulesEngine initialized",
            extra={"mode": self.rules_mode},
        )

    @property
    def version(self) -> str:
        return self.VERSION

    def generate_candidates(
        self,
        report_text: str,
        registry: Optional[Dict[str, Any]] = None,
    ) -> RuleEngineResult:
        """Generate code candidates from the procedure note.

        This method:
        1. Extracts groups and evidence using IPCodingKnowledgeBase
        2. Gets initial candidate codes from the groups
        3. Builds EvidenceContext for rule evaluation
        4. Applies CodingRulesEngine rules (R001-R014)
        5. Returns RuleEngineResult for HybridPolicy

        Args:
            report_text: The procedure note text.
            registry: Optional registry form data (for registry-gated rules)

        Returns:
            RuleEngineResult with candidates and warnings.
        """
        if not self._ip_kb_available:
            logger.warning("IP KB not available, falling back to keyword matching")
            return self._fallback_keyword_matching(report_text)

        warnings: list[str] = []

        try:
            # Step 1: Extract groups and evidence from text using IP KB
            groups = self._ip_kb.groups_from_text(report_text)
            evidence = dict(self._ip_kb.last_group_evidence)

            # Step 2: Get initial candidate codes from groups
            candidates = self._get_candidates_from_groups(groups)

            # Step 3: Extract additional context
            term_hits = self._extract_term_hits(report_text)
            navigation_context = self._extract_navigation_context(registry or {})
            radial_context = self._extract_radial_context(registry or {})

            # Step 4: Build EvidenceContext
            context = EvidenceContext(
                groups=groups,
                evidence=evidence,
                registry=registry or {},
                candidates=candidates,
                term_hits=term_hits,
                navigation_context=navigation_context,
                radial_context=radial_context,
                text_lower=report_text.lower(),
            )

            # Step 5: Apply coding rules (R001-R014)
            valid_cpts = self.kb_repo.get_all_codes() if self.kb_repo else None
            rules_result = self._coding_rules_engine.apply_rules(context, valid_cpts)

            # Step 6: Convert to RuleEngineResult
            # Parse applied_rules to find rules that affected each code
            code_rules: Dict[str, list[str]] = {}
            for rule_entry in rules_result.applied_rules:
                # Format: "RULE_ID:action:code"
                parts = rule_entry.split(":")
                if len(parts) >= 3:
                    rule_id, action, affected_code = parts[0], parts[1], parts[2]
                    if action == "add":
                        if affected_code not in code_rules:
                            code_rules[affected_code] = []
                        code_rules[affected_code].append(rule_id)

            result_candidates = []
            for code in rules_result.codes:
                # Compute confidence based on evidence strength
                confidence = self._compute_confidence(code, groups, evidence)

                # Build rationale from applied rules
                applied_rules = code_rules.get(code, [])
                rationale = self._build_rationale(code, groups, applied_rules)

                result_candidates.append(
                    RuleCandidate(
                        code=code,
                        confidence=confidence,
                        rationale=rationale,
                        rule_path=",".join(applied_rules) if applied_rules else "group_detection",
                    )
                )

            # Add any warnings from rules
            warnings.extend(rules_result.warnings)

            # Optional NCCI bundling notes (merged KB + external config injected via DI).
            # We do not remove codes here; we only surface a warning so downstream
            # policy/validation layers can decide how to handle bundled codes.
            if self.ncci_data:
                try:
                    from app.coder.ncci import NCCIEngine

                    ncci_engine = NCCIEngine(ptp_cfg=self.ncci_data)
                    ncci_result = ncci_engine.apply(set(rules_result.codes))
                    for secondary, primary in sorted(ncci_result.bundled.items()):
                        warnings.append(f"NCCI_BUNDLE: {secondary} bundled into {primary}")
                except Exception as exc:
                    warnings.append(f"NCCI_BUNDLE_FAILED: {type(exc).__name__}")

            return RuleEngineResult(candidates=result_candidates, warnings=warnings)

        except Exception as e:
            logger.error(f"Error in rule engine: {e}", exc_info=True)
            warnings.append(f"Rule engine error: {str(e)}")
            return self._fallback_keyword_matching(report_text, warnings)

    def _get_candidates_from_groups(self, groups: Set[str]) -> Set[str]:
        """Get candidate codes from detected groups."""
        candidates: Set[str] = set()

        # Group to code mapping (from IP KB)
        # Both 31652 and 31653 are added for linear EBUS - R010 filters based on station count
        group_code_map = {
            "bronchoscopy_bal": {"31624"},
            "bronchoscopy_navigation": {"+31627"},
            "bronchoscopy_tbna": {"31629"},
            "bronchoscopy_tbbx": {"31628", "+31632"},  # Include add-on for multi-lobe
            "bronchoscopy_biopsy_parenchymal": {"31628"},  # TBLB
            "bronchoscopy_biopsy_parenchymal_additional": {"+31632"},  # Additional lobe
            "bronchoscopy_ebus_linear": {"31652", "31653"},  # R010 filters by station count
            "bronchoscopy_ebus_linear_additional": {"31653"},  # 3+ stations
            "bronchoscopy_ebus_radial": {"+31654"},
            "bronchoscopy_therapeutic_stent": {"31631", "31636", "31637", "31638"},  # Include all stent codes
            "bronchoscopy_stent_revision": {"31638"},  # Stent revision
            "bronchoscopy_therapeutic_debulking": {"31640", "31641"},
            "bronchoscopy_airway_stenosis": {"31641"},  # Ablation/destruction
            "bronchoscopy_therapeutic_aspiration": {"31645", "31646"},
            "bronchoscopy_airway_dilation": {"31630"},
            "bronchoscopy_dilation": {"31630"},
            "bronchoscopy_foreign_body": {"31635"},
            "pleural_drainage": {"32556", "32557"},
            "tunneled_pleural_catheter_placement": {"32550"},
            "tunneled_pleural_catheter_removal": {"32552"},
            "ipc_insertion": {"32550"},
            "ipc_removal": {"32552"},
            "thoracentesis": {"32554", "32555"},
            # Thoracoscopy - site-specific groups map to single codes
            "thoracoscopy_diagnostic_only": {"32601"},
            "thoracoscopy_pleural_biopsy": {"32609"},
            "thoracoscopy_pericardial_biopsy": {"32604"},
            "thoracoscopy_mediastinal_biopsy": {"32606"},
            "thoracoscopy_lung_biopsy": {"32607"},
            "thoracoscopy_surgical_pleurodesis_decortication": {"32650"},
            # Generic thoracoscopy (fallback)
            "thoracoscopy": {"32601"},
        }

        for group in groups:
            if group in group_code_map:
                candidates.update(group_code_map[group])

        return candidates

    def _extract_term_hits(self, report_text: str) -> Dict[str, list]:
        """Extract term hits for rule evaluation."""
        text_lower = report_text.lower()
        term_hits: Dict[str, list] = {}

        # Procedure categories
        categories = []
        if any(kw in text_lower for kw in ["bronchoscopy", "bronch"]):
            categories.append("bronchoscopy")
        if any(kw in text_lower for kw in ["thoracoscopy", "pleuroscopy", "vats"]):
            categories.append("thoracoscopy")
        if any(kw in text_lower for kw in ["pleural", "chest tube", "thoracentesis"]):
            categories.append("pleural")

        term_hits["procedure_categories"] = categories
        return term_hits

    def _extract_navigation_context(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract navigation-specific context from registry."""
        return {
            "nav_platform": registry.get("nav_platform"),
            "nav_tool_in_lesion": registry.get("nav_tool_in_lesion", False),
            "nav_concept_reached": registry.get("nav_concept_reached", False),
            "nav_direct_visualization": registry.get("nav_direct_visualization", False),
        }

    def _extract_radial_context(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract radial EBUS context from registry."""
        return {
            "radial_target_visualized": registry.get("radial_target_visualized", False),
            "radial_biopsy_performed": registry.get("radial_biopsy_performed", False),
        }

    def _compute_confidence(
        self,
        code: str,
        groups: Set[str],
        evidence: Dict[str, Any],
    ) -> float:
        """Compute confidence score based on evidence strength."""
        # Base confidence from group detection
        base_confidence = 0.7

        # Boost for explicit evidence
        if code == "31624" and evidence.get("bronchoscopy_bal", {}).get("bal_explicit"):
            return 0.95

        if code in ("31652", "31653"):
            ebus_ev = evidence.get("bronchoscopy_ebus_linear", {})
            if ebus_ev.get("station_count", 0) >= 1:
                return 0.9

        if code in ("31631", "31636"):
            stent_ev = evidence.get("bronchoscopy_therapeutic_stent", {})
            if stent_ev.get("stent_word") and stent_ev.get("placement_action"):
                return 0.9

        # Lower confidence for add-on codes
        if code.startswith("+"):
            return base_confidence - 0.1

        return base_confidence

    def _build_rationale(
        self,
        code: str,
        groups: Set[str],
        applied_rules: list[str],
    ) -> str:
        """Build human-readable rationale for the code."""
        parts = []

        # Group detection
        relevant_groups = [g for g in groups if self._group_relates_to_code(g, code)]
        if relevant_groups:
            parts.append(f"Groups detected: {', '.join(relevant_groups)}")

        # Applied rules
        if applied_rules:
            parts.append(f"Rules: {', '.join(applied_rules)}")

        return "; ".join(parts) if parts else "Rule-based detection"

    def _group_relates_to_code(self, group: str, code: str) -> bool:
        """Check if a group is related to a code."""
        code_base = code.lstrip("+")

        group_code_relations = {
            "bronchoscopy_bal": ["31624"],
            "bronchoscopy_navigation": ["31627"],
            "bronchoscopy_ebus_linear": ["31652", "31653"],
            "bronchoscopy_ebus_linear_additional": ["31653"],
            "bronchoscopy_ebus_radial": ["31654"],
            "bronchoscopy_tbbx": ["31628", "31632"],
            "bronchoscopy_biopsy_parenchymal": ["31628"],
            "bronchoscopy_biopsy_parenchymal_additional": ["31632"],
            "bronchoscopy_therapeutic_stent": ["31631", "31636", "31637", "31638"],
            "bronchoscopy_stent_revision": ["31638"],
            "bronchoscopy_therapeutic_debulking": ["31640", "31641"],
            "bronchoscopy_airway_stenosis": ["31641"],
            "bronchoscopy_therapeutic_aspiration": ["31645", "31646"],
            "bronchoscopy_airway_dilation": ["31630"],
            "bronchoscopy_dilation": ["31630"],
            "bronchoscopy_foreign_body": ["31635"],
            "pleural_drainage": ["32556", "32557"],
            "tunneled_pleural_catheter_placement": ["32550"],
            "tunneled_pleural_catheter_removal": ["32552"],
            "thoracentesis": ["32554", "32555"],
            "thoracoscopy_diagnostic_only": ["32601"],
            "thoracoscopy_pleural_biopsy": ["32609"],
            "thoracoscopy_pericardial_biopsy": ["32604"],
            "thoracoscopy_mediastinal_biopsy": ["32606"],
            "thoracoscopy_lung_biopsy": ["32607"],
            "thoracoscopy_surgical_pleurodesis_decortication": ["32650"],
        }

        return code_base in group_code_relations.get(group, [])

    def _fallback_keyword_matching(
        self,
        report_text: str,
        existing_warnings: Optional[list[str]] = None,
    ) -> RuleEngineResult:
        """Fallback to simple keyword matching when IP KB unavailable.

        This is a simplified version that doesn't use the full rules engine.
        It's only used when the IP Knowledge Base fails to load.
        """
        candidates: list[RuleCandidate] = []
        warnings = existing_warnings or []
        text_lower = report_text.lower()

        # Navigation
        if any(kw in text_lower for kw in ["navigation", "enb", "superdimension", "ion", "monarch"]):
            candidates.append(
                RuleCandidate(
                    code="31627",
                    confidence=0.7,
                    rationale="Navigation keyword detected (fallback mode)",
                    rule_path="fallback_navigation",
                )
            )

        # EBUS-TBNA
        if "ebus" in text_lower or "endobronchial ultrasound" in text_lower:
            station_count = self._count_ebus_stations(text_lower)
            if station_count >= 3:
                candidates.append(
                    RuleCandidate(
                        code="31653",
                        confidence=0.7,
                        rationale=f"EBUS with {station_count} stations (fallback mode)",
                        rule_path="fallback_ebus_3plus",
                    )
                )
            elif station_count >= 1:
                candidates.append(
                    RuleCandidate(
                        code="31652",
                        confidence=0.7,
                        rationale=f"EBUS with {station_count} stations (fallback mode)",
                        rule_path="fallback_ebus_1-2",
                    )
                )

        # BAL
        if any(kw in text_lower for kw in ["bronchoalveolar lavage", " bal ", "bal performed"]):
            candidates.append(
                RuleCandidate(
                    code="31624",
                    confidence=0.75,
                    rationale="BAL keyword detected (fallback mode)",
                    rule_path="fallback_bal",
                )
            )

        # Transbronchial biopsy
        if any(kw in text_lower for kw in ["transbronchial biopsy", "tblb", "tbbx"]):
            candidates.append(
                RuleCandidate(
                    code="31628",
                    confidence=0.75,
                    rationale="TBLB keyword detected (fallback mode)",
                    rule_path="fallback_tblb",
                )
            )

        # Stent placement
        if any(kw in text_lower for kw in ["stent placed", "stent deployed", "stent insertion"]):
            if "trachea" in text_lower:
                candidates.append(
                    RuleCandidate(
                        code="31631",
                        confidence=0.7,
                        rationale="Tracheal stent keyword detected (fallback mode)",
                        rule_path="fallback_tracheal_stent",
                    )
                )
            else:
                candidates.append(
                    RuleCandidate(
                        code="31636",
                        confidence=0.7,
                        rationale="Bronchial stent keyword detected (fallback mode)",
                        rule_path="fallback_bronchial_stent",
                    )
                )

        # Tumor destruction
        if any(kw in text_lower for kw in [
            "tumor destruction", "debulking", "cryotherapy", "laser", "electrocautery",
            "ablation", "apc", "argon plasma"
        ]):
            candidates.append(
                RuleCandidate(
                    code="31641",
                    confidence=0.7,
                    rationale="Tumor destruction keyword detected (fallback mode)",
                    rule_path="fallback_destruction",
                )
            )

        warnings.append("Using fallback keyword matching - IP KB unavailable")
        return RuleEngineResult(candidates=candidates, warnings=warnings)

    def _count_ebus_stations(self, text_lower: str) -> int:
        """Count the number of EBUS stations mentioned."""
        # Common lymph node station patterns
        station_patterns = [
            r"\b4[rl]\b",  # 4R, 4L
            r"\b7\b",  # Station 7
            r"\b10[rl]\b",  # 10R, 10L
            r"\b11[rl]\b",  # 11R, 11L
            r"\bsubcarinal\b",
            r"\bparatracheal\b",
            r"\bhilar\b",
        ]

        stations_found = set()
        for pattern in station_patterns:
            if re.search(pattern, text_lower):
                stations_found.add(pattern)

        # Also check for explicit station counts
        count_match = re.search(r"(\d+)\s*stations?\b", text_lower)
        if count_match:
            return int(count_match.group(1))

        return len(stations_found)
