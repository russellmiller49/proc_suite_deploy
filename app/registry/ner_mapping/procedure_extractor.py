"""Procedure extraction from NER entities.

Maps NER entities to procedure boolean flags in RegistryRecord.

The granular NER model may emit procedure *devices* (e.g., `DEV_STENT`) instead
of `PROC_METHOD`/`PROC_ACTION` spans; those device entities must still drive the
corresponding clinical performed flags to avoid falling back to regex uplift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from app.ner.inference import NEREntity, NERExtractionResult


@dataclass
class ProcedureExtractionResult:
    """Result from procedure extraction."""

    procedure_flags: Dict[str, bool]
    """Map of procedure name to performed flag."""

    evidence: Dict[str, List[str]]
    """Map of procedure name to evidence texts."""

    procedure_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Optional per-procedure attributes derived from entity context."""

    warnings: List[str] = field(default_factory=list)


# Mapping from keyword patterns to procedure field names
PROCEDURE_MAPPINGS: Dict[str, Tuple[Set[str], str]] = {
    # NOTE: These keys are used as the procedure_key allowlist for reporter LLM findings.
    # Keep keys aligned with RegistryRecord field names (or provide explicit field paths).

    # EBUS procedures
    "linear_ebus": (
        {"ebus", "convex ebus", "linear ebus", "endobronchial ultrasound"},
        "procedures_performed.linear_ebus.performed",
    ),
    "radial_ebus": (
        {"radial ebus", "radial endobronchial ultrasound", "rebus", "miniprobe", "radial probe"},
        "procedures_performed.radial_ebus.performed",
    ),

    # Navigation
    "navigational_bronchoscopy": (
        {
            "navigation",
            "enb",
            "superdimension",
            "ion",
            "monarch",
            "electromagnetic navigation",
            "robotic bronchoscopy",
            "robotic-assisted bronchoscopy",
            "robotic assisted bronchoscopy",
        },
        "procedures_performed.navigational_bronchoscopy.performed",
    ),

    # Biopsies
    "transbronchial_biopsy": (
        {"transbronchial biopsy", "tbbx", "tblb", "transbronchial lung biopsy"},
        "procedures_performed.transbronchial_biopsy.performed",
    ),
    "transbronchial_cryobiopsy": (
        {"cryobiopsy", "cryo biopsy", "transbronchial cryobiopsy", "cryo tbbx"},
        "procedures_performed.transbronchial_cryobiopsy.performed",
    ),
    "endobronchial_biopsy": (
        {"endobronchial biopsy", "ebbx", "bronchial biopsy"},
        "procedures_performed.endobronchial_biopsy.performed",
    ),

    # Sampling procedures
    "bal": (
        {"bal", "bronchoalveolar lavage", "lavage"},
        "procedures_performed.bal.performed",
    ),
    "brushings": (
        {"brushing", "brushings", "brush", "cytology brush", "bronchial brushing", "bronchial brush"},
        "procedures_performed.brushings.performed",
    ),
    "tbna_conventional": (
        {"tbna", "transbronchial needle aspiration", "transbronchial needle"},
        "procedures_performed.tbna_conventional.performed",
    ),

    # Therapeutic procedures
    "therapeutic_aspiration": (
        {"therapeutic aspiration", "suctioning", "mucus plug"},
        "procedures_performed.therapeutic_aspiration.performed",
    ),
    "airway_dilation": (
        {"dilation", "balloon dilation", "bougie", "airway dilation"},
        "procedures_performed.airway_dilation.performed",
    ),
    "airway_stent": (
        {"stent", "dumon", "y-stent", "silicone stent", "metal stent"},
        "procedures_performed.airway_stent.performed",
    ),

    # Balloon occlusion / endobronchial blockers (often Chartis/BLVR or bleeding control)
    "balloon_occlusion": (
        {
            "balloon occlusion",
            "serial occlusion",
            "endobronchial blocker",
            "bronchial blocker",
            "balloon blocker",
            "uniblocker",
            "arndt",
            "ardnt",
            "fogarty",
            "cohen flexitip",
        },
        "procedures_performed.balloon_occlusion.performed",
    ),

    # Ablation procedures
    "thermal_ablation": (
        {"apc", "argon plasma", "electrocautery", "laser", "thermal ablation"},
        "procedures_performed.thermal_ablation.performed",
    ),
    "cryotherapy": (
        {"cryotherapy", "cryo ablation", "cryo debulking"},
        "procedures_performed.cryotherapy.performed",
    ),
    "tumor_debulking": (
        {"debulking", "tumor debulking", "mechanical debridement"},
        "procedures_performed.mechanical_debulking.performed",
    ),

    # BLVR
    "blvr": (
        {"blvr", "bronchoscopic lung volume reduction", "valve", "zephyr"},
        "procedures_performed.blvr.performed",
    ),

    # Pleural procedures
    "thoracentesis": (
        {"thoracentesis", "pleural tap"},
        "pleural_procedures.thoracentesis.performed",
    ),
    "chest_tube": (
        {"chest tube", "pigtail catheter", "tube thoracostomy"},
        "pleural_procedures.chest_tube.performed",
    ),
    "ipc": (
        {"ipc", "indwelling pleural catheter", "pleurx"},
        "pleural_procedures.ipc.performed",
    ),
    "medical_thoracoscopy": (
        {"thoracoscopy", "pleuroscopy", "medical thoracoscopy"},
        "pleural_procedures.medical_thoracoscopy.performed",
    ),
    "pleurodesis": (
        {"pleurodesis", "talc", "chemical pleurodesis"},
        "pleural_procedures.pleurodesis.performed",
    ),

    # Imaging
    "chest_ultrasound": (
        {"chest ultrasound", "thoracic ultrasound", "pleural ultrasound"},
        "procedures_performed.chest_ultrasound.performed",
    ),

    # Tracheostomy
    "percutaneous_tracheostomy": (
        {"tracheostomy", "percutaneous tracheostomy", "pdt"},
        "procedures_performed.percutaneous_tracheostomy.performed",
    ),

    # Other
    "rigid_bronchoscopy": (
        {"rigid bronchoscopy", "rigid scope"},
        "procedures_performed.rigid_bronchoscopy.performed",
    ),
    "foreign_body_removal": (
        {"foreign body", "foreign body removal", "retrieval"},
        "procedures_performed.foreign_body_removal.performed",
    ),
}


class ProcedureExtractor:
    """Extract procedure flags from NER entities."""

    def __init__(self) -> None:
        # Pre-compile patterns
        self._patterns = self._compile_patterns()
        self._stent_placement_re = re.compile(
            r"\b(?:place(?:d|ment)?|deploy(?:ed|ment)?|insert(?:ed|ion)?|implant(?:ed|ation)?)\b",
            re.IGNORECASE,
        )
        self._stent_removal_re = re.compile(
            r"\b(?:remove(?:d|al)?|retriev(?:e|ed|al)|extract(?:ed|ion)|exchange(?:d|s)?|replac(?:e|ed|ement))\b",
            re.IGNORECASE,
        )
        self._stent_revision_re = re.compile(
            r"\b(?:reposition(?:ed|ing)?|revision|revis(?:ed|ion)|adjust(?:ed|ment))\b",
            re.IGNORECASE,
        )
        self._stent_negated_placement_re = re.compile(
            r"\b(?:no|not|without|declined|refused)\b[^.\n]{0,40}\b(?:stent|stents)\b[^.\n]{0,40}"
            r"\b(?:place(?:d|ment)?|deploy(?:ed|ment)?|insert(?:ed|ion)?)\b",
            re.IGNORECASE,
        )
        self._stent_assessment_only_re = re.compile(
            r"\b(?:stent\s+in\s+place|stent\s+in\s+good\s+position|stent\s+well\s+positioned|"
            r"well\s+positioned\s+stent|known\s+stent|existing\s+stent|stent\s+surveillance|"
            r"inspection\s+of\s+stent|stent\s+patent|patent\s+stent)\b",
            re.IGNORECASE,
        )

    def field_path_for(self, proc_name: str) -> str | None:
        """Return the RegistryRecord field path for a procedure key."""
        pattern = self._patterns.get(proc_name)
        if not pattern:
            return None
        return pattern[1]

    def _keyword_hit(self, text_lower: str, needle: str) -> bool:
        """Return True if needle matches text.

        Short tokens (e.g., "ion", "enb", "bal") require word boundaries to
        avoid false positives from substrings like "aspiratION".
        """
        needle_lower = (needle or "").lower()
        if " " in needle_lower or len(needle_lower) >= 5:
            return needle_lower in text_lower
        return re.search(rf"\b{re.escape(needle_lower)}\b", text_lower) is not None

    def _compile_patterns(self) -> Dict[str, Tuple[List[str], str]]:
        """Convert keyword sets to sorted lists for consistent matching."""
        patterns = {}
        for proc_name, (keywords, field_path) in PROCEDURE_MAPPINGS.items():
            # Sort by length (longest first) for greedy matching
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            patterns[proc_name] = (sorted_keywords, field_path)
        return patterns

    @staticmethod
    def _stent_action_priority(action: str | None) -> int:
        if action == "Removal":
            return 3
        if action == "Revision":
            return 2
        if action == "Placement":
            return 1
        return 0

    def _detect_stent_action(self, *contexts: str) -> str | None:
        for context in contexts:
            text = (context or "").lower()
            if "stent" not in text:
                continue
            if self._stent_removal_re.search(text):
                return "Removal"
            if self._stent_revision_re.search(text):
                return "Revision"
            if self._stent_placement_re.search(text) and not self._stent_negated_placement_re.search(text):
                return "Placement"
        return None

    def _is_stent_assessment_only(self, context: str) -> bool:
        text = (context or "").lower()
        if "stent" not in text:
            return False
        return self._stent_assessment_only_re.search(text) is not None

    def extract(self, ner_result: NERExtractionResult) -> ProcedureExtractionResult:
        """
        Extract procedure flags from NER entities.

        Args:
            ner_result: NER extraction result with entities

        Returns:
            ProcedureExtractionResult with procedure flags
        """
        proc_methods = ner_result.entities_by_type.get("PROC_METHOD", [])
        proc_actions = ner_result.entities_by_type.get("PROC_ACTION", [])
        neg_stent_entities = ner_result.entities_by_type.get("NEG_STENT", [])
        ctx_stent_entities = ner_result.entities_by_type.get("CTX_STENT_PRESENT", [])
        device_hints = (
            ner_result.entities_by_type.get("DEV_STENT", [])
            + ner_result.entities_by_type.get("DEV_INSTRUMENT", [])
            + ner_result.entities_by_type.get("DEV_DEVICE", [])
        )

        # Combine method and action entities for procedure detection
        all_proc_entities = proc_methods + proc_actions + device_hints

        procedure_flags: Dict[str, bool] = {}
        evidence: Dict[str, List[str]] = {}
        procedure_attributes: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []
        stent_suppression_signals = bool(neg_stent_entities or ctx_stent_entities)
        stent_suppression_warned = False
        stent_assessment_warned = False

        raw_text = ner_result.raw_text or ""
        raw_lower = raw_text.lower()
        ebus_context_re = re.compile(
            r"\b(?:ebus|endobronchial\s+ultrasound|convex\s+probe|ebus[-\s]?tbna)\b",
            re.IGNORECASE,
        )
        peripheral_context_re = re.compile(
            r"\b(?:ion|robotic|navigation|navigational|target|lesion|nodule|mass|cone\s+beam|cbct|tool[- ]?in[- ]?lesion|extended\s+working\s+channel)\b",
            re.IGNORECASE,
        )
        stent_size_entities = ner_result.entities_by_type.get("DEV_STENT_SIZE", [])
        stent_material_entities = ner_result.entities_by_type.get("DEV_STENT_MATERIAL", [])
        stent_device_entities = ner_result.entities_by_type.get("DEV_STENT", [])
        stent_brand_entities = ner_result.entities_by_type.get("DEV_DEVICE", [])

        def _normalize_stent_type(raw: str | None) -> str | None:
            if not raw:
                return None
            t = raw.strip().lower()
            if not t:
                return None
            if re.search(r"\by-?\s*stent\b|\by\s+stent\b|\by-?shaped\b", t):
                return "Y-Stent"
            if "dumon" in t:
                return "Silicone - Dumon"
            if "novatech" in t:
                return "Silicone - Novatech"
            if "hood" in t:
                return "Silicone - Hood"
            if "hybrid" in t:
                return "Hybrid"
            if "partially" in t and "cover" in t:
                return "SEMS - Partially covered"
            if "cover" in t:
                return "SEMS - Covered"
            if "bare" in t or "uncovered" in t or "metal" in t or "metallic" in t:
                return "SEMS - Uncovered"
            return None

        def _select_stent_size() -> str | None:
            if not stent_size_entities:
                return None
            # Prefer the longest/most specific size span.
            best = max(stent_size_entities, key=lambda e: len((e.text or "")))
            return (best.text or "").strip() or None

        def _select_stent_brand() -> str | None:
            candidates = []
            generic = {"stent", "airway stent", "y-stent", "y stent"}
            for ent in stent_brand_entities + stent_device_entities:
                text = (ent.text or "").strip()
                if not text:
                    continue
                lower = text.lower()
                if lower in generic:
                    continue
                if "stent" in lower and "self" in lower and "expand" in lower:
                    continue
                if lower in {"self-expandable airway stent", "self expandable airway stent", "self-expanding airway stent", "self expanding airway stent"}:
                    continue
                if "stent" in lower and lower in generic:
                    continue
                candidates.append(ent)
            if not candidates:
                return None
            if stent_device_entities:
                # Prefer device mentions closest to a stent entity.
                def _distance(ent):
                    return min(abs(ent.start_char - s.start_char) for s in stent_device_entities)
                best = min(candidates, key=_distance)
            else:
                best = candidates[0]
            return (best.text or "").strip() or None

        for entity in all_proc_entities:
            text_lower = entity.text.lower()
            if raw_lower and entity.start_char is not None and entity.end_char is not None:
                window_start = max(0, entity.start_char - 100)
                window_end = min(len(raw_lower), entity.end_char + 100)
                entity_context = raw_lower[window_start:window_end]
            else:
                entity_context = ""
            radial_keywords = self._patterns.get("radial_ebus", ([], ""))[0]
            radial_hit = any(self._keyword_hit(text_lower, kw) for kw in radial_keywords)

            for proc_name, (keywords, field_path) in self._patterns.items():
                if proc_name == "linear_ebus" and radial_hit:
                    continue
                for keyword in keywords:
                    if self._keyword_hit(text_lower, keyword):
                        if proc_name == "tbna_conventional" and entity_context:
                            # Peripheral/lung TBNA (e.g., navigation/ION targets) should not be
                            # treated as nodal conventional TBNA.
                            if peripheral_context_re.search(entity_context):
                                procedure_flags["peripheral_tbna"] = True
                                evidence.setdefault("peripheral_tbna", []).append(entity.text)
                                break
                            # EBUS-TBNA is captured under linear_ebus; do not also set conventional TBNA.
                            if ebus_context_re.search(entity_context):
                                break
                        if proc_name == "airway_stent":
                            stent_action = self._detect_stent_action(entity_context, text_lower, raw_lower)
                            assessment_only = self._is_stent_assessment_only(entity_context or text_lower)
                            if stent_suppression_signals and not stent_action:
                                if not stent_suppression_warned:
                                    warnings.append(
                                        "Suppressed airway_stent from CTX_STENT_PRESENT/NEG_STENT without action cues."
                                    )
                                    stent_suppression_warned = True
                                break
                            if assessment_only and not stent_action:
                                if not stent_assessment_warned:
                                    warnings.append(
                                        "Suppressed airway_stent from assessment-only stent context."
                                    )
                                    stent_assessment_warned = True
                                break
                        # Found a match
                        procedure_flags[proc_name] = True

                        if proc_name not in evidence:
                            evidence[proc_name] = []
                        evidence[proc_name].append(entity.text)

                        if proc_name == "airway_stent":
                            stent_action = self._detect_stent_action(entity_context, text_lower, raw_lower)
                            if stent_action:
                                attrs = procedure_attributes.setdefault("airway_stent", {})
                                current_action = attrs.get("action")
                                if self._stent_action_priority(stent_action) >= self._stent_action_priority(
                                    str(current_action) if current_action else None
                                ):
                                    attrs["action"] = stent_action
                                attrs["airway_stent_removal"] = attrs.get("action") == "Removal"

                        break  # Only match first keyword per entity

        # Backfill airway stent attributes from NER device entities.
        if procedure_flags.get("airway_stent"):
            attrs = procedure_attributes.setdefault("airway_stent", {})
            if not attrs.get("device_size"):
                size_val = _select_stent_size()
                if size_val:
                    attrs["device_size"] = size_val
            if not attrs.get("stent_type"):
                stent_type_val = None
                for ent in stent_material_entities + stent_device_entities:
                    stent_type_val = _normalize_stent_type(ent.text)
                    if stent_type_val:
                        break
                if stent_type_val:
                    attrs["stent_type"] = stent_type_val
            if not attrs.get("stent_brand"):
                brand_val = _select_stent_brand()
                if brand_val:
                    attrs["stent_brand"] = brand_val

        return ProcedureExtractionResult(
            procedure_flags=procedure_flags,
            evidence=evidence,
            procedure_attributes=procedure_attributes,
            warnings=warnings,
        )

    def extract_valve_count(self, ner_result: NERExtractionResult) -> int:
        """
        Extract valve count from NER entities for BLVR codes.

        Args:
            ner_result: NER extraction result

        Returns:
            Number of valves mentioned (0 if not found)
        """
        # Look for DEV_VALVE entities
        valves = ner_result.entities_by_type.get("DEV_VALVE", [])

        # Also check MEAS_COUNT entities near valve mentions
        counts = ner_result.entities_by_type.get("MEAS_COUNT", [])

        # Simple heuristic: count unique valve mentions
        return len(valves)

    def extract_lobe_locations(self, ner_result: NERExtractionResult) -> List[str]:
        """
        Extract lobe locations for transbronchial biopsy codes.

        Args:
            ner_result: NER extraction result

        Returns:
            List of unique lobe codes (RUL, RML, etc.)
        """
        lung_locs = ner_result.entities_by_type.get("ANAT_LUNG_LOC", [])

        lobes = set()
        lobe_patterns = {
            "rul": "RUL",
            "rml": "RML",
            "rll": "RLL",
            "lul": "LUL",
            "lll": "LLL",
            "lingula": "Lingula",
            "right upper": "RUL",
            "right middle": "RML",
            "right lower": "RLL",
            "left upper": "LUL",
            "left lower": "LLL",
        }

        for entity in lung_locs:
            text_lower = entity.text.lower()
            for pattern, lobe_code in lobe_patterns.items():
                if pattern in text_lower:
                    lobes.add(lobe_code)
                    break

        return sorted(lobes)
