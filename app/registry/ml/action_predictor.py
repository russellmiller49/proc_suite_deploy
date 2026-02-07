"""ActionPredictor: Atomic clinical action extraction for extraction-first architecture.

This module is the cornerstone of the extraction-first pivot. It extracts
structured clinical actions from procedure notes, serving as the source
of truth for deterministic CPT code derivation.

Architecture:
    Text → ActionPredictor.predict() → ClinicalActions → RegistryBasedCoder → CPT Codes

Extraction Strategy (Hybrid):
    1. Deterministic extractors (high precision, regex-based)
    2. ML predictions for coverage validation
    3. Optional LLM for complex/ambiguous cases

Usage:
    predictor = ActionPredictor()
    result = predictor.predict(note_text)

    # Structured actions for CPT derivation
    if result.actions.ebus.performed:
        stations = result.actions.ebus.stations
        # → Derive 31652 or 31653 based on station count

    # Evidence for auditing
    ebus_evidence = result.field_extractions.get("ebus.stations")
    if ebus_evidence:
        print(f"Evidence: {[s.text for s in ebus_evidence.evidence]}")
"""

from __future__ import annotations

import re
from typing import Sequence

from app.common.logger import get_logger
from app.common.sectionizer import Section, SectionizerService
from app.common.spans import Span

from app.registry.slots.base import SlotResult, section_for_offset
from app.registry.slots.ebus import EbusExtractor
from app.registry.slots.tblb import TBLBExtractor
from app.registry.slots.pleura import PleuraExtractor
from app.registry.slots.sedation import SedationExtractor
from app.registry.slots.stent import StentExtractor

# Note: ComplicationsExtractor has a broken import (COMPLICATIONS not in schema.py)
# We implement a simple inline version here instead

from .models import (
    ActionResult,
    ClinicalActions,
    EBUSActions,
    BiopsyActions,
    NavigationActions,
    BALActions,
    BrushingsActions,
    BronchialWashActions,
    PleuralActions,
    CAOActions,
    StentActions,
    BLVRActions,
    TherapeuticActions,
    ComplicationActions,
    SedationActions,
    PredictionResult,
)


logger = get_logger("registry.ml.action_predictor")


# =============================================================================
# Additional Regex Patterns for Actions Not Covered by Existing Slot Extractors
# =============================================================================

# BAL patterns
BAL_PATTERNS = [
    re.compile(r"bronchoalveolar\s+lavage", re.IGNORECASE),
    re.compile(r"\bBAL\b"),
    re.compile(r"lavage\s+(?:was\s+)?(?:performed|obtained|sent)", re.IGNORECASE),
]

# Brushings patterns
BRUSHINGS_PATTERNS = [
    re.compile(r"brush(?:ing)?s?\s+(?:was|were)?\s*(?:performed|obtained|sent)", re.IGNORECASE),
    re.compile(r"bronchial\s+brush", re.IGNORECASE),
    re.compile(r"protected\s+(?:specimen\s+)?brush", re.IGNORECASE),
]

# Bronchial wash patterns
WASH_PATTERNS = [
    re.compile(r"bronchial\s+wash", re.IGNORECASE),
    re.compile(r"wash(?:ing)?s?\s+(?:was|were)?\s*(?:performed|obtained|sent)", re.IGNORECASE),
]

# Navigation platform patterns
NAV_PLATFORM_PATTERNS = {
    "superDimension": re.compile(r"super\s*dimension|superdimension", re.IGNORECASE),
    "Ion": re.compile(r"\bion\b(?:\s+robotic)?", re.IGNORECASE),
    "Monarch": re.compile(r"monarch", re.IGNORECASE),
    "Illumisite": re.compile(r"illumisite", re.IGNORECASE),
    "SPiN": re.compile(r"\bspin\b", re.IGNORECASE),
    "electromagnetic": re.compile(r"electromagnetic\s+navigation|EMN\b", re.IGNORECASE),
}

# Robotic navigation patterns
ROBOTIC_PATTERNS = [
    re.compile(r"robotic\s+(?:assisted\s+)?(?:bronchoscopy|navigation)", re.IGNORECASE),
    re.compile(r"\bion\b", re.IGNORECASE),
    re.compile(r"monarch", re.IGNORECASE),
]

# Cone beam CT patterns
CBCT_PATTERNS = [
    re.compile(r"cone\s*beam\s*CT", re.IGNORECASE),
    re.compile(r"\bCBCT\b"),
    re.compile(r"fluoroscopic\s+CT", re.IGNORECASE),
]

# BLVR patterns
BLVR_PATTERNS = [
    re.compile(r"valve(?:s)?\s+(?:was|were)?\s*(?:placed|inserted|deployed)", re.IGNORECASE),
    re.compile(r"endobronchial\s+valve", re.IGNORECASE),
    re.compile(r"zephyr\s+valve", re.IGNORECASE),
    re.compile(r"lung\s+volume\s+reduction", re.IGNORECASE),
    re.compile(r"\bBLVR\b"),
]

# Chartis patterns
CHARTIS_PATTERNS = [
    re.compile(r"chartis", re.IGNORECASE),
    re.compile(r"collateral\s+ventilation", re.IGNORECASE),
]

# CAO/therapeutic patterns
THERMAL_ABLATION_PATTERNS = [
    re.compile(r"APC|argon\s+plasma", re.IGNORECASE),
    re.compile(r"electrocautery", re.IGNORECASE),
    re.compile(r"laser\s+(?:ablation|therapy)", re.IGNORECASE),
    re.compile(r"thermal\s+ablation", re.IGNORECASE),
]

CRYO_PATTERNS = [
    re.compile(r"cryo(?:therapy|ablation|spray)", re.IGNORECASE),
]

DILATION_PATTERNS = [
    re.compile(r"balloon\s+(?:bronchoplasty|dilation)", re.IGNORECASE),
    re.compile(r"(?:airway|bronchial)\s+dilation", re.IGNORECASE),
]

# Foreign body patterns
FOREIGN_BODY_PATTERNS = [
    re.compile(r"foreign\s+body", re.IGNORECASE),
    re.compile(r"aspiration\s+of\s+(?:foreign|object)", re.IGNORECASE),
]

# Transbronchial biopsy patterns (supplements TBLBExtractor which requires lobe)
# NOTE: Be careful not to match endobronchial biopsy or EBUS-TBNA
TRANSBRONCHIAL_BIOPSY_PATTERNS = [
    re.compile(r"transbronchial\s+(?:lung\s+)?biops(?:y|ies)", re.IGNORECASE),
    re.compile(r"\bTBLB\b", re.IGNORECASE),
    re.compile(r"\bTBBx?\b", re.IGNORECASE),  # TBB or TBBx
    # Forceps biopsy in context of lung/lobe suggests transbronchial
    re.compile(r"(?:forceps|transbronchial)\s+biops(?:y|ies)\s+(?:of|from|at)\s+(?:the\s+)?(?:RUL|RML|RLL|LUL|LLL|lingula|upper|middle|lower|lung|peripheral)", re.IGNORECASE),
    # Specific patterns for parenchymal/lung biopsy
    re.compile(r"(?:lung|parenchymal|peripheral\s+lung)\s+biops(?:y|ies)", re.IGNORECASE),
    re.compile(r"biops(?:y|ies)\s+(?:of|from|at)\s+(?:the\s+)?(?:RUL|RML|RLL|LUL|LLL|lingula)", re.IGNORECASE),
    # Biopsy under fluoroscopic guidance (common for TBLB)
    re.compile(r"(?:fluoroscop(?:y|ic)|fluoro)\s+(?:guided\s+)?biops(?:y|ies)", re.IGNORECASE),
    re.compile(r"biops(?:y|ies)\s+(?:under|with)\s+(?:fluoroscop(?:y|ic)|fluoro)", re.IGNORECASE),
    # "tissue biopsy" from lung context
    re.compile(r"(?:tissue|lung\s+tissue)\s+biops(?:y|ies)(?:\s+(?:was|were))?\s+(?:performed|obtained|taken)", re.IGNORECASE),
    # Forceps biopsies (specific tool for TBLB, not needle)
    re.compile(r"forceps\s+biops(?:y|ies)", re.IGNORECASE),
]

# Endobronchial biopsy patterns - DISTINCT from transbronchial
ENDOBRONCHIAL_BIOPSY_PATTERNS = [
    re.compile(r"endobronchial\s+biops(?:y|ies)", re.IGNORECASE),
    re.compile(r"\bEBBx?\b"),  # EBB or EBBx
    re.compile(r"biops(?:y|ies)\s+(?:of|from)\s+(?:the\s+)?(?:mass|tumor|lesion|mucosa|airway)", re.IGNORECASE),
]

# Conventional TBNA patterns (non-EBUS guided needle aspiration)
CONVENTIONAL_TBNA_PATTERNS = [
    re.compile(r"(?:conventional|blind)\s+TBNA", re.IGNORECASE),
    re.compile(r"transbronchial\s+needle\s+aspiration", re.IGNORECASE),
    re.compile(r"\bTBNA\b(?!\s*-?\s*EBUS)", re.IGNORECASE),  # TBNA not followed by EBUS
    re.compile(r"Wang\s+needle", re.IGNORECASE),  # Wang needle is conventional TBNA
]

# Cryobiopsy patterns
CRYOBIOPSY_PATTERNS = [
    re.compile(r"cryo(?:biopsy|biopsies)", re.IGNORECASE),
    re.compile(r"transbronchial\s+cryobiopsy", re.IGNORECASE),
]

# IPC patterns
IPC_PATTERNS = [
    re.compile(r"tunneled\s+(?:pleural\s+)?catheter", re.IGNORECASE),
    re.compile(r"pleur\s*x", re.IGNORECASE),
    re.compile(r"indwelling\s+pleural\s+catheter", re.IGNORECASE),
    re.compile(r"\bIPC\b"),
]

# Thoracoscopy patterns
THORACOSCOPY_PATTERNS = [
    re.compile(r"(?:medical\s+)?thoracoscopy", re.IGNORECASE),
    re.compile(r"pleuroscopy", re.IGNORECASE),
]

# Radial EBUS patterns (distinct from linear EBUS)
RADIAL_EBUS_PATTERNS = [
    re.compile(r"radial\s+(?:probe\s+)?(?:endobronchial\s+ultrasound|EBUS)", re.IGNORECASE),
    re.compile(r"r-?EBUS\b", re.IGNORECASE),
    re.compile(r"radial\s+probe", re.IGNORECASE),
    re.compile(r"miniprobe", re.IGNORECASE),
    re.compile(r"UM-S20-17S", re.IGNORECASE),  # Olympus radial probe model
]

# Rigid bronchoscopy patterns
RIGID_BRONCH_PATTERNS = [
    re.compile(r"rigid\s+bronchoscop(?:y|e)", re.IGNORECASE),
    re.compile(r"rigid\s+scope", re.IGNORECASE),
    re.compile(r"under\s+general\s+anesthesia.*bronchoscop", re.IGNORECASE),
]

# =============================================================================
# Negation Handling Patterns
# =============================================================================

# Negation patterns that indicate procedure was NOT performed
NEGATION_PREFIX_PATTERN = re.compile(
    r"(?:not?\s+|without\s+|no\s+|denied\s+|declined\s+|deferred\s+)"
    r"(?:\w+\s+){0,3}",  # Allow up to 3 words between negation and procedure
    re.IGNORECASE,
)

NEGATION_SUFFIX_PATTERNS = [
    re.compile(r"(?:was|were)\s+not\s+(?:performed|done|obtained|attempted)", re.IGNORECASE),
    re.compile(r"not\s+(?:performed|done|obtained|attempted)", re.IGNORECASE),
    re.compile(r"(?:was|were)\s+(?:deferred|declined|avoided)", re.IGNORECASE),
]

# Context window for checking negation (characters before/after match)
NEGATION_WINDOW = 50

# Complication patterns (inline since ComplicationsExtractor has broken import)
COMPLICATION_PATTERNS = {
    "Bleeding": re.compile(r"\bbleeding\b", re.IGNORECASE),
    "Pneumothorax": re.compile(r"\bpneumothorax\b", re.IGNORECASE),
    "Hypoxia": re.compile(r"\bhypoxia\b|\bdesaturation\b", re.IGNORECASE),
    "Respiratory failure": re.compile(r"respiratory\s+failure", re.IGNORECASE),
    "Cardiac arrhythmia": re.compile(r"arrhythmia|a(?:trial\s+)?fib", re.IGNORECASE),
    "Laryngospasm": re.compile(r"laryngospasm", re.IGNORECASE),
    "Bronchospasm": re.compile(r"bronchospasm", re.IGNORECASE),
}

NO_COMPLICATION_PATTERN = re.compile(
    r"(?:complications?|adverse\s+events?)[\s:]*(?:none|no|nil)",
    re.IGNORECASE,
)


# =============================================================================
# ActionPredictor Class
# =============================================================================


class ActionPredictor:
    """Extracts atomic clinical actions from procedure notes.

    This predictor implements the extraction-first architecture by:
    1. Using deterministic slot extractors for high-precision extraction
    2. Applying regex patterns for coverage of additional procedure types
    3. Optionally validating with ML predictions
    4. Producing structured ClinicalActions for CPT derivation

    Attributes:
        sectionizer: Service to parse note into sections
        _slot_extractors: Dict of slot extractors by name
        _ml_predictor: Optional ML predictor for validation
    """

    def __init__(
        self,
        sectionizer: SectionizerService | None = None,
        ml_predictor: object | None = None,
    ) -> None:
        """Initialize the ActionPredictor.

        Args:
            sectionizer: Section parser for procedure notes. If None, creates default.
            ml_predictor: Optional RegistryMLPredictor for validation.
        """
        self.sectionizer = sectionizer or SectionizerService()
        self._ml_predictor = ml_predictor

        # Initialize slot extractors
        # Note: ComplicationsExtractor removed due to broken import
        # Complications are extracted via regex patterns instead
        self._slot_extractors = {
            "ebus": EbusExtractor(),
            "tblb": TBLBExtractor(),
            "pleura": PleuraExtractor(),
            "sedation": SedationExtractor(),
            "stent": StentExtractor(),
        }

    def predict(self, note_text: str) -> PredictionResult:
        """Extract clinical actions from procedure note text.

        This is the main entry point for extraction-first processing.

        Args:
            note_text: Raw procedure note text

        Returns:
            PredictionResult containing:
            - actions: Structured ClinicalActions
            - field_extractions: Per-field evidence and confidence
            - warnings/errors: Any extraction issues
        """
        if not note_text or not note_text.strip():
            return PredictionResult(
                actions=ClinicalActions(),
                confidence_overall=0.0,
                warnings=["Empty note text provided"],
            )

        # Parse sections for context-aware extraction
        sections = self.sectionizer.sectionize(note_text)

        # Initialize result containers
        field_extractions: dict[str, ActionResult] = {}
        warnings: list[str] = []
        errors: list[str] = []

        # Run slot extractors first (high precision)
        slot_results = self._run_slot_extractors(note_text, sections)

        # Run additional regex extractors
        regex_results = self._run_regex_extractors(note_text, sections)

        # Build ClinicalActions from extraction results
        actions = self._build_clinical_actions(slot_results, regex_results, field_extractions)

        # Derive diagnostic bronchoscopy flag
        actions.diagnostic_bronchoscopy = self._is_bronchoscopy(actions)

        # Calculate overall confidence
        confidences = [r.confidence for r in field_extractions.values() if r.confidence > 0]
        confidence_overall = min(confidences) if confidences else 0.0

        # Optional: Validate with ML predictions
        if self._ml_predictor:
            ml_warnings = self._validate_with_ml(actions, note_text)
            warnings.extend(ml_warnings)

        return PredictionResult(
            actions=actions,
            field_extractions=field_extractions,
            confidence_overall=confidence_overall,
            extraction_method="hybrid",
            warnings=warnings,
            errors=errors,
        )

    def _run_slot_extractors(
        self, text: str, sections: Sequence[Section]
    ) -> dict[str, SlotResult]:
        """Run all slot extractors and collect results."""
        results = {}
        for name, extractor in self._slot_extractors.items():
            try:
                results[name] = extractor.extract(text, list(sections))
            except Exception as e:
                logger.warning(f"Slot extractor {name} failed: {e}")
                results[name] = SlotResult(None, [], 0.0)
        return results

    def _run_regex_extractors(
        self, text: str, sections: Sequence[Section]
    ) -> dict[str, SlotResult]:
        """Run regex-based extractors for actions not covered by slots."""
        results = {}

        # BAL
        results["bal"] = self._extract_with_patterns(
            text, sections, BAL_PATTERNS, "BAL"
        )

        # Brushings
        results["brushings"] = self._extract_with_patterns(
            text, sections, BRUSHINGS_PATTERNS, "brushings"
        )

        # Bronchial wash
        results["wash"] = self._extract_with_patterns(
            text, sections, WASH_PATTERNS, "bronchial_wash"
        )

        # Navigation platform
        results["navigation"] = self._extract_navigation_platform(text, sections)

        # BLVR
        results["blvr"] = self._extract_with_patterns(
            text, sections, BLVR_PATTERNS, "BLVR"
        )

        # Chartis
        results["chartis"] = self._extract_with_patterns(
            text, sections, CHARTIS_PATTERNS, "Chartis"
        )

        # Thermal ablation
        results["thermal"] = self._extract_with_patterns(
            text, sections, THERMAL_ABLATION_PATTERNS, "thermal_ablation"
        )

        # Cryotherapy
        results["cryo"] = self._extract_with_patterns(
            text, sections, CRYO_PATTERNS, "cryotherapy"
        )

        # Dilation
        results["dilation"] = self._extract_with_patterns(
            text, sections, DILATION_PATTERNS, "dilation"
        )

        # Transbronchial biopsy (supplements slot extractor)
        results["tbb"] = self._extract_with_patterns(
            text, sections, TRANSBRONCHIAL_BIOPSY_PATTERNS, "transbronchial_biopsy"
        )

        # Conventional TBNA (non-EBUS guided)
        results["conventional_tbna"] = self._extract_with_patterns(
            text, sections, CONVENTIONAL_TBNA_PATTERNS, "conventional_tbna"
        )

        # Cryobiopsy
        results["cryobiopsy"] = self._extract_with_patterns(
            text, sections, CRYOBIOPSY_PATTERNS, "cryobiopsy"
        )

        # Endobronchial biopsy (distinct from transbronchial)
        results["endobronchial_biopsy"] = self._extract_with_patterns(
            text, sections, ENDOBRONCHIAL_BIOPSY_PATTERNS, "endobronchial_biopsy"
        )

        # IPC
        results["ipc"] = self._extract_with_patterns(
            text, sections, IPC_PATTERNS, "IPC"
        )

        # Thoracoscopy
        results["thoracoscopy"] = self._extract_with_patterns(
            text, sections, THORACOSCOPY_PATTERNS, "thoracoscopy"
        )

        # Foreign body
        results["foreign_body"] = self._extract_with_patterns(
            text, sections, FOREIGN_BODY_PATTERNS, "foreign_body"
        )

        # CBCT
        results["cbct"] = self._extract_with_patterns(
            text, sections, CBCT_PATTERNS, "CBCT"
        )

        # Robotic
        results["robotic"] = self._extract_with_patterns(
            text, sections, ROBOTIC_PATTERNS, "robotic"
        )

        # Radial EBUS (distinct from linear EBUS)
        results["radial_ebus"] = self._extract_with_patterns(
            text, sections, RADIAL_EBUS_PATTERNS, "radial_ebus"
        )

        # Rigid bronchoscopy
        results["rigid_bronch"] = self._extract_with_patterns(
            text, sections, RIGID_BRONCH_PATTERNS, "rigid_bronch"
        )

        # Complications (using inline patterns)
        results["complications"] = self._extract_complications(text, sections)

        return results

    def _extract_with_patterns(
        self,
        text: str,
        sections: Sequence[Section],
        patterns: list[re.Pattern],
        label: str,
        check_negation: bool = True,
    ) -> SlotResult:
        """Extract using a list of regex patterns with optional negation checking.

        Args:
            text: Full note text
            sections: Parsed sections
            patterns: List of regex patterns to match
            label: Label for logging
            check_negation: Whether to filter out negated matches

        Returns:
            SlotResult with found=True only if non-negated matches exist
        """
        evidence: list[Span] = []
        found = False

        for pattern in patterns:
            for match in pattern.finditer(text):
                # Check for negation if enabled
                if check_negation and self._is_negated(text, match.start(), match.end()):
                    logger.debug(f"Negated match for {label}: '{match.group(0)}'")
                    continue

                found = True
                evidence.append(
                    Span(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        section=section_for_offset(sections, match.start()),
                    )
                )

        return SlotResult(found, evidence, 0.8 if found else 0.0)

    def _is_negated(self, text: str, match_start: int, match_end: int) -> bool:
        """Check if a match is negated by surrounding context.

        Args:
            text: Full text
            match_start: Start position of the match
            match_end: End position of the match

        Returns:
            True if the match appears to be negated
        """
        # Get context window before the match
        context_start = max(0, match_start - NEGATION_WINDOW)
        prefix_context = text[context_start:match_start].lower()

        # Check for negation prefix (e.g., "no EBUS", "not performed")
        if NEGATION_PREFIX_PATTERN.search(prefix_context):
            # Make sure negation is close to the match (within window)
            neg_match = NEGATION_PREFIX_PATTERN.search(prefix_context)
            if neg_match and (len(prefix_context) - neg_match.end()) < 20:
                return True

        # Get context window after the match
        context_end = min(len(text), match_end + NEGATION_WINDOW)
        suffix_context = text[match_end:context_end].lower()

        # Check for negation suffix (e.g., "was not performed")
        for pattern in NEGATION_SUFFIX_PATTERNS:
            if pattern.search(suffix_context):
                return True

        return False

    def _extract_navigation_platform(
        self, text: str, sections: Sequence[Section]
    ) -> SlotResult:
        """Extract navigation platform with specific platform identification."""
        evidence: list[Span] = []
        platform = None

        for platform_name, pattern in NAV_PLATFORM_PATTERNS.items():
            for match in pattern.finditer(text):
                platform = platform_name
                evidence.append(
                    Span(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        section=section_for_offset(sections, match.start()),
                    )
                )
                break  # Take first match for platform
            if platform:
                break

        return SlotResult(
            {"platform": platform, "performed": platform is not None},
            evidence,
            0.9 if platform else 0.0,
        )

    def _extract_complications(
        self, text: str, sections: Sequence[Section]
    ) -> SlotResult:
        """Extract complications from text."""
        evidence: list[Span] = []
        complications: list[str] = []

        # Check for explicit "no complications"
        if NO_COMPLICATION_PATTERN.search(text):
            match = NO_COMPLICATION_PATTERN.search(text)
            evidence.append(
                Span(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    section=section_for_offset(sections, match.start()),
                )
            )
            return SlotResult(["None"], evidence, 0.8)

        # Search for specific complications
        for complication, pattern in COMPLICATION_PATTERNS.items():
            for match in pattern.finditer(text):
                complications.append(complication)
                evidence.append(
                    Span(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        section=section_for_offset(sections, match.start()),
                    )
                )

        unique_complications = list(dict.fromkeys(complications))
        return SlotResult(
            unique_complications if unique_complications else [],
            evidence,
            0.7 if unique_complications else 0.0,
        )

    def _build_clinical_actions(
        self,
        slot_results: dict[str, SlotResult],
        regex_results: dict[str, SlotResult],
        field_extractions: dict[str, ActionResult],
    ) -> ClinicalActions:
        """Build ClinicalActions from extraction results."""

        # EBUS
        ebus_result = slot_results.get("ebus")
        ebus_value = ebus_result.value if ebus_result else {}
        if isinstance(ebus_value, dict):
            stations = ebus_value.get("stations", [])
            navigation = ebus_value.get("navigation", False)
            radial = ebus_value.get("radial", False)
        else:
            stations = []
            navigation = False
            radial = False

        ebus_actions = EBUSActions(
            performed=len(stations) > 0,
            stations=stations,
        )
        if ebus_result and ebus_result.evidence:
            field_extractions["ebus.stations"] = ActionResult(
                value=stations,
                evidence=ebus_result.evidence,
                confidence=ebus_result.confidence,
                source="deterministic",
            )

        # Navigation
        nav_result = regex_results.get("navigation")
        nav_value = nav_result.value if nav_result else {}
        robotic_result = regex_results.get("robotic")
        cbct_result = regex_results.get("cbct")
        radial_ebus_result = regex_results.get("radial_ebus")

        if isinstance(nav_value, dict):
            nav_platform = nav_value.get("platform")
            nav_performed = nav_value.get("performed", False) or navigation
        else:
            nav_platform = None
            nav_performed = navigation

        # Radial EBUS can be detected via slot extractor OR regex pattern
        radial_detected = radial or bool(radial_ebus_result and radial_ebus_result.value)

        nav_actions = NavigationActions(
            performed=nav_performed,
            platform=nav_platform,
            is_robotic=bool(robotic_result and robotic_result.value),
            radial_ebus_used=radial_detected,
            cone_beam_ct_used=bool(cbct_result and cbct_result.value),
        )
        if nav_result and nav_result.evidence:
            field_extractions["navigation.platform"] = ActionResult(
                value=nav_platform,
                evidence=nav_result.evidence,
                confidence=nav_result.confidence,
                source="deterministic",
            )

        # Biopsy (TBLB from slot extractor + regex patterns)
        tblb_result = slot_results.get("tblb")
        tblb_lobes = tblb_result.value if tblb_result and tblb_result.value else []
        tbb_regex_result = regex_results.get("tbb")
        cryobiopsy_result = regex_results.get("cryobiopsy")
        conventional_tbna_result = regex_results.get("conventional_tbna")
        endobronchial_biopsy_result = regex_results.get("endobronchial_biopsy")

        # TBB detected if slot extractor found lobes OR regex patterns matched
        tbb_detected = len(tblb_lobes) > 0 or bool(tbb_regex_result and tbb_regex_result.value)

        # Endobronchial biopsy is distinct from transbronchial
        ebb_detected = bool(endobronchial_biopsy_result and endobronchial_biopsy_result.value)

        biopsy_actions = BiopsyActions(
            transbronchial_performed=tbb_detected,
            transbronchial_sites=tblb_lobes if isinstance(tblb_lobes, list) else [],
            cryobiopsy_performed=bool(cryobiopsy_result and cryobiopsy_result.value),
            tbna_conventional_performed=bool(conventional_tbna_result and conventional_tbna_result.value),
            endobronchial_performed=ebb_detected,
        )
        if tblb_result and tblb_result.evidence:
            field_extractions["biopsy.transbronchial_sites"] = ActionResult(
                value=tblb_lobes,
                evidence=tblb_result.evidence,
                confidence=tblb_result.confidence,
                source="deterministic",
            )

        # BAL
        bal_result = regex_results.get("bal")
        bal_actions = BALActions(
            performed=bool(bal_result and bal_result.value),
        )
        if bal_result and bal_result.evidence:
            field_extractions["bal.performed"] = ActionResult(
                value=True,
                evidence=bal_result.evidence,
                confidence=bal_result.confidence,
                source="deterministic",
            )

        # Brushings
        brushings_result = regex_results.get("brushings")
        brushings_actions = BrushingsActions(
            performed=bool(brushings_result and brushings_result.value),
        )
        if brushings_result and brushings_result.evidence:
            field_extractions["brushings.performed"] = ActionResult(
                value=True,
                evidence=brushings_result.evidence,
                confidence=brushings_result.confidence,
                source="deterministic",
            )

        # Bronchial wash
        wash_result = regex_results.get("wash")
        wash_actions = BronchialWashActions(
            performed=bool(wash_result and wash_result.value),
        )

        # Pleural
        pleura_result = slot_results.get("pleura")
        pleura_values = pleura_result.value if pleura_result and pleura_result.value else []
        ipc_result = regex_results.get("ipc")
        thoracoscopy_result = regex_results.get("thoracoscopy")

        pleural_actions = PleuralActions(
            thoracentesis_performed="Thoracentesis" in pleura_values,
            chest_tube_performed="Chest Tube" in pleura_values,
            pleurodesis_performed="Pleurodesis" in pleura_values,
            ipc_performed=bool(ipc_result and ipc_result.value),
            thoracoscopy_performed=bool(thoracoscopy_result and thoracoscopy_result.value),
        )
        if pleura_result and pleura_result.evidence:
            field_extractions["pleural.procedures"] = ActionResult(
                value=pleura_values,
                evidence=pleura_result.evidence,
                confidence=pleura_result.confidence,
                source="deterministic",
            )

        # CAO
        thermal_result = regex_results.get("thermal")
        cryo_result = regex_results.get("cryo")
        dilation_result = regex_results.get("dilation")

        cao_actions = CAOActions(
            performed=(
                bool(thermal_result and thermal_result.value)
                or bool(cryo_result and cryo_result.value)
                or bool(dilation_result and dilation_result.value)
            ),
            thermal_ablation_performed=bool(thermal_result and thermal_result.value),
            cryotherapy_performed=bool(cryo_result and cryo_result.value),
            dilation_performed=bool(dilation_result and dilation_result.value),
        )

        # Stent
        stent_result = slot_results.get("stent")
        stent_value = stent_result.value if stent_result else None
        stent_actions = StentActions(
            performed=bool(stent_value),
        )
        if stent_result and stent_result.evidence:
            field_extractions["stent.performed"] = ActionResult(
                value=True,
                evidence=stent_result.evidence,
                confidence=stent_result.confidence,
                source="deterministic",
            )

        # BLVR
        blvr_result = regex_results.get("blvr")
        chartis_result = regex_results.get("chartis")

        blvr_actions = BLVRActions(
            performed=bool(blvr_result and blvr_result.value),
            chartis_performed=bool(chartis_result and chartis_result.value),
        )
        if blvr_result and blvr_result.evidence:
            field_extractions["blvr.performed"] = ActionResult(
                value=True,
                evidence=blvr_result.evidence,
                confidence=blvr_result.confidence,
                source="deterministic",
            )

        # Therapeutic
        foreign_body_result = regex_results.get("foreign_body")
        therapeutic_actions = TherapeuticActions(
            foreign_body_removal_performed=bool(
                foreign_body_result and foreign_body_result.value
            ),
        )

        # Complications (from regex results, not slot results)
        complications_result = regex_results.get("complications")
        complications_values = (
            complications_result.value
            if complications_result and complications_result.value
            else []
        )
        has_complications = (
            bool(complications_values) and complications_values != ["None"]
        )

        complication_actions = ComplicationActions(
            any_complication=has_complications,
            complications=complications_values if has_complications else [],
            bleeding="Bleeding" in complications_values,
            pneumothorax="Pneumothorax" in complications_values,
            hypoxia="Hypoxia" in complications_values or "Desaturation" in complications_values,
        )
        if complications_result and complications_result.evidence:
            field_extractions["complications"] = ActionResult(
                value=complications_values,
                evidence=complications_result.evidence,
                confidence=complications_result.confidence,
                source="deterministic",
            )

        # Sedation
        sedation_result = slot_results.get("sedation")
        sedation_value = sedation_result.value if sedation_result else None
        sedation_actions = SedationActions()
        if sedation_value and isinstance(sedation_value, dict):
            sedation_type = sedation_value.get("sedation_type")
            if sedation_type in ("MAC", "moderate", "general", "topical"):
                sedation_actions.sedation_type = sedation_type

        # Rigid bronchoscopy detection
        rigid_bronch_result = regex_results.get("rigid_bronch")
        is_rigid_bronch = bool(rigid_bronch_result and rigid_bronch_result.value)

        return ClinicalActions(
            ebus=ebus_actions,
            biopsy=biopsy_actions,
            navigation=nav_actions,
            bal=bal_actions,
            brushings=brushings_actions,
            bronchial_wash=wash_actions,
            pleural=pleural_actions,
            cao=cao_actions,
            stent=stent_actions,
            blvr=blvr_actions,
            therapeutic=therapeutic_actions,
            complications=complication_actions,
            sedation=sedation_actions,
            rigid_bronchoscopy=is_rigid_bronch,
        )

    def _is_bronchoscopy(self, actions: ClinicalActions) -> bool:
        """Determine if any bronchoscopic procedure was performed."""
        return (
            actions.ebus.performed
            or actions.navigation.performed
            or actions.biopsy.transbronchial_performed
            or actions.biopsy.cryobiopsy_performed
            or actions.biopsy.endobronchial_performed
            or actions.bal.performed
            or actions.brushings.performed
            or actions.bronchial_wash.performed
            or actions.cao.performed
            or actions.stent.performed
            or actions.blvr.performed
            or actions.therapeutic.foreign_body_removal_performed
            or actions.therapeutic.whole_lung_lavage_performed
            or actions.therapeutic.bronchial_thermoplasty_performed
        )

    def _validate_with_ml(
        self, actions: ClinicalActions, note_text: str
    ) -> list[str]:
        """Validate extracted actions against ML predictions.

        Returns warnings for discrepancies between extraction and ML.
        """
        warnings = []

        if not self._ml_predictor or not hasattr(self._ml_predictor, "predict"):
            return warnings

        try:
            ml_fields = self._ml_predictor.predict(note_text)
            extracted_procedures = set(actions.get_performed_procedures())

            # Check for ML predictions not found by extraction
            for field in ml_fields:
                if field not in extracted_procedures:
                    warnings.append(
                        f"ML predicted '{field}' but extraction did not find it"
                    )

            # Check for extractions not predicted by ML
            for proc in extracted_procedures:
                if proc not in ml_fields:
                    warnings.append(
                        f"Extraction found '{proc}' but ML did not predict it"
                    )

        except Exception as e:
            logger.warning(f"ML validation failed: {e}")
            warnings.append(f"ML validation unavailable: {e}")

        return warnings


__all__ = ["ActionPredictor"]
