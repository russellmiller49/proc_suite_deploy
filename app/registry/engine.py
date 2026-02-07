"""Registry extraction orchestrator."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Set

from app.common.logger import get_logger
from app.common.sectionizer import SectionizerService
from app.common.spans import Span
from app.registry.extractors.llm_detailed import LLMDetailedExtractor
from app.registry.postprocess import POSTPROCESSORS, apply_cross_field_consistency, derive_global_ebus_rose_result
from app.registry.transform import build_nested_registry_payload
from app.registry.deterministic_extractors import run_deterministic_extractors
from app.registry.processing.masking import mask_offset_preserving
from app.registry.normalization import normalize_registry_enums
from app.registry.tags import FIELD_APPLICABLE_TAGS, PROCEDURE_FAMILIES

from .schema import RegistryRecord


logger = get_logger("registry_engine")

# List-like fields that must contain only non-empty strings (no None/"").
# Centralize this so enum-array fields don't trigger record-wide validation failures.
_STRING_ENUM_LIST_FIELDS: set[str] = {
    "sampling_tools_used",
    "specimen_sent_for",
    "hemostasis_methods",
    "destinations",
}


def _format_payload_path(path: tuple[Any, ...]) -> str:
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            if rendered:
                rendered += "."
            rendered += str(part)
    return rendered or "<root>"


def _sanitize_string_enum_lists(payload: Any) -> list[str]:
    """Drop invalid items (None/blank/non-str) from known list-of-enum fields.

    This runs before RegistryRecord validation to prevent broad pruning/fallbacks
    when a single list contains an invalid element.
    """
    warnings: list[str] = []

    def _walk(obj: Any, path: tuple[Any, ...]) -> None:
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                next_path = path + (key,)
                if key in _STRING_ENUM_LIST_FIELDS and isinstance(value, list):
                    kept: list[str] = []
                    dropped: list[Any] = []
                    for item in value:
                        if item is None:
                            dropped.append(item)
                            continue
                        if not isinstance(item, str):
                            dropped.append(item)
                            continue
                        if not item.strip():
                            dropped.append(item)
                            continue
                        kept.append(item)

                    if dropped:
                        obj[key] = kept
                        warnings.append(
                            f"Dropped invalid list items from {_format_payload_path(next_path)}: {dropped!r}"
                        )
                    continue

                _walk(value, next_path)
            return

        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, path + (idx,))

    _walk(payload, ())
    return warnings


def _summarize_list(items: list[str], *, max_items: int, sep: str = "; ") -> str:
    if not items:
        return ""
    shown = items[:max_items]
    suffix = f"{sep}(+{len(items) - max_items} more)" if len(items) > max_items else ""
    return sep.join(shown) + suffix


def _drop_none_items_from_lists(obj: Any) -> None:
    """In-place removal of None items from any lists in a payload structure."""
    if isinstance(obj, dict):
        for value in obj.values():
            _drop_none_items_from_lists(value)
        return

    if isinstance(obj, list):
        obj[:] = [item for item in obj if item is not None]
        for item in obj:
            _drop_none_items_from_lists(item)

# PROCEDURE_FAMILIES / FIELD_APPLICABLE_TAGS are now sourced from app.registry.tags


def filter_inapplicable_fields(data: Dict[str, Any], families: Set[str]) -> Dict[str, Any]:
    """Null out fields that don't apply to detected procedure families.

    This prevents phantom data from being extracted for procedures not performed.
    """
    filtered = dict(data)
    for field, applicable_tags in FIELD_APPLICABLE_TAGS.items():
        if field in filtered and not applicable_tags.intersection(families):
            # Field doesn't apply to any detected procedure family - null it out
            filtered[field] = None
    return filtered


def validate_evidence_spans(
    note_text: str,
    evidence: Dict[str, list[Any]],
    similarity_threshold: float = 0.7,
) -> Dict[str, list[Any]]:
    """Filter out hallucinated evidence spans that don't match the source text.

    LLMs sometimes produce evidence spans with incorrect start/end offsets or
    text that doesn't actually appear in the source document. This function
    validates each span and removes any that fail validation.

    Args:
        note_text: The original procedure note text.
        evidence: Dict mapping field names to lists of Span objects.
        similarity_threshold: Minimum ratio of matching characters for fuzzy match.

    Returns:
        Filtered evidence dict with only validated spans.
    """
    from difflib import SequenceMatcher

    validated: Dict[str, list[Any]] = {}

    for field, spans in evidence.items():
        valid_spans = []
        for span in spans:
            if not hasattr(span, "text") or not hasattr(span, "start") or not hasattr(span, "end"):
                continue

            span_text = span.text
            start = span.start
            end = span.end

            # Skip if offsets are clearly invalid
            if start is None or end is None:
                continue
            if start < 0 or end > len(note_text) or start >= end:
                continue

            # Extract actual text at the given offsets
            actual_text = note_text[start:end]

            # Check for exact match first
            if span_text == actual_text:
                valid_spans.append(span)
                continue

            # Fuzzy match - allow minor OCR/formatting differences
            if span_text and actual_text:
                ratio = SequenceMatcher(None, span_text.lower(), actual_text.lower()).ratio()
                if ratio >= similarity_threshold:
                    valid_spans.append(span)
                    continue

            # Check if span text appears anywhere in the note (offset might be wrong but text is real)
            if span_text and span_text in note_text:
                # Special handling for station evidence - station "7" alone is too ambiguous
                # It could match dates, times, ages, etc. Require explicit station context.
                is_station_field = field in ("linear_ebus_stations", "ebus_stations_sampled", "ebus_stations_detail")
                if is_station_field and span_text.strip() in ("7", "station 7"):
                    # For station 7, require EXPLICIT "station 7" with sampling context
                    # or "7:" followed by EBUS details (not just any "7" in the text)
                    station_7_strict_pattern = r"(?:station\s+7\s*[:\-]|station\s*7\s+(?:was\s+)?(?:sampl|biops|needle|pass)|subcarinal\s+(?:lymph\s+)?node)"
                    if not re.search(station_7_strict_pattern, note_text, re.IGNORECASE):
                        # Skip this span - "7" appears but not in clear EBUS station context
                        continue

                # Fix the offsets to where it actually appears
                actual_start = note_text.find(span_text)
                if actual_start >= 0:
                    span.start = actual_start
                    span.end = actual_start + len(span_text)
                    span.text = note_text[span.start:span.end]
                    valid_spans.append(span)
                    continue

        if valid_spans:
            deduped: list[Any] = []
            seen_keys: set[tuple[int, int, str]] = set()
            for span in valid_spans:
                key = (span.start, span.end, span.text)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Enforce exact text match after any corrections
                try:
                    span.text = note_text[span.start:span.end]
                except Exception:
                    pass
                deduped.append(span)
            if deduped:
                validated[field] = deduped

    return validated


def classify_procedure_families(note_text: str) -> Set[str]:
    """Return tags describing the procedures actually performed.

    This function analyzes the procedure note to identify which procedure families
    were actually performed (not just indicated or mentioned in history).

    The classification focuses on:
    - Procedural action verbs (performed, placed, obtained, sampled)
    - Specific equipment/technique mentions in procedure context
    - Explicit procedure names in PROCEDURE or TECHNIQUE sections

    Returns:
        Set of procedure family tags from PROCEDURE_FAMILIES.
    """
    families: Set[str] = set()
    lowered = note_text.lower()

    # Extract relevant sections for more accurate detection
    # Focus on PROCEDURE, TECHNIQUE, DESCRIPTION sections
    procedure_section_text = _extract_procedure_sections(note_text)
    proc_lowered = procedure_section_text.lower() if procedure_section_text else lowered

    # --- EBUS Detection ---
    # Linear EBUS (not radial) with actual sampling
    # Must be careful to exclude mentions of stations in non-EBUS context (e.g., CT findings)
    # NOTE: This classifier gates downstream EBUS-specific extraction (e.g. station parsing).
    # Keep patterns conservative, but tolerate common dictation typos like "staion" and "biospy".
    ebus_indicators = [
        # EBUS + sampling context (including common "biopsy"/"bx" shorthand and typos).
        r"\bebus\b.*(?:tbna|sampl|aspirat|needle|biops|biopsy|biospy|\bbx\b)",
        r"(?:tbna|sampl|aspirat|needle|biops|biopsy|biospy|\bbx\b).*\bebus\b",
        r"linear\s+(?:ebus|endobronchial ultrasound)",
        r"ebus[-\s]*findings",
        r"overall\s+rose\s+diagnosis",
        # ROSE results commonly appear only when sampling was performed.
        r"\bebus\b.{0,80}\brose\b.{0,60}(?:positive|negative|adequate|inadequate|malignan|benign|nondiagnostic|insufficient)",
        r"\brose\b.{0,80}(?:positive|negative|adequate|inadequate|malignan|benign|nondiagnostic|insufficient).{0,80}\bebus\b",
        # Station sampling - require close proximity and exclude negatives. Accept "staion" typo.
        r"sta(?:t)?ion\s*(?:2r|2l|4r|4l|7|10r|10l|11r|11l).{0,30}(?:sampl|pass|needle|aspirat|biops|biopsy|biospy|rose)",
        r"(?:sampl|pass|needle|aspirat|biops|biopsy|biospy|rose).{0,30}sta(?:t)?ion\s*(?:2r|2l|4r|4l|7|10r|10l|11r|11l)",
        r"endobronchial ultrasound.{0,50}(?:guided|needle|aspirat|biops|biopsy|biospy)",
    ]
    # Exclude EBUS if there's explicit negative context about procedure not being done
    ebus_exclusion_indicators = [
        r"(?:not|no|without)\s+(?:ebus|tbna)\s+(?:performed|done|planned)",
        r"ebus\s+(?:was\s+)?not\s+(?:performed|done)",
        r"no\s+ebus\s+(?:was\s+)?performed",
        r"tbna\s+(?:was\s+)?not\s+(?:performed|done|obtained)",
    ]
    ebus_match = any(re.search(pat, proc_lowered) for pat in ebus_indicators)
    ebus_excluded = any(re.search(pat, proc_lowered) for pat in ebus_exclusion_indicators)

    # Additional check: if "station X" is mentioned, verify it's not in negative context
    if ebus_match and not ebus_excluded:
        # Check if station mentions are actually sampled (not just described)
        station_pattern = r"sta(?:t)?ion\s*(2r|2l|4r|4l|7|10r|10l|11r|11l)"
        station_matches = list(re.finditer(station_pattern, proc_lowered))
        if station_matches:
            # For each station mention, check surrounding context for negative phrases
            all_stations_negative = True
            for match in station_matches:
                # Get surrounding context (100 chars around the match)
                start = max(0, match.start() - 50)
                end = min(len(proc_lowered), match.end() + 80)
                context = proc_lowered[start:end]
                # Check for actual sampling in this context
                if re.search(r"(?:sampl|pass|needle|aspirat|biops|biopsy|biospy|rose)", context):
                    # Also check it's not negative
                    if not re.search(r"(?:not|no|wasn't|were\s+not|was\s+not)\s+sampl", context):
                        all_stations_negative = False
                        break
            if all_stations_negative:
                ebus_match = False

    if ebus_match and not ebus_excluded:
        families.add("EBUS")

    # --- NAVIGATION Detection ---
    nav_indicators = [
        r"(?:electromagnetic|emn)\s+navigation",
        r"\bion\b.*(?:catheter|target|nodule|bronchoscop)",
        r"\bmonarch\b.*(?:robot|bronchoscop|navigat)",
        r"\bauris\b",
        r"navigat(?:ed|ion)\s+(?:bronchoscopy|to|biopsy)",
        r"superDimension",
        r"illumisite",
        r"veran",
        r"spin(?:drive)?.*(?:navigat|target)",
    ]
    # Navigation details are often documented outside strict PROCEDURE/TECHNIQUE
    # headers (e.g., planning and platform setup narratives). Check both the
    # extracted procedure section text and the full note to avoid false negatives.
    if any(re.search(pat, proc_lowered, re.IGNORECASE) for pat in nav_indicators) or any(
        re.search(pat, lowered, re.IGNORECASE) for pat in nav_indicators
    ):
        families.add("NAVIGATION")

    # --- CAO (Central Airway Obstruction) Detection ---
    cao_indicators = [
        r"debulk",
        r"tumor\s+(?:resect|ablat|destruct|remov|treat)",
        r"(?:recanaliz|recanalis)",
        r"central\s+airway\s+obstruct",
        r"airway\s+(?:obstruct|stenosis).*(?:treat|interven)",
        r"(?:apc|argon\s+plasma).*(?:ablat|coagul|tumor)",
        r"(?:electrocautery|cautery).*(?:tumor|lesion|debulk)",
        r"cryotherapy.*(?:tumor|ablat|destruct)",
        r"laser.*(?:ablat|resect|tumor)",
        r"mechanical.*(?:debulk|core.?out|resect)",
        r"endobronchial\s+(?:tumor|mass|lesion).*(?:resect|remov|debulk|treat)",
        # Therapeutic modalities applied to endobronchial tumors
        r"endobronchial\s+(?:tumor|mass|lesion).{0,50}(?:apc|cryotherapy|cryo|cautery|laser)",
        r"(?:apc|cryotherapy|cryo|cautery|laser).{0,50}endobronchial\s+(?:tumor|mass|lesion)",
        # Treatment with interventional modalities
        r"(?:tumor|mass|lesion).{0,30}(?:treated|ablated).{0,30}(?:apc|cryotherapy|cryo|cautery|laser)",
    ]
    if any(re.search(pat, proc_lowered) for pat in cao_indicators):
        families.add("CAO")

    # --- STENT Detection ---
    # Must be an actual stent procedure, not just history/mention
    stent_placement_indicators = [
        r"stent\s+(?:plac|deploy|insert)",
        r"(?:plac|deploy|insert).*stent",
        r"(?:silicone|metallic|hybrid|dumon|y-stent).*(?:plac|deploy|insert)",
        r"stent.*(?:remov|retriev|exchang)\w*",
        r"stent\s+was\s+(?:plac|deploy|insert)",
    ]
    # Exclude history-only mentions
    stent_history_indicators = [
        r"(?:history|prior|previous)\s+(?:of\s+)?(?:.*\s+)?stent",
        r"stent\s+(?:removed|was\s+removed)\s+\d+",  # "stent removed 2 years ago"
        r"old\s+stent",
        r"prior\s+stent",
    ]
    has_stent_procedure = any(re.search(pat, proc_lowered) for pat in stent_placement_indicators)
    is_history_only = any(re.search(pat, proc_lowered) for pat in stent_history_indicators)

    # Only add STENT if there's a procedure AND it's not history-only (unless there's also a new procedure)
    if has_stent_procedure and not is_history_only:
        families.add("STENT")
        # Airway stenting/removal is a CAO intervention family in this registry.
        families.add("CAO")
    elif has_stent_procedure and is_history_only:
        # Check if there's explicit new stent action beyond the history mention
        new_action_patterns = [
            r"(?:today|now|this\s+procedure).*stent",
            r"stent.*(?:today|now|performed|deployed|placed)\b",
            r"new\s+stent",
        ]
        if any(re.search(pat, proc_lowered) for pat in new_action_patterns):
            families.add("STENT")
            families.add("CAO")

    # --- PLEURAL Detection ---
    # Be specific to avoid false positives from EBUS needle "aspiration"
    pleural_indicators = [
        r"thoracentesis",
        r"pleural\s+(?:tap|drain|fluid\s+remov)",
        r"pleural\s+effusion.*(?:drain|remov|tap)",
        r"(?:drain|remov|tap).*pleural\s+effusion",
        r"chest\s+tube\s+(?:plac|insert|remov|exchange)",
        r"(?:plac|insert|remov|exchange).*chest\s+tube",
        r"pigtail\s+(?:catheter|drain)",
        r"tunneled\s+(?:pleural\s+)?catheter",
        r"indwelling\s+pleural\s+catheter",
        r"\bpleurx\b",  # PleurX catheter brand
        r"\baspira\s+(?:catheter|drain)",  # Aspira catheter brand - not just "aspira" alone
        r"\bipc\b(?!\s*\d)",  # IPC (Indwelling Pleural Catheter) - but not ipc followed by numbers (like IP addresses)
        r"catheter\s+(?:exchange|replac).*pleural",  # Only pleural catheter exchange
        r"ultrasound.{0,20}guid.{0,30}(?:thoracentesis|pleural)",
    ]
    # Exclusion patterns to prevent false positives
    pleural_exclusions = [
        r"needle\s+aspiration",  # EBUS-TBNA, FNA
        r"tbna",  # Transbronchial needle aspiration
        r"transbronchial.*aspiration",
        r"fine\s+needle\s+aspiration",
    ]
    pleural_match = any(re.search(pat, proc_lowered) for pat in pleural_indicators)
    pleural_excluded = any(re.search(pat, proc_lowered) for pat in pleural_exclusions)
    # Only add PLEURAL if indicators found AND (no exclusions OR explicit pleural procedure terms)
    if pleural_match:
        # If we have needle aspiration context but also explicit pleural procedure, still add it
        has_explicit_pleural = any(re.search(pat, proc_lowered) for pat in [
            r"thoracentesis", r"chest\s+tube", r"tunneled.*catheter",
            r"pleural\s+(?:tap|drain|fluid)", r"\bpleurx\b", r"pigtail"
        ])
        if has_explicit_pleural or not pleural_excluded:
            families.add("PLEURAL")

    # --- THORACOSCOPY Detection ---
    thoracoscopy_indicators = [
        r"(?:medical\s+)?thoracoscopy",
        r"pleuroscopy",
        r"(?:vats|video.?assisted).*(?:biops|pleurodesis|inspect)",
        r"thoracoscop.*(?:biops|pleurodesis|inspect)",
        r"talc\s+(?:poudrage|pleurodesis|insufflat)",
        r"chemical\s+pleurodesis",
    ]
    if any(re.search(pat, proc_lowered) for pat in thoracoscopy_indicators):
        families.add("THORACOSCOPY")

    # --- BLVR Detection ---
    # BLVR may have its own section (BLVR:) so check full note text too
    blvr_indicators = [
        r"(?:zephyr|spiration)\s+valve",
        r"endobronchial\s+valve",
        r"(?:ebv|valve)\s+(?:plac|deploy|insert)",
        r"lung\s+volume\s+reduction",
        r"chartis\s+(?:assess|measur|catheter)",
    ]
    if any(re.search(pat, proc_lowered) for pat in blvr_indicators):
        families.add("BLVR")
    elif any(re.search(pat, lowered) for pat in blvr_indicators):  # Check full note
        families.add("BLVR")

    # --- BAL Detection ---
    bal_indicators = [
        r"bronchoalveolar\s+lavage",
        r"bronchial\s+alveolar\s+lavage",
        r"\bbal\b.*(?:perform|obtain|sent|collect)",
        r"(?:perform|obtain).*\bbal\b",
        r"lavage.{0,20}(?:perform|performed|sent|obtain|obtained|specimen|collect|collected)",
    ]
    if any(re.search(pat, proc_lowered) for pat in bal_indicators) or any(re.search(pat, lowered) for pat in bal_indicators):
        families.add("BAL")

    # --- BIOPSY Detection (general transbronchial/endobronchial) ---
    biopsy_indicators = [
        r"transbronchial\s+(?:biops|forceps)",
        r"endobronchial\s+biops",
        r"(?:forceps|brush)\s+biops",
        r"biops(?:y|ies)\s+(?:obtain|perform|taken|sent)",
        r"tissue\s+sampl(?:e|ing|ed)",
    ]
    if any(re.search(pat, proc_lowered) for pat in biopsy_indicators):
        families.add("BIOPSY")

    # --- CRYO_BIOPSY Detection ---
    cryo_biopsy_indicators = [
        r"cryobiops",
        r"transbronchial\s+cryo",
        r"cryo\s*(?:probe)?.*(?:biops|sampl)",
        r"(?:biops|sampl).*cryo\s*probe",
    ]
    if any(re.search(pat, proc_lowered) for pat in cryo_biopsy_indicators):
        families.add("CRYO_BIOPSY")

    # --- FOREIGN_BODY Detection ---
    fb_indicators = [
        r"foreign\s+body\s+(?:remov|retriev|extract)",
        r"(?:remov|retriev|extract).*foreign\s+body",
        r"aspirat(?:ed|ion)\s+(?:object|material).*(?:remov|retriev)",
    ]
    if any(re.search(pat, proc_lowered) for pat in fb_indicators):
        families.add("FOREIGN_BODY")

    # --- HEMOPTYSIS Detection ---
    hemoptysis_indicators = [
        r"hemoptysis.*(?:control|manag|treat|tamponade)",
        r"(?:control|manag|treat).*hemoptysis",
        r"balloon\s+tamponade",
        r"(?:cold|iced)\s+saline.*hemostasis",
        r"bleeding.*(?:control|cauteriz|coagulat)",
    ]
    if any(re.search(pat, proc_lowered) for pat in hemoptysis_indicators):
        families.add("HEMOPTYSIS")

    # --- THERMOPLASTY Detection ---
    thermoplasty_indicators = [
        r"bronchial\s+thermoplasty",
        r"alair",
        r"thermoplasty.*(?:treat|activat|session)",
    ]
    if any(re.search(pat, proc_lowered) for pat in thermoplasty_indicators):
        families.add("THERMOPLASTY")

    # --- DIAGNOSTIC Detection ---
    # Only if no other interventional procedures detected
    diagnostic_indicators = [
        r"diagnostic\s+bronchoscopy",
        r"inspection\s+only",
        r"airway\s+(?:survey|inspection|exam)",
        r"flexible\s+bronchoscopy.*(?:inspect|survey|exam)",
    ]
    if not families and any(re.search(pat, proc_lowered) for pat in diagnostic_indicators):
        families.add("DIAGNOSTIC")

    # If still empty but procedure note exists, at least mark as DIAGNOSTIC
    if not families and _has_procedure_content(note_text):
        families.add("DIAGNOSTIC")

    return families


def _extract_procedure_sections(note_text: str) -> str:
    """Extract text from procedure-relevant sections for more accurate classification.

    Includes both the header line and the section content.
    """
    relevant_headers = [
        r"procedure[s]?",
        r"technique",
        r"description",
        r"operative\s+note",
        r"findings",
        r"intervention",
    ]

    extracted_parts = []
    lines = note_text.split("\n")
    in_relevant_section = False
    current_section_text = []

    # Match header with optional content after colon (e.g., "PROCEDURE: Thoracentesis")
    header_pattern = re.compile(
        r"^\s*(" + "|".join(relevant_headers) + r")\s*[:]\s*(.*)?$",
        re.IGNORECASE
    )
    # Match header-only lines (no content on same line)
    header_only_pattern = re.compile(
        r"^\s*(?:" + "|".join(relevant_headers) + r")\s*[:]*\s*$",
        re.IGNORECASE
    )
    any_header_pattern = re.compile(r"^\s*[A-Z][A-Z\s]{2,30}:\s*$")

    for line in lines:
        header_match = header_pattern.match(line)
        if header_match:
            # Save previous section if any
            if current_section_text:
                extracted_parts.append("\n".join(current_section_text))
            in_relevant_section = True
            current_section_text = []
            # Include the header line content (everything after the colon)
            content_after_colon = header_match.group(2)
            if content_after_colon and content_after_colon.strip():
                current_section_text.append(content_after_colon.strip())
        elif in_relevant_section and any_header_pattern.match(line):
            # Hit a different section header
            if current_section_text:
                extracted_parts.append("\n".join(current_section_text))
            in_relevant_section = False
            current_section_text = []
        elif in_relevant_section:
            current_section_text.append(line)

    # Don't forget last section
    if current_section_text:
        extracted_parts.append("\n".join(current_section_text))

    return "\n\n".join(extracted_parts) if extracted_parts else note_text


def _has_procedure_content(note_text: str) -> bool:
    """Check if note has actual procedure content vs just being a consult note."""
    procedural_verbs = [
        r"performed",
        r"inserted",
        r"placed",
        r"obtained",
        r"biopsied",
        r"sampled",
        r"advanced",
        r"visualized",
        r"inspected",
    ]
    lowered = note_text.lower()
    return any(re.search(verb, lowered) for verb in procedural_verbs)


class RegistryEngine:
    """Coordinates sectionization, LLM extraction, and record assembly."""

    def __init__(
        self,
        sectionizer: SectionizerService | None = None,
        llm_extractor: LLMDetailedExtractor | None = None,
    ) -> None:
        self.sectionizer = sectionizer or SectionizerService()
        self.llm_extractor = llm_extractor or LLMDetailedExtractor()

    def run(
        self,
        note_text: str,
        *,
        explain: bool = False,
        include_evidence: bool = True,
        schema_version: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> RegistryRecord | tuple[RegistryRecord, dict[str, list[Span]]]:
        record, _warnings = self.run_with_warnings(
            note_text,
            include_evidence=include_evidence,
            schema_version=schema_version,
            context=context,
        )
        if explain:
            return record, record.evidence
        return record

    def run_with_warnings(
        self,
        note_text: str,
        *,
        include_evidence: bool = True,
        schema_version: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[RegistryRecord, list[str]]:
        note_text = mask_offset_preserving(note_text)
        sections = self.sectionizer.sectionize(note_text)
        evidence: Dict[str, list[Span]] = {}
        seed_data: Dict[str, Any] = {}
        warnings: list[str] = []

        # Run deterministic extractors FIRST to seed commonly missed fields
        # These provide reliable extraction for demographics, ASA, sedation, etc.
        deterministic_data = run_deterministic_extractors(note_text)
        deterministic_evidence = None
        if isinstance(deterministic_data, dict):
            deterministic_evidence = deterministic_data.pop("evidence", None)
        seed_data.update(deterministic_data)
        if include_evidence and isinstance(deterministic_evidence, dict):
            for field, spans in deterministic_evidence.items():
                if not isinstance(field, str) or not isinstance(spans, list):
                    continue
                evidence.setdefault(field, []).extend([s for s in spans if isinstance(s, Span)])

        # Classify procedure families FIRST - this gates downstream extraction
        procedure_families = classify_procedure_families(note_text)
        seed_data["procedure_families"] = list(procedure_families)

        # Only extract EBUS station data if EBUS procedure family detected
        # This prevents hallucinated station "7" in CAO/rigid bronchoscopy cases
        station_list: list[str] = []
        station_spans: list[Span] = []
        if "EBUS" in procedure_families:
            station_list, station_spans = self._extract_linear_station_spans(note_text)

        mrn_match = re.search(r"MRN:?\s*(\w+)", note_text, re.IGNORECASE)
        if mrn_match:
            seed_data["patient_mrn"] = mrn_match.group(1)
            if include_evidence:
                evidence.setdefault("patient_mrn", []).append(
                    Span(text=mrn_match.group(0).strip(), start=mrn_match.start(), end=mrn_match.end())
                )

        llm_result = None
        llm_data: dict[str, Any] = {}
        raw_llm_evidence = None
        try:
            effective_context = dict(context) if context else {}
            if schema_version:
                effective_context.setdefault("schema_version", schema_version)
            llm_result = self.llm_extractor.extract(note_text, sections, context=effective_context)
        except TimeoutError as exc:
            warnings.append("REGISTRY_LLM_TIMEOUT_FALLBACK_TO_ENGINE")
            try:
                logger.warning("LLM timeout; falling back to deterministic engine extractors (%s)", exc)
            except Exception:
                pass
        except Exception as exc:
            warnings.append("REGISTRY_LLM_TIMEOUT_FALLBACK_TO_ENGINE")
            try:
                logger.warning("LLM extractor error; falling back to deterministic engine extractors (%s)", exc)
            except Exception:
                pass

        if llm_result is not None:
            llm_data_raw = llm_result.value
            llm_data = llm_data_raw if isinstance(llm_data_raw, dict) else {}

            # Treat empty dict/null as a degraded LLM extraction and proceed with seeded/heuristic extraction.
            if not llm_data:
                warnings.append("REGISTRY_LLM_TIMEOUT_FALLBACK_TO_ENGINE")

            # LLM responses sometimes return an "evidence" map with offsets only. Strip or
            # normalize them so pydantic validation does not fail when building the record.
            raw_llm_evidence = llm_data.pop("evidence", None) if isinstance(llm_data, dict) else None

            # Guardrail: billing/CPT codes are derived deterministically; do not trust LLM output for them.
            if isinstance(llm_data, dict):
                llm_data.pop("billing", None)
                llm_data.pop("cpt_codes", None)  # legacy flat CPT list (if present)
                llm_data.pop("coding_support", None)

        merged_data = self._merge_llm_and_seed(llm_data, seed_data)

        # Seed linear EBUS station list early so downstream normalization and heuristics can use it.
        if station_list and "EBUS" in procedure_families and not merged_data.get("linear_ebus_stations"):
            merged_data["linear_ebus_stations"] = station_list

        # Apply field-specific normalization/postprocessing before validation
        for field, func in POSTPROCESSORS.items():
            if field in merged_data:
                merged_data[field] = func(merged_data.get(field))

        # Apply heuristics for EBUS and new fields
        # Pass procedure_families to gate EBUS-specific extractions
        self._apply_ebus_heuristics(merged_data, note_text, procedure_families)
        self._apply_pleural_heuristics(merged_data, note_text, procedure_families)
        self._apply_bronchoscopy_therapeutics_heuristics(merged_data, note_text)
        self._apply_navigation_fiducial_heuristics(merged_data, note_text)

        # Clear bronch_tbbx fields for EBUS-only cases
        # EBUS-TBNA (transbronchial needle aspiration) is NOT the same as TBBx (transbronchial biopsy)
        # For pure EBUS staging, bronch_num_tbbx and bronch_tbbx_tool should be null
        self._apply_ebus_only_bronch_cleanup(merged_data, note_text, procedure_families)

        # Validate EBUS station fields: filter out any hallucinated stations not in the text
        if "EBUS" in procedure_families:
            if merged_data.get("linear_ebus_stations"):
                validated_stations = self._validate_station_mentions(note_text, merged_data["linear_ebus_stations"])
                merged_data["linear_ebus_stations"] = validated_stations if validated_stations else None
            if merged_data.get("ebus_stations_sampled"):
                valid_sampled = self._validate_station_mentions(note_text, merged_data["ebus_stations_sampled"])
                merged_data["ebus_stations_sampled"] = valid_sampled if valid_sampled else None
            if merged_data.get("ebus_stations_detail"):
                filtered_detail = []
                for entry in merged_data.get("ebus_stations_detail") or []:
                    station = entry.get("station")
                    if not station:
                        continue
                    if self._validate_station_mentions(note_text, [station]):
                        filtered_detail.append(entry)
                merged_data["ebus_stations_detail"] = filtered_detail if filtered_detail else None

        lowered_note = note_text.lower()

        # Defaults based on cross-field context
        # airway_type schema enum: ["Native", "ETT", "Tracheostomy", "LMA", "iGel"]
        # NOTE: "Rigid Bronchoscope" is NOT an airway_type - it's a procedure type.
        # For rigid bronchoscopy, airway is managed by the rigid scope itself.
        sedation_val = merged_data.get("sedation_type")
        airway_val = merged_data.get("airway_type")
        if airway_val in (None, "", []):
            # Skip airway_type inference for rigid bronchoscopy - the rigid scope is the airway
            if re.search(r"rigid\s+bronch", lowered_note):
                pass  # Leave airway_type as None for rigid bronchoscopy
            elif re.search(r"\bi-?gel\b", lowered_note):
                merged_data["airway_type"] = "iGel"
            elif re.search(r"\blma\b", lowered_note) or "laryngeal mask" in lowered_note:
                merged_data["airway_type"] = "LMA"
            elif re.search(r"\btrach(?:eostomy| tube| stoma)?\b", lowered_note):
                merged_data["airway_type"] = "Tracheostomy"
            elif re.search(r"\bett\b", lowered_note) or "endotracheal tube" in lowered_note or "intubated" in lowered_note:
                merged_data["airway_type"] = "ETT"
            elif sedation_val in ("Moderate", "Deep", "MAC"):
                merged_data["airway_type"] = "Native"

        # If pleural procedure present but no guidance, default to Blind; otherwise null out accidental guidance
        if not merged_data.get("pleural_procedure_type"):
            merged_data["pleural_guidance"] = None
        elif not merged_data.get("pleural_guidance"):
            merged_data["pleural_guidance"] = "Blind"

        # Ensure version is set if missing (Pydantic default might not trigger if key is missing in dict passed to **)
        if not merged_data.get("version"):
            merged_data["version"] = "0.5.0"

        # Pleural laterality and access site heuristics
        if merged_data.get("pleural_side") is None:
            if re.search(r"\bright (?:pleural )?effusion", lowered_note):
                merged_data["pleural_side"] = "Right"
            elif re.search(r"\bleft (?:pleural )?effusion", lowered_note):
                merged_data["pleural_side"] = "Left"
            elif re.search(r"\bright hemithorax", lowered_note):
                merged_data["pleural_side"] = "Right"
            elif re.search(r"\bleft hemithorax", lowered_note):
                merged_data["pleural_side"] = "Left"

        if merged_data.get("pleural_intercostal_space") is None:
            ics_match = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:intercostal\s*space|ics)", note_text, re.IGNORECASE)
            if ics_match:
                num = int(ics_match.group(1))
                if 10 <= num % 100 <= 20:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")
                formatted_ics = f"{num}{suffix}"
                merged_data["pleural_intercostal_space"] = formatted_ics
                merged_data.setdefault("intercostal_space", formatted_ics)
                if not merged_data.get("entry_location"):
                    window = note_text[max(0, ics_match.start() - 40) : ics_match.end() + 60]
                    loc_match = re.search(
                        r"(mid[-\\s]?axillary|anterior axillary|posterior axillary|midclavicular)",
                        window,
                        re.IGNORECASE,
                    )
                    if loc_match:
                        merged_data["entry_location"] = loc_match.group(1)

        # Filter out fields that don't apply to the detected procedure families
        # This prevents phantom data (e.g., EBUS fields when no EBUS performed)
        merged_data = filter_inapplicable_fields(merged_data, procedure_families)

        # Apply cross-field consistency checks
        # This ensures related fields are consistent (e.g., pneumothorax_intervention null when pneumothorax=false)
        merged_data = apply_cross_field_consistency(merged_data)

        nested_payload = build_nested_registry_payload(merged_data)

        # Apply enum normalization layer (gender, bronchus_sign, sedation, etc.)
        # This runs BEFORE API normalization to handle common LLM output variations
        nested_payload = normalize_registry_enums(nested_payload)

        # Apply normalization layer to clean up noisy LLM outputs before validation
        from app.api.normalization import normalize_registry_payload

        nested_payload = normalize_registry_payload(nested_payload)

        sanitization_warnings = _sanitize_string_enum_lists(nested_payload)
        if sanitization_warnings:
            warnings.extend(sanitization_warnings)
            for warning in sanitization_warnings:
                try:
                    logger.warning(warning)
                except Exception:
                    # Logging must never affect extraction correctness.
                    pass

        # Attempt to create RegistryRecord with better error handling
        try:
            record = RegistryRecord(**nested_payload)
        except Exception as e:
            # Provide more helpful error message for Pydantic validation errors
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                # Demo-friendly recovery: prune invalid fields and retry once.
                #
                # The LLM may emit values that are "close" but not schema-valid (e.g., lists where a
                # literal is expected). For demo and interactive UI usage, it's better to drop those
                # specific fields and return a partially-filled record than to fail the whole request.
                initial_errors = e.errors()
                pruned_payload = deepcopy(nested_payload)
                pruned_paths: list[str] = []
                error_summaries: list[str] = []

                def _null_out_path(obj: Any, loc: tuple[Any, ...] | list[Any]) -> None:
                    if not loc:
                        return
                    cur: Any = obj
                    for step in loc[:-1]:
                        if isinstance(step, int):
                            if isinstance(cur, list) and 0 <= step < len(cur):
                                cur = cur[step]
                            else:
                                return
                        else:
                            if isinstance(cur, dict) and step in cur:
                                cur = cur[step]
                            else:
                                return
                    last = loc[-1]
                    if isinstance(last, int):
                        if isinstance(cur, list) and 0 <= last < len(cur):
                            cur[last] = None
                        return
                    if isinstance(cur, dict) and last in cur:
                        # Prefer deletion to let defaults apply if available.
                        try:
                            del cur[last]
                        except Exception:
                            cur[last] = None

                for err in initial_errors:
                    loc = err.get("loc", [])
                    loc_tuple = tuple(loc) if isinstance(loc, (list, tuple)) else (loc,)
                    path_str = _format_payload_path(loc_tuple)
                    pruned_paths.append(path_str)
                    err_type = err.get("type") or "validation_error"
                    error_summaries.append(f"{path_str}: {err_type}")
                    _null_out_path(pruned_payload, loc_tuple)

                # If pruning sets list indices to None, strip them so enum-array validation can succeed.
                _drop_none_items_from_lists(pruned_payload)

                try:
                    record = RegistryRecord(**pruned_payload)
                except Exception as retry_exc:
                    # Second recovery: drop entire top-level sections that still fail.
                    from pydantic import ValidationError as _ValidationError

                    if isinstance(retry_exc, _ValidationError):
                        retry_errors = retry_exc.errors()
                        top_pruned_payload = deepcopy(pruned_payload)
                        top_keys_pruned: list[str] = []
                        retry_summaries: list[str] = []

                        for err in retry_errors:
                            loc = err.get("loc", [])
                            if not loc:
                                continue
                            retry_loc_tuple = tuple(loc) if isinstance(loc, (list, tuple)) else (loc,)
                            retry_path = _format_payload_path(retry_loc_tuple)
                            err_type = err.get("type") or "validation_error"
                            retry_summaries.append(f"{retry_path}: {err_type}")
                            top_key = loc[0]
                            if isinstance(top_key, str) and isinstance(top_pruned_payload, dict):
                                top_pruned_payload[top_key] = None

                                if top_key not in top_keys_pruned:
                                    top_keys_pruned.append(top_key)
                        try:
                            record = RegistryRecord(**top_pruned_payload)
                        except Exception as final_exc:
                            try:
                                logger.error(
                                    "RegistryRecord validation failed after pruning; returning empty record",
                                    extra={"error": str(final_exc)},
                                )
                            except Exception:
                                pass
                            record = RegistryRecord()
                        else:
                            message = (
                                "RegistryRecord validation required top-level pruning; returning partial record "
                                f"(errors={len(retry_errors)}). "
                                f"Top-level pruned: {_summarize_list(top_keys_pruned, max_items=5, sep=', ')}. "
                                f"Error summary: {_summarize_list(retry_summaries, max_items=3)}"
                            )
                            warnings.append(message)
                            try:
                                logger.warning(message)
                            except Exception:
                                pass
                    else:
                        try:
                            logger.error(
                                "RegistryRecord validation failed after pruning; returning empty record",
                                extra={"error": str(retry_exc)},
                            )
                        except Exception:
                            pass
                        record = RegistryRecord()
                else:
                    message = (
                        "RegistryRecord validation required pruning; returning partial record "
                        f"(errors={len(initial_errors)}). "
                        f"Pruned: {_summarize_list(pruned_paths, max_items=5, sep=', ')}. "
                        f"Error summary: {_summarize_list(error_summaries, max_items=3)}"
                    )
                    warnings.append(message)
                    try:
                        logger.warning(message)
                    except Exception:
                        pass
            else:
                try:
                    logger.error(
                        "RegistryRecord validation failed with non-validation error; returning empty record",
                        extra={"error": str(e)},
                    )
                except Exception:
                    pass
                record = RegistryRecord()
        
        normalized_evidence: dict[str, list[Span]] = {}
        if include_evidence:
            normalized_evidence = self._normalize_evidence(note_text, raw_llm_evidence)

            # Merge evidence gathered from regex seeding (e.g., MRN) with any usable LLM
            # evidence. The normalize helper already guards against malformed entries.
            for field, spans in evidence.items():
                normalized_evidence.setdefault(field, []).extend(spans)
            if station_spans:
                normalized_evidence.setdefault("linear_ebus_stations", []).extend(station_spans)

            # Validate evidence spans against source text to filter hallucinations
            normalized_evidence = validate_evidence_spans(note_text, normalized_evidence)

        record.evidence = {field: spans for field, spans in normalized_evidence.items()}
        return record, warnings

    def _extract_linear_station_spans(self, text: str) -> tuple[list[str], list[Span]]:
        """Extract linear EBUS station mentions and their spans from raw text."""
        # Require a boundary before the station number to avoid matching inside "12R" -> "2R"
        pattern = r"(?mi)(?:station\s*)?(?<![0-9A-Za-z])(2R|2L|4R|4L|7|10R|10L|11R|11L)(?:[sSiI])?\b\s*[:\-]?"
        stations: list[str] = []
        spans: list[Span] = []
        for match in re.finditer(pattern, text):
            station = match.group(1).upper()
            if station not in stations:
                stations.append(station)
            spans.append(Span(text=match.group(0).strip(), start=match.start(), end=match.end()))
        return stations, spans

    def _validate_station_mentions(self, text: str, stations: list[str]) -> list[str]:
        """Validate that each station in the list actually appears in the source text.

        This prevents hallucinated stations (like "7" appearing when only "11L" and "4R" exist).
        Station references must appear in a recognizable format in the text.
        """
        if not stations:
            return []

        validated = []
        for station in stations:
            station_upper = station.upper()
            # Build patterns that match the station in typical EBUS contexts
            # Be careful with station "7" which is just a number - require more context
            if station_upper == "7":
                # Station 7 needs EXPLICIT "station 7" with sampling context
                # This is more strict than before to prevent false positives from
                # random "7" occurrences (times, dates, ages, etc.)
                patterns = [
                    rf"(?m)^\s*7\s*[:\-]\s*[^\n]{{0,80}}(?:\d{{1,2}}\s*g\b|needle|pass|rose|sampl|biops)",  # "7: 22G, 4 passes"
                    rf"station\s+7\s*[:\-]",  # "station 7:" or "station 7-"
                    rf"station\s*7\s+(?:was\s+)?(?:sampl|biops|needle|pass|aspirat)",  # "station 7 was sampled"
                    rf"subcarinal\s+(?:lymph\s+)?node",  # "subcarinal node" (station 7 synonym)
                    rf"\b7\b[^.\n]{{0,40}}subcarinal",  # "7 (subcarinal) node"
                    rf"subcarinal[^.\n]{{0,40}}\b7\b",  # "subcarinal ... 7"
                    rf"(?:sampl|biops|needle|pass).{{0,30}}station\s*7\b",  # sampling context before station 7
                ]
            else:
                # For stations like 4R, 11L - more flexible matching
                # These are alphanumeric and unlikely to match random text
                patterns = [
                    rf"\b{re.escape(station_upper)}\b",
                    rf"station\s*{re.escape(station_upper)}",
                ]

            found = False
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found = True
                    break

            if found:
                validated.append(station_upper)

        return validated

    def _parse_ebus_station_sizes(self, text: str) -> list[tuple[str, float | None]]:
        """Parse station size mentions like 'station 11L (5.4mm)', '5.5 mm node at 4R', or '1.2 x 0.8 cm at 4R'."""
        results: list[tuple[str, float | None]] = []
        station_pattern = r"(2R|2L|4R|4L|7|10R|10L|11R|11L)"

        def _to_mm(val_str: str, unit: str | None) -> float | None:
            try:
                val = float(val_str)
            except Exception:
                return None
            if unit and unit.lower().startswith("c"):  # cm -> mm
                return val * 10
            return val

        # Station followed by single dimension
        patterns = [
            re.compile(rf"\b{station_pattern}\b\s*\(?\s*(\d+(?:\.\d+)?)\s*(mm|cm)\b", re.IGNORECASE),
            re.compile(rf"station\s*{station_pattern}\s*\(?\s*(\d+(?:\.\d+)?)\s*(mm|cm)\b", re.IGNORECASE),
            re.compile(rf"(\d+(?:\.\d+)?)\s*(mm|cm)[^.\n]{{0,60}}?\b{station_pattern}\b", re.IGNORECASE),
        ]

        # Patterns with two dimensions (short-axis = smaller number)
        dim_patterns = [
            re.compile(rf"\b{station_pattern}\b[^.\n]{{0,80}}?(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm)", re.IGNORECASE),
            re.compile(rf"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm)[^.\n]{{0,80}}?\b{station_pattern}\b", re.IGNORECASE),
        ]

        for pat in patterns:
            for match in pat.finditer(text):
                if pat.pattern.startswith("\\b(") or pat.pattern.startswith("station"):
                    station = match.group(1).upper()
                    size_val = _to_mm(match.group(2), match.group(3))
                else:
                    size_val = _to_mm(match.group(1), match.group(2))
                    station = match.group(3).upper()
                results.append((station, size_val))

        for pat in dim_patterns:
            for match in pat.finditer(text):
                # Depending on pattern order, station may be first or last
                if pat.pattern.startswith("\\b("):
                    station = match.group(1).upper()
                    dim1, dim2, unit = match.group(2), match.group(3), match.group(4)
                else:
                    dim1, dim2, unit = match.group(1), match.group(2), match.group(3)
                    station = match.group(4).upper()
                dims = [_to_mm(dim1, unit), _to_mm(dim2, unit)]
                dims_filtered = [d for d in dims if d is not None]
                size_val = min(dims_filtered) if dims_filtered else None
                results.append((station, size_val))

        # Deduplicate keeping first occurrence
        deduped: list[tuple[str, float | None]] = []
        seen: set[str] = set()
        for station, size in results:
            if station in seen:
                continue
            seen.add(station)
            deduped.append((station, size))
        return deduped

    def _parse_ebus_station_passes(self, text: str) -> dict[str, int]:
        """Parse per-station needle pass counts from the narrative."""
        lowered = text.lower()
        station_passes: dict[str, int] = {}
        station_pattern = r"(?<!\d)(2r|2l|4r|4l|7|10r|10l|11r|11l)\b"
        alphanumeric_station_pattern = r"(?<![a-z0-9])(2r|2l|4r|4l|10r|10l|11r|11l)\b"
        word_to_int = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }

        pass_count_pattern = r"\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten"
        pass_phrase_pattern = (
            rf"\b(?P<count>{pass_count_pattern})\b\s+"
            rf"(?:needle\s+)?(?:passes?|biops(?:y|ies)|samples?)\b"
        )

        def _parse_count(raw: str) -> int | None:
            if not raw:
                return None
            raw = raw.strip().lower()
            if raw.isdigit():
                return int(raw)
            return word_to_int.get(raw)

        def _set_passes(station_raw: str, count_raw: str) -> None:
            station = station_raw.strip().upper()
            count = _parse_count(count_raw)
            if count is None:
                return
            station_passes.setdefault(station, count)

        # Station-first patterns ("station 11L ... five passes", "11L: five passes")
        station_first_patterns: list[re.Pattern[str]] = [
            re.compile(rf"\bstation\s*(?P<station>{station_pattern})", re.IGNORECASE),
            re.compile(rf"(?P<station>{alphanumeric_station_pattern})", re.IGNORECASE),
            # Station 7 is ambiguous; only accept label-ish forms (e.g., "7: ...", "7 (subcarinal) ...")
            re.compile(r"(?<![a-z0-9])(?P<station>7)\b(?=\s*(?:[:\-]|\())", re.IGNORECASE),
        ]

        for pat in station_first_patterns:
            for match in pat.finditer(lowered):
                station = match.group("station")
                window = lowered[match.end() : match.end() + 160]
                pass_match = re.search(pass_phrase_pattern, window, re.IGNORECASE)
                if pass_match:
                    _set_passes(station, pass_match.group("count"))

        # Passes-first patterns ("five needle passes at station 11L", "5 passes ... 11L")
        for match in re.finditer(pass_phrase_pattern, lowered, re.IGNORECASE):
            count = match.group("count")
            window = lowered[match.end() : match.end() + 160]
            station_match = re.search(rf"\bstation\s*(?P<station>{station_pattern})", window, re.IGNORECASE)
            if station_match:
                _set_passes(station_match.group("station"), count)
                continue
            alpha_station_match = re.search(rf"(?P<station>{alphanumeric_station_pattern})", window, re.IGNORECASE)
            if alpha_station_match:
                _set_passes(alpha_station_match.group("station"), count)

        return station_passes

    def _infer_station_rose(self, text: str, station: str) -> str | None:
        """Infer station-specific ROSE result from nearby wording."""
        lowered = text.lower()
        priority = {
            "Malignant": 5,
            "Granuloma": 4,
            "Benign": 3,
            "Nondiagnostic": 2,
            "Atypical cells present": 1,
        }
        best_val: str | None = None

        # Prefer direct keyword-station associations (either order)
        keyword_map = [
            (r"malignan", "Malignant"),
            (r"granuloma", "Granuloma"),
            (r"non[-\\s]?diagnostic", "Nondiagnostic"),
            (r"benign", "Benign"),
            (r"atypical", "Atypical cells present"),
        ]
        for pattern, label in keyword_map:
            if re.search(rf"{pattern}[^.;\n]{{0,50}}{re.escape(station.lower())}", lowered):
                return label
            if re.search(rf"{re.escape(station.lower())}[^.;\n]{{0,50}}{pattern}", lowered):
                return label

        def _maybe_update(candidate: str | None) -> None:
            nonlocal best_val
            if not candidate:
                return
            if best_val is None or priority.get(candidate, 0) > priority.get(best_val, 0):
                best_val = candidate

        station_mentions = list(re.finditer(r"(station\s*)?(2r|2l|4r|4l|7|10r|10l|11r|11l)", lowered))
        for idx, match in enumerate(station_mentions):
            if match.group(2).upper() != station.upper():
                continue
            next_start = len(lowered)
            if idx + 1 < len(station_mentions):
                next_start = station_mentions[idx + 1].start()
            window = lowered[max(0, match.start() - 80) : min(next_start, match.end() + 120)]
            if "malignan" in window:
                _maybe_update("Malignant")
            if "benign" in window:
                _maybe_update("Benign")
            if "granuloma" in window:
                _maybe_update("Granuloma")
            if "non-diagnostic" in window or "nondiagnostic" in window:
                _maybe_update("Nondiagnostic")
            if "atypical" in window:
                _maybe_update("Atypical cells present")
        if "rose" in lowered:
            for segment in re.split(r"[.;]\s*", lowered):
                if "rose" not in segment:
                    continue
                if station.lower() in segment:
                    if "malignan" in segment:
                        _maybe_update("Malignant")
                    if "benign" in segment:
                        _maybe_update("Benign")
                    if "granuloma" in segment:
                        _maybe_update("Granuloma")
                    if "non-diagnostic" in segment or "nondiagnostic" in segment:
                        _maybe_update("Nondiagnostic")
                    if "atypical" in segment:
                        _maybe_update("Atypical cells present")
        return best_val

    def _apply_ebus_heuristics(
        self, data: dict[str, Any], text: str, procedure_families: Set[str] | None = None
    ) -> None:
        """Apply regex/keyword heuristics for EBUS, sedation reversal, and basic BLVR.

        EBUS-specific extractions are gated by procedure_families to prevent
        hallucinating EBUS data in non-EBUS procedures like CAO/rigid bronchoscopy.
        """
        lowered = text.lower()
        is_ebus_procedure = procedure_families is None or "EBUS" in procedure_families
        if not data.get("nav_platform"):
            # Use canonical schema values directly
            if re.search(r"\belectromagnetic navigation\b", lowered) or re.search(r"\bemn\b", lowered):
                data["nav_platform"] = "superDimension"
            elif re.search(r"\bion\b", lowered):
                data["nav_platform"] = "Ion"
            elif re.search(r"\bmonarch\b", lowered) or re.search(r"\bauris\b", lowered):
                data["nav_platform"] = "Monarch"

        if data.get("nav_rebus_used") is None:
            if "radial ebus" in lowered or "rebus" in lowered:
                data["nav_rebus_used"] = True
        if not data.get("sedation_type"):
            if "monitored anesthesia care" in lowered or re.search(r"\bmac\b", lowered):
                data["sedation_type"] = "Deep"
            elif "moderate sedation" in lowered:
                data["sedation_type"] = "Moderate"
            elif "deep sedation" in lowered:
                data["sedation_type"] = "Deep"
            elif "general anesthesia" in lowered:
                data["sedation_type"] = "General"
        
        # --- EBUS Heuristics (gated by procedure family) ---
        # Only extract EBUS-specific data if this is an EBUS procedure
        if is_ebus_procedure:
            station_order = {
                "2R": 0,
                "2L": 1,
                "4R": 2,
                "4L": 3,
                "7": 4,
                "10R": 5,
                "10L": 6,
                "11R": 7,
                "11L": 8,
            }

            def _sort_stations(stations: set[str]) -> list[str]:
                return sorted({s.upper() for s in stations if s}, key=lambda s: (station_order.get(s, 999), s))

            # ebus_scope_brand
            if "Olympus" in text and ("BF-UC" in text or "EBUS" in text):
                data["ebus_scope_brand"] = "Olympus"
            elif "Fujifilm" in text or "EB-530" in text or "Fuji" in text:
                data["ebus_scope_brand"] = "Fuji"
            elif "Pentax" in text or "EB-1970" in text:
                data["ebus_scope_brand"] = "Pentax"

            # ebus_stations_sampled
            stations_found = set()
            station_pattern = r"Station\s*(?<!\d)(2R|2L|4R|4L|7|10R|10L|11R|11L)\b"
            for match in re.finditer(station_pattern, text, re.IGNORECASE):
                snippet = text[match.end():match.end()+100].lower()
                if any(kw in snippet for kw in ["pass", "sample", "needle", "tbna", "aspirat"]):
                    if "not sampled" not in snippet:
                        stations_found.add(match.group(1).upper())
            if not stations_found and data.get("linear_ebus_stations"):
                # Fallback: if we have station numbers and the note indicates TBNA sampling,
                # treat the documented stations as sampled.
                lowered = text.lower()
                has_sampling = any(
                    kw in lowered
                    for kw in [
                        "tbna",
                        "transbronchial needle",
                        "needle aspiration",
                        "needle aspirat",
                        "needle pass",
                        "passes",
                    ]
                )
                if has_sampling:
                    stations_found.update(data.get("linear_ebus_stations") or [])
            if stations_found:
                data["ebus_stations_sampled"] = _sort_stations(stations_found)

            # Station-level detail (size, passes, rose_result) when present
            detail_entries = data.get("ebus_stations_detail") or []
            detail_by_station: dict[str, dict[str, Any]] = {d.get("station"): dict(d) for d in detail_entries if d.get("station")}
            station_pass_map = self._parse_ebus_station_passes(text)

            for station, size in self._parse_ebus_station_sizes(text):
                entry = detail_by_station.setdefault(station, {"station": station})
                if size is not None:
                    entry.setdefault("size_mm", size)

            for station, passes in station_pass_map.items():
                entry = detail_by_station.setdefault(station, {"station": station})
                entry.setdefault("passes", passes)

            stations_for_rose = set(detail_by_station.keys()) or stations_found
            for station in stations_for_rose:
                rose_val = self._infer_station_rose(text, station)
                if rose_val:
                    entry = detail_by_station.setdefault(station, {"station": station})
                    entry["rose_result"] = rose_val

            # Ensure each planned station has a detail entry (even if only station is known).
            for station in (data.get("linear_ebus_stations") or []):
                detail_by_station.setdefault(station, {"station": station})

            if detail_by_station:
                for entry in detail_by_station.values():
                    entry.setdefault("shape", None)
                    entry.setdefault("margin", None)
                    entry.setdefault("echogenicity", None)
                    entry.setdefault("chs_present", None)
                    entry.setdefault("appearance_category", None)
                    entry.setdefault("rose_result", entry.get("rose_result", None))
                data["ebus_stations_detail"] = [
                    detail_by_station[st]
                    for st in _sort_stations(set(detail_by_station.keys()))
                    if st in detail_by_station
                ]
                if data.get("ebus_stations_sampled"):
                    merged = set(data["ebus_stations_sampled"]) | set(detail_by_station.keys())
                    data["ebus_stations_sampled"] = _sort_stations(merged)
                else:
                    data["ebus_stations_sampled"] = _sort_stations(set(detail_by_station.keys()))

            # ebus_needle_gauge - expanded pattern to capture "22 gauge needle" formats
            # Schema enum: [19, 21, 22, 25] (integers, not strings)
            if not data.get("ebus_needle_gauge"):
                # Pattern 1: "22G" or "22-gauge" adjacent format
                gauge_match = re.search(r"\b(19|21|22|25)\s*[-]?\s*(?:G|gauge)\b", text, re.IGNORECASE)
                if gauge_match:
                    data["ebus_needle_gauge"] = int(gauge_match.group(1))
                else:
                    # Pattern 2: "22 gauge needle" with space
                    gauge_match2 = re.search(r"\b(19|21|22|25)\s+gauge\s+needle\b", text, re.IGNORECASE)
                    if gauge_match2:
                        data["ebus_needle_gauge"] = int(gauge_match2.group(1))

            # ebus_needle_type
            # Schema enum: ["Standard FNA", "Core/ProCore", "Acquire"]
            if any(kw in text for kw in ["FNB", "core biopsy", "ProCore"]):
                data["ebus_needle_type"] = "Core/ProCore"
            elif "Acquire" in text:
                data["ebus_needle_type"] = "Acquire"
            elif "needle" in text.lower() or "tbna" in text.lower():
                if data.get("ebus_needle_type") not in ["Core/ProCore", "Acquire"]:
                    data["ebus_needle_type"] = "Standard FNA"

            # ebus_systematic_staging
            if re.search(r"Systematic.*(evaluation|staging|N3)", text, re.IGNORECASE):
                if "No systematic" in text or "not systematic" in text.lower():
                    data["ebus_systematic_staging"] = False
                else:
                    data["ebus_systematic_staging"] = True
            elif "No systematic" in text:
                data["ebus_systematic_staging"] = False

            # ebus_rose_available
            if "ROSE" in text or "rapid on-site" in text.lower() or "rapid onsite" in lowered or "rapid on site" in lowered:
                data["ebus_rose_available"] = True
                if "ROSE not available" in text or "no ROSE" in text.lower():
                    data["ebus_rose_available"] = False

            # ebus_rose_result - derive from per-station ROSE results if available
            # Schema enum: ["Adequate - malignant", "Adequate - benign lymphocytes",
            #               "Adequate - granulomas", "Adequate - other", "Inadequate", "Not performed"]
            # Use priority: malignant > other (atypical) > granulomas > benign > inadequate
            if data.get("ebus_rose_available") and not data.get("ebus_rose_result"):
                # First, try to derive from ebus_stations_detail if we have per-station ROSE
                station_details = data.get("ebus_stations_detail") or []
                rose_results = [d.get("rose_result") for d in station_details if d.get("rose_result")]

                if rose_results:
                    # Priority-based aggregation using schema enum values
                    priority_order = [
                        "Adequate - malignant",
                        "Adequate - other",  # Covers atypical cases
                        "Adequate - granulomas",
                        "Adequate - benign lymphocytes",
                        "Inadequate",
                    ]
                    best_result = None
                    best_priority = len(priority_order)
                    for result in rose_results:
                        result_lower = result.lower() if isinstance(result, str) else ""
                        # Check each priority level
                        for idx, prio_val in enumerate(priority_order):
                            if prio_val.lower() == result_lower or prio_val == result:
                                if idx < best_priority:
                                    best_priority = idx
                                    best_result = prio_val
                                break
                            # Fuzzy match for legacy values
                            elif "malignant" in result_lower and idx == 0:
                                if idx < best_priority:
                                    best_priority = idx
                                    best_result = prio_val
                                break
                            elif ("atypical" in result_lower or "lymphoid" in result_lower) and idx == 1:
                                if idx < best_priority:
                                    best_priority = idx
                                    best_result = prio_val
                                break
                            elif "granulom" in result_lower and idx == 2:
                                if idx < best_priority:
                                    best_priority = idx
                                    best_result = prio_val
                                break
                            elif ("benign" in result_lower or "reactive" in result_lower) and idx == 3:
                                if idx < best_priority:
                                    best_priority = idx
                                    best_result = prio_val
                                break
                            elif ("inadequate" in result_lower or "nondiagnostic" in result_lower or "insufficient" in result_lower) and idx == 4:
                                if idx < best_priority:
                                    best_priority = idx
                                    best_result = prio_val
                                break
                    if best_result:
                        data["ebus_rose_result"] = best_result
                else:
                    # Fallback to text extraction if no per-station data
                    rose_snippets = []
                    for match in re.finditer(r"ROSE\b.*?(?::|-|is|shows|demonstrates|positive|negative)?\s*(.*?)(?:\n|\.|;)", text, re.IGNORECASE):
                        rose_snippets.append(match.group(1).lower())

                    combined_rose = " ".join(rose_snippets)
                    if "malignan" in combined_rose or "adenocarcinoma" in combined_rose or "squamous" in combined_rose or "tumor" in combined_rose or "carcinoma" in combined_rose:
                        data["ebus_rose_result"] = "Adequate - malignant"
                    elif "granuloma" in combined_rose:
                        data["ebus_rose_result"] = "Adequate - granulomas"
                    elif "lymphoma" in combined_rose or "lymphoid proliferation" in combined_rose or "atypical" in combined_rose:
                        data["ebus_rose_result"] = "Adequate - other"
                    elif "benign" in combined_rose or "reactive" in combined_rose or "lymphocytes" in combined_rose:
                        data["ebus_rose_result"] = "Adequate - benign lymphocytes"
                    elif "nondiagnostic" in combined_rose or "insufficient" in combined_rose or "inadequate" in combined_rose:
                        data["ebus_rose_result"] = "Inadequate"

            # ebus_intranodal_forceps_used
            if "intranodal forceps" in text.lower() or "ebus-ifb" in text.lower():
                data["ebus_intranodal_forceps_used"] = True

            # ebus_photodocumentation_complete
            if re.search(r"(Complete\s*)?(Photodocumentation|Photodoc|Photos).*(all.*(accessible.*)?stations|complete|taken|archived)|all.*stations.*photographed", text, re.IGNORECASE):
                data["ebus_photodocumentation_complete"] = True
            elif "photos all stations" in text.lower():
                data["ebus_photodocumentation_complete"] = True

            # ebus elastography
            if "elastograph" in lowered:
                if "no elastograph" in lowered or "without elastograph" in lowered:
                    data.setdefault("ebus_elastography_used", False)
                elif data.get("ebus_elastography_used") is None:
                    data["ebus_elastography_used"] = True

            if data.get("ebus_elastography_pattern") is None and data.get("ebus_elastography_used"):
                for sentence in re.split(r"[\n\.]", text):
                    if "elastograph" not in sentence.lower():
                        continue
                    match = re.search(r"elastograph\w*(?:\s*(?:pattern|patterns|score|shows|was|were|with|used)?[^:;\\-]*)[:,-]\s*([^.;\\n]+)", sentence, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        if candidate:
                            data["ebus_elastography_pattern"] = candidate
                            break
                    lowered_sentence = sentence.lower()
                    if any(color in lowered_sentence for color in ["blue", "green", "heterogeneous"]):
                        data["ebus_elastography_pattern"] = sentence.strip().rstrip(".;")
                        break

        # --- Sedation Reversal ---
        reversal_pattern = r"(Flumazenil|Naloxone|Narcan|Romazicon).*?(given|administered|IV)"
        reversal_match = re.search(reversal_pattern, text, re.IGNORECASE)
        if reversal_match:
            data["sedation_reversal_given"] = True
            agent = reversal_match.group(1).capitalize()
            if agent == "Narcan": agent = "Naloxone"
            if agent == "Romazicon": agent = "Flumazenil"
            data["sedation_reversal_agent"] = agent
        else:
            if "no reversal agents" in text.lower():
                 data["sedation_reversal_given"] = False
                 data["sedation_reversal_agent"] = None
            elif "reversal agents" in text.lower() and "administered" not in text.lower() and "given" not in text.lower():
                 if "no reversal" in text.lower() or "reversal agents: none" in text.lower() or "(x) none" in text.lower() or "reversal agents... available" in text.lower() or "at bedside" in text.lower():
                     data["sedation_reversal_given"] = False
                     data["sedation_reversal_agent"] = None

        # --- BLVR Heuristics (Basic) ---
        # valve_type schema enum: ["Zephyr (Pulmonx)", "Spiration (Olympus)"]
        if "valve" in text.lower():
            # Valve Type - use canonical schema values
            if "Zephyr" in text or "Pulmonx" in text:
                data["blvr_valve_type"] = "Zephyr (Pulmonx)"
            elif "Spiration" in text or "Olympus" in text:
                data["blvr_valve_type"] = "Spiration (Olympus)"

            # Target Lobe
            # Look for lobe mention near "valve" or "placed"
            # Simple check for lobe presence if not already set
            if not data.get("blvr_target_lobe"):
                if "left lower lobe" in text.lower() or "LLL" in text:
                    data["blvr_target_lobe"] = "LLL"
                elif "left upper lobe" in text.lower() or "LUL" in text:
                    data["blvr_target_lobe"] = "LUL"
                elif "right lower lobe" in text.lower() or "RLL" in text:
                    data["blvr_target_lobe"] = "RLL"
                elif "right middle lobe" in text.lower() or "RML" in text:
                    data["blvr_target_lobe"] = "RML"
                elif "right upper lobe" in text.lower() or "RUL" in text:
                    data["blvr_target_lobe"] = "RUL"

        # --- CAO (Central Airway Obstruction) Heuristics ---
        self._apply_cao_heuristics(data, text)

        # --- Disposition Heuristics ---
        if not data.get("disposition"):
            if re.search(r"\bpacu\b", lowered) or "post-anesthesia" in lowered or "recovery room" in lowered:
                data["disposition"] = "PACU Recovery"
            elif re.search(r"\bicu\b", lowered) or "intensive care" in lowered:
                data["disposition"] = "ICU Admission"
            elif "floor" in lowered and ("admit" in lowered or "transfer" in lowered):
                data["disposition"] = "Floor Admission"
            elif "discharge" in lowered and "home" in lowered:
                data["disposition"] = "Discharge Home"

    def _apply_bronchoscopy_therapeutics_heuristics(self, data: dict[str, Any], text: str) -> None:
        """Deterministically seed common therapeutic bronchoscopy actions.

        The extraction engine may run in stub/offline mode (or the LLM may fail). These
        heuristics ensure high-salience actions are reflected in procedures_performed
        when they are explicitly documented in the note.
        """
        lowered = text.lower()

        existing = data.get("procedures_performed")
        procedures: dict[str, Any]
        if isinstance(existing, dict):
            procedures = dict(existing)
        elif existing is None:
            procedures = {}
        else:
            # Unexpected type (e.g., narrative string). Leave untouched.
            return

        def _is_performed(obj: Any) -> bool:
            return isinstance(obj, dict) and obj.get("performed") is True

        def _set_if_missing(name: str, payload: dict[str, Any]) -> None:
            current = procedures.get(name)
            if _is_performed(current):
                return
            if isinstance(current, dict):
                merged = dict(current)
                for k, v in payload.items():
                    if merged.get(k) in (None, "", [], {}):
                        merged[k] = v
                procedures[name] = merged
            else:
                procedures[name] = payload

        # Conventional TBNA (31629) and transbronchial biopsy (31628) frequently appear
        # in sampling bullets; seed them explicitly when documented.
        tbna_trigger_re = re.compile(
            r"\b(?:tbna|transbronchial\s+needle\s+aspiration|transbronchial\s+needle)\b"
        )
        tbna_negation_re = re.compile(
            r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,60}\b(?:tbna|transbronchial\s+needle\s+aspiration|transbronchial\s+needle)\b"
        )
        ebus_context_re = re.compile(
            r"\b(?:ebus|endobronchial\s+ultrasound|convex\s+probe|ebus[-\s]?tbna)\b"
        )

        def _has_non_ebus_tbna(text_lower: str) -> bool:
            for match in tbna_trigger_re.finditer(text_lower):
                window = text_lower[max(0, match.start() - 160) : min(len(text_lower), match.end() + 160)]
                if tbna_negation_re.search(window):
                    continue

                # Treat TBNA mentions inside an EBUS paragraph as EBUS-TBNA.
                lookback_start = max(0, match.start() - 800)
                paragraph_break = text_lower.rfind("\n\n", lookback_start, match.start())
                if paragraph_break != -1:
                    lookback_start = paragraph_break + 2
                ebus_lookback = text_lower[lookback_start:match.start()]
                ebus_lookahead = text_lower[match.end() : min(len(text_lower), match.end() + 40)]
                if ebus_context_re.search(ebus_lookback) or ebus_context_re.search(ebus_lookahead):
                    continue
                return True
            return False

        if _has_non_ebus_tbna(lowered):
            _set_if_missing("tbna_conventional", {"performed": True})

        has_tblb = bool(re.search(r"\btblb\b", lowered) or re.search(r"transbronchial\s+lung\s+biops", lowered))
        has_cryo = bool(re.search(r"cryo\s*biops|cryobiops", lowered))
        biopsy_negated = bool(
            re.search(r"\b(?:no|not|without)\b[^.\n]{0,60}\b(?:tblb|transbronchial)\b", lowered)
            or re.search(r"\b(?:biops(?:y|ies)|cryobiops)\b[^.\n]{0,40}\bnot\b[^.\n]{0,40}\b(?:perform|done|obtain|take)\w*\b", lowered)
        )
        if (has_tblb or has_cryo) and not biopsy_negated:
            if has_cryo:
                _set_if_missing("transbronchial_cryobiopsy", {"performed": True})
            else:
                _set_if_missing("transbronchial_biopsy", {"performed": True})

        if "fluoroscopy" in lowered and data.get("fluoroscopy_used") is None:
            data["fluoroscopy_used"] = True

        # Therapeutic aspiration (31645)
        # Anchor on explicit phrase to avoid confusing suctioning with BAL.
        if re.search(r"\btherapeutic\s+aspiration\b", lowered) and not re.search(
            r"\b(?:no|not|without)\b[^.\n]{0,40}\btherapeutic\s+aspiration\b",
            lowered,
        ):
            aspiration: dict[str, Any] = {"performed": True}
            if "mucus" in lowered:
                aspiration["material"] = "Mucus plug"
            _set_if_missing("therapeutic_aspiration", aspiration)

        # Airway dilation (31630)
        # Require balloon + dilat* (or a known balloon brand) and avoid explicit negation.
        has_dilation = bool(
            re.search(r"\bmustang\s+balloon\b", lowered)
            or re.search(r"\bballoon\b[^.\n]{0,60}\bdilat", lowered)
            or re.search(r"\bdilat\w*\b[^.\n]{0,60}\bballoon\b", lowered)
        )
        dilation_negated = bool(
            re.search(r"\b(?:no|not|without)\b[^.\n]{0,60}\bdilat", lowered)
            or re.search(r"\bdilat\w*\b[^.\n]{0,30}\b(?:not|no)\b[^.\n]{0,30}\bperformed", lowered)
        )
        if has_dilation and not dilation_negated:
            dilation: dict[str, Any] = {"performed": True, "method": "Balloon"}
            size_match = re.search(r"\b(\d+(?:\.\d+)?)\s*mm\b[^.\n]{0,60}\bballoon\b", lowered)
            if size_match:
                try:
                    dilation["balloon_diameter_mm"] = float(size_match.group(1))
                except ValueError:
                    pass
            _set_if_missing("airway_dilation", dilation)

        # Airway stent removal (31638)
        removal_match = bool(
            re.search(
                r"\bstent\b[^.\n]{0,60}\b(?:remov\w*|retriev\w*|extract\w*|explant\w*|exchang\w*)",
                lowered,
            )
            or re.search(
                r"\b(?:remov\w*|retriev\w*|extract\w*|explant\w*|exchang\w*)\b[^.\n]{0,60}\bstent\b",
                lowered,
            )
        )
        removal_negated = bool(
            re.search(r"\b(?:no|not|without)\b[^.\n]{0,60}\bstent\b[^.\n]{0,60}\bremov\w*", lowered)
        )
        removal_history = bool(
            re.search(r"\bstent\b[^.\n]{0,40}\bremoved\b[^.\n]{0,40}\b(?:year|yr|month|day)s?\s+ago", lowered)
            or re.search(r"\b(?:history|prior|previous)\b[^.\n]{0,80}\bstent\b", lowered)
        )
        if removal_match and not removal_negated and not removal_history:
            stent_payload = {"performed": True, "action": "Removal", "airway_stent_removal": True}
            _set_if_missing("airway_stent", stent_payload)

        if procedures:
            data["procedures_performed"] = procedures

    def _apply_navigation_fiducial_heuristics(self, data: dict[str, Any], text: str) -> None:
        """Deterministically extract fiducial marker placement into granular navigation targets."""
        from app.registry.processing.navigation_fiducials import apply_navigation_fiducials

        apply_navigation_fiducials(data, text)

    def _apply_ebus_only_bronch_cleanup(
        self, data: dict[str, Any], text: str, procedure_families: Set[str]
    ) -> None:
        """Clear bronch_tbbx fields for EBUS-only staging cases.

        Per specification (§3 - Distinguishing EBUS vs bronchial biopsies):
        - EBUS-TBNA (TransBronchial Needle Aspiration) is NOT the same as TBBx (TransBronchial Biopsy)
        - bronch_*tbbx* fields are for parenchymal/mucosal transbronchial biopsy, not nodal TBNA
        - For pure EBUS staging cases without actual parenchymal/mucosal biopsy:
          - bronch_num_tbbx should be null
          - bronch_tbbx_tool should be null
          - bronch_guidance should be null (unless there's explicit guidance for a parenchymal target)
        - EBUS nodal aspirates belong in EBUS fields, not the parenchymal TBBx fields

        This method clears any bronch_tbbx fields that may have been hallucinated
        by the LLM confusing TBNA with TBBx.
        """
        # Only apply cleanup for EBUS-only cases
        # If other biopsy procedures are present, keep bronch fields
        biopsy_families = {"BIOPSY", "CRYO_BIOPSY", "CAO", "NAVIGATION"}

        # Check if this is an EBUS-only case (no other biopsy procedures)
        is_ebus_only = "EBUS" in procedure_families and not procedure_families.intersection(biopsy_families)

        if not is_ebus_only:
            return

        # Additional text-based validation: check if there's explicit TBBx/forceps biopsy
        # Even if procedure_families doesn't include BIOPSY, text might have it
        lowered = text.lower()
        has_actual_tbbx = any([
            re.search(r"transbronchial\s+(?:biops|forceps)", lowered),
            re.search(r"endobronchial\s+biops", lowered),
            re.search(r"forceps\s+biops", lowered),
            re.search(r"tbbx|tbb\b", lowered),
            re.search(r"parenchymal\s+biops", lowered),
            re.search(r"mucosal\s+biops", lowered),
        ])

        if has_actual_tbbx:
            return

        # Check for explicit parenchymal guidance (radial EBUS, fluoroscopy, EMN for lung nodule)
        has_parenchymal_guidance = any([
            re.search(r"radial\s+ebus\s+(?:guid|for|to)", lowered),
            re.search(r"fluoroscopy\s+(?:guid|for|to)\s+(?:nodule|lesion|mass)", lowered),
            re.search(r"emn\s+(?:guid|for|to)\s+(?:nodule|lesion|mass)", lowered),
            re.search(r"(?:nodule|lesion|mass)\s+(?:in|within|at)\s+(?:the\s+)?(?:lung|lobe)", lowered),
        ])

        # This is a pure EBUS-TBNA case - clear bronch_tbbx fields
        bronch_tbbx_fields = [
            "bronch_num_tbbx",
            "bronch_tbbx_tool",
        ]

        for field in bronch_tbbx_fields:
            if field in data and data[field] is not None:
                data[field] = None

        # Also clear bronch_guidance unless there's explicit parenchymal guidance
        # Per spec: bronch_guidance should be null unless there's explicit guidance for a parenchymal target
        if not has_parenchymal_guidance:
            if data.get("bronch_guidance") == "EBUS":
                # "EBUS" as guidance is incorrect for pure EBUS staging - that's for nodal sampling
                data["bronch_guidance"] = None

    def _apply_pleural_heuristics(
        self, data: dict[str, Any], text: str, procedure_families: Set[str] | None = None
    ) -> None:
        """Deterministic parsing for pleural procedures (thoracentesis, chest tubes, IPC).

        IMPORTANT: If procedure_families is set and doesn't include PLEURAL/THORACOSCOPY,
        this method actively clears any spurious pleural data that may have been hallucinated.
        """
        families = procedure_families or set()

        # If we have procedure families and PLEURAL/THORACOSCOPY is NOT among them,
        # actively clear any spurious pleural fields that LLM may have hallucinated
        if families and not families.intersection({"PLEURAL", "THORACOSCOPY"}):
            pleural_fields_to_clear = [
                "pleural_procedure_type", "pleural_side", "pleural_fluid_volume",
                "pleural_volume_drained_ml", "pleural_fluid_appearance", "pleural_guidance",
                "pleural_intercostal_space", "pleural_catheter_type", "pleural_pleurodesis_agent",
                "pleural_opening_pressure_measured", "pleural_opening_pressure_cmh2o",
                "pleural_thoracoscopy_findings", "pleurodesis_performed", "pleurodesis_agent",
            ]
            for field in pleural_fields_to_clear:
                if field in data:
                    data[field] = None
            return

        lowered = text.lower()

        # Procedure type (avoid overwriting explicit values)
        if not data.get("pleural_procedure_type"):
            if re.search(r"(medical\s+)?thoracoscopy|pleuroscopy", lowered):
                data["pleural_procedure_type"] = "Medical Thoracoscopy"
            elif "talc pleurodesis" in lowered or "chemical pleurodesis" in lowered:
                data["pleural_procedure_type"] = "Chemical Pleurodesis"
            elif re.search(r"(tunneled|tunnelled).*catheter|pleurx|aspira|ipc|indwelling pleural catheter", lowered):
                if re.search(r"exchange|replac", lowered):
                    data["pleural_procedure_type"] = "Tunneled Catheter Exchange"
                elif re.search(r"\bdrain(ed|age)?\b", lowered) and not re.search(r"\bplace|insert", lowered):
                    data["pleural_procedure_type"] = "IPC Drainage"
                else:
                    data["pleural_procedure_type"] = "Tunneled Catheter"
            elif re.search(r"chest\s+tube|pigtail", lowered):
                if re.search(r"remov", lowered):
                    data["pleural_procedure_type"] = "Chest Tube Removal"
                else:
                    data["pleural_procedure_type"] = "Chest Tube"
            elif re.search(r"thoracentesis|pleural tap", lowered):
                data["pleural_procedure_type"] = "Thoracentesis"

        if data.get("pleural_side") is None:
            side_match = re.search(r"\b(right|left)\b\s+(?:pleural\s+)?(?:effusion|thoracentesis|chest tube|hemithorax)", lowered)
            if side_match:
                side = side_match.group(1).lower()
                data["pleural_side"] = "Right" if side.startswith("r") else "Left"

        if data.get("pleural_guidance") is None and data.get("pleural_procedure_type"):
            if "ultrasound" in lowered or "sonograph" in lowered or "u/s" in lowered:
                data["pleural_guidance"] = "Ultrasound"
            elif "ct-guid" in lowered or "computed tomography" in lowered or re.search(r"\bct\b", lowered):
                data["pleural_guidance"] = "CT"

        if data.get("pleural_volume_drained_ml") is None:
            vol_match = re.search(r"(\d+(?:\.\d+)?)\s*(l|liter|litre|liters|litres)\b", lowered)
            ml_match = re.search(r"(\d{2,5})(?:\s*|\s*-?\s*)(?:ml|mL|cc)\b", text, re.IGNORECASE)
            if vol_match:
                try:
                    liters = float(vol_match.group(1))
                    data["pleural_volume_drained_ml"] = int(liters * 1000)
                except ValueError:
                    pass
            elif ml_match:
                try:
                    data["pleural_volume_drained_ml"] = int(ml_match.group(1))
                except ValueError:
                    pass

        if data.get("pleural_fluid_appearance") is None:
            appearance_map = {
                "serous": "Serous",
                "serosanguinous": "Serosanguinous",
                "sero-sanguinous": "Serosanguinous",
                "sanguinous": "Sanguinous",
                "bloody": "Sanguinous",
                "purulent": "Purulent",
                "pus": "Purulent",
                "chylous": "Chylous",
                "milky": "Chylous",
                "turbid": "Turbid",
            }
            for key, val in appearance_map.items():
                if key in lowered:
                    data["pleural_fluid_appearance"] = val
                    break

        if data.get("pleural_opening_pressure_cmh2o") is None:
            pressure_match = re.search(
                r"opening pressure\D{0,20}(\d+(?:\.\d+)?)\s*(?:cm\s*h2o|cmh2o)",
                lowered,
                re.IGNORECASE,
            )
            if pressure_match:
                try:
                    data["pleural_opening_pressure_cmh2o"] = float(pressure_match.group(1))
                    data["pleural_opening_pressure_measured"] = True
                except ValueError:
                    pass
            elif "opening pressure" in lowered and data.get("pleural_opening_pressure_measured") is None:
                data["pleural_opening_pressure_measured"] = True


    def _apply_cao_heuristics(self, data: dict[str, Any], text: str) -> None:
        """Apply regex/keyword heuristics for Central Airway Obstruction (CAO) procedures.

        Supports multi-site CAO extraction, storing detailed per-site interventions
        in cao_interventions array while also populating legacy flat fields for
        backwards compatibility.
        """
        lowered = text.lower()

        # Detect if this is a CAO/debulking procedure
        is_cao_procedure = any(kw in lowered for kw in [
            "debulk", "tumor debulk", "airway obstruction", "central airway",
            "recanalization", "recanaliz", "obstruct", "endobronchial tumor",
            "endobronchial lesion", "endobronchial mass", "airway tumor"
        ])

        if not is_cao_procedure:
            return

        # Extract multi-site CAO interventions
        cao_interventions = self._extract_cao_interventions(text)
        if cao_interventions:
            data["cao_interventions"] = cao_interventions

            # Populate legacy flat fields from the primary (most clinically significant) site
            primary_site = self._get_primary_cao_site(cao_interventions)
            if primary_site:
                if not data.get("cao_tumor_location"):
                    # Normalize location abbreviations to schema enum values
                    location = primary_site.get("location")
                    location_mapping = {
                        "BI": "Bronchus Intermedius",
                        "bi": "Bronchus Intermedius",
                        "distal_trachea": "Trachea",
                    }
                    data["cao_tumor_location"] = location_mapping.get(location, location)
                if not data.get("cao_obstruction_pre_pct") and primary_site.get("pre_obstruction_pct") is not None:
                    data["cao_obstruction_pre_pct"] = primary_site["pre_obstruction_pct"]
                if not data.get("cao_obstruction_post_pct") and primary_site.get("post_obstruction_pct") is not None:
                    data["cao_obstruction_post_pct"] = primary_site["post_obstruction_pct"]
                if not data.get("cao_primary_modality") and primary_site.get("modalities"):
                    # Map modality to schema enum
                    modality_mapping = {
                        "APC": "APC",
                        "apc": "APC",
                        "argon": "APC",
                        "cryo": "Cryotherapy",
                        "cryotherapy": "Cryotherapy",
                        "electrocautery": "Electrocautery",
                        "cautery": "Electrocautery",
                        "laser": "Laser",
                        "mechanical": "Mechanical Core",
                        "forceps": "Mechanical Core",
                        "rigid_core": "Mechanical Core",
                    }
                    for mod in primary_site["modalities"]:
                        mapped = modality_mapping.get(mod.lower(), mod)
                        if mapped in ("APC", "Cryotherapy", "Electrocautery", "Laser", "Mechanical Core", "Other"):
                            data["cao_primary_modality"] = mapped
                            break

        # Fallback to simple extraction if multi-site extraction didn't find anything
        if not cao_interventions:
            # --- cao_primary_modality ---
            if not data.get("cao_primary_modality"):
                if re.search(r"\bapc\b", lowered) or "argon plasma" in lowered:
                    data["cao_primary_modality"] = "APC"
                elif "cryotherapy" in lowered or "cryo" in lowered or "cryoprobe" in lowered:
                    data["cao_primary_modality"] = "Cryotherapy"
                elif "electrocautery" in lowered or "cautery" in lowered or "hot biopsy" in lowered:
                    data["cao_primary_modality"] = "Electrocautery"
                elif "laser" in lowered or "nd:yag" in lowered:
                    data["cao_primary_modality"] = "Laser"
                elif "microdebrider" in lowered:
                    data["cao_primary_modality"] = "Other"
                elif "forceps" in lowered or "mechanical" in lowered or "core out" in lowered:
                    data["cao_primary_modality"] = "Mechanical Core"

            # --- cao_tumor_location ---
            if not data.get("cao_tumor_location"):
                tumor_locations = []
                if re.search(r"\bbronchus intermedius\b", lowered) or re.search(r"\bBI\b", text):
                    tumor_locations.append("Bronchus Intermedius")
                if re.search(r"\bright mainstem\b", lowered) or re.search(r"\bRMS\b", text) or re.search(r"\bright main.?stem\b", lowered):
                    tumor_locations.append("RMS")
                if re.search(r"\bleft mainstem\b", lowered) or re.search(r"\bLMS\b", text) or re.search(r"\bleft main.?stem\b", lowered):
                    tumor_locations.append("LMS")
                if "trachea" in lowered:
                    tumor_locations.append("Trachea")
                if re.search(r"\bright upper lobe\b", lowered) or re.search(r"\bRUL\b", text):
                    tumor_locations.append("RUL")
                if re.search(r"\bright middle lobe\b", lowered) or re.search(r"\bRML\b", text):
                    tumor_locations.append("RML")
                if re.search(r"\bright lower lobe\b", lowered) or re.search(r"\bRLL\b", text):
                    tumor_locations.append("RLL")
                if re.search(r"\bleft upper lobe\b", lowered) or re.search(r"\bLUL\b", text):
                    tumor_locations.append("LUL")
                if re.search(r"\bleft lower lobe\b", lowered) or re.search(r"\bLLL\b", text):
                    tumor_locations.append("LLL")

                priority_order = ["Trachea", "RMS", "LMS", "Bronchus Intermedius", "RUL", "RML", "RLL", "LUL", "LLL"]
                for loc in priority_order:
                    if loc in tumor_locations:
                        data["cao_tumor_location"] = loc
                        break

        # --- cao_location (broader category) ---
        if not data.get("cao_location"):
            tumor_loc = data.get("cao_tumor_location")
            if tumor_loc == "Trachea":
                data["cao_location"] = "Trachea"
            elif tumor_loc in ("RMS", "LMS", "Mainstem"):
                data["cao_location"] = "Mainstem"
            elif tumor_loc in ("Bronchus Intermedius", "RUL", "RML", "RLL", "LUL", "LLL", "Lobar"):
                data["cao_location"] = "Lobar"

        # Extract biopsy sites for CAO/bronchoscopy procedures
        biopsy_sites = self._extract_biopsy_sites(text)
        if biopsy_sites:
            data["bronch_biopsy_sites"] = biopsy_sites
            # Set primary bronch_location_lobe from the most significant biopsy site
            if not data.get("bronch_location_lobe"):
                for site in biopsy_sites:
                    if site.get("lobe"):
                        data["bronch_location_lobe"] = site["lobe"]
                        break

    def _extract_cao_interventions(self, text: str) -> list[dict]:
        """Extract multi-site CAO intervention data from procedure text.

        Identifies distinct anatomic sites and extracts per-site:
        - pre_obstruction_pct
        - post_obstruction_pct
        - modalities used
        - contextual notes
        """
        interventions: list[dict] = []
        lowered = text.lower()

        # Define location patterns with their canonical names
        location_patterns = [
            (r"\bright middle lobe\b|\bRML\b", "RML"),
            (r"\bright lower lobe\b|\bRLL\b", "RLL"),
            (r"\bright upper lobe\b|\bRUL\b", "RUL"),
            (r"\bleft upper lobe\b|\bLUL\b", "LUL"),
            (r"\bleft lower lobe\b|\bLLL\b", "LLL"),
            (r"\bbronchus intermedius\b|\bBI\b", "BI"),
            (r"\bright mainstem\b|\bRMS\b|\bright main.?stem\b", "RMS"),
            (r"\bleft mainstem\b|\bLMS\b|\bleft main.?stem\b", "LMS"),
            (r"\bdistal trachea\b", "distal_trachea"),
            (r"\btrachea\b(?!\s+bifurc)", "Trachea"),
        ]

        # Modality patterns
        modality_patterns = [
            (r"\bapc\b|argon plasma", "APC"),
            (r"cryotherap|cryoprobe|\bcryo\b", "cryo"),
            (r"electrocauter|cautery|hot biopsy", "electrocautery"),
            (r"\blaser\b|nd:yag", "laser"),
            (r"forceps|mechanical|core.?out|rigid.{0,20}(?:debulk|shave)", "mechanical"),
            (r"balloon.{0,20}dilat|dilat.{0,20}balloon", "balloon"),
            (r"microdebrider", "microdebrider"),
        ]

        # Split text into sentences for context-aware extraction
        sentences = re.split(r"[.;]\s*", text)

        # Track which locations we've found
        found_locations: dict[str, dict] = {}

        # Track current context location for sentences without explicit location
        # (e.g., "LMS: 80% obstruction. Mechanical debulking performed. APC applied.")
        current_context_location: str | None = None

        for sentence in sentences:
            sent_lower = sentence.lower()

            # Find locations mentioned in this sentence
            locations_in_sentence = []
            for pattern, canonical in location_patterns:
                if re.search(pattern, sent_lower if canonical != "BI" else sentence, re.IGNORECASE):
                    locations_in_sentence.append(canonical)

            # Update context location if we found locations
            if locations_in_sentence:
                current_context_location = locations_in_sentence[0]

            # Find modalities mentioned in this sentence
            modalities_in_sentence = []
            for pattern, mod_name in modality_patterns:
                if re.search(pattern, sent_lower):
                    modalities_in_sentence.append(mod_name)

            # Extract obstruction percentages from this sentence
            pre_pct = None
            post_pct = None

            # "completely obstructed" or "100% obstruction"
            if "completely obstruct" in sent_lower or "total obstruct" in sent_lower:
                pre_pct = 100
            elif re.search(r"(\d{1,3})\s*(?:-\s*(\d{1,3}))?\s*%\s*(?:obstruct|occlu|stenosis|narrow|block)", sent_lower):
                match = re.search(r"(\d{1,3})\s*(?:-\s*(\d{1,3}))?\s*%\s*(?:obstruct|occlu|stenosis|narrow|block)", sent_lower)
                if match:
                    val1 = int(match.group(1))
                    val2 = int(match.group(2)) if match.group(2) else val1
                    pre_pct = max(val1, val2)

            # Recanalization percentage (e.g., "40% recanalization" means post = 60% obstruction)
            recan_match = re.search(r"(\d{1,3})\s*%\s*(?:recanaliz|patent|open)", sent_lower)
            if recan_match:
                patency = int(recan_match.group(1))
                if patency <= 100:
                    post_pct = 100 - patency

            # "complete recanalization" means 0% obstruction
            if "complete recanaliz" in sent_lower or "fully patent" in sent_lower:
                post_pct = 0

            # Post-procedure obstruction patterns (e.g., "Post-procedure: 20% obstruction")
            post_proc_match = re.search(
                r"(?:post[-\s]?(?:procedure|intervention|treatment|op)|final(?:ly)?|result(?:ing)?).{0,30}?(\d{1,3})\s*%\s*(?:obstruct|occlu|stenosis|narrow|block)",
                sent_lower
            )
            if post_proc_match and post_pct is None:
                post_pct = int(post_proc_match.group(1))

            # Also detect "improved to X% obstruction" as post
            improved_match = re.search(
                r"(?:improv|reduc|decreas).{0,30}?(\d{1,3})\s*%\s*(?:obstruct|occlu|stenosis|narrow|block)",
                sent_lower
            )
            if improved_match and post_pct is None:
                post_pct = int(improved_match.group(1))

            # If no explicit location in sentence but we have modalities/percentages,
            # associate with the current context location
            if not locations_in_sentence and current_context_location:
                if modalities_in_sentence or post_pct is not None:
                    locations_in_sentence = [current_context_location]

            # Associate data with locations
            for loc in locations_in_sentence:
                if loc not in found_locations:
                    found_locations[loc] = {
                        "location": loc,
                        "pre_obstruction_pct": None,
                        "post_obstruction_pct": None,
                        "modalities": [],
                        "notes": None,
                    }

                entry = found_locations[loc]

                # Update pre/post percentages (don't overwrite if already set)
                if pre_pct is not None and entry["pre_obstruction_pct"] is None:
                    entry["pre_obstruction_pct"] = pre_pct
                if post_pct is not None and entry["post_obstruction_pct"] is None:
                    entry["post_obstruction_pct"] = post_pct

                # Add modalities
                for mod in modalities_in_sentence:
                    if mod not in entry["modalities"]:
                        entry["modalities"].append(mod)

        # Convert to list
        interventions = list(found_locations.values())
        return interventions

    def _get_primary_cao_site(self, interventions: list[dict]) -> dict | None:
        """Select the primary (most clinically significant) CAO site.

        Priority is based on:
        1. Most proximal (central) location
        2. Highest pre-procedure obstruction percentage
        """
        if not interventions:
            return None

        # Priority order (most proximal first)
        priority_order = [
            "Trachea", "distal_trachea", "RMS", "LMS",
            "BI", "Bronchus Intermedius",
            "RUL", "RML", "RLL", "LUL", "LLL"
        ]

        def sort_key(site: dict) -> tuple:
            loc = site.get("location", "")
            try:
                loc_priority = priority_order.index(loc)
            except ValueError:
                loc_priority = len(priority_order)
            # Secondary: highest pre_obstruction (use negative so higher values sort first)
            pre_pct = site.get("pre_obstruction_pct") or 0
            return (loc_priority, -pre_pct)

        sorted_sites = sorted(interventions, key=sort_key)
        return sorted_sites[0] if sorted_sites else None

    def _extract_biopsy_sites(self, text: str) -> list[dict]:
        """Extract multiple biopsy site locations from procedure text.

        Supports non-lobar locations like "distal trachea", "carina", etc.
        """
        biopsy_sites: list[dict] = []
        lowered = text.lower()

        # Only extract if biopsy-related keywords present
        if not any(kw in lowered for kw in ["biopsy", "biopsies", "biopsied", "sampled", "specimens"]):
            return []

        # Location patterns for biopsies
        location_patterns = [
            (r"distal\s+trachea", "distal_trachea", None),
            (r"proximal\s+trachea", "proximal_trachea", None),
            (r"carina", "carina", None),
            (r"right\s+mainstem|RMS", "RMS", None),
            (r"left\s+mainstem|LMS", "LMS", None),
            (r"bronchus\s+intermedius|BI\b", "BI", None),
            (r"right\s+upper\s+lobe|RUL", "RUL", "RUL"),
            (r"right\s+middle\s+lobe|RML", "RML", "RML"),
            (r"right\s+lower\s+lobe|RLL", "RLL", "RLL"),
            (r"left\s+upper\s+lobe|LUL", "LUL", "LUL"),
            (r"left\s+lower\s+lobe|LLL", "LLL", "LLL"),
        ]

        # Find biopsy mentions with locations
        # Pattern: "biopsies ... from/of [location]" or "[location] ... biopsied"
        for pattern, location, lobe in location_patterns:
            # Check for location near biopsy keywords
            context_patterns = [
                rf"biops\w*\s+(?:were\s+)?(?:taken\s+|obtained\s+)?from\s+[^.]*?{pattern}",
                rf"{pattern}[^.]*?biops\w*",
                rf"from\s+(?:the\s+)?{pattern}[^.]*?(?:and|,)",
            ]
            for ctx_pat in context_patterns:
                if re.search(ctx_pat, lowered, re.IGNORECASE):
                    site_entry = {
                        "location": location,
                        "lobe": lobe,
                        "segment": None,
                        "specimens_count": None,
                    }
                    # Check if this location is already added
                    if not any(s["location"] == location for s in biopsy_sites):
                        biopsy_sites.append(site_entry)
                    break

        return biopsy_sites

    @staticmethod
    def _merge_llm_and_seed(llm_data: dict[str, Any], seed_data: dict[str, Any]) -> dict[str, Any]:
        """Merge deterministic seed data into LLM output.

        Deterministic extractors are intended as high-precision "safety nets" and
        must not be dropped simply because the LLM returned a non-empty parent dict
        (e.g., procedures_performed without the key we need).
        """

        def _merge_missing(dst: Any, src: Any) -> Any:
            # If destination is empty-ish, prefer the source.
            if dst in (None, "", [], {}):
                return src

            # Deterministic seeds should be able to flip a boolean performed flag to True.
            if isinstance(dst, bool) and isinstance(src, bool):
                return dst or src

            # Deep-merge dicts, filling missing/empty values recursively.
            if isinstance(dst, dict) and isinstance(src, dict):
                merged_dict = dict(dst)
                for k, v in src.items():
                    if v is None:
                        continue
                    if k not in merged_dict:
                        merged_dict[k] = v
                        continue
                    merged_dict[k] = _merge_missing(merged_dict.get(k), v)
                return merged_dict

            # If destination is an empty list and source is non-empty, take it.
            if isinstance(dst, list) and isinstance(src, list):
                return src if not dst and src else dst

            return dst

        merged: dict[str, Any] = dict(llm_data)
        for key, value in seed_data.items():
            if value is None:
                continue
            merged[key] = _merge_missing(merged.get(key), value)
        return merged

    @staticmethod
    def _normalize_evidence(note_text: str, raw_evidence: Any) -> dict[str, list[Span]]:
        """Convert loose evidence payloads into Span objects.

        The LLM can emit evidence as dicts with start/end offsets but no text, or even
        as a single dict instead of a list. We defensively coerce these into the
        expected ``dict[str, list[Span]]`` shape and drop anything malformed.
        """

        normalized: dict[str, list[Span]] = {}
        if not isinstance(raw_evidence, dict):
            return normalized

        for field, spans in raw_evidence.items():
            span_candidates = spans if isinstance(spans, list) else [spans]
            for span_data in span_candidates:
                if not isinstance(span_data, dict):
                    continue

                start = span_data.get("start") or span_data.get("start_offset")
                end = span_data.get("end") or span_data.get("end_offset")
                if start is None or end is None:
                    continue

                try:
                    start_i = int(start)
                    end_i = int(end)
                except (TypeError, ValueError):
                    continue

                text = span_data.get("text")
                if text is None:
                    text = note_text[start_i:end_i]

                section = span_data.get("section")
                confidence = span_data.get("confidence")

                normalized.setdefault(field, []).append(
                    Span(text=str(text), start=start_i, end=end_i, section=section, confidence=confidence)
                )

        return normalized


__all__ = [
    "RegistryEngine",
    "classify_procedure_families",
    "filter_inapplicable_fields",
    "validate_evidence_spans",
    "PROCEDURE_FAMILIES",
    "FIELD_APPLICABLE_TAGS",
]
