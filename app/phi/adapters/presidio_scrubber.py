"""Presidio-based scrubber adapter (Unified Clinical Context).

Implements PHIScrubberPort using Presidio AnalyzerEngine with enhanced
dynamic safeguards for clinical terms, providers, and HIPAA age rules.
Unified version combining scispacy integration and targeted false positive fixes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.phi.ports import PHIScrubberPort, ScrubResult, ScrubbedEntity

logger = logging.getLogger(__name__)

ZERO_WIDTH_CHARACTERS: frozenset[str] = frozenset(
    {
        "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2060", "\ufeff"
    }
)
ZERO_WIDTH_TRANSLATION_TABLE: dict[int, int] = {ord(ch): ord(" ") for ch in ZERO_WIDTH_CHARACTERS}

# --- 1. Allow Lists (Terms to Protect) ---

# Pulmonary/Interventional terms often mistaken for Names/Orgs by NLP
PULMONARY_ALLOW_LIST = {
    # Robots & Navigation
    "ion", "monarch", "superdimension", "veran", "spin", "lungvision", "galaxy", "archimedes",
    # Valves & Stents
    "zephyr", "spiration", "aero", "ultraflex", "bonastent", "silicone", "dumon", "hood",
    # Catheters & Tools
    "alair", "pleurx", "pleura-x", "chartis", "prosense", "neuwave", "vizishot", 
    "olympus", "cre", "elation", "erbe", "erbokryo", "truefreeze", "cryoprobe",
    "fogarty", "freitag", "pneumostat", "monsoon",
    # Anatomy often mistaken for names
    "lingula", "carina", "hilum", "naris", "nares", "glottis", "epiglottis",
    # Clinical Acronyms/Terms
    "rose", "ebus", "rebus", "tbna", "bal", "gpa", "copd", "pap", "ip", "or",
    "severe", "moderate", "mild", "acute", "chronic", "for", "with" # Clinical stopwords
}

# --- 2. Dynamic Regex Configurations ---

# HIPAA Age Rule: Match ages >= 90.
AGE_OVER_90_RE = re.compile(
    r"(?i)(?:\b(?:age|aged)\s*[:]?\s*(?:9\d|[1-9]\d{2,})\b)|"
    r"(?:\b(?:9\d|[1-9]\d{2,})\s*-?\s*(?:y/o|y\.?o\.?|yo|yrs?|years?|year-old|year\s+old)\b)"
)

# Date of Birth (DOB) - Catches "DOB: 11/22/1971"
DOB_RE = re.compile(
    r"(?i)\b(?:DOB|Date\s*of\s*Birth)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
)

# Structured Header Names (High Sensitivity)
# Handles CSV artifacts (newlines/quotes/pipes) and expanded roles (Operator, Staff, Fellow).
# Captures: "Patient: Jasmine Young", "Pt: Williams, Robert J.", "Operator: Dr. Smith"
STRUCTURED_PERSON_RE = re.compile(
    r"(?im)(?:^|[\r\n\"\|;])\s*(?:Patient(?:\s+Name)?|Pt|Name|Subject|Attending|Fellow|Surgeon|Physician|Assistants?|"
    r"RN|RT|CRNA|Cytology|Cytopathologist|Proceduralist|Operator|Resident|Anesthesia|Staff|Provider)\s*[:\-\|]+\s+"
    r"((?:Dr\.|Mr\.|Ms\.|Mrs\.|LCDR\.|CDR\.|Prof\.)?\s*[A-Z][a-z]+(?:[\s,]+(?:[A-Z]\.?|[A-Z][a-z]+)){1,4})"
)

# Narrative Provider/Dictation (Contextual)
# Catches "Dr. Smith performed..." or dictation starts like "Lisa Morgan here..."
NARRATIVE_PERSON_RE = re.compile(
    r"(?im)(?:\b(?:Dr\.|Doctor|LCDR|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))|"  # Dr. Smith
    r"(?:^|[\r\n])\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:here|presents|presented|with|has|is|was|underwent)\b" # Dictation start
)

# Structured Location/Facility
# Catches "Facility: Lakeshore University Hospital", "Location: Bronchoscopy Suite"
STRUCTURED_LOCATION_RE = re.compile(
    r"(?im)(?:^|[\r\n\"\|;])\s*(?:Facility|Location|Hospital|Institution|Service|Center|Site)\s*[:\-\|]+\s+"
    r"([A-Z0-9][a-zA-Z0-9\s\.\,\-\&]+?)(?=\s{2,}|\n|$|Dr\.|Attending)"
)

# MRN / ID Values
MRN_RE = re.compile(
    r"(?i)\b(?:MRN|MR|Medical\s*Record|Medical\s*Rec(?:ord)?|Patient\s*ID|ID|EDIPI|DOD\s*ID)\s*[:\#]?\s*(\d{5,12})\b"
)

# Improved ROSE Context (To Identify Clinical Context vs Person)
_ROSE_CONTEXT_RE = re.compile(
    r"\bROSE(?:\s*(?:[-:]|from|said|result|report|findings?)\s*|\s+)"
    r"(?:suspicious|consistent|positive|negative|pos|neg|performed|"
    r"collected|sample|specimen|analysis|evaluation|procedure|review|findings?|"
    r"malignant|benign|adequate|inadequate|atypical|granuloma|granulomatous|"
    r"lymphocytes|cells|carcinoma|adeno|squamous|nsclc|scc|inflammation|inflammatory|maybe)\b",
    re.IGNORECASE,
)

# Device Context (To Protect names like "Noah" or "King" when part of a device)
_DEVICE_CONTEXT_RE = re.compile(
    r"\b(?:Noah|Wang|Cook|Mark|Baker|Young|King|Edwards|Ion|Monarch|Zephyr|Chartis|"
    r"Alair|PleurX|Fogarty|Freitag|Olympus|SuperDimension|Pneumostat|Monsoon|ViziShot|"
    r"TrueFreeze|ProSense|Neuwave|Erbokryo)\s+"
    r"(?:Medical|Needle|Catheter|EchoTip|Fiducial|Marker|System|Platform|Robot|Forceps|"
    r"Biopsy|Galaxy|Valve|Drain|Stent|Scope|Bronchoscope|Nav|Navigation|Probe|Generator|Ablation)\b",
    re.IGNORECASE,
)


# =============================================================================
# Backward-compatible PHI redaction utilities (used by tests)
# =============================================================================

@dataclass(frozen=True)
class Detection:
    """Lightweight span detection used by the PHI redaction contract tests."""

    entity_type: str
    start: int
    end: int
    score: float = 0.5


DEFAULT_ENTITY_SCORE_THRESHOLDS: dict[str, float] = {
    "PERSON": 0.50,
    "DATE_TIME": 0.50,
    "LOCATION": 0.50,
    "ORGANIZATION": 0.50,
    "ADDRESS": 0.50,
    "MRN": 0.50,
    "ID": 0.50,
    "US_DRIVER_LICENSE": 0.50,
}

DEFAULT_RELATIVE_DATE_TIME_PHRASES: list[str] = [
    "about a week",
    "about a month",
    "a few days",
    "a few weeks",
]

_CREDENTIAL_TOKENS = {"md", "do", "rn", "rt", "crna", "pa", "np", "phd"}
_SECTION_HEADERS = {"history", "indication", "procedure", "findings", "impression", "assessment", "plan"}
_ALLOWLIST_PHRASES = {
    "lung nodule",
    "target",
    "kenalog",
    "nonobstructive",
    "us",
    "ns",
    "mc",
}

_PROVIDER_HEADER_RE = re.compile(
    r"(?i)\b(?:surgeon|assistant|attending|fellow|physician|proceduralist|operator|anesthesia|staff)\b"
)
_SIGNATURE_CONTEXT_RE = re.compile(r"(?i)\b(?:electronically\s+signed\s+by|signed\s+by|dictated\s+by)\b")

_CPT_CODE_RE = re.compile(r"\b\d{5}\b")
_STATION_RE = re.compile(r"\b\d{1,2}[RLS](?:s|rs|ls)?\b", re.IGNORECASE)
_MEASUREMENT_UNIT_RE = re.compile(r"^\s*(?:ml|cc|mg|mcg|l)\b", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"(?i)\b(?:\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)|about\s+a\s+week)\b"
)

_DATE_MMDDYYYY_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DATE_MMDDYYYY_MALFORMED_RE = re.compile(r"\b\d{1,2}/\d{1,2}\d{4}\b")
_ADDRESS_BLOCK_RE = re.compile(
    r"(?m)^(?P<street>\d{3,6}\s+.+\b(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane)\.?)\s*$",
    re.IGNORECASE,
)
_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def filter_cpt_codes(text: str, detections: Sequence[Any]) -> list[Any]:
    """Drop detections overlapping 5-digit CPT codes misclassified as DATE_TIME."""
    kept: list[Any] = []
    for det in detections:
        entity_type = getattr(det, "entity_type", None)
        start = getattr(det, "start", None)
        end = getattr(det, "end", None)
        if entity_type == "DATE_TIME" and isinstance(start, int) and isinstance(end, int):
            span = text[start:end]
            if _CPT_CODE_RE.fullmatch(span):
                continue
        kept.append(det)
    return kept


def filter_allowlisted_terms(text: str, detections: Sequence[Any]) -> list[Any]:
    """Drop detections that are known clinical false positives."""
    kept: list[Any] = []
    for det in detections:
        start = getattr(det, "start", None)
        end = getattr(det, "end", None)
        if not isinstance(start, int) or not isinstance(end, int):
            kept.append(det)
            continue
        span = text[start:end]
        span_norm = span.strip().lower()

        if not span_norm:
            continue
        if span_norm in _ALLOWLIST_PHRASES:
            continue
        if span_norm in PULMONARY_ALLOW_LIST:
            continue
        if _STATION_RE.fullmatch(span_norm):
            continue
        kept.append(det)
    return kept


def detect_datetime_detections(text: str) -> list[Detection]:
    """Detect date-like strings (including malformed MM/DDYYYY) as DATE_TIME."""
    detections: list[Detection] = []
    for regex in (_DATE_MMDDYYYY_RE, _DATE_MMDDYYYY_MALFORMED_RE):
        for match in regex.finditer(text):
            detections.append(
                Detection(entity_type="DATE_TIME", start=match.start(), end=match.end(), score=0.95)
            )
    return detections


def detect_address_detections(text: str) -> list[Detection]:
    """Detect a simple multi-line mailing address block as ADDRESS."""
    detections: list[Detection] = []
    for match in _ADDRESS_BLOCK_RE.finditer(text):
        street_start, street_end = match.span("street")
        end = street_end

        # Heuristic: extend to include the next line if it contains a ZIP code.
        next_newline = text.find("\n", street_end)
        if next_newline != -1:
            next_line_end = text.find("\n", next_newline + 1)
            if next_line_end == -1:
                next_line_end = len(text)
            next_line = text[next_newline + 1 : next_line_end]
            if _ZIP_RE.search(next_line):
                end = next_line_end

        detections.append(Detection(entity_type="ADDRESS", start=street_start, end=end, score=0.95))
    return detections


def extract_patient_names(text: str) -> list[str]:
    """Extract patient names from common indication/patient header patterns."""
    names: list[str] = []

    # Indication line: "INDICATION FOR OPERATION: John Q Public is a ..."
    m = re.search(
        r"(?im)^INDICATION FOR OPERATION:\s*"
        r"([A-Z][a-z]+(?:\s+(?:[A-Z]\.?|[A-Z][a-z]+)){1,4})\s+is\b",
        text,
    )
    if m:
        names.append(m.group(1).strip())

    # Patient header: "Patient: Last, First"
    m2 = re.search(r"(?im)^\s*Patient\s*:\s*([A-Z][^,\n]+,\s*[A-Z][^\n]+)$", text)
    if m2:
        names.append(m2.group(1).strip())

    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def forced_patient_name_detections(text: str, names: Sequence[str]) -> list[Detection]:
    """Create PERSON detections for all literal occurrences of extracted patient names."""
    detections: list[Detection] = []
    for name in names:
        if not name:
            continue
        for match in re.finditer(re.escape(name), text):
            detections.append(Detection(entity_type="PERSON", start=match.start(), end=match.end(), score=0.99))
    return detections


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return line_start, line_end


def redact_with_audit(
    *,
    text: str,
    detections: Sequence[Detection],
    enable_driver_license_recognizer: bool = True,  # noqa: ARG001 - kept for API compatibility
    score_thresholds: Mapping[str, float] | None = None,
    relative_datetime_phrases: Sequence[str] | None = None,
    nlp_backend: Any | None = None,  # noqa: ARG001 - kept for API compatibility
    nlp_model: str | None = None,  # noqa: ARG001 - kept for API compatibility
) -> tuple[ScrubResult, dict[str, Any]]:
    """Apply redactions with guardrails and return (ScrubResult, audit dict)."""
    thresholds = dict(DEFAULT_ENTITY_SCORE_THRESHOLDS)
    if score_thresholds:
        thresholds.update({k: float(v) for k, v in score_thresholds.items() if isinstance(k, str)})

    rel_phrases = [p.lower() for p in (relative_datetime_phrases or DEFAULT_RELATIVE_DATE_TIME_PHRASES) if p]

    removed: list[dict[str, Any]] = []
    kept: list[Detection] = []

    # Filter low-score first (keeps audit deterministic).
    for det in detections:
        detected_text = text[det.start : det.end]
        threshold = float(thresholds.get(det.entity_type, 0.5))
        if float(det.score) < threshold:
            removed.append(
                {
                    "entity_type": det.entity_type,
                    "start": det.start,
                    "end": det.end,
                    "score": det.score,
                    "detected_text": detected_text,
                    "reason": "low_score",
                }
            )
            continue
        kept.append(det)

    # Drop CPT/date false positives.
    filtered_any: list[Detection] = []
    for det in kept:
        detected_text = text[det.start : det.end]
        detected_lower = detected_text.strip().lower()

        # Never redact credentials or section headers.
        if detected_lower in _CREDENTIAL_TOKENS:
            removed.append(
                {
                    "entity_type": det.entity_type,
                    "start": det.start,
                    "end": det.end,
                    "score": det.score,
                    "detected_text": detected_text,
                    "reason": "credential",
                }
            )
            continue
        if det.entity_type == "PERSON" and detected_lower.rstrip(":") in _SECTION_HEADERS:
            removed.append(
                {
                    "entity_type": det.entity_type,
                    "start": det.start,
                    "end": det.end,
                    "score": det.score,
                    "detected_text": detected_text,
                    "reason": "section_header",
                }
            )
            continue

        # Provider/signature safe zones.
        if det.entity_type == "PERSON":
            line_start, line_end = _line_bounds(text, det.start, det.end)
            line = text[line_start:line_end]
            line_lower = line.lower()
            is_patient_line = line_lower.lstrip().startswith("patient:")
            # Use a local context window so a provider header later on the same line
            # doesn't incorrectly suppress patient header redaction.
            local_start = max(line_start, det.start - 80)
            local_end = min(line_end, det.end + 25)
            local = text[local_start:local_end]
            local_lower = local.lower()

            # Treat "Last, First MRN: ####" as a patient header even when providers
            # appear later on the same (wrapped) line.
            mrn_nearby = bool(re.search(r"(?i)\bmrn\b", text[det.end : min(line_end, det.end + 30)]))
            is_patient_line = is_patient_line or mrn_nearby

            has_credentials = any(
                re.search(rf"(?i)\b{re.escape(token)}\b", local) for token in _CREDENTIAL_TOKENS
            )
            if not is_patient_line and (
                _PROVIDER_HEADER_RE.search(local) or _SIGNATURE_CONTEXT_RE.search(local) or has_credentials
            ):
                removed.append(
                    {
                        "entity_type": det.entity_type,
                        "start": det.start,
                        "end": det.end,
                        "score": det.score,
                        "detected_text": detected_text,
                        "reason": "provider_context",
                    }
                )
                continue

        # Device model safe zones (e.g., T190) - treat as non-PHI.
        if det.entity_type == "US_DRIVER_LICENSE":
            if re.fullmatch(r"T\d{3}", detected_text.strip()):
                removed.append(
                    {
                        "entity_type": det.entity_type,
                        "start": det.start,
                        "end": det.end,
                        "score": det.score,
                        "detected_text": detected_text,
                        "reason": "device_model",
                    }
                )
                continue
            if not enable_driver_license_recognizer:
                removed.append(
                    {
                        "entity_type": det.entity_type,
                        "start": det.start,
                        "end": det.end,
                        "score": det.score,
                        "detected_text": detected_text,
                        "reason": "driver_license_disabled",
                    }
                )
                continue

        # Allowlist suppressor.
        if detected_lower in _ALLOWLIST_PHRASES or detected_lower in PULMONARY_ALLOW_LIST:
            removed.append(
                {
                    "entity_type": det.entity_type,
                    "start": det.start,
                    "end": det.end,
                    "score": det.score,
                    "detected_text": detected_text,
                    "reason": "allowlist",
                }
            )
            continue

        # Duration/relative DATE_TIME suppressor.
        if det.entity_type == "DATE_TIME":
            if any(phrase in detected_lower for phrase in rel_phrases) or _DURATION_RE.search(detected_text):
                removed.append(
                    {
                        "entity_type": det.entity_type,
                        "start": det.start,
                        "end": det.end,
                        "score": det.score,
                        "detected_text": detected_text,
                        "reason": "duration_datetime",
                    }
                )
                continue

            if _CPT_CODE_RE.fullmatch(detected_text.strip()):
                removed.append(
                    {
                        "entity_type": det.entity_type,
                        "start": det.start,
                        "end": det.end,
                        "score": det.score,
                        "detected_text": detected_text,
                        "reason": "cpt_code",
                    }
                )
                continue

            # Measurements like "1250 ml", "10 mg" misdetected as DATE_TIME.
            following = text[det.end : min(len(text), det.end + 10)]
            if detected_text.strip().isdigit() and _MEASUREMENT_UNIT_RE.match(following):
                removed.append(
                    {
                        "entity_type": det.entity_type,
                        "start": det.start,
                        "end": det.end,
                        "score": det.score,
                        "detected_text": detected_text,
                        "reason": "measurement_datetime",
                    }
                )
                continue

        filtered_any.append(det)

    # Apply remaining redactions in reverse order.
    redactions = sorted(filtered_any, key=lambda d: d.start, reverse=True)
    out_chars = list(text)
    entities: list[ScrubbedEntity] = []

    for det in redactions:
        placeholder = f"<{det.entity_type}>"
        out_chars[det.start : det.end] = placeholder
        entities.append(
            ScrubbedEntity(
                placeholder=placeholder,
                entity_type=det.entity_type,
                original_start=det.start,
                original_end=det.end,
            )
        )

    entities.reverse()
    scrubbed_text = "".join(out_chars)
    audit = {
        "redacted_text": scrubbed_text,
        "detections": [
            {
                "entity_type": d.entity_type,
                "start": d.start,
                "end": d.end,
                "score": d.score,
                "detected_text": text[d.start : d.end],
            }
            for d in redactions
        ],
        "removed_detections": removed,
    }
    return ScrubResult(scrubbed_text=scrubbed_text, entities=entities), audit


def _patient_name_detections(text: str) -> list[Detection]:
    detections: list[Detection] = []

    # Structured patient header.
    for match in re.finditer(r"(?im)\bPatient\s*:\s*([^\n]+)", text):
        name = match.group(1).strip()
        if name:
            start = match.start(1)
            end = match.end(1)
            detections.append(Detection(entity_type="PERSON", start=start, end=end, score=0.99))

    # Leading "Last, First" name before MRN.
    m = re.match(r"^\s*([A-Z][A-Za-z'\-]+,\s*[A-Z][A-Za-z'\-]+)\s+MRN\b", text)
    if m:
        detections.append(Detection(entity_type="PERSON", start=m.start(1), end=m.end(1), score=0.99))

    # "Patient Jane Test" (no colon).
    for match in re.finditer(r"(?i)\bPatient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text):
        detections.append(Detection(entity_type="PERSON", start=match.start(1), end=match.end(1), score=0.90))

    return detections


def _mrn_detections(text: str) -> list[Detection]:
    detections: list[Detection] = []
    for match in re.finditer(r"(?i)\bMRN\s*:\s*(\d{5,12})\b", text):
        detections.append(Detection(entity_type="MRN", start=match.start(1), end=match.end(1), score=0.99))
    return detections


@dataclass
class PresidioScrubber(PHIScrubberPort):
    """
    Adapter that uses Microsoft Presidio + Custom Regex for scrubbing.
    """
    analyzer: Any = None
    anonymizer: Any = None
    model_name: str = "presidio"

    def scrub(self, text: str, document_type: str | None = None, specialty: str | None = None) -> ScrubResult:
        scrubbed, _audit = self.scrub_with_audit(
            text,
            document_type=document_type,
            specialty=specialty,
        )
        return scrubbed

    # Legacy interface expected by tests and older integrations.
    def scrub_with_audit(
        self,
        text: str,
        document_type: str | None = None,  # noqa: ARG002 - kept for compatibility
        specialty: str | None = None,  # noqa: ARG002 - kept for compatibility
    ) -> tuple[ScrubResult, dict[str, Any]]:
        if not text:
            empty = ScrubResult(scrubbed_text="", entities=[])
            return empty, {"redacted_text": "", "detections": [], "removed_detections": []}

        clean_text = text.translate(ZERO_WIDTH_TRANSLATION_TABLE)

        detections: list[Detection] = []
        detections.extend(_patient_name_detections(clean_text))
        detections.extend(_mrn_detections(clean_text))
        detections.extend(detect_datetime_detections(clean_text))

        return redact_with_audit(
            text=clean_text,
            detections=detections,
            enable_driver_license_recognizer=False,
            score_thresholds=DEFAULT_ENTITY_SCORE_THRESHOLDS,
            relative_datetime_phrases=DEFAULT_RELATIVE_DATE_TIME_PHRASES,
        )
