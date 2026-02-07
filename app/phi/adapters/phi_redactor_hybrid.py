"""
Hybrid Regex + DistilBERT NER PHI Redaction for Interventional Pulmonology Procedural Notes
==========================================================================================

Combines rule-based regex patterns with ML-based entity recognition (DistilBERT NER)
for comprehensive PHI de-identification while preserving clinically relevant information.

Design Philosophy:
1. REGEX FIRST: Fast, deterministic rules for structured PHI patterns
2. ML SECOND: NER catches contextual PHI that regex misses
3. PROTECT LAYER: Whitelist patterns that should never be redacted

Author: Russell (IP Physician / Developer)
"""

import json
import os
import re
import uuid
import logging
import argparse
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION & ENTITY TYPES
# =============================================================================

class RedactionAction(Enum):
    REDACT = "REDACT"
    PROTECT = "PROTECT"
    REVIEW = "REVIEW"


@dataclass
class Detection:
    """Represents a detected entity span."""
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float
    source: str  # "regex" or "ner"
    action: RedactionAction = RedactionAction.REDACT


@dataclass
class RedactionConfig:
    """Configuration for PHI redaction behavior."""
    redact_patient_names: bool = True
    redact_mrn: bool = True
    redact_dob: bool = True
    redact_procedure_dates: bool = True  # Set False if dates needed for analysis
    redact_facilities: bool = True
    redact_ages_over_89: bool = True  # HIPAA rule
    protect_physician_names: bool = True
    protect_device_names: bool = True
    protect_anatomical_terms: bool = True
    ner_threshold: float = 0.5


# =============================================================================
# 1. PROTECTION PATTERNS (ALLOW LISTS) - Applied FIRST
# =============================================================================

# --- Physician/Provider Role Headers ---
PHYSICIAN_TITLES = (
    r"(?:Dr\.|Doctor|Attending|Fellow|Surgeon|Physician|Pulmonologist|"
    r"Anesthesiologist|Oncologist|Radiologist|Pathologist|Cytopathologist|"
    r"Assistants?|MD|DO|RN|RT|CRNA|PA|NP|Operator|Staff|Proctored\s+by|"
    r"Supervising|Resident|Intern|Chief|Director)"
)

PHYSICIAN_HEADER_RE = re.compile(
    fr"(?im)^[\t ]*(?:{PHYSICIAN_TITLES})\s*[:\-]?\s+"
    r"((?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+(?:[\s,]+(?:[A-Z]\.?|[A-Z][a-z]+)){0,4})",
    re.MULTILINE
)

# Inline physician mentions
INLINE_PHYSICIAN_RE = re.compile(
    r"(?i)\b(?:Dr\.|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)

# Procedure performed by / Electronically signed by
SIGNATURE_RE = re.compile(
    r"(?im)(?:performed\s+(?:under\s+)?(?:direct\s+)?supervision\s+of|"
    r"electronically\s+(?:signed|attested)|procedure\s+performed\s+by|"
    r"dictated\s+by|reviewed\s+by|cosigned\s+by)\s*[:\-]?\s*"
    r"((?:Dr\.|Mr\.|Ms\.)?\s*[A-Z][a-z]+(?:[\s,]+(?:[A-Z]\.?|[A-Z][a-z]+)){0,3})",
    re.IGNORECASE
)

# Standalone physician name with credentials (prevents Location false positives)
# Matches "Maria Santos, MD" or "John Smith MD" or "Name, MD (title)"
PHYSICIAN_CREDENTIAL_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z]'?[A-Za-z]+)?),?\s*(?:MD|DO|RN|CRNA|PA|NP|PhD)\b"
    r"(?:\s*\([^)]+\))?",
    re.IGNORECASE
)

# --- ROSE Context (Rapid On-Site Evaluation) ---
ROSE_CONTEXT_RE = re.compile(
    r"\bROSE(?:\s*[:\-]?\s*|\s+)(?:suspicious|consistent|positive|negative|pos|neg|"
    r"performed|collected|sample|specimen|analysis|evaluation|procedure|review|"
    r"findings?|malignant|benign|adequate|inadequate|atypical|granuloma|"
    r"granulomatous|lymphocytes|cells|carcinoma|adeno|squamous|nsclc|scc|"
    r"small\s+cell|non-?small\s+cell|cytopathologist|present|available|result)",
    re.IGNORECASE
)

# --- Medical Device Manufacturers (Ambiguous Names) ---
DEVICE_MANUFACTURERS = {
    "noah", "wang", "cook", "mark", "baker", "young", "king", "edwards",
    "olympus", "boston", "stryker", "intuitive", "auris", "fujifilm", 
    "pentax", "medtronic", "merit", "conmed", "erbe", "karl storz"
}

DEVICE_CONTEXT_RE = re.compile(
    r"\b(?:Noah|Wang|Cook|Mark|Baker|Young|King|Edwards|Olympus|Boston|Stryker|"
    r"Intuitive|Auris|Fujifilm|Pentax|Medtronic|Merit|Conmed|Erbe|Karl\s+Storz)\s+"
    r"(?:Medical|Needle|Catheter|EchoTip|Fiducial|Marker|System|Platform|Robot|"
    r"Forceps|Biopsy|Galaxy|Scientific|Surgical|Healthcare|Endoscopy|bronchoscope|scope)",
    re.IGNORECASE
)

# Specific device/system names that look like person names
PROTECTED_DEVICE_NAMES = {
    "ion", "monarch", "galaxy", "superdimension", "illumisite", "lungvision",
    "archimedes", "spin", "veran", "inreach", "chartis", "zephyr", "spiration",
    "pleurx", "aspira", "dumon", "ultraflex", "alair", "emprint", "neuwave"
}

# --- Robotic Platform Pattern ---
ROBOTIC_PLATFORM_RE = re.compile(
    r"\b(?:Ion|Monarch|Galaxy)\s+(?:robotic\s+)?(?:bronchoscopy\s+)?(?:system|platform|robot|catheter)",
    re.IGNORECASE
)

# --- Anatomical Terms (commonly flagged as locations/names) ---
ANATOMICAL_TERMS = {
    # Lobes and segments
    "rul", "rml", "rll", "lul", "lll", "lingula", "lingular",
    "right upper lobe", "right middle lobe", "right lower lobe",
    "left upper lobe", "left lower lobe", "upper lobe", "lower lobe", "middle lobe",
    "apical", "apicoposterior", "anterior", "posterior", "superior", "inferior",
    "medial", "lateral", "basal", "basilar",
    # Segments (B1-B10)
    "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10",
    "b1+2", "b7+8", "rb4", "rb5", "rb6", "lb4", "lb5", "lb6",
    # Airways
    "trachea", "carina", "bronchus", "bronchi", "mainstem", "main stem",
    "bronchus intermedius", "segmental", "subsegmental",
    "right mainstem", "left mainstem", "rms", "lms",
    # Lymph node stations
    "station", "stations", "4r", "4l", "7", "10r", "10l", "11r", "11l",
    "2r", "2l", "1r", "1l", "subcarinal", "paratracheal", "hilar",
    # Other anatomy
    "vocal cords", "glottis", "subglottis", "epiglottis", "larynx",
    "pleura", "pleural", "mediastinum", "mediastinal", "hilum"
}

# --- Clinical/Procedure Terms Often Misidentified ---
CLINICAL_ALLOW_LIST = {
    # ROSE results
    "rose", "rapid on-site evaluation", "adequate lymphocytes", "atypical cells",
    "granuloma", "suspicious for malignancy", "malignant", "benign",
    # Pathology terms
    "adenocarcinoma", "squamous cell carcinoma", "nsclc", "scc", "sclc",
    "small cell carcinoma", "non-small cell", "carcinoid",
    # Procedures
    "ebus", "tbna", "bal", "tbbx", "bronchoscopy", "thoracoscopy",
    "rigid bronchoscopy", "flexible bronchoscopy", "navigation bronchoscopy",
    "radial ebus", "linear ebus", "endobronchial ultrasound",
    # Tools
    "forceps", "needle", "catheter", "scope", "probe", "basket", "snare",
    "cryoprobe", "apc", "microdebrider", "stent", "balloon",
    # Common abbreviations
    "ga", "ett", "asa", "npo", "nkda", "ebl", "ptx", "cxr", "cbct"
    # medicationa
    "lidocaine", "fentanyl", "midazolam", "versed", "propofol", "epinephrine",
}


# =============================================================================
# 2. REDACTION PATTERNS (PHI to Remove)
# =============================================================================

# --- Patient Header Pattern ---
# Handles: "Patient: Thompson, Margaret A.", "Pt: John Smith", "Patient Name: Virginia Richardson"
# Uses greedy matching to capture full names - order alternatives from longest to shortest
PATIENT_HEADER_RE = re.compile(
    r"(?im)^[\t ]*(?:Patient(?:\s+Name)?|Pt|Name|Subject)\s*[:\-]?\s*"
    r"("
    r"[A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?"  # Last, First M. format
    r"|"
    r"[A-Z][a-z]+\s+[A-Z]'?[A-Za-z]+"  # First O'Last format
    r"|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"  # First Last (multiple words) - GREEDY
    r"|"
    r"[A-Z][a-z]+\s+[A-Z]\."  # First M. format
    r"|"
    r"[A-Z][a-z]+"  # Single name (fallback)
    r")",
    re.MULTILINE
)

# --- MRN / ID Patterns ---
MRN_RE = re.compile(
    r"(?i)\b(?:MRN|MR|Medical\s*Record|Patient\s*ID|ID|EDIPI|DOD\s*ID)\s*"
    r"[:\#]?\s*([A-Z0-9\-]{4,15})\b"
)

# --- Date Patterns ---
# Full dates (MM/DD/YYYY, YYYY-MM-DD, etc.)
DATE_FULL_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|"  # MM/DD/YYYY or MM-DD-YYYY
    r"\d{4}[/\-]\d{1,2}[/\-]\d{1,2}|"     # YYYY-MM-DD
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"  # Month DD, YYYY
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}"    # DD Month YYYY
    r")\b",
    re.IGNORECASE
)

# DOB specifically
DOB_RE = re.compile(
    r"(?i)\b(?:DOB|Date\s*of\s*Birth|Birth\s*Date|Born)\s*[:\-]?\s*"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"
)

# --- Age Patterns (HIPAA: ages > 89 must be aggregated) ---
AGE_HIPAA_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:age|aged)\s*[:\-]?\s*(9\d|[1-9]\d{2,})\b|"  # age: 90+
    r"\b(9\d|[1-9]\d{2,})\s*-?\s*(?:y/?o|yo|yrs?|years?|year[\s-]old)\b"  # 90 y/o, 90-year-old
    r")"
)

# Regular age mentions (keep but track)
AGE_RE = re.compile(
    r"(?i)\b(\d{1,3})\s*-?\s*(?:y/?o|yo|yrs?|years?\s*old|year[\s-]old)\b"
)

# --- Facility / Location Patterns ---
FACILITY_RE = re.compile(
    r"(?im)^[\t ]*(?:Facility|Location|Hospital|Institution|Site|Center)\s*[:\-]?\s*"
    r"(.+?)(?:\n|$)",
    re.MULTILINE
)

# City, State pattern - require it to be preceded by comma or common location indicators
# Avoid matching "Name, MD" pattern by requiring state abbreviation to NOT be followed by common suffixes
CITY_STATE_RE = re.compile(
    r"(?:(?:,\s+)|(?:in\s+)|(?:at\s+)|(?:Location[:\s]+)|(?:Facility[:\s]+)|(?:Hospital[:\s]+))?"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s*"
    r"(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MA|MI|MN|MS|"  # Note: MD removed
    r"MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)"
    r"(?:\s+\d{5}(?:-\d{4})?)?\b"  # Optional ZIP code
)

# Special pattern for Maryland locations (to distinguish from ", MD" credential)
# Only matches when there's a ZIP code or when preceded by explicit location marker
MD_STATE_RE = re.compile(
    r"(?:(?:Location[:\s]+)|(?:Facility[:\s]+)|(?:Hospital[:\s]+)|(?:,\s+[A-Z][a-z]+,\s+))"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*MD"
    r"(?:\s+\d{5}(?:-\d{4})?)?\b"
)

# --- Phone Numbers ---
PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# --- SSN Pattern ---
SSN_RE = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)

# --- Email Pattern ---
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)


# =============================================================================
# 3. NER CONFIGURATION
# =============================================================================

# Map NER labels to redaction actions
NER_LABEL_ACTIONS = {
    # Redact these (standard PHI labels)
    "GIVENNAME": RedactionAction.REDACT,
    "SURNAME": RedactionAction.REDACT,
    "USERNAME": RedactionAction.REDACT,
    "FIRSTNAME": RedactionAction.REDACT,
    "LASTNAME": RedactionAction.REDACT,
    "PATIENT": RedactionAction.REDACT,
    "STREETNAME": RedactionAction.REDACT,
    "BUILDINGNUM": RedactionAction.REDACT,
    "ZIPCODE": RedactionAction.REDACT,
    "CITY": RedactionAction.REDACT,
    "GEO": RedactionAction.REDACT,
    "DATE": RedactionAction.REDACT,
    "SSN": RedactionAction.REDACT,
    "PHONE": RedactionAction.REDACT,
    "EMAIL": RedactionAction.REDACT,
    "MRN": RedactionAction.REDACT,
    "ID": RedactionAction.REDACT,
    "PASSWORD": RedactionAction.REDACT,
    
    # Protect these (provider/clinical context)
    "PROVIDER": RedactionAction.PROTECT,
    "DOCTOR": RedactionAction.PROTECT,
    "PHYSICIAN": RedactionAction.PROTECT,
}


# =============================================================================
# 4. CORE REDACTION ENGINE
# =============================================================================

class PHIRedactor:
    """
    Hybrid PHI redactor combining regex patterns with a DistilBERT NER model.

    Pipeline:
    1. Identify protection zones (physicians, devices, anatomy)
    2. Apply regex patterns for structured PHI
    3. Apply NER for contextual PHI
    4. Resolve overlaps (protection wins)
    5. Apply redactions
    """

    def __init__(self, config: Optional[RedactionConfig] = None, use_ner_model: bool = True):
        self.config = config or RedactionConfig()
        self.use_ner_model = use_ner_model
        self.ner_pipeline = None

        if use_ner_model:
            self._load_ner_model()

    def _load_ner_model(self):
        """Load the PHI NER model with a local-first policy."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
            from pathlib import Path
        except ImportError:
            logger.warning("Transformers not installed - using regex-only mode")
            self.use_ner_model = False
            return

        model_id = os.getenv("PHI_NER_MODEL_ID")
        model_path = Path(os.getenv("PHI_NER_MODEL_DIR", "artifacts/phi_distilbert_ner"))

        if model_id:
            try:
                logger.info("Loading PHI NER model from HuggingFace: %s", model_id)
                model = AutoModelForTokenClassification.from_pretrained(model_id)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=-1,  # CPU by default
                )
                logger.info("Loaded PHI NER model from HuggingFace: %s", model_id)
                return
            except Exception as exc:
                logger.warning("Could not load from HuggingFace (%s), trying local path...", exc)

        if model_path.exists():
            try:
                model = AutoModelForTokenClassification.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                )
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=-1,  # CPU by default
                )
                logger.info("Loaded PHI NER model from local path: %s", model_path)
                return
            except Exception as exc:
                logger.warning("Could not load PHI NER model from local path: %s - using regex-only mode", exc)
        else:
            logger.warning("PHI NER model not found at %s - using regex-only mode", model_path)

        self.use_ner_model = False
    
    def _find_protection_zones(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Identify spans that should NOT be redacted.
        Returns list of (start, end, reason) tuples.
        """
        protected = []
        
        # Physician headers
        for match in PHYSICIAN_HEADER_RE.finditer(text):
            protected.append((match.start(), match.end(), "physician_header"))
        
        # Inline physicians (Dr. Smith)
        for match in INLINE_PHYSICIAN_RE.finditer(text):
            protected.append((match.start(), match.end(), "inline_physician"))
        
        # Signatures
        for match in SIGNATURE_RE.finditer(text):
            protected.append((match.start(), match.end(), "signature"))
        
        # Physician names with credentials (Name, MD)
        for match in PHYSICIAN_CREDENTIAL_RE.finditer(text):
            protected.append((match.start(), match.end(), "physician_credential"))
        
        # ROSE context
        for match in ROSE_CONTEXT_RE.finditer(text):
            protected.append((match.start(), match.end(), "rose_context"))
        
        # Device manufacturers
        for match in DEVICE_CONTEXT_RE.finditer(text):
            protected.append((match.start(), match.end(), "device_manufacturer"))
        
        # Robotic platforms
        for match in ROBOTIC_PLATFORM_RE.finditer(text):
            protected.append((match.start(), match.end(), "robotic_platform"))
        
        # Protected device names (Ion, Monarch, Galaxy)
        for device in PROTECTED_DEVICE_NAMES:
            pattern = re.compile(rf"\b{re.escape(device)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                protected.append((match.start(), match.end(), "device_name"))
        
        # Anatomical terms
        for term in ANATOMICAL_TERMS:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                protected.append((match.start(), match.end(), "anatomical"))
        
        # Clinical terms
        for term in CLINICAL_ALLOW_LIST:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                protected.append((match.start(), match.end(), "clinical"))
        
        return protected
    
    def _apply_regex_patterns(self, text: str) -> List[Detection]:
        """Apply regex patterns to detect PHI."""
        detections = []
        
        # Patient names from headers
        for match in PATIENT_HEADER_RE.finditer(text):
            name = match.group(1)
            if name and len(name) > 2:
                detections.append(Detection(
                    entity_type="PATIENT_NAME",
                    start=match.start(1),
                    end=match.end(1),
                    text=name,
                    confidence=0.95,
                    source="regex",
                    action=RedactionAction.REDACT
                ))
        
        # MRN/ID
        for match in MRN_RE.finditer(text):
            detections.append(Detection(
                entity_type="MRN",
                start=match.start(1),
                end=match.end(1),
                text=match.group(1),
                confidence=0.95,
                source="regex",
                action=RedactionAction.REDACT
            ))
        
        # DOB
        for match in DOB_RE.finditer(text):
            detections.append(Detection(
                entity_type="DOB",
                start=match.start(1),
                end=match.end(1),
                text=match.group(1),
                confidence=0.95,
                source="regex",
                action=RedactionAction.REDACT
            ))
        
        # Full dates (if configured)
        if self.config.redact_procedure_dates:
            for match in DATE_FULL_RE.finditer(text):
                detections.append(Detection(
                    entity_type="DATE",
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    confidence=0.85,
                    source="regex",
                    action=RedactionAction.REDACT
                ))
        
        # Ages over 89 (HIPAA)
        if self.config.redact_ages_over_89:
            for match in AGE_HIPAA_RE.finditer(text):
                # Get the age value from whichever group matched
                age_text = match.group(1) or match.group(2)
                if age_text:
                    try:
                        age = int(age_text)
                        if age > 89:
                            detections.append(Detection(
                                entity_type="AGE_OVER_89",
                                start=match.start(),
                                end=match.end(),
                                text=match.group(),
                                confidence=0.95,
                                source="regex",
                                action=RedactionAction.REDACT
                            ))
                    except ValueError:
                        pass
        
        # Facility names
        if self.config.redact_facilities:
            for match in FACILITY_RE.finditer(text):
                facility = match.group(1).strip()
                if facility and len(facility) > 3:
                    detections.append(Detection(
                        entity_type="FACILITY",
                        start=match.start(1),
                        end=match.end(1),
                        text=facility,
                        confidence=0.90,
                        source="regex",
                        action=RedactionAction.REDACT
                    ))
            
            # City, State patterns
            for match in CITY_STATE_RE.finditer(text):
                detections.append(Detection(
                    entity_type="LOCATION",
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    confidence=0.85,
                    source="regex",
                    action=RedactionAction.REDACT
                ))
        
        # Phone numbers
        for match in PHONE_RE.finditer(text):
            detections.append(Detection(
                entity_type="PHONE",
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=0.95,
                source="regex",
                action=RedactionAction.REDACT
            ))
        
        # SSN
        for match in SSN_RE.finditer(text):
            # Avoid matching dates that look like SSN
            detected_text = match.group()
            if not DATE_FULL_RE.match(detected_text):
                detections.append(Detection(
                    entity_type="SSN",
                    start=match.start(),
                    end=match.end(),
                    text=detected_text,
                    confidence=0.90,
                    source="regex",
                    action=RedactionAction.REDACT
                ))
        
        # Email
        for match in EMAIL_RE.finditer(text):
            detections.append(Detection(
                entity_type="EMAIL",
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=0.95,
                source="regex",
                action=RedactionAction.REDACT
            ))
        
        return detections
    
    def _apply_ner_model(self, text: str) -> List[Detection]:
        """Apply the NER model to detect contextual PHI."""
        if not self.use_ner_model or self.ner_pipeline is None:
            return []
        
        detections = []
        try:
            # NER pipeline returns entities with start, end, entity_group, score, word
            entities = self.ner_pipeline(text)
            
            for entity in entities:
                # Get label from entity_group, entity, or label field (NER uses entity_group)
                label = (
                    entity.get("entity_group") or 
                    entity.get("entity") or 
                    entity.get("label") or 
                    ""
                ).upper()
                
                if not label:
                    continue
                
                score = entity.get("score", 0.5)
                
                # Filter by threshold
                if score < self.config.ner_threshold:
                    continue
                
                action = NER_LABEL_ACTIONS.get(label, RedactionAction.REDACT)
                
                # Get text from word field (aggregation_strategy="simple" provides this)
                entity_text = entity.get("word", "")
                if not entity_text:
                    # Fallback: extract text from original using start/end
                    start = entity.get("start", 0)
                    end = entity.get("end", 0)
                    entity_text = text[start:end] if start < len(text) and end <= len(text) else ""
                
                detections.append(Detection(
                    entity_type=label,
                    start=entity.get("start", 0),
                    end=entity.get("end", 0),
                    text=entity_text,
                    confidence=score,
                    source="ner",
                    action=action
                ))
        except Exception as e:
            logger.error(f"NER prediction error: {e}")
        
        return detections
    
    def _learn_patient_names(self, text: str, regex_detections: List[Detection]) -> Set[str]:
        """
        Extract patient names from detected headers to catch subsequent mentions.
        """
        patient_names = set()
        
        for det in regex_detections:
            if det.entity_type == "PATIENT_NAME":
                full_name = det.text.strip()
                patient_names.add(full_name)
                
                # Split into parts for partial matching
                # Handle "Last, First" format
                if "," in full_name:
                    parts = [p.strip() for p in full_name.split(",")]
                else:
                    parts = full_name.split()
                
                for part in parts:
                    # Only add if it looks like a name (not initials)
                    if len(part) > 2 and part[0].isupper():
                        # Avoid common words
                        if part.lower() not in {"the", "and", "with", "for", "from"}:
                            patient_names.add(part)
        
        return patient_names
    
    def _detect_name_mentions(self, text: str, patient_names: Set[str]) -> List[Detection]:
        """
        Find subsequent mentions of learned patient names.
        """
        detections = []
        
        for name in patient_names:
            # Case-insensitive, word-boundary search
            pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                detections.append(Detection(
                    entity_type="PATIENT_NAME_MENTION",
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    confidence=0.90,
                    source="learned",
                    action=RedactionAction.REDACT
                ))
        
        return detections
    
    def _is_protected(self, detection: Detection, protected_zones: List[Tuple[int, int, str]]) -> bool:
        """Check if a detection overlaps with a protected zone."""
        for start, end, reason in protected_zones:
            # Check for any overlap
            if detection.start < end and detection.end > start:
                return True
        return False
    
    def _resolve_overlaps(self, detections: List[Detection]) -> List[Detection]:
        """
        Resolve overlapping detections, keeping highest confidence.
        """
        if not detections:
            return []
        
        # Sort by start position, then by length (longer first)
        sorted_dets = sorted(detections, key=lambda d: (d.start, -(d.end - d.start)))
        
        resolved = []
        last_end = -1
        
        for det in sorted_dets:
            if det.start >= last_end:
                resolved.append(det)
                last_end = det.end
            elif det.confidence > resolved[-1].confidence:
                # Higher confidence, replace previous
                resolved[-1] = det
                last_end = det.end
        
        return resolved
    
    def _apply_redactions(self, text: str, detections: List[Detection]) -> str:
        """
        Apply redactions to text, replacing PHI with placeholders.
        """
        # Sort by position (descending) to avoid offset issues
        sorted_dets = sorted(detections, key=lambda d: d.start, reverse=True)
        
        result = text
        for det in sorted_dets:
            if det.action == RedactionAction.REDACT:
                placeholder = f"[REDACTED_{det.entity_type}]"
                result = result[:det.start] + placeholder + result[det.end:]
        
        return result
    
    def scrub(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main scrubbing method.
        
        Returns:
            Tuple of (scrubbed_text, audit_info)
        """
        if not text or not isinstance(text, str):
            return text, {"error": "Invalid input"}
        
        audit = {
            "original_length": len(text),
            "detections": [],
            "protected_zones": [],
            "redaction_count": 0
        }
        
        # Step 1: Find protection zones
        protected_zones = self._find_protection_zones(text)
        audit["protected_zones"] = [
            {"start": s, "end": e, "reason": r} for s, e, r in protected_zones
        ]
        
        # Step 2: Apply regex patterns
        regex_detections = self._apply_regex_patterns(text)
        
        # Step 3: Learn patient names
        patient_names = self._learn_patient_names(text, regex_detections)
        name_mentions = self._detect_name_mentions(text, patient_names)
        
        # Step 4: Apply NER model
        ner_detections = self._apply_ner_model(text)
        
        # Step 5: Combine all detections
        all_detections = regex_detections + name_mentions + ner_detections
        
        # Step 6: Filter out protected zones
        filtered_detections = [
            d for d in all_detections 
            if not self._is_protected(d, protected_zones) and d.action == RedactionAction.REDACT
        ]
        
        # Step 7: Resolve overlaps
        final_detections = self._resolve_overlaps(filtered_detections)
        
        # Step 8: Apply redactions
        scrubbed_text = self._apply_redactions(text, final_detections)
        
        # Audit info
        audit["detections"] = [
            {
                "type": d.entity_type,
                "text": d.text,
                "start": d.start,
                "end": d.end,
                "confidence": d.confidence,
                "source": d.source
            }
            for d in final_detections
        ]
        audit["redaction_count"] = len(final_detections)
        audit["scrubbed_length"] = len(scrubbed_text)
        
        return scrubbed_text, audit


# =============================================================================
# 5. JSON PROCESSING
# =============================================================================

def process_json_structure(data: Any, redactor: PHIRedactor, target_fields: Optional[List[str]] = None) -> Any:
    """
    Recursively traverse JSON to scrub string fields.
    
    Args:
        data: JSON data (dict, list, or primitive)
        redactor: PHIRedactor instance
        target_fields: Specific keys to scrub (None = all strings)
    
    Returns:
        Scrubbed JSON structure
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if target_fields and k not in target_fields and not isinstance(v, (dict, list)):
                new_dict[k] = v
            elif isinstance(v, str):
                # Only scrub strings longer than 3 chars
                if len(v) > 3:
                    scrubbed, _ = redactor.scrub(v)
                    new_dict[k] = scrubbed
                else:
                    new_dict[k] = v
            else:
                new_dict[k] = process_json_structure(v, redactor, target_fields)
        return new_dict
    elif isinstance(data, list):
        return [process_json_structure(item, redactor, target_fields) for item in data]
    else:
        return data


# =============================================================================
# 6. COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Regex + DistilBERT NER PHI Redaction for Procedural Notes"
    )
    parser.add_argument("input_file", help="Path to input file (JSON or text)")
    parser.add_argument("output_file", help="Path to save redacted output")
    parser.add_argument("--fields", nargs='+', 
                        help="Specific JSON keys to scrub (e.g., 'note_text' 'body')")
    parser.add_argument("--no-ner", action="store_true",
                        help="Disable NER model (regex-only mode)")
    parser.add_argument("--keep-dates", action="store_true",
                        help="Do not redact procedure dates")
    parser.add_argument("--audit", action="store_true",
                        help="Output audit log alongside results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="NER confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Configure
    config = RedactionConfig(
        redact_procedure_dates=not args.keep_dates,
        ner_threshold=args.threshold
    )
    
    # Initialize redactor
    print(f"Initializing PHI Redactor (NER: {not args.no_ner})...")
    redactor = PHIRedactor(config=config, use_ner_model=not args.no_ner)
    
    # Read input
    print(f"Reading {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Detect format
        try:
            data = json.loads(content)
            is_json = True
        except json.JSONDecodeError:
            data = content
            is_json = False
        
        # Process
        if is_json:
            if args.fields:
                print(f"Scrubbing specific fields: {args.fields}")
            else:
                print("Scrubbing ALL string fields...")
            
            scrubbed_data = process_json_structure(data, redactor, args.fields)
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(scrubbed_data, f, indent=2, ensure_ascii=False)
        else:
            print("Processing as plain text...")
            scrubbed_text, audit = redactor.scrub(data)
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(scrubbed_text)
            
            if args.audit:
                audit_file = args.output_file.rsplit('.', 1)[0] + '_audit.json'
                with open(audit_file, 'w', encoding='utf-8') as f:
                    json.dump(audit, f, indent=2)
                print(f"Audit log saved to {audit_file}")
        
        print(f"Success! Saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
