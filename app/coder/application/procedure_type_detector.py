"""Procedure type auto-detection from report text and codes.

This module provides heuristic-based detection of procedure types from:
- Procedure note text (keywords, phrases)
- CPT codes (when available)

Supported procedure types:
- bronch_diagnostic: Standard diagnostic bronchoscopy
- bronch_ebus: EBUS bronchoscopy with TBNA
- bronch_therapeutic: Therapeutic bronchoscopy (stent, ablation, etc.)
- pleural: Pleural procedures (thoracentesis, chest tube, pleuroscopy)
- blvr: Bronchoscopic lung volume reduction (valves, coils, etc.)
- unknown: Cannot determine (default)

Usage:
    from app.coder.application.procedure_type_detector import detect_procedure_type

    # From text only
    proc_type = detect_procedure_type("EBUS-TBNA performed at stations 4R, 7...")

    # From text and codes
    proc_type = detect_procedure_type(
        report_text="...",
        codes=["31652", "31624"],
    )
"""

from __future__ import annotations

import re
from typing import Optional

# ============================================================================
# CPT Code Mappings
# ============================================================================

# EBUS-related codes
EBUS_CODES = {
    "31652",  # EBUS-TBNA 1-2 stations
    "31653",  # EBUS-TBNA 3+ stations
    "31654",  # EBUS-TBNA with ultrasound
}

# BLVR-related codes
BLVR_CODES = {
    "31647",  # Bronchoscopic valve insertion
    "31648",  # Bronchoscopic valve removal
    "31649",  # Bronchoscopic valve each additional
    "31651",  # Bronchoscopic thermoplasty
}

# Therapeutic bronchoscopy codes
THERAPEUTIC_BRONCH_CODES = {
    "31630",  # Bronchoscopy with tracheal dilation
    "31631",  # Bronchoscopy with tracheal stent
    "31632",  # Bronchoscopy lung biopsy single lobe
    "31633",  # Bronchoscopy lung biopsy add lobe
    "31634",  # Bronchoscopy with balloon occlusion
    "31636",  # Bronchoscopy with tumor excision
    "31637",  # Bronchoscopy with stent placement
    "31638",  # Bronchoscopy with tumor debulking
    "31640",  # Bronchoscopy with foreign body removal
    "31641",  # Bronchoscopy with ablation
    "31645",  # Bronchoscopy with aspiration
    "31646",  # Bronchoscopy with radiofrequency ablation
}

# Diagnostic bronchoscopy codes
DIAGNOSTIC_BRONCH_CODES = {
    "31622",  # Diagnostic bronchoscopy
    "31623",  # Bronchoscopy with brushing
    "31624",  # Bronchoscopy with BAL
    "31625",  # Bronchoscopy with biopsy
    "31626",  # Bronchoscopy with transbronchial biopsy markers
    "31627",  # Navigation bronchoscopy
    "31628",  # Transbronchial lung biopsy
    "31629",  # Transbronchial biopsy add lobe
}

# Pleural procedure codes
PLEURAL_CODES = {
    "32550",  # Pleurodesis
    "32551",  # Chest tube insertion
    "32554",  # Thoracentesis with catheter
    "32555",  # Thoracentesis with imaging
    "32556",  # Thoracentesis with tube
    "32557",  # Chest tube removal
    "32601",  # Thoracoscopy diagnostic
    "32602",  # Thoracoscopy with biopsy
    "32603",  # Thoracoscopy with lymph node biopsy
    "32604",  # Thoracoscopy with excision
    "32606",  # Thoracoscopy with decortication
    "32607",  # Medical thoracoscopy
    "32608",  # Medical thoracoscopy with biopsy
    "32609",  # Medical thoracoscopy with pleurodesis
}

# ============================================================================
# Keyword Patterns
# ============================================================================

# EBUS keywords (case-insensitive patterns)
EBUS_KEYWORDS = [
    r"\bebus\b",
    r"\bendobronchial\s+ultrasound\b",
    r"\btbna\b",
    r"\btransbronchial\s+needle\s+aspiration\b",
    r"\bstation\s+\d+[RL]?\b",  # e.g., "station 4R", "station 7"
    r"\bmediastinal\s+lymph\s+node\b",
    r"\bhilar\s+lymph\s+node\b",
]

# BLVR keywords
BLVR_KEYWORDS = [
    r"\bblvr\b",
    r"\blung\s+volume\s+reduction\b",
    r"\bvalve\s+placement\b",
    r"\bendobronchial\s+valve\b",
    r"\bzephyr\b",
    r"\bspiration\b",
    r"\bchartis\b",
    r"\bcollateral\s+ventilation\b",
    r"\bcoil\s+placement\b",
    r"\bbronchoscopic\s+thermal\s+vapor\b",
]

# Therapeutic bronchoscopy keywords
THERAPEUTIC_KEYWORDS = [
    r"\bstent\s+placement\b",
    r"\bstent\s+removal\b",
    r"\bairway\s+stent\b",
    r"\btracheal\s+stent\b",
    r"\bbronchial\s+stent\b",
    r"\bablation\b",
    r"\bcryotherapy\b",
    r"\blaser\b",
    r"\belectrocautery\b",
    r"\bargon\s+plasma\b",
    r"\bforeign\s+body\s+removal\b",
    r"\btumor\s+debulking\b",
    r"\bairway\s+dilation\b",
    r"\bballoon\s+dilation\b",
    r"\bthermoplasty\b",
]

# Pleural procedure keywords
PLEURAL_KEYWORDS = [
    r"\bpleural\b",
    r"\bpleuroscopy\b",
    r"\bthoracoscopy\b",
    r"\bmedical\s+thoracoscopy\b",
    r"\bthoracentesis\b",
    r"\bchest\s+tube\b",
    r"\bpleural\s+effusion\b",
    r"\bpleurodesis\b",
    r"\btalc\s+poudrage\b",
    r"\bpleural\s+biopsy\b",
    r"\bindwelling\s+pleural\s+catheter\b",
    r"\bipc\b",
    r"\bpleurx\b",
]

# Diagnostic bronchoscopy keywords (broad, as fallback)
DIAGNOSTIC_KEYWORDS = [
    r"\bdiagnostic\s+bronchoscopy\b",
    r"\bbronchoalveolar\s+lavage\b",
    r"\bbal\b",
    r"\bbronchial\s+wash\b",
    r"\btransbronchial\s+lung\s+biopsy\b",
    r"\btransbronchial\s+biopsy\b",
    r"\btblb\b",
    r"\bbronchoscopy\b",  # Generic, low priority
]


# ============================================================================
# Detection Logic
# ============================================================================


def _count_keyword_matches(text: str, patterns: list[str]) -> int:
    """Count how many keyword patterns match in the text."""
    count = 0
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            count += 1
    return count


def _codes_intersect(codes: list[str], code_set: set[str]) -> bool:
    """Check if any codes are in the given code set."""
    return bool(set(codes) & code_set)


def detect_procedure_type(
    report_text: str,
    codes: list[str] | None = None,
) -> str:
    """Detect procedure type from report text and optional codes.

    The detection uses a priority-based approach:
    1. BLVR (most specific) - if BLVR codes or keywords are present
    2. EBUS - if EBUS codes or significant EBUS keywords
    3. Therapeutic - if therapeutic codes or keywords
    4. Pleural - if pleural codes or keywords
    5. Diagnostic bronchoscopy - if diagnostic codes or keywords
    6. Unknown - if no clear indicators

    Args:
        report_text: The procedure note text to analyze.
        codes: Optional list of CPT codes to assist detection.

    Returns:
        Procedure type string: "bronch_diagnostic", "bronch_ebus",
        "bronch_therapeutic", "pleural", "blvr", or "unknown".
    """
    codes = codes or []
    text = report_text.lower() if report_text else ""

    # Score each category
    scores: dict[str, float] = {
        "blvr": 0,
        "bronch_ebus": 0,
        "bronch_therapeutic": 0,
        "pleural": 0,
        "bronch_diagnostic": 0,
    }

    # Check codes first (stronger signal)
    if _codes_intersect(codes, BLVR_CODES):
        scores["blvr"] += 10
    if _codes_intersect(codes, EBUS_CODES):
        scores["bronch_ebus"] += 10
    if _codes_intersect(codes, THERAPEUTIC_BRONCH_CODES):
        scores["bronch_therapeutic"] += 10
    if _codes_intersect(codes, PLEURAL_CODES):
        scores["pleural"] += 10
    if _codes_intersect(codes, DIAGNOSTIC_BRONCH_CODES):
        scores["bronch_diagnostic"] += 5  # Lower weight for diagnostic

    # Check keywords
    scores["blvr"] += _count_keyword_matches(text, BLVR_KEYWORDS) * 3
    scores["bronch_ebus"] += _count_keyword_matches(text, EBUS_KEYWORDS) * 2
    scores["bronch_therapeutic"] += _count_keyword_matches(text, THERAPEUTIC_KEYWORDS) * 2
    scores["pleural"] += _count_keyword_matches(text, PLEURAL_KEYWORDS) * 2
    scores["bronch_diagnostic"] += _count_keyword_matches(text, DIAGNOSTIC_KEYWORDS) * 1

    # Find the highest scoring category
    max_score = max(scores.values())

    if max_score == 0:
        return "unknown"

    # Return the highest scoring type
    for proc_type, score in sorted(scores.items(), key=lambda x: -x[1]):
        if score == max_score:
            return proc_type

    return "unknown"


def detect_procedure_type_from_codes(codes: list[str]) -> str:
    """Detect procedure type from CPT codes only.

    Useful when no report text is available.

    Args:
        codes: List of CPT codes.

    Returns:
        Procedure type string.
    """
    return detect_procedure_type(report_text="", codes=codes)
