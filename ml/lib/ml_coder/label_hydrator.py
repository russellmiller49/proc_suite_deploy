"""Label hydration module for registry ML training data.

Provides 3-tier extraction with fallback hydration:
1. Tier 1 (structured): Use extract_v2_booleans() from registry_entry
2. Tier 2 (CPT): Use derive_booleans_from_json() from CPT codes
3. Tier 3 (keyword): Use keyword patterns on note_text

This ensures training data has minimal all-zero label rows by imputing
labels from text when structured data is incomplete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

from app.registry.v2_booleans import PROCEDURE_BOOLEAN_FIELDS, extract_v2_booleans

# Import CPT-based derivation function
from .data_prep import derive_booleans_from_json
from .registry_label_constraints import apply_label_constraints

__all__ = [
    "HydratedLabels",
    "hydrate_labels_from_text",
    "extract_labels_with_hydration",
    "KEYWORD_TO_PROCEDURE_MAP",
]

# =============================================================================
# Keyword-to-Procedure Mapping
# =============================================================================
# Maps regex patterns to (procedure_field, confidence) tuples.
# Patterns are case-insensitive and should use word boundaries.
# Multiple patterns can map to the same procedure for redundancy.

KEYWORD_TO_PROCEDURE_MAP: Dict[str, List[Tuple[str, float]]] = {
    # =========================================================================
    # EBUS / TBNA (Linear EBUS)
    # =========================================================================
    r"\bebus\b": [
        ("linear_ebus", 0.7),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bendobronchial\s+ultrasound\b": [
        ("linear_ebus", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\btbna\b": [
        ("linear_ebus", 0.75),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\btransbronchial\s+needle\s+aspiration\b": [
        ("linear_ebus", 0.8),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bstation\s+\d+[RL]?i?\b": [
        ("linear_ebus", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bmediastinal\s+lymph\s+node\b": [
        ("linear_ebus", 0.6),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31652\b": [("linear_ebus", 0.95)],
    r"\b31653\b": [("linear_ebus", 0.95)],

    # =========================================================================
    # Radial EBUS
    # =========================================================================
    r"\bradial\s+ebus\b": [
        ("radial_ebus", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\br-ebus\b": [
        ("radial_ebus", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bradial\s+probe\b": [
        ("radial_ebus", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bminiprobe\b": [
        ("radial_ebus", 0.8),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bconcentric\s+view\b": [
        ("radial_ebus", 0.85),
    ],
    r"\beccentric\s+view\b": [
        ("radial_ebus", 0.85),
    ],
    r"\b31654\b": [("radial_ebus", 0.95)],

    # =========================================================================
    # Navigational Bronchoscopy
    # =========================================================================
    r"\belectromagnetic\s+navigation\b": [
        ("navigational_bronchoscopy", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\benb\b": [
        ("navigational_bronchoscopy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bion\s+system\b": [
        ("navigational_bronchoscopy", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bmonarch\s+system\b": [
        ("navigational_bronchoscopy", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\brobotic\s+bronchoscopy\b": [
        ("navigational_bronchoscopy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bsuperdimension\b": [
        ("navigational_bronchoscopy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bnavigat(?:ion|ional)\s+bronchoscopy\b": [
        ("navigational_bronchoscopy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bnav[\s-]guided\b": [
        ("navigational_bronchoscopy", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31627\b": [("navigational_bronchoscopy", 0.95)],

    # =========================================================================
    # BLVR (Bronchoscopic Lung Volume Reduction)
    # =========================================================================
    r"\bblvr\b": [
        ("blvr", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\blung\s+volume\s+reduction\b": [
        ("blvr", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bendobronchial\s+valve\b": [
        ("blvr", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bzephyr\s+valve\b": [
        ("blvr", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bchartis\b": [
        ("blvr", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bcollateral\s+ventilation\b": [
        ("blvr", 0.8),
    ],
    r"\b31647\b": [("blvr", 0.95)],
    r"\b31648\b": [("blvr", 0.95)],
    r"\b31649\b": [("blvr", 0.95)],

    # =========================================================================
    # Transbronchial Biopsy
    # =========================================================================
    r"\btransbronchial\s+lung\s+biopsy\b": [
        ("transbronchial_biopsy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\btblb\b": [
        ("transbronchial_biopsy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\btbbx\b": [
        ("transbronchial_biopsy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31628\b": [("transbronchial_biopsy", 0.95)],
    r"\b31632\b": [("transbronchial_biopsy", 0.95)],

    # =========================================================================
    # Transbronchial Cryobiopsy
    # =========================================================================
    r"\bcryobiopsy\b": [
        ("transbronchial_cryobiopsy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\btransbronchial\s+cryobiopsy\b": [
        ("transbronchial_cryobiopsy", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bcryo\s+biopsy\b": [
        ("transbronchial_cryobiopsy", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],

    # =========================================================================
    # BAL (Bronchoalveolar Lavage)
    # =========================================================================
    r"\bbronchoalveolar\s+lavage\b": [
        ("bal", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bbal\b": [
        ("bal", 0.8),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31624\b": [("bal", 0.95)],

    # =========================================================================
    # Bronchial Wash
    # =========================================================================
    r"\bbronchial\s+wash(?:ing)?\b": [
        ("bronchial_wash", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31622\b": [("bronchial_wash", 0.7)],

    # =========================================================================
    # Brushings
    # =========================================================================
    r"\bbrush(?:ing)?s?\b": [
        ("brushings", 0.7),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bcytology\s+brush\b": [
        ("brushings", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31623\b": [("brushings", 0.95)],

    # =========================================================================
    # Endobronchial Biopsy
    # =========================================================================
    r"\bendobronchial\s+biopsy\b": [
        ("endobronchial_biopsy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bebb\b": [
        ("endobronchial_biopsy", 0.75),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31625\b": [("endobronchial_biopsy", 0.95)],

    # =========================================================================
    # TBNA Conventional (non-EBUS)
    # =========================================================================
    r"\bconventional\s+tbna\b": [
        ("tbna_conventional", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31629\b": [("tbna_conventional", 0.8)],

    # =========================================================================
    # Therapeutic Aspiration
    # =========================================================================
    r"\btherapeutic\s+aspiration\b": [
        ("therapeutic_aspiration", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bmucus\s+plug\s+removal\b": [
        ("therapeutic_aspiration", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31645\b": [("therapeutic_aspiration", 0.95)],
    r"\b31646\b": [("therapeutic_aspiration", 0.95)],

    # =========================================================================
    # Foreign Body Removal
    # =========================================================================
    r"\bforeign\s+body\s+remov(?:al|ed)\b": [
        ("foreign_body_removal", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bforeign\s+body\s+extraction\b": [
        ("foreign_body_removal", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31635\b": [("foreign_body_removal", 0.95)],

    # =========================================================================
    # Airway Dilation
    # =========================================================================
    r"\bairway\s+dilation\b": [
        ("airway_dilation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bballoon\s+dilat(?:ion|ation)\b": [
        ("airway_dilation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\btracheal\s+dilation\b": [
        ("airway_dilation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bbronchial\s+dilation\b": [
        ("airway_dilation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31630\b": [("airway_dilation", 0.95)],
    r"\b31631\b": [("airway_dilation", 0.95)],
    r"\b31634\b": [("airway_dilation", 0.95)],

    # =========================================================================
    # Airway Stent
    # =========================================================================
    r"\bairway\s+stent\b": [
        ("airway_stent", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\btracheal\s+stent\b": [
        ("airway_stent", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bbronchial\s+stent\b": [
        ("airway_stent", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bstent\s+placement\b": [
        ("airway_stent", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bdumon\s+stent\b": [
        ("airway_stent", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31636\b": [("airway_stent", 0.95)],
    r"\b31637\b": [("airway_stent", 0.95)],
    r"\b31638\b": [("airway_stent", 0.95)],

    # =========================================================================
    # Thermal Ablation
    # =========================================================================
    r"\bthermal\s+ablation\b": [
        ("thermal_ablation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\belectrocautery\b": [
        ("thermal_ablation", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bargon\s+plasma\s+coagulation\b": [
        ("thermal_ablation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bapc\b": [
        ("thermal_ablation", 0.75),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\blaser\s+ablation\b": [
        ("thermal_ablation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31641\b": [("thermal_ablation", 0.85)],

    # =========================================================================
    # Cryotherapy (Central)
    # =========================================================================
    r"\bcryotherapy\b": [
        ("cryotherapy", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bcryoablation\b": [
        ("cryotherapy", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bcryo\s+spray\b": [
        ("cryotherapy", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],

    # =========================================================================
    # Peripheral Ablation
    # =========================================================================
    r"\bperipheral\s+ablation\b": [
        ("peripheral_ablation", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bmicrowave\s+ablation\b": [
        ("peripheral_ablation", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bneuwave\b": [
        ("peripheral_ablation", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bnavblate\b": [
        ("peripheral_ablation", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],

    # =========================================================================
    # Bronchial Thermoplasty
    # =========================================================================
    r"\bbronchial\s+thermoplasty\b": [
        ("bronchial_thermoplasty", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bbt\s+treatment\b": [
        ("bronchial_thermoplasty", 0.7),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31660\b": [("bronchial_thermoplasty", 0.95)],
    r"\b31661\b": [("bronchial_thermoplasty", 0.95)],

    # =========================================================================
    # Whole Lung Lavage
    # =========================================================================
    r"\bwhole\s+lung\s+lavage\b": [
        ("whole_lung_lavage", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bwll\b": [
        ("whole_lung_lavage", 0.8),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b32997\b": [("whole_lung_lavage", 0.95)],

    # =========================================================================
    # Rigid Bronchoscopy
    # =========================================================================
    r"\brigid\s+bronchoscopy\b": [
        ("rigid_bronchoscopy", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\brigid\s+scope\b": [
        ("rigid_bronchoscopy", 0.85),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\b31600\b": [("rigid_bronchoscopy", 0.95), ("percutaneous_tracheostomy", 0.95)],
    r"\b31601\b": [("rigid_bronchoscopy", 0.95), ("percutaneous_tracheostomy", 0.95)],

    # =========================================================================
    # Other Interventions
    # =========================================================================
    r"\bpercutaneous\s+tracheostomy\b": [("percutaneous_tracheostomy", 0.95)],
    r"\bpercutaneous\s+dilat\w*\s+tracheostomy\b": [("percutaneous_tracheostomy", 0.95)],
    r"\bdilational\s+trach(?:eostomy)?\b": [("percutaneous_tracheostomy", 0.85)],
    r"\bperc(?:utaneous)?\s+dilat\w*\s+trach(?:eostomy)?\b": [("percutaneous_tracheostomy", 0.9)],
    r"\bperc(?:utaneous)?\s+trach(?:eostomy)?\b": [("percutaneous_tracheostomy", 0.85)],
    r"\bbedside\s+trach(?:eostomy)?\b": [("percutaneous_tracheostomy", 0.75)],
    r"\btrach\s*(?:\+|and|/|&)\s*peg\b": [("percutaneous_tracheostomy", 0.7), ("peg_insertion", 0.7)],
    r"\btracheostomy\s*(?:\+|and|/|&)\s*peg\b": [("percutaneous_tracheostomy", 0.7), ("peg_insertion", 0.7)],
    r"\btracheostomy\b[\s:,-]{0,10}\bpercutaneous\b": [("percutaneous_tracheostomy", 0.9)],
    r"\btrach(?:eostomy)?\b[\s\S]{0,200}\b(needle|guide\s*wire|wire|dilat\w*|dilator|ett\s+withdrawn|bronch\w*)\b": [
        ("percutaneous_tracheostomy", 0.8)
    ],
    r"\bblue\s+rhino\b": [("percutaneous_tracheostomy", 0.9)],
    r"\bciaglia\b": [("percutaneous_tracheostomy", 0.8)],
    r"\bseldinger\b": [("percutaneous_tracheostomy", 0.75)],
    r"\bshiley\s+trach(?:eostomy)?\b": [("percutaneous_tracheostomy", 0.75)],
    r"\b31612\b": [("percutaneous_tracheostomy", 0.9)],

    r"\bpercutaneous\s+endoscopic\s+gastrostomy\b": [("peg_insertion", 0.95)],
    r"\bpeg\s+(tube|placement|insertion)\b": [("peg_insertion", 0.9)],
    r"\bpeg\b": [("peg_insertion", 0.7)],
    r"\b43246\b": [("peg_insertion", 0.95)],
    r"\b49440\b": [("peg_insertion", 0.9)],

    # =========================================================================
    # Fiducial Placement
    # =========================================================================
    r"\bfiducial\s+marker\b": [
        ("fiducial_placement", 0.9),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bfiducial\s+placement\b": [
        ("fiducial_placement", 0.95),
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bgold\s+marker\b": [
        ("fiducial_placement", 0.7),
        ("diagnostic_bronchoscopy", 0.9),
    ],

    # =========================================================================
    # PLEURAL PROCEDURES
    # =========================================================================

    # Thoracentesis
    r"\bthoracentesis\b": [
        ("thoracentesis", 0.95),
    ],
    r"\bpleural\s+tap\b": [
        ("thoracentesis", 0.85),
    ],
    r"\bpleural\s+fluid\s+drainage\b": [
        ("thoracentesis", 0.8),
    ],
    r"\b32554\b": [("thoracentesis", 0.95)],
    r"\b32555\b": [("thoracentesis", 0.95)],

    # Chest Tube
    r"\bchest\s+tube\b": [
        ("chest_tube", 0.9),
    ],
    r"\btube\s+thoracostomy\b": [
        ("chest_tube", 0.95),
    ],
    r"\bintercostal\s+drain\b": [
        ("chest_tube", 0.85),
    ],
    r"\b32551\b": [("chest_tube", 0.95)],

    # IPC (Indwelling Pleural Catheter)
    r"\bindwelling\s+pleural\s+catheter\b": [
        ("ipc", 0.95),
    ],
    r"\bipc\b": [
        ("ipc", 0.8),
    ],
    r"\bpleurx\b": [
        ("ipc", 0.95),
    ],
    r"\btunneled\s+pleural\s+catheter\b": [
        ("ipc", 0.95),
    ],
    r"\b32550\b": [("ipc", 0.95)],

    # Medical Thoracoscopy
    r"\bmedical\s+thoracoscopy\b": [
        ("medical_thoracoscopy", 0.95),
    ],
    r"\bpleuroscopy\b": [
        ("medical_thoracoscopy", 0.9),
    ],
    r"\b32601\b": [("medical_thoracoscopy", 0.95)],

    # Pleurodesis
    r"\bpleurodesis\b": [
        ("pleurodesis", 0.95),
    ],
    r"\btalc\s+poudrage\b": [
        ("pleurodesis", 0.95),
    ],
    r"\btalc\s+slurry\b": [
        ("pleurodesis", 0.9),
    ],
    r"\bchemical\s+pleurodesis\b": [
        ("pleurodesis", 0.9),
    ],
    r"\b32650\b": [("pleurodesis", 0.95)],

    # Pleural Biopsy
    r"\bpleural\s+biopsy\b": [
        ("pleural_biopsy", 0.95),
    ],
    r"\bparietal\s+pleural\s+biopsy\b": [
        ("pleural_biopsy", 0.95),
    ],

    # Fibrinolytic Therapy
    r"\bfibrinolytic\s+therapy\b": [
        ("fibrinolytic_therapy", 0.95),
    ],
    r"\btpa\s+instillation\b": [
        ("fibrinolytic_therapy", 0.9),
    ],
    r"\bfibrinolysis\b": [
        ("fibrinolytic_therapy", 0.85),
    ],
    r"\bdnase\b": [
        ("fibrinolytic_therapy", 0.8),
    ],
    r"\b32560\b": [("fibrinolytic_therapy", 0.95)],

    # =========================================================================
    # Generic Diagnostic Bronchoscopy (low priority, fallback)
    # =========================================================================
    r"\bdiagnostic\s+bronchoscopy\b": [
        ("diagnostic_bronchoscopy", 0.9),
    ],
    r"\bbronchoscopy\b": [
        ("diagnostic_bronchoscopy", 0.6),
    ],
}

# Negation patterns to filter false positives (patterns that appear BEFORE the keyword)
NEGATION_PATTERNS = [
    r"\bno\s+",
    r"\bnot\s+",
    r"\bwithout\s+",
    r"\bnegative\s+for\s+",
    r"\bdenied\s+",
    r"\bno\s+evidence\s+of\s+",
    r"\bruled\s+out\s+",
]

# Post-keyword negation patterns (patterns that appear AFTER the keyword)
POST_NEGATION_PATTERNS = [
    r"\bwas\s+not\s+performed\b",
    r"\bwas\s+not\s+done\b",
    r"\bnot\s+performed\b",
    r"\bnot\s+done\b",
    r"\bwas\s+deferred\b",
    r"\bwas\s+cancelled\b",
    r"\bwas\s+avoided\b",
]

_STENT_PRESENCE_RE = re.compile(
    r"\b(?:"
    r"well[- ]positioned"
    r"|in\s+(?:good|adequate)\s+position"
    r"|adequately\s+positioned"
    r"|in\s+place"
    r"|patent"
    r"|intact"
    r"|present"
    r"|stent\s+check"
    r")\b",
    re.IGNORECASE,
)
_STENT_HISTORY_RE = re.compile(r"\b(?:known|existing|prior|previous|history\s+of)\b", re.IGNORECASE)
_STENT_ACTION_RE = re.compile(
    r"\b(?:"
    r"plac(?:e|ed|ement)"
    r"|insert(?:ed|ion)"
    r"|deploy(?:ed|ment)"
    r"|remove(?:d|al)"
    r"|retriev(?:e|ed|al)"
    r"|extract(?:ed|ion)"
    r"|explant(?:ed|ation)"
    r"|revision"
    r"|reposition"
    r"|exchange"
    r"|replace(?:d|ment)?"
    r")\b",
    re.IGNORECASE,
)

_CHEST_TUBE_REMOVE_RE = re.compile(
    r"\b(?:"
    r"remove(?:d|al)?"
    r"|discontinue(?:d|ation)?"
    r"|d/c"
    r"|\bdc\b"
    r"|pull(?:ed)?"
    r"|withdrawn"
    r"|taken\s+out"
    r")\b",
    re.IGNORECASE,
)
_CHEST_TUBE_INSERT_RE = re.compile(
    r"\b(?:place(?:d|ment)?|insert(?:ed|ion)?|introduc(?:e|ed|tion)|position(?:ed)?)\b",
    re.IGNORECASE,
)


@dataclass
class HydratedLabels:
    """Result of label extraction with hydration.

    Attributes:
        labels: Dict mapping procedure field names to 0/1 values.
        confidence: Overall confidence score (0.0-1.0).
        source: Extraction source tier ("structured", "cpt", "keyword", "empty").
        hydrated_fields: List of fields that were filled by hydration (Tier 2 or 3).
    """
    labels: Dict[str, int]
    confidence: float
    source: Literal["structured", "cpt", "keyword", "empty"]
    hydrated_fields: List[str] = field(default_factory=list)


def _is_negated(text: str, match_start: int, match_end: int, window_size: int = 30) -> bool:
    """Check if a match is negated (before or after the match).

    Args:
        text: Full text being searched.
        match_start: Start position of the match.
        match_end: End position of the match.
        window_size: How many characters before/after match to check.

    Returns:
        True if a negation phrase is found near the match.
    """
    # Check window BEFORE the match
    window_start = max(0, match_start - window_size)
    before_window = text[window_start:match_start].lower()

    for neg_pattern in NEGATION_PATTERNS:
        if re.search(neg_pattern, before_window, re.IGNORECASE):
            return True

    # Check window AFTER the match
    after_end = min(len(text), match_end + window_size)
    after_window = text[match_end:after_end].lower()

    for neg_pattern in POST_NEGATION_PATTERNS:
        if re.search(neg_pattern, after_window, re.IGNORECASE):
            return True

    return False


def hydrate_labels_from_text(
    note_text: str,
    threshold: float = 0.6,
) -> Dict[str, float]:
    """Extract procedure labels from note text using keyword patterns.

    This is Tier 3 extraction that uses keyword matching to identify
    procedures mentioned in the text. Applies negation filtering to
    avoid false positives like "no EBUS performed".

    Args:
        note_text: The procedure note text to analyze.
        threshold: Minimum confidence to include a label.

    Returns:
        Dict mapping procedure field names to confidence scores.
        Only includes fields with confidence >= threshold.
    """
    if not note_text:
        return {}

    text_lower = note_text.lower()
    field_scores: Dict[str, float] = {}

    def _context_false_positive(field_name: str, match_start: int, match_end: int) -> bool:
        if not note_text:
            return False
        window_start = max(0, match_start - 120)
        window_end = min(len(note_text), match_end + 180)
        window = note_text[window_start:window_end].lower()

        if field_name == "airway_stent":
            presence_only = bool(_STENT_PRESENCE_RE.search(window) or _STENT_HISTORY_RE.search(window))
            has_action = bool(_STENT_ACTION_RE.search(window))
            if presence_only and not has_action:
                return True

        if field_name == "chest_tube":
            has_remove = bool(_CHEST_TUBE_REMOVE_RE.search(window))
            has_insert = bool(_CHEST_TUBE_INSERT_RE.search(window))
            if has_remove and not has_insert:
                return True

        return False

    for pattern, mappings in KEYWORD_TO_PROCEDURE_MAP.items():
        # Find all matches of this pattern
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            # Skip if negated (before or after the match)
            if _is_negated(note_text, match.start(), match.end()):
                continue

            # Apply confidence scores for each mapped field
            for field_name, confidence in mappings:
                if field_name not in PROCEDURE_BOOLEAN_FIELDS:
                    continue
                if _context_false_positive(field_name, match.start(), match.end()):
                    continue

                # Keep the highest confidence seen for each field
                current = field_scores.get(field_name, 0.0)
                field_scores[field_name] = max(current, confidence)

    # Filter by threshold
    return {
        field: score
        for field, score in field_scores.items()
        if score >= threshold
    }


def extract_labels_with_hydration(
    entry: Dict[str, Any],
    note_text: str | None = None,
    keyword_threshold: float = 0.6,
) -> HydratedLabels:
    """Extract procedure labels using 3-tier fallback strategy.

    Tier 1: Structured extraction from registry_entry using extract_v2_booleans().
    Tier 2: CPT-based derivation using derive_booleans_from_json().
    Tier 3: Keyword hydration from note_text.

    Args:
        entry: Full golden JSON entry with registry_entry, cpt_codes, note_text.
        note_text: Override note text (uses entry["note_text"] if not provided).
        keyword_threshold: Minimum confidence for Tier 3 keyword matches.

    Returns:
        HydratedLabels with extracted labels, confidence, source, and hydrated fields.
    """
    # Initialize empty labels
    labels = {field: 0 for field in PROCEDURE_BOOLEAN_FIELDS}
    hydrated_fields: List[str] = []

    # Get note text
    if note_text is None:
        note_text = entry.get("note_text", "")

    # =========================================================================
    # TIER 1: Structured Extraction (confidence 0.95)
    # =========================================================================
    registry_entry = entry.get("registry_entry", {})
    if registry_entry:
        structured_labels = extract_v2_booleans(registry_entry)
        labels.update(structured_labels)

        if any(v == 1 for v in structured_labels.values()):
            apply_label_constraints(labels, note_text=note_text)
            return HydratedLabels(
                labels=labels,
                confidence=0.95,
                source="structured",
                hydrated_fields=[],
            )

    # =========================================================================
    # TIER 2: CPT-Based Derivation (confidence 0.80)
    # =========================================================================
    cpt_codes = entry.get("cpt_codes", [])
    if cpt_codes:
        cpt_labels = derive_booleans_from_json(entry)

        # Track which fields were filled by CPT
        for field, value in cpt_labels.items():
            if value == 1 and labels.get(field, 0) == 0:
                labels[field] = 1
                hydrated_fields.append(field)

        if any(v == 1 for v in labels.values()):
            apply_label_constraints(labels, note_text=note_text)
            return HydratedLabels(
                labels=labels,
                confidence=0.80,
                source="cpt",
                hydrated_fields=hydrated_fields,
            )

    # =========================================================================
    # TIER 3: Keyword Hydration (confidence 0.60)
    # =========================================================================
    if note_text:
        from app.registry.processing.masking import mask_extraction_noise

        masked_note_text, _mask_meta = mask_extraction_noise(note_text)
        keyword_scores = hydrate_labels_from_text(masked_note_text, keyword_threshold)

        for field, score in keyword_scores.items():
            if score >= keyword_threshold and labels.get(field, 0) == 0:
                labels[field] = 1
                hydrated_fields.append(field)

        if any(v == 1 for v in labels.values()):
            # Average confidence of matched keywords
            avg_confidence = sum(keyword_scores.values()) / len(keyword_scores) if keyword_scores else 0.6
            apply_label_constraints(labels, note_text=note_text)
            return HydratedLabels(
                labels=labels,
                confidence=min(0.60, avg_confidence),
                source="keyword",
                hydrated_fields=hydrated_fields,
            )

    # =========================================================================
    # TIER 4: Empty (no labels found)
    # =========================================================================
    return HydratedLabels(
        labels=labels,
        confidence=0.0,
        source="empty",
        hydrated_fields=[],
    )
