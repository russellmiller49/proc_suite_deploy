"""
Canonical CPT coding rules derived from:
- ip_golden_knowledge_v2_2.json (golden coding rules)
- data/synthetic_CPT_corrected.json (validated coding patterns)

This module is regenerated from the above canonical sources.
DO NOT edit manually - update the source files and regenerate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ============================================================================
# BUNDLING RULES derived from synthetic_CPT_corrected.json excluded_or_bundled_codes
# and ip_golden_knowledge_v2_2.json global rules
# ============================================================================

# Rule: Radial EBUS (+31654) cannot be billed with Linear EBUS sampling (31652/31653)
RADIAL_LINEAR_EBUS_EXCLUSIVE: Dict[str, Set[str]] = {
    "linear_ebus_codes": {"31652", "31653"},
    "radial_ebus_codes": {"31654"},
}

# Rule: Brushing (31623) is bundled into transbronchial biopsy (31628) in same lobe
TBLB_BUNDLES_BRUSH: Dict[str, Set[str]] = {
    "dominant_codes": {"31628"},
    "bundled_codes": {"31623"},
}

# Rule: Ultrasound guidance (76942) bundled with IPC placement (32550)
IPC_BUNDLES_US_GUIDANCE: Dict[str, Set[str]] = {
    "dominant_codes": {"32550"},
    "bundled_codes": {"76942"},
}

# Rule: Diagnostic thoracoscopy/biopsy (32609) bundled with surgical thoracoscopy (32650)
THORACOSCOPY_SURGICAL_BUNDLES_DIAGNOSTIC: Dict[str, Set[str]] = {
    "dominant_codes": {"32650"},
    "bundled_codes": {"32609", "32601", "32604", "32606"},
}

# Rule: Pleurodesis via chest tube (32560) bundled with surgical thoracoscopy (32650)
THORACOSCOPY_BUNDLES_PLEURODESIS: Dict[str, Set[str]] = {
    "dominant_codes": {"32650"},
    "bundled_codes": {"32560"},
}

# ============================================================================
# THORACOSCOPY CODE SELECTION RULES
# Per ip_golden_knowledge_v2_2.json:
# - 32601: Diagnostic thoracoscopy, NO biopsy
# - 32604: Diagnostic thoracoscopy, PERICARDIAL SAC with biopsy
# - 32606: Diagnostic thoracoscopy, MEDIASTINAL space with biopsy
# - 32609: Diagnostic thoracoscopy, PLEURA with biopsy
# - 32602/32607/32608: Lung parenchyma (wedge, resection)
#
# RULE: Only ONE thoracoscopy code per hemithorax per session
# RULE: Biopsy codes trump diagnostic-only (32601)
# ============================================================================

# All diagnostic thoracoscopy codes (with and without biopsy)
THORACOSCOPY_ALL_DIAGNOSTIC_CODES: Set[str] = {
    "32601",  # Diagnostic, no biopsy
    "32604",  # Pericardial sac, with biopsy
    "32606",  # Mediastinal space, with biopsy
    "32609",  # Pleura, with biopsy
}

# Thoracoscopy codes with biopsy - these trump 32601
THORACOSCOPY_BIOPSY_CODES: Set[str] = {
    "32604",  # Pericardial
    "32606",  # Mediastinal
    "32609",  # Pleural
}

# Lung parenchyma thoracoscopy codes
THORACOSCOPY_LUNG_CODES: Set[str] = {
    "32602",  # Lung biopsy (unspecified)
    "32607",  # Lung biopsy, first
    "32608",  # Lung biopsy, additional
}

# All thoracoscopy codes (diagnostic)
ALL_THORACOSCOPY_CODES: Set[str] = (
    THORACOSCOPY_ALL_DIAGNOSTIC_CODES | THORACOSCOPY_LUNG_CODES
)

# Anatomic site synonyms for thoracoscopy code selection
THORACOSCOPY_PLEURAL_SYNONYMS: List[str] = [
    "pleura", "parietal pleura", "visceral pleura", "pleural plaques",
    "pleural nodules", "pleural surface", "pleural biops", "pleural biopsy",
    "pleuroscopy", "medical thoracoscopy",
]

THORACOSCOPY_PERICARDIAL_SYNONYMS: List[str] = [
    "pericardial sac", "pericardium", "pericardial biopsy", "pericardial",
    "pericardial space", "pericardial effusion biopsy",
]

THORACOSCOPY_MEDIASTINAL_SYNONYMS: List[str] = [
    "mediastinal space", "mediastinum", "mediastinal biopsy",
    "mediastinal mass", "mediastinal node",
]

THORACOSCOPY_LUNG_SYNONYMS: List[str] = [
    "lung parenchyma", "wedge resection", "wedge biopsy", "pulmonary nodule",
    "lung nodule", "lung biopsy", "lung mass", "parenchymal biopsy",
]

# Temporary drain terms (bundled into thoracoscopy - not separately billable)
THORACOSCOPY_BUNDLED_DRAIN_TERMS: List[str] = [
    "pigtail catheter", "pigtail removed", "catheter removed at end",
    "drain removed", "evacuated", "evacuating", "access catheter",
    "placed and removed", "temporary drain", "temporary catheter",
]

# Drains that ARE separately billable (left in place for ongoing drainage)
THORACOSCOPY_SEPARATE_DRAIN_TERMS: List[str] = [
    "left in place", "secured to skin", "connected to drainage",
    "chest tube left", "drain secured", "tunneled", "pleurx",
    "indwelling", "ongoing drainage", "to gravity drainage",
]

# Rule: Diagnostic bronchoscopy (31622) bundled with PDT light therapy (96570)
PDT_BUNDLES_DIAGNOSTIC: Dict[str, Set[str]] = {
    "dominant_codes": {"96570"},
    "bundled_codes": {"31622"},
}

# Rule: TBLB additional lobe (31632) only valid when multiple lobes sampled
TBLB_ADDITIONAL_REQUIRES_MULTIPLE_LOBES = {
    "add_on_code": "31632",
    "primary_code": "31628",
}

# Rule: Balloon dilation (31630) bundled with stent placement (31631) same site
STENT_BUNDLES_DILATION: Dict[str, Set[str]] = {
    "stent_codes": {"31631", "31636"},
    "dilation_codes": {"31630"},
}

# Rule: Tumor debulking (31640/31641) bundled with stent placement (31631) same site
STENT_BUNDLES_ABLATION: Dict[str, Set[str]] = {
    "stent_codes": {"31631", "31636"},
    "ablation_codes": {"31640", "31641"},
}

# Rule: Excision (31640) and destruction (31641) same site - choose one
EXCISION_DESTRUCTION_EXCLUSIVE: Set[str] = {"31640", "31641"}

# Rule: Ablation/destruction (31641) bundles dilation (31630) when same site
ABLATION_BUNDLES_DILATION: Dict[str, Set[str]] = {
    "ablation_codes": {"31641"},
    "dilation_codes": {"31630"},
}

# Rule: Balloon tamponade for hemorrhage bundled with therapeutic bronchoscopy
HEMORRHAGE_BUNDLES_BALLOON: Dict[str, Set[str]] = {
    "dominant_codes": {"31645"},
    "bundled_codes": {"31634"},
}

# Rule: Endobronchial blocker for bleeding control bundled into biopsy service
BIOPSY_BUNDLES_BLOCKER: Dict[str, Set[str]] = {
    "dominant_codes": {"31628"},
    "bundled_codes": {"31634"},
}

# Rule: TBNA (31629) bundled into TBLB (31628) when same lobe
TBLB_BUNDLES_TBNA: Dict[str, Set[str]] = {
    "dominant_codes": {"31628"},
    "bundled_codes": {"31629"},
}

# Rule: Endobronchial biopsy (31625) bundled into TBLB (31628) same lobe
TBLB_BUNDLES_EBB: Dict[str, Set[str]] = {
    "dominant_codes": {"31628"},
    "bundled_codes": {"31625"},
}

# Rule: EBUS 1-2 stations (31652) replaced by 3+ stations (31653) when >= 3 stations
EBUS_STATION_UPGRADE: Dict[str, str] = {
    "base_code": "31652",
    "upgrade_code": "31653",
    "threshold": 3,
}

# Rule: BAL (31624) bundled into TBLB (31628) in same lobe
BAL_BUNDLED_WITH_TBLB_SAME_LOBE: Dict[str, Set[str]] = {
    "dominant_codes": {"31628"},
    "bundled_codes": {"31624"},
}

# Rule: Fiducial markers (31626) at same lesion as biopsy often bundled
FIDUCIAL_AT_BIOPSY_SITE_BUNDLED: Dict[str, Set[str]] = {
    "dominant_codes": {"31628"},
    "bundled_codes": {"31626"},
}

# Rule: Complication management not coded separately (e.g., chest tube for airway injury)
COMPLICATION_MANAGEMENT_NOT_CODED: Set[str] = {"32557"}


# ============================================================================
# SYNONYM TRIGGERS derived from both sources
# These map clinical phrases to CPT code groups
# ============================================================================

# Navigation synonyms from golden knowledge + corrected patterns
NAVIGATION_SYNONYMS: List[str] = [
    "ion",
    "ion robotic",
    "ion robotic bronchoscopy",
    "robotic bronchoscopy",
    "robotic navigation",
    "robotic navigational bronchoscopy",
    "enb",
    "emn",
    "emn-guided",
    "emn guided",
    "emn bronchoscopy",
    "emn navigation",
    "electromagnetic navigation",
    "navigational bronchoscopy",
    "navigation bronchoscopy",
    "computer-assisted navigation",
    "computer-assisted, image-guided navigation",
    "superdimension",
    "spin thoracic navigation",
    "spin navigation",
    "shape-sensing",
    "shape sensing",
    "ct-based pathway",
    "ct based pathway",
    "virtual pathway",
    "pathway planned",
    "pathway planning",
    "3d navigation",
    "ct-to-body",
    "registration",
    "full registration",
]

# Radial EBUS synonyms
RADIAL_EBUS_SYNONYMS: List[str] = [
    "radial ebus",
    "radial probe",
    "radial endobronchial ultrasound",
    "rebus",
    "rebus-guided",
    "peripheral ebus",
    "miniprobe",
    "tool-in-lesion",
    "concentric view",
    "concentric lesion",
    "concentric pattern",
    "eccentric view",
]

# Linear EBUS synonyms
LINEAR_EBUS_SYNONYMS: List[str] = [
    "linear ebus",
    "ebus-tbna",
    "ebus tbna",
    "endobronchial ultrasound",
    "systematic nodal staging",
    "mediastinal staging",
    "nodal survey",
    "mediastinal survey",
    "hilar and mediastinal",
    "mediastinal and hilar",
]

# EBUS station terms (triggers linear EBUS)
EBUS_STATION_SYNONYMS: List[str] = [
    "station 4r",
    "station 4l",
    "station 7",
    "station 10r",
    "station 10l",
    "station 11r",
    "station 11l",
    "station 2r",
    "station 2l",
    "station 5",
    "stations 4r",
    "stations 7",
    "subcarinal",
    "4r, 7",
    "4r and 7",
    "7, 4r",
    "n2 disease",
    "n2 nodes",
]

# Transbronchial biopsy synonyms
TBLB_SYNONYMS: List[str] = [
    "transbronchial lung biopsy",
    "transbronchial biopsies",
    "transbronchial biopsy",
    "tblb",
    "tbblx",
    "tbbx",
    "cryobiopsy",
    "transbronchial cryobiopsy",
    "cryo-tbb",
    "forceps biopsy",
    "forceps biopsies",
    "peripheral biopsy",
    "peripheral biopsies",
    "lung biopsy",
    "lung biopsies",
]

# Ablation/destruction synonyms (31641)
ABLATION_SYNONYMS: List[str] = [
    "ablation",
    "radiofrequency ablation",
    "rfa",
    "cryoablation",
    "cryotherapy",
    "argon plasma coagulation",
    "apc",
    "electrocautery",
    "laser ablation",
    "tumor destruction",
    "destruction of tumor",
    "relief of stenosis",
    "debulking",
    "tumor debulking",
    "recanalization",
]

# Stent placement synonyms
STENT_SYNONYMS: List[str] = [
    "stent",
    "stent placement",
    "stent placed",
    "stent deployed",
    "stent deployment",
    "stent inserted",
    "stent insertion",
    "stent positioned",
    "bronchial stent",
    "tracheal stent",
    "silicone stent",
    "dumon stent",
    "metallic stent",
    "covered stent",
    "y-stent",
    "y stent",
    "sems",
    "self-expanding metal stent",
]

# Dilation synonyms
DILATION_SYNONYMS: List[str] = [
    "balloon dilation",
    "balloon dilatation",
    "serial balloon dilation",
    "cre balloon",
    "airway dilation",
    "bronchial dilation",
    "tracheal dilation",
]

# Therapeutic aspiration synonyms
THERAPEUTIC_ASPIRATION_SYNONYMS: List[str] = [
    "therapeutic aspiration",
    "airway toilet",
    "airway toileting",
    "bronchial toilet",
    "secretion clearance",
    "mucus plugging",
    "extensive suctioning",
    "aggressive suctioning",
    "hemoptysis",
    "bleeding control",
    "hemorrhage control",
    "tamponade",
]

# Foreign body removal synonyms (includes valve retrieval)
FOREIGN_BODY_SYNONYMS: List[str] = [
    "foreign body removal",
    "foreign body retrieval",
    "valve removal",
    "valve retrieval",
    "retrieval of valve",
    "retrieval of valves",
    "retrieval of three zephyr",
    "retrieval of zephyr",
    "zephyr valve removed",
    "zephyr valves removed",
    "valves removed",
    "removed valve",
    "removed valves",
    "endobronchial valves removed",
    "endobronchial valve removal",
    "aspirated object removal",
    "stent removal",
    "stent retrieved",
]

# IPC/tunneled pleural catheter synonyms
IPC_SYNONYMS: List[str] = [
    "tunneled pleural catheter",
    "indwelling pleural catheter",
    "ipc placement",
    "ipc",
    "pleurx catheter",
    "pleurx",
    "tunneled pleurx",
    "tunneling device",
    "subcutaneous cuff",
]

# Thoracentesis synonyms
THORACENTESIS_SYNONYMS: List[str] = [
    "thoracentesis",
    "pleural tap",
    "pleural fluid aspiration",
    "pleural fluid drainage",
    "ultrasound-guided thoracentesis",
    "therapeutic thoracentesis",
    "diagnostic thoracentesis",
]

# Pleurodesis synonyms
PLEURODESIS_SYNONYMS: List[str] = [
    "pleurodesis",
    "talc pleurodesis",
    "talc slurry",
    "talc poudrage",
    "chemical pleurodesis",
    "mechanical pleurodesis",
    "instillation of agent for pleurodesis",
]

# Thoracoscopy/pleuroscopy synonyms
THORACOSCOPY_SYNONYMS: List[str] = [
    "thoracoscopy",
    "medical thoracoscopy",
    "pleuroscopy",
    "thoracoscopy with pleurodesis",
    "surgical thoracoscopy",
    "vats",
    "video-assisted thoracoscopic",
]

# BAL synonyms
BAL_SYNONYMS: List[str] = [
    "bronchoalveolar lavage",
    "bal",
    "lavage",
]

# PDT synonyms
PDT_SYNONYMS: List[str] = [
    "photodynamic therapy",
    "pdt",
    "photofrin",
    "light application",
    "630 nm laser",
]


def load_canonical_patterns() -> List[Dict]:
    """Load the canonical synthetic_CPT_corrected.json patterns."""
    # Path is: app/autocode/ip_kb -> autocode -> app -> repo_root
    repo_root = Path(__file__).parent.parent.parent.parent
    patterns_path = repo_root / "data" / "synthetic_CPT_corrected.json"
    if patterns_path.exists():
        with open(patterns_path, "r") as f:
            return json.load(f)
    return []


def load_golden_rules() -> Dict:
    """Load the golden ip_golden_knowledge_v2_2.json rules."""
    # Path is: app/autocode/ip_kb -> autocode -> app -> repo_root
    repo_root = Path(__file__).parent.parent.parent.parent
    rules_path = repo_root / "ip_golden_knowledge_v2_2.json"
    if rules_path.exists():
        with open(rules_path, "r") as f:
            return json.load(f)
    return {}


def extract_bundling_rules_from_patterns(patterns: List[Dict]) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Extract bundling rules from the excluded_or_bundled_codes in synthetic_CPT_corrected.json.

    Returns dict of:
        bundled_code -> [(context_code, description, reason), ...]
    """
    rules: Dict[str, List[Tuple[str, str, str]]] = {}

    for pattern in patterns:
        coding = pattern.get("coding_and_billing", {})
        billed = coding.get("billed_codes", [])
        excluded = coding.get("excluded_or_bundled_codes", [])

        billed_cpts = {c["cpt_code"] for c in billed}

        for exc in excluded:
            exc_code = exc.get("cpt_code", "").replace("/", ",").split(",")[0].strip()
            reason = exc.get("reason", "")
            desc = exc.get("description", "")

            if exc_code:
                if exc_code not in rules:
                    rules[exc_code] = []
                # Record which billed codes trigger this bundling
                for bc in billed_cpts:
                    rules[exc_code].append((bc, desc, reason))

    return rules


def get_code_for_station_count(station_count: int) -> str:
    """
    Return the appropriate EBUS code based on station count.
    Per canonical rules:
    - 1-2 stations: 31652
    - 3+ stations: 31653
    """
    if station_count >= 3:
        return "31653"
    return "31652"
