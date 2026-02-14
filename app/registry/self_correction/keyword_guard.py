"""Guardrails for registry extraction self-correction (Phase 6).

Includes:
- CPT keyword gating (prevents self-correction when evidence text lacks keywords)
- Omission detection (detects high-value terms in raw text that were missed)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from app.common.logger import get_logger
from app.common.spans import Span
from app.registry.schema import RegistryRecord

logger = get_logger("keyword_guard")

# Minimal CPT → keywords mapping for allowlisted targets.
# This is intentionally conservative: if a CPT has no keywords configured, the
# guard fails and self-correction is skipped for that CPT.
CPT_KEYWORDS: dict[str, list[str]] = {
    # Pleural: indwelling pleural catheter (IPC / PleurX)
    "32550": ["pleurx", "indwelling pleural catheter", "tunneled pleural catheter", "ipc"],
    # Pleural: thoracentesis
    "32554": ["thoracentesis", "pleural fluid removed", "tap"],
    "32555": ["thoracentesis", "pleural fluid removed", "tap"],
    # Pleural: chest tube / thoracostomy
    "32551": ["chest tube", "tube thoracostomy", "thoracostomy"],
    # Pleural: intrapleural fibrinolysis (initial/subsequent day)
    "32561": ["fibrinolysis", "fibrinolytic", "tpa", "alteplase", "dnase", "dornase"],
    "32562": ["fibrinolysis", "fibrinolytic", "tpa", "alteplase", "dnase", "dornase", "subsequent day"],
    # Pleural: percutaneous pleural drainage catheter (pigtail) without/with imaging
    "32556": ["pigtail catheter", "pleural drainage", "seldinger"],
    "32557": ["pigtail catheter", "pleural drainage", "ultrasound", "imaging guidance", "seldinger"],
    # Diagnostic chest ultrasound
    "76604": ["chest ultrasound", "ultrasound findings", "with image documentation", "image saved"],
    # Bronchoscopy add-ons / performed flags
    "31623": ["brushing", "brushings", "bronchial brushing"],
    "31624": [
        "bronchoalveolar lavage",
        "broncho alveolar lavage",
        "broncho-alveolar lavage",
        "bronchial alveolar lavage",
        "bal",
        "lavage",
    ],
    "31626": ["fiducial", "fiducial marker", "fiducial placement"],
    "31628": [
        "transbronchial biopsy",
        "transbronchial biops",
        "transbronchial bx",
        "transbronchial forceps biopsy",
        "transbronchial forceps biops",
        "tbbx",
        "tblb",
    ],
    "31632": [
        "transbronchial biopsy",
        "transbronchial biops",
        "transbronchial bx",
        "additional lobe",
        "second lobe",
        "third lobe",
        "multiple lobes",
        "tbbx",
        "tblb",
    ],
    "31629": ["tbna", "transbronchial needle aspiration", "transbronchial needle"],
    "31633": [
        "tbna",
        "transbronchial needle aspiration",
        "transbronchial needle",
        "additional lobe",
        "second lobe",
        "third lobe",
        "multiple lobes",
    ],
    "31652": [
        "ebus",
        "endobronchial ultrasound",
        "tbna",
        "transbronchial needle aspiration",
        "transbronchial needle",
    ],
    "31653": [
        "ebus",
        "endobronchial ultrasound",
        "tbna",
        "transbronchial needle aspiration",
        "transbronchial needle",
        "ebus lymph nodes sampled",
        "lymph nodes sampled",
        "lymph node stations",
        "site 1",
        "site 2",
        "site 3",
        "site 4",
        "subcarinal",
        "11l",
        "11rs",
        "11ri",
        "4r",
        "4l",
        "10r",
        "10l",
    ],
    # Tumor debulking / destruction
    "31640": [
        "31640",
        "mechanical debulk",
        "mechanical excision",
        "tumor excision",
        "forceps debulk",
        "rigid coring",
        "microdebrider",
        "snare resection",
    ],
    "31641": [
        "31641",
        "apc",
        "argon plasma",
        "electrocautery",
        "laser",
        "ablation",
        "tumor base ablation",
        "cryotherapy",
    ],
    "31654": ["radial ebus", "radial ultrasound", "rp-ebus", "r-ebus", "rebus", "miniprobe"],
    "31627": [
        "navigational bronchoscopy",
        "navigation",
        "electromagnetic navigation",
        "enb",
        "ion",
        "intuitive ion",
        "robotic bronchoscopy",
        "monarch",
        "galaxy",
        "planning station",
    ],
    "43238": [
        "eus-b",
        "eus b",
        "endoscopic ultrasound",
        "transesophageal",
        "transgastric",
        "left adrenal",
        "adrenal mass",
        "eusb",
    ],
    "76982": [
        "elastography",
        "elastrography",
        "type 1 elastographic",
        "type 2 elastographic",
        "stiff",
        "soft (green",
        "blue)",
    ],
    "76983": [
        "elastography",
        "elastrography",
        "additional target",
        "additional targets",
        "type 1 elastographic",
        "type 2 elastographic",
    ],
    "76981": [
        "elastography",
        "elastrography",
        "type 1 elastographic",
        "type 2 elastographic",
        "stiff",
        "soft (green",
        "blue)",
    ],
    "77012": [
        "cone beam ct",
        "cone-beam ct",
        "cios",
        "spin system",
        "ct guidance",
        "ct guided",
        "3d reconstruction",
    ],
    "76377": [
        "3d rendering",
        "3-d reconstruction",
        "3d reconstruction",
        "planning station",
        "ion planning station",
    ],
    # Therapeutics: dilation
    "31630": ["balloon", "dilation", "dilate", "dilated"],
    "31631": ["balloon", "dilation", "dilate", "dilated"],
    # Therapeutics: airway stent
    "31636": ["stent", "silicone", "metal", "metallic", "hybrid", "y-stent", "dumon", "ultraflex", "aero"],
    "31637": ["stent", "silicone", "metal", "metallic", "hybrid", "y-stent", "dumon", "ultraflex", "aero"],
    "31638": ["stent", "removal", "removed", "retrieved", "extracted", "forceps", "silicone", "metal", "metallic"],
    # Therapeutics: foreign body removal
    "31635": ["foreign body", "removed", "remove", "extracted", "retrieved", "forceps"],
    # Therapeutics: therapeutic aspiration (initial/subsequent episode)
    "31645": [
        "therapeutic aspiration",
        "mucus plug",
        "mucous plug",
        "mucus plugging",
        "clot removal",
        "copious secretions",
        "tenacious secretions",
        "airway cleared",
        "suctioned",
        "aspirated",
    ],
    "31646": [
        "therapeutic aspiration",
        "subsequent",
        "repeat aspiration",
        "second aspiration",
        "mucus plug",
        "clot removal",
    ],
    # BLVR valve family (initial + add-on lobe)
    "31647": ["valve", "zephyr", "spiration", "endobronchial valve", "blvr"],
    "31648": ["valve removal", "valve removed", "remove valve", "zephyr", "spiration", "endobronchial valve", "blvr"],
    "31649": [
        "valve removal",
        "valve removed",
        "remove valve",
        "additional lobe",
        "second lobe",
        "multiple lobes",
        "zephyr",
        "spiration",
        "endobronchial valve",
        "blvr",
    ],
    "31651": ["valve", "zephyr", "spiration", "endobronchial valve", "blvr"],
    # Pleurodesis (chemical / thoracoscopic)
    "32560": ["pleurodesis", "chemical pleurodesis", "talc", "slurry", "poudrage", "sclerosing agent"],
    "32650": ["pleurodesis", "chemical pleurodesis", "talc", "slurry", "poudrage", "sclerosing agent"],
    # Thoracoscopy / pleuroscopy (biopsy / lysis of adhesions)
    "32609": ["thoracoscopy", "pleuroscopy", "biopsy of pleura", "lysis of adhesions"],
    "32653": ["thoracoscopy", "pleuroscopy", "biopsy of pleura", "lysis of adhesions"],
}

# High-confidence bypass: allow self-correction when RAW-ML has very high confidence
# but evidence text is partially masked (e.g., CPT/menu blocks removed).
HIGH_CONF_BYPASS_CPTS: frozenset[str] = frozenset({"31647", "31651", "32609", "32653"})
HIGH_CONF_BYPASS_THRESHOLD = 0.90

# Optional generated CPT-keyword mapping support.
DEFAULT_GENERATED_CPT_KEYWORDS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "keyword_mappings" / "cpt_keywords.generated.json"
)

_EFFECTIVE_CPT_KEYWORDS_CACHE: dict[str, list[str]] | None = None
_EFFECTIVE_CPT_KEYWORDS_CACHE_KEY: tuple[bool, str, int] | None = None


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y"}


def _normalize_keyword_value(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _dedupe_keywords(values: list[object], *, min_length: int = 1) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = _normalize_keyword_value(value)
        if len(norm) < min_length:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _load_generated_cpt_keywords(path: Path) -> dict[str, list[str]] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(
            "REGISTRY_KEYWORD_GUARD_USE_GENERATED enabled but generated keyword file not found",
            extra={"path": str(path)},
        )
        return None
    except json.JSONDecodeError as exc:
        logger.warning(
            "Generated keyword file is invalid JSON; falling back to baseline keywords",
            extra={"path": str(path), "error": str(exc)},
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Failed reading generated keyword file; falling back to baseline keywords",
            extra={"path": str(path), "error": str(exc)},
        )
        return None

    if not isinstance(raw, dict):
        logger.warning(
            "Generated keyword file must contain an object; falling back to baseline keywords",
            extra={"path": str(path)},
        )
        return None

    generated: dict[str, list[str]] = {}
    for cpt, phrases in raw.items():
        if not isinstance(phrases, list):
            continue
        cleaned = _dedupe_keywords(phrases, min_length=3)
        if cleaned:
            generated[str(cpt)] = cleaned
    return generated


def get_effective_cpt_keywords(*, force_refresh: bool = False) -> dict[str, list[str]]:
    """Return effective CPT keywords (baseline or baseline+generated)."""
    global _EFFECTIVE_CPT_KEYWORDS_CACHE
    global _EFFECTIVE_CPT_KEYWORDS_CACHE_KEY

    use_generated = _truthy_env("REGISTRY_KEYWORD_GUARD_USE_GENERATED")
    generated_path_raw = os.getenv("REGISTRY_KEYWORD_GUARD_GENERATED_PATH", "").strip()
    generated_path = Path(generated_path_raw) if generated_path_raw else DEFAULT_GENERATED_CPT_KEYWORDS_PATH
    generated_mtime_ns = -1
    if use_generated:
        try:
            generated_mtime_ns = generated_path.stat().st_mtime_ns
        except OSError:
            generated_mtime_ns = -1

    cache_key = (use_generated, str(generated_path), generated_mtime_ns)
    if not force_refresh and _EFFECTIVE_CPT_KEYWORDS_CACHE_KEY == cache_key and _EFFECTIVE_CPT_KEYWORDS_CACHE is not None:
        return _EFFECTIVE_CPT_KEYWORDS_CACHE

    effective = {cpt: _dedupe_keywords(keywords, min_length=1) for cpt, keywords in CPT_KEYWORDS.items()}
    if use_generated:
        generated = _load_generated_cpt_keywords(generated_path)
        if generated:
            for cpt, keywords in generated.items():
                baseline = effective.get(cpt, [])
                effective[cpt] = _dedupe_keywords([*baseline, *keywords], min_length=1)

    _EFFECTIVE_CPT_KEYWORDS_CACHE = effective
    _EFFECTIVE_CPT_KEYWORDS_CACHE_KEY = cache_key
    return effective

# -----------------------------------------------------------------------------
# Omission Detection ("Safety Net")
# -----------------------------------------------------------------------------
# Dictionary mapping a Registry Field Path -> List of (Regex, Failure Message).
# If the Regex matches the text, the Registry Field MUST be True (or populated).
# -----------------------------------------------------------------------------
REQUIRED_PATTERNS: dict[str, list[tuple[str, str]]] = {
    # Fixes missed tracheostomy (Report #3)
    "procedures_performed.percutaneous_tracheostomy.performed": [
        (
            r"(?is)\b(?:perform|create|place|insert)\w*\b.{0,20}\btracheostomy\b",
            "Text indicates tracheostomy creation but extraction missed it.",
        ),
        (
            r"(?is)\b(?:perform|create|place|insert)\w*\b.{0,20}\bperc(?:utaneous)?\s+trach\b",
            "Text indicates percutaneous trach creation but extraction missed it.",
        ),
        (
            r"(?is)\bpercutaneous\s+tracheostomy\b.{0,40}\b(?:perform|create|place|insert)\w*\b",
            "Text indicates percutaneous tracheostomy but extraction missed it.",
        ),
    ],
    # Fixes missed endobronchial biopsy (Report #2)
    "procedures_performed.endobronchial_biopsy.performed": [
        (r"(?i)\bendobronchial\s+biopsy\b", "Text explicitly states 'endobronchial biopsy'."),
        (r"(?i)\bebbx\b", "Text contains 'EBBx' (endobronchial biopsy abbreviation)."),
        (r"(?i)\blesions?\s+were\s+biopsied\b", "Text states 'lesions were biopsied' (likely endobronchial)."),
        (r"(?i)\bbiopsy\s+of\s+(?:the\s+)?(?:lesion|mass|polyp)\b", "Text describes biopsy of a lesion/mass/polyp."),
    ],
    # Fix for missed BAL
    "procedures_performed.bal.performed": [
        (r"(?i)\bbroncho[-\s]?alveolar\s+lavage\b", "Text contains 'bronchoalveolar lavage' but extraction missed it."),
        (r"(?i)\bbronchial\s+alveolar\s+lavage\b", "Text contains 'bronchial alveolar lavage' but extraction missed it."),
        (r"(?i)\bbal\b(?!\s*score)", "Text contains 'BAL' but extraction missed it."),
    ],
    # Fix for missed radial EBUS
    "procedures_performed.radial_ebus.performed": [
        (r"(?i)\bradial\s+ebus\b", "Text contains 'radial EBUS' but extraction missed it."),
        (r"(?i)\bradial\s+probe\s+ebus\b", "Text contains 'radial probe EBUS' but extraction missed it."),
        (r"(?i)\bradial\s+probe\b", "Text contains 'radial probe' (radial EBUS) but extraction missed it."),
        (r"(?i)\br-?ebus\b", "Text contains 'rEBUS' but extraction missed it."),
        (r"(?i)\brp-?ebus\b", "Text contains 'rp-EBUS' but extraction missed it."),
        (r"(?i)\bminiprobe\b", "Text contains 'miniprobe' (radial EBUS) but extraction missed it."),
    ],
    # Fix for missed linear EBUS
    "procedures_performed.linear_ebus.performed": [
        (r"(?i)\b(?:linear|convex)\s+ebus\b", "Text contains 'linear/convex EBUS' but extraction missed it."),
        (r"(?i)\bebus[- ]?tbna\b", "Text contains 'EBUS-TBNA' but extraction missed linear EBUS."),
        (r"(?i)EBUS[- ]Findings", "Text contains 'EBUS Findings' but extraction missed linear EBUS."),
        (r"(?i)EBUS Lymph Nodes Sampled", "Text contains 'EBUS Lymph Nodes Sampled' but extraction missed linear EBUS."),
        (
            r"(?is)\b(?:ebus|endobronchial\s+ultrasound)\b.{0,200}\b(?:lymph\s+node(?:s)?|lymph\s+nodes\s+sampled|lymph\s+node\s+stations?)\b",
            "Text mentions EBUS lymph node sampling but extraction missed linear EBUS.",
        ),
        (
            r"(?is)\b(?:ebus|endobronchial\s+ultrasound)\b.{0,200}\b(?:station|level)\s*\d+[RL]?\b",
            "Text mentions EBUS station/level numbers but extraction missed linear EBUS.",
        ),
    ],
    # Fix for missed EBUS elastography (schema uses linear_ebus.* fields).
    "procedures_performed.linear_ebus.elastography_used": [
        (
            r"(?i)\b(?:ebus[-\s]*)?elastograph(?:y|ic)\b",
            "Text indicates EBUS elastography but extraction missed it.",
        ),
        (
            r"(?i)\btype\s*[123]\s*elastographic\s+pattern\b",
            "Text documents elastographic pattern types but extraction missed elastography.",
        ),
    ],
    # Fix for missed EUS-B
    "procedures_performed.eus_b.performed": [
        (r"(?i)\bEUS-?B\b", "Text contains 'EUS-B' but extraction missed EUS-B."),
        (r"(?i)\bleft adrenal\b", "Text contains left adrenal mass evaluation but extraction missed EUS-B."),
        (r"(?i)\btransgastric\b|\btransesophageal\b", "Text contains transgastric/transesophageal sampling but extraction missed EUS-B."),
    ],
    # Fix for missed cryotherapy / tumor destruction
    "procedures_performed.cryotherapy.performed": [
        (r"(?i)\bcryotherap(?:y|ies)\b", "Text mentions 'cryotherapy' but extraction missed it."),
        (r"(?i)\bcryo(?:therapy|ablation|debulk(?:ing)?)\b", "Text mentions cryotherapy/cryo debulking but extraction missed it."),
    ],
    # Fixes missed neck ultrasound (Report #3)
    "procedures_performed.neck_ultrasound.performed": [
        (r"(?i)\bneck\s+ultrasound\b", "Text contains 'neck ultrasound'."),
        (r"(?i)\bultrasound\s+of\s+(?:the\s+)?neck\b", "Text contains 'ultrasound of the neck'."),
    ],
    # Fix for missed brushings (Report #1 & #7)
    "procedures_performed.brushings.performed": [
        (r"(?i)\bbrush(?:ings?)?\b", "Text mentions 'brush' or 'brushings'."),
        (r"(?i)triple\s+needle", "Text mentions 'triple needle' (implies brushing/sampling)."),
    ],
    # Fix for missed transbronchial biopsy (forceps TBLB/TBBx) in peripheral/radial cases.
    "procedures_performed.transbronchial_biopsy.performed": [
        (
            r"(?i)\b(?:transbronchial\s+(?:lung\s+)?biops(?:y|ies)|transbronchial\s+forceps\s+biops(?:y|ies)|tbbx|tblb)\b",
            "Text indicates transbronchial (lung) biopsy but extraction missed it.",
        ),
        (
            r"(?is)\b(?:radial\s+(?:ebus|ultrasound)|rebus|miniprobe)\b.{0,250}\b(?:forceps\b.{0,40}\bbiops(?:y|ies)|biops(?:y|ies)\b.{0,40}\bforceps)\b",
            "Text indicates peripheral biopsy with radial EBUS guidance but extraction missed transbronchial biopsy.",
        ),
        (
            r"(?is)\b(?:guide\s+sheath|sheath\s+catheter|large\s+sheath|guide\s+catheter)\b.{0,250}\b(?:forceps\b.{0,40}\bbiops(?:y|ies)|biops(?:y|ies)\b.{0,40}\bforceps)\b",
            "Text indicates peripheral biopsy through a sheath but extraction missed transbronchial biopsy.",
        ),
        (
            r"(?is)\bfluoro(?:scop\w*)?\b.{0,250}\b(?:guide\s+sheath|sheath)\b.{0,250}\b(?:biops(?:y|ies)|forceps)\b",
            "Text indicates peripheral biopsy with fluoroscopic + sheath guidance but extraction missed transbronchial biopsy.",
        ),
        (
            r"(?i)\bperipheral\s+needle\s+forceps\b",
            "Text indicates peripheral forceps biopsy but extraction missed transbronchial biopsy.",
        ),
    ],
    # Fix for missed peripheral TBNA (lung/peripheral targets).
    # NOTE: Generic "TBNA" also appears in EBUS sections; patterns here require
    # navigation/peripheral-lesion context to avoid forcing nodal TBNA flags.
    "procedures_performed.peripheral_tbna.performed": [
        (
            r"(?is)\b(?:ion|robotic|navigation|navigational|\benb\b|monarch|galaxy|superdimension|peripheral|target\s+lesion|lesion|nodule|(?:lung|pulmonary)\s+(?:lesion|nodule|mass))\b.{0,250}\b(?:tbna|transbronchial\s+needle\s+aspiration|transbronchial\s+needle)\b",
            "Text indicates peripheral/lung TBNA but extraction missed it.",
        ),
        (
            r"(?is)\b(?:tbna|transbronchial\s+needle\s+aspiration|transbronchial\s+needle)\b.{0,250}\b(?:ion|robotic|navigation|navigational|\benb\b|monarch|galaxy|superdimension|peripheral|target\s+lesion|lesion|nodule|(?:lung|pulmonary)\s+(?:lesion|nodule|mass))\b",
            "Text indicates peripheral/lung TBNA but extraction missed it.",
        ),
        (
            r"(?is)\bendobronchial\s+needle\s+biops(?:y|ies)\b.{0,120}\b(?:nodule|lesion|tumou?r|mass)\b",
            "Text indicates non-EBUS needle biopsy of a lesion but extraction missed it.",
        ),
        (
            r"(?is)\b(?:nodule|lesion|tumou?r|mass)\b.{0,120}\bendobronchial\s+needle\s+biops(?:y|ies)\b",
            "Text indicates non-EBUS needle biopsy of a lesion but extraction missed it.",
        ),
    ],
    # Fix for missed conventional (non-EBUS) nodal TBNA.
    "procedures_performed.tbna_conventional.performed": [
        (r"(?i)\bconventional\s+tbna\b", "Text explicitly states 'conventional TBNA' but extraction missed it."),
        (r"(?i)\bblind\s+tbna\b", "Text explicitly states 'blind TBNA' but extraction missed it."),
        (
            r"(?is)\b(?:station|ln|lymph\s+node)\b[^.\n]{0,80}\b(?:2R|2L|4R|4L|5|7|8|9|10R|10L|11R(?:S|I)?|11L(?:S|I)?|12R|12L)\b[^.\n]{0,120}\b(?:tbna|transbronchial\s+needle)\b",
            "Text indicates nodal TBNA at a lymph node station but extraction missed it.",
        ),
    ],
    # Fix for missed navigational bronchoscopy
    "procedures_performed.navigational_bronchoscopy.performed": [
        (r"(?i)\bnavigational\s+bronchoscopy\b", "Text mentions navigational bronchoscopy."),
        (r"(?i)\belectromagnetic\s+navigation\b", "Text mentions electromagnetic navigation."),
        (r"(?i)\benb\b", "Text mentions ENB navigation."),
        (r"(?i)\bion\b", "Text mentions ION navigation."),
        (r"(?i)\bmonarch\b", "Text mentions Monarch navigation."),
        (r"(?i)\brobotic\s+bronchoscopy\b", "Text mentions robotic bronchoscopy."),
        (r"(?i)\bsuperdimension\b", "Text mentions SuperDimension navigation."),
    ],
    # Fix for missed transbronchial cryobiopsy
    "procedures_performed.transbronchial_cryobiopsy.performed": [
        (r"(?i)\btransbronchial\s+cryo\b", "Text mentions transbronchial cryo biopsy."),
        (r"(?i)\bcryo\s*biops(?:y|ies)\b", "Text mentions cryobiopsy."),
        (r"(?i)\bcryobiops(?:y|ies)\b", "Text mentions cryobiopsy."),
        (r"(?i)\btblc\b", "Text mentions TBLC."),
    ],
    # Fix for missed fiducial marker placement
    "granular_data.navigation_targets.fiducial_marker_placed": [
        (r"(?i)\bfiducial\s+marker\b", "Text mentions fiducial marker placement."),
        (r"(?i)\bfiducial\s+placement\b", "Text mentions fiducial placement."),
        (r"(?i)\bfiducials?\b[^.\n]{0,40}\bplaced\b", "Text mentions fiducials placed."),
    ],
    # Fix for missed peripheral ablation
    "procedures_performed.peripheral_ablation.performed": [
        (r"(?i)\bmicrowave\s+ablation\b", "Text mentions microwave ablation."),
        (r"(?i)\bmwa\b", "Text mentions MWA."),
        (r"(?i)\bradiofrequency\s+ablation\b", "Text mentions radiofrequency ablation."),
        (r"(?i)\brfa\b", "Text mentions RFA."),
        (r"(?i)\bcryoablation\b", "Text mentions cryoablation."),
        (r"(?i)\bcryo\s*ablation\b", "Text mentions cryo ablation."),
    ],
    # Fix for missed rigid bronchoscopy (Report #3)
    "procedures_performed.rigid_bronchoscopy.performed": [
        (r"(?i)rigid\s+bronchoscop", "Text mentions 'rigid bronchoscopy'."),
        (r"(?i)rigid\s+optic", "Text mentions 'rigid optic'."),
        (r"(?i)rigid\s+barrel", "Text mentions 'rigid barrel'."),
    ],
    # Fix for missed thermal/ablation keywords (Report #3)
    "procedures_performed.thermal_ablation.performed": [
        (r"(?i)electrocautery", "Text mentions 'electrocautery'."),
        (r"(?i)\blaser\b", "Text mentions 'laser'."),
        (r"(?i)\bapc\b", "Text mentions 'APC' (Argon Plasma Coagulation)."),
        (r"(?i)argon\s+plasma", "Text mentions 'Argon Plasma'."),
    ],
    # Fix for missed mechanical debulking / excision (31640 family)
    "procedures_performed.mechanical_debulking.performed": [
        (r"(?i)\bmechanical\s+debulk(?:ing)?\b", "Text explicitly mentions mechanical debulking."),
        (
            r"(?is)\b(?:snare|microdebrider|microdebrid\w*|rigid\s+coring)\b.{0,220}\b(?:en\s+bloc|resect|excise|excision|remove(?:d)?)\b",
            "Text indicates mechanical excision/debulking (e.g., snare resection / en bloc removal).",
        ),
    ],
    # Pleural: chest tube / pleural drainage catheter placement
    "pleural_procedures.chest_tube.performed": [
        (r"(?i)\bpigtail\s+catheter\b", "Text mentions 'pigtail catheter' (pleural drain)."),
        (
            r"(?is)\b(?:placed|placement|insert(?:ed|ion)?|tube\s+thoracostomy|thoracostomy|seldinger)\b"
            r"[^.\n]{0,80}\bchest\s+tube\b"
            r"|\bchest\s+tube\b[^.\n]{0,80}\b(?:placed|placement|insert(?:ed|ion)?|tube\s+thoracostomy|thoracostomy|seldinger)\b",
            "Text indicates chest tube placement/insertion but extraction missed it.",
        ),
        (r"(?i)\btube\s+thoracostomy\b", "Text mentions 'tube thoracostomy'."),
    ],
    # Pleural: intrapleural fibrinolysis via chest tube/catheter (32561/32562)
    "pleural_procedures.fibrinolytic_therapy.performed": [
        (r"\b32561\b", "Text lists CPT 32561 (intrapleural fibrinolysis; initial day)."),
        (r"\b32562\b", "Text lists CPT 32562 (intrapleural fibrinolysis; subsequent day)."),
        (
            r"(?is)\binstillat\w*\b[^.\n]{0,120}\b(?:tpa|alteplase|dnase|dornase|fibrinolys(?:is|tic))\b",
            "Text indicates intrapleural fibrinolytic instillation (tPA/DNase).",
        ),
        (
            r"(?is)\b(?:tpa|alteplase)\b[^.\n]{0,120}\b(?:dnase|dornase)\b",
            "Text indicates combined tPA/DNase intrapleural therapy.",
        ),
    ],
    # Diagnostic chest ultrasound (76604)
    "procedures_performed.chest_ultrasound.performed": [
        (r"(?i)\bchest\s+ultrasound\s+findings\b", "Text contains 'Chest ultrasound findings'."),
        (r"(?i)\bultrasound,\s*chest\b", "Text contains 'Ultrasound, chest'."),
        (r"\b76604\b", "Text lists CPT 76604 (chest ultrasound)."),
    ],
}

_NEGATION_CUES = r"(?:no|not|without|declined|deferred|aborted)"

# Field-specific "do not treat as performed" cues.
#
# Example: "D/c chest tube" should not trigger a chest tube placement override.
_CHEST_TUBE_REMOVAL_CUES_RE = re.compile(
    r"(?i)(?:\bd/c\b|\bdc\b|\bdiscontinu(?:e|ed|ation)\b|\bremove(?:d|al)?\b|\bpull(?:ed)?\b|\bwithdrawn\b)"
)
_CHEST_TUBE_INSERTION_CUES_RE = re.compile(
    r"(?i)\b(?:place(?:d|ment)?|insert(?:ed|ion)?|tube\s+thoracostomy|thoracostomy|pigtail|seldinger)\b"
)
_TBNA_EBUS_CONTEXT_RE = re.compile(
    r"(?i)\b(?:ebus|endobronchial\s+ultrasound|convex\s+probe|ebus[-\s]?tbna)\b"
)
_EBUS_STATION_TOKEN_RE = re.compile(
    r"(?i)\b(?:2R|2L|4R|4L|7|10R|10L|11R(?:S|I)?|11L(?:S|I)?)\b"
)
_TBNA_TERM_RE = re.compile(
    r"(?i)\b(?:tbna|transbronchial\s+needle\s+aspiration|transbronchial\s+needle)\b"
)

_MUCUS_CUE_RE = re.compile(r"(?i)\b(?:mucous|mucus|secretions?|clot|plug|mucostasis)\b")
_TISSUE_DEBULKING_CUE_RE = re.compile(
    r"(?i)\b(?:tumou?r|mass|lesion|mycetoma|granulation|neoplasm|endobronchial\s+(?:tumou?r|mass|lesion))\b"
)
_PERIPHERAL_ABLATION_PERIPHERAL_CUE_RE = re.compile(
    r"(?i)\b(?:peripheral|nodule|lesion|mass|parenchym|target\s+lesion|lung\s+nodule|pulmonary\s+nodule|"
    r"navigation|navigational|robotic|ion|cbct|cone\s*beam|tool[- ]?in[- ]?lesion)\b"
)
_PERIPHERAL_ABLATION_ENDOBRONCHIAL_CUE_RE = re.compile(
    r"(?i)\b(?:endobronch|airway|trachea|carina|main(?:\s*|-)?stem|bronch(?:us|ial)|stenos|stricture)\b"
)


def _sentence_window(note_text: str, *, start: int, end: int) -> str:
    if not note_text:
        return ""
    start = max(0, min(len(note_text), start))
    end = max(0, min(len(note_text), end))

    left = max(
        note_text.rfind(".", 0, start),
        note_text.rfind("!", 0, start),
        note_text.rfind("?", 0, start),
        note_text.rfind("\n", 0, start),
    )
    left = 0 if left == -1 else left + 1

    right_candidates = [
        note_text.find(".", end),
        note_text.find("!", end),
        note_text.find("?", end),
        note_text.find("\n", end),
    ]
    right_candidates = [pos for pos in right_candidates if pos != -1]
    right = min(right_candidates) if right_candidates else len(note_text)

    return note_text[left:right]


def _looks_like_mucus_debulking_only(note_text: str, match: re.Match[str]) -> bool:
    """Return True when 'mechanical debulking' refers to mucus/secretions (not tissue excision)."""
    if not note_text:
        return False
    sentence = _sentence_window(note_text, start=match.start(), end=match.end())
    if not sentence:
        return False
    if not _MUCUS_CUE_RE.search(sentence):
        return False
    if _TISSUE_DEBULKING_CUE_RE.search(sentence):
        return False
    return True


def _looks_like_ebus_nodal_tbna_only(note_text: str, match: re.Match[str]) -> bool:
    """Return True when the match sits in an EBUS nodal TBNA paragraph (not peripheral TBNA)."""
    if not note_text:
        return False
    # Prefer a local window around the match: notes often contain both an EBUS
    # section (stations) and separate non-EBUS needle sampling elsewhere; using
    # the full paragraph can incorrectly suppress distinct-site TBNA.
    local_start = max(0, match.start() - 300)
    local_end = min(len(note_text), match.end() + 300)
    window = note_text[local_start:local_end]

    tbna_hit = _TBNA_TERM_RE.search(window)
    if not tbna_hit:
        return False

    # Focus the disambiguation around the TBNA wording itself (not an entire
    # long paragraph that may include unrelated peripheral-biopsy content).
    tbna_abs_start = local_start + tbna_hit.start()
    tbna_abs_end = local_start + tbna_hit.end()
    tbna_window = note_text[max(0, tbna_abs_start - 180) : min(len(note_text), tbna_abs_end + 180)]

    # Specimen sections often omit the literal token "EBUS" but still reference
    # nodal stations (e.g., "TBNA station 11R and 4R"). Treat station tokens as
    # sufficient evidence of nodal (non-peripheral) TBNA in local context.
    if _EBUS_STATION_TOKEN_RE.search(tbna_window) or re.search(r"(?i)\bstation\(s\)?\b", tbna_window):
        return True
    return bool(_TBNA_EBUS_CONTEXT_RE.search(tbna_window))


def _looks_like_ebus_nodal_context(note_text: str, match: re.Match[str]) -> bool:
    """Return True when a match sits in an EBUS lymph-node context (not lung parenchyma)."""
    if not note_text:
        return False

    local_start = max(0, match.start() - 260)
    local_end = min(len(note_text), match.end() + 260)
    window = note_text[local_start:local_end]

    if not _TBNA_EBUS_CONTEXT_RE.search(window):
        return False
    if _EBUS_STATION_TOKEN_RE.search(window):
        return True
    if re.search(r"(?i)\blymph\s+node(?:s)?\b", window):
        return True
    return False


def _looks_like_endobronchial_ablation_context(note_text: str, match: re.Match[str]) -> bool:
    """Return True when ablation language is airway/endobronchial (not peripheral nodule ablation)."""
    if not note_text:
        return False

    local_start = max(0, match.start() - 260)
    local_end = min(len(note_text), match.end() + 260)
    window = note_text[local_start:local_end]

    if not _PERIPHERAL_ABLATION_ENDOBRONCHIAL_CUE_RE.search(window):
        return False

    peripheral_positive = False
    for peripheral_match in _PERIPHERAL_ABLATION_PERIPHERAL_CUE_RE.finditer(window):
        lead = window[max(0, peripheral_match.start() - 24) : peripheral_match.start()]
        if re.search(r"(?i)\b(?:no|not|without|absent)\b[^.\n]{0,20}$", lead):
            continue
        peripheral_positive = True
        break

    if peripheral_positive:
        return False
    return True


def _match_is_negated(note_text: str, match: re.Match[str], *, field_path: str | None = None) -> bool:
    """Return True when a keyword match is negated in local context."""
    if not note_text:
        return False

    start, end = match.start(), match.end()
    before = note_text[max(0, start - 120) : start]
    after = note_text[end : end + 120]

    if re.search(rf"(?i)\b{_NEGATION_CUES}\b[^.\n]{{0,60}}$", before):
        return True

    if re.search(rf"(?i)^[^.\n]{{0,60}}\b{_NEGATION_CUES}\b", after):
        return True

    if field_path == "procedures_performed.percutaneous_tracheostomy.performed":
        window = note_text[max(0, start - 80) : min(len(note_text), end + 80)]
        if re.search(r"(?i)\bexisting\s+tracheostomy\b", window):
            return True
        if re.search(r"(?i)\b(?:through|via)\s+tracheostomy\b", window):
            return True
        if re.search(r"(?i)\btracheostomy\s+tube\b", window):
            return True

    if field_path == "pleural_procedures.chest_tube.performed":
        window = note_text[max(0, start - 80) : min(len(note_text), end + 160)]
        if re.search(r"(?i)\bdate\s+of\s+(?:the\s+)?chest\s+tube\s+insertion\b", window):
            return True
        if _CHEST_TUBE_REMOVAL_CUES_RE.search(window) and not _CHEST_TUBE_INSERTION_CUES_RE.search(window):
            return True

    if field_path == "procedures_performed.tbna_conventional.performed":
        # TBNA language inside an EBUS paragraph should not trigger conventional TBNA.
        lookback_start = max(0, start - 800)
        paragraph_break = note_text.rfind("\n\n", lookback_start, start)
        if paragraph_break != -1:
            lookback_start = paragraph_break + 2
        paragraph_end = note_text.find("\n\n", end)
        if paragraph_end == -1:
            paragraph_end = min(len(note_text), end + 800)
        window = note_text[lookback_start:paragraph_end]
        if _TBNA_EBUS_CONTEXT_RE.search(window):
            return True

    return False


def scan_for_omissions(note_text: str, record: RegistryRecord) -> list[str]:
    """Scan raw text for required patterns missing from the extracted record.

    Returns warning strings suitable for surfacing in API responses and for
    triggering manual review or a retry/self-correction loop.
    """
    warnings: list[str] = []

    for field_path, rules in REQUIRED_PATTERNS.items():
        # TBNA is satisfied by either peripheral TBNA or EBUS-TBNA sampling; do not
        # emit a conventional TBNA omission when those are present.
        if field_path == "procedures_performed.tbna_conventional.performed":
            if _is_field_populated(record, "procedures_performed.peripheral_tbna.performed"):
                continue
            if _is_field_populated(record, "procedures_performed.linear_ebus.node_events") or _is_field_populated(
                record, "procedures_performed.linear_ebus.stations_sampled"
            ):
                continue
        if _is_field_populated(record, field_path):
            continue

        for pattern, msg in rules:
            match = re.search(pattern, note_text or "")
            if match and not _match_is_negated(note_text or "", match, field_path=field_path):
                if field_path == "procedures_performed.peripheral_tbna.performed":
                    if _looks_like_ebus_nodal_tbna_only(note_text or "", match):
                        continue
                if field_path == "procedures_performed.transbronchial_biopsy.performed":
                    if _looks_like_ebus_nodal_context(note_text or "", match):
                        continue
                if field_path == "procedures_performed.mechanical_debulking.performed":
                    if _looks_like_mucus_debulking_only(note_text or "", match):
                        continue
                if field_path == "procedures_performed.peripheral_ablation.performed":
                    if _looks_like_endobronchial_ablation_context(note_text or "", match):
                        continue
                warning = f"SILENT_FAILURE: {msg} (Pattern: '{pattern}')"
                warnings.append(warning)
                logger.warning(warning, extra={"field": field_path, "pattern": pattern})
                break

    return warnings


def apply_required_overrides(note_text: str, record: RegistryRecord) -> tuple[RegistryRecord, list[str]]:
    """Force required procedure flags when high-signal patterns appear."""
    if record is None:
        return RegistryRecord(), []

    warnings: list[str] = []
    record_data = record.model_dump()
    evidence = record_data.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}

    updated = False
    for field_path, rules in REQUIRED_PATTERNS.items():
        # Never force conventional nodal TBNA when peripheral TBNA or EBUS-TBNA
        # sampling is already present (prevents phantom tbna_conventional alongside EBUS).
        if field_path == "procedures_performed.tbna_conventional.performed":
            if _is_field_populated(record, "procedures_performed.peripheral_tbna.performed"):
                continue
            if _is_field_populated(record, "procedures_performed.linear_ebus.node_events") or _is_field_populated(
                record, "procedures_performed.linear_ebus.stations_sampled"
            ):
                continue
        if _is_field_populated(record, field_path):
            continue

        for pattern, msg in rules:
            match = re.search(pattern, note_text or "")
            if not match:
                continue
            if _match_is_negated(note_text or "", match, field_path=field_path):
                continue
            if field_path == "procedures_performed.peripheral_tbna.performed":
                if _looks_like_ebus_nodal_tbna_only(note_text or "", match):
                    continue
            if field_path == "procedures_performed.transbronchial_biopsy.performed":
                if _looks_like_ebus_nodal_context(note_text or "", match):
                    continue
            if field_path == "procedures_performed.mechanical_debulking.performed":
                if _looks_like_mucus_debulking_only(note_text or "", match):
                    continue
            if field_path == "procedures_performed.peripheral_ablation.performed":
                if _looks_like_endobronchial_ablation_context(note_text or "", match):
                    continue

            if field_path == "pleural_procedures.fibrinolytic_therapy.performed":
                pleural = record_data.get("pleural_procedures")
                if pleural is None or not isinstance(pleural, dict):
                    pleural = {}

                fibrinolytic = pleural.get("fibrinolytic_therapy")
                if fibrinolytic is None or not isinstance(fibrinolytic, dict):
                    fibrinolytic = {}

                fibrinolytic["performed"] = True

                agents: list[str] = []
                if re.search(r"(?i)\b(?:tpa|alteplase)\b", note_text or ""):
                    agents.append("tPA")
                if re.search(r"(?i)\b(?:dnase|dornase)", note_text or ""):
                    agents.append("DNase")
                if agents:
                    seen: set[str] = set()
                    fibrinolytic["agents"] = [a for a in agents if not (a in seen or seen.add(a))]

                dose_match = re.search(
                    r"(?is)\b(\d+(?:\.\d+)?)\s*mg\s*/\s*(\d+(?:\.\d+)?)\s*mg\b[^.\n]{0,60}"
                    r"\b(?:tpa|alteplase)\b[^.\n]{0,60}\b(?:dnase|dornase)",
                    note_text or "",
                )
                if dose_match:
                    fibrinolytic["tpa_dose_mg"] = float(dose_match.group(1))
                    fibrinolytic["dnase_dose_mg"] = float(dose_match.group(2))

                dose_num = re.search(
                    r"(?i)dose\s*#\s*[:=]?\s*[_ ]*(\d{1,2})(?=[_ ]|\b)",
                    note_text or "",
                )
                if dose_num:
                    fibrinolytic["number_of_doses"] = int(dose_num.group(1))
                elif re.search(r"(?i)\bsubsequent\s+day\b", note_text or "") or re.search(r"\b32562\b", note_text or ""):
                    fibrinolytic["number_of_doses"] = 2
                if fibrinolytic.get("number_of_doses") in (0, "0") and re.search(r"\b32562\b", note_text or ""):
                    fibrinolytic["number_of_doses"] = 2

                pleural["fibrinolytic_therapy"] = fibrinolytic
                record_data["pleural_procedures"] = pleural
                evidence.setdefault(field_path, []).append(
                    Span(
                        text=match.group(0).strip(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
                warnings.append(f"HARD_OVERRIDE: {msg} -> {field_path}=true")
                updated = True
                break

            if field_path.startswith("granular_data.navigation_targets"):
                granular = record_data.get("granular_data")
                if granular is None or not isinstance(granular, dict):
                    granular = {}

                targets_raw = granular.get("navigation_targets")
                if isinstance(targets_raw, list):
                    targets = [t for t in targets_raw if isinstance(t, dict)]
                else:
                    targets = []

                if not targets:
                    targets = [
                        {
                            "target_number": 1,
                            "target_location_text": "Unknown target",
                            "fiducial_marker_placed": True,
                        }
                    ]
                else:
                    latest = dict(targets[-1])
                    if latest.get("fiducial_marker_placed") is not True:
                        latest["fiducial_marker_placed"] = True
                    targets[-1] = latest

                granular["navigation_targets"] = targets
                record_data["granular_data"] = granular
                evidence.setdefault(field_path, []).append(
                    Span(
                        text=match.group(0).strip(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
                warnings.append(f"HARD_OVERRIDE: {msg} -> {field_path}=true")
                updated = True
                break

            _set_nested_field(record_data, field_path, True)
            evidence.setdefault(field_path, []).append(
                Span(
                    text=match.group(0).strip(),
                    start=match.start(),
                    end=match.end(),
                )
            )
            warnings.append(f"HARD_OVERRIDE: {msg} -> {field_path}=true")
            updated = True
            break

    if updated:
        record_data["evidence"] = evidence
        record = RegistryRecord(**record_data)

    return record, warnings


def _is_field_populated(record: RegistryRecord, path: str) -> bool:
    """Safely navigate the RegistryRecord using dot-notation and check truthiness."""
    try:
        def _walk(current: object | None, parts: list[str]) -> bool:
            if current is None:
                return False
            if not parts:
                return bool(current)

            part = parts[0]
            remaining = parts[1:]

            if isinstance(current, list):
                return any(_walk(item, parts) for item in current)

            if hasattr(current, part):
                return _walk(getattr(current, part), remaining)

            if isinstance(current, dict) and part in current:
                return _walk(current[part], remaining)

            return False

        return _walk(record, path.split("."))
    except Exception as exc:  # pragma: no cover
        logger.error("Error checking field population for %s: %s", path, exc)
        return False


def _set_nested_field(data: dict, path: str, value: object) -> None:
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def keyword_guard_passes(*, cpt: str, evidence_text: str) -> bool:
    """Return True if any configured keywords hit in evidence_text (case-insensitive)."""
    ok, _reason = keyword_guard_check(cpt=cpt, evidence_text=evidence_text)
    return ok


def keyword_guard_check(*, cpt: str, evidence_text: str, ml_prob: float | None = None) -> tuple[bool, str]:
    """Return (passes, reason) for keyword gating."""
    if ml_prob is not None:
        try:
            prob = float(ml_prob)
        except (TypeError, ValueError):
            prob = None
        if prob is not None and prob >= HIGH_CONF_BYPASS_THRESHOLD and str(cpt) in HIGH_CONF_BYPASS_CPTS:
            return True, f"high_conf_prob>={HIGH_CONF_BYPASS_THRESHOLD:.2f} bypass"

    keywords = get_effective_cpt_keywords().get(str(cpt), [])
    if not keywords:
        return False, "no keywords configured"

    text = (evidence_text or "").lower()
    if not text.strip():
        return False, "empty evidence text"

    for keyword in keywords:
        needle = (keyword or "").strip().lower()
        if not needle:
            continue
        if _keyword_hit(text, needle):
            return True, f"matched '{needle}'"
    return False, "no keyword hit"


def _keyword_hit(text_lower: str, needle_lower: str) -> bool:
    if " " in needle_lower or len(needle_lower) >= 5:
        return needle_lower in text_lower
    # Short token (e.g., "ipc", "bal", "tap"): require word boundary to reduce false positives.
    return re.search(rf"\b{re.escape(needle_lower)}\b", text_lower) is not None


__all__ = [
    "CPT_KEYWORDS",
    "DEFAULT_GENERATED_CPT_KEYWORDS_PATH",
    "HIGH_CONF_BYPASS_CPTS",
    "REQUIRED_PATTERNS",
    "apply_required_overrides",
    "get_effective_cpt_keywords",
    "keyword_guard_check",
    "keyword_guard_passes",
    "scan_for_omissions",
]
