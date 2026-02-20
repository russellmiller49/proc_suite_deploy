#!/usr/bin/env python3
"""Build a lightweight UMLS concept map for interventional pulmonology.

Reads the UMLS 2025AA RRF files directly (MRCONSO, MRSTY, MRREL, MRDEF)
and extracts only terminology relevant to the thoracic cavity, respiratory
diseases, and bronchoscopic equipment. Outputs a compact JSON file
(ip_umls_map.json) that replaces the heavy (~1GB) runtime UMLS/scispaCy
linker on memory-constrained deployments like Railway.

Usage
-----
    # Default: reads ~/UMLS/2025AA/META, writes data/knowledge/ip_umls_map.json
    python ops/tools/build_ip_umls_map.py

    # Explicit UMLS path
    python ops/tools/build_ip_umls_map.py --umls-dir ~/UMLS/2025AA/META

    # Include MRREL parent/child expansion (broader, slower)
    python ops/tools/build_ip_umls_map.py --expand-rels

    # Include definitions from MRDEF
    python ops/tools/build_ip_umls_map.py --include-defs

    # Verbose logging
    python ops/tools/build_ip_umls_map.py -v

Data Sources
------------
    ~/UMLS/2025AA/META/MRCONSO.RRF  — concept names & synonyms
    ~/UMLS/2025AA/META/MRSTY.RRF    — semantic types per CUI
    ~/UMLS/2025AA/META/MRREL.RRF    — relationships (parent/child/sibling)
    ~/UMLS/2025AA/META/MRDEF.RRF    — definitions (optional)

Notes
-----
    MRCONSO is ~10M lines, MRSTY ~3.7M, MRREL ~37M. This script streams
    them line-by-line so memory stays bounded. Typical run: ~2-3 min,
    output ~3-8 MB depending on options.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# UMLS RRF paths
# ---------------------------------------------------------------------------

DEFAULT_UMLS_DIR = Path.home() / "UMLS" / "2025AA" / "META"

# ---------------------------------------------------------------------------
# Semantic types relevant to interventional pulmonology
# ---------------------------------------------------------------------------

ANATOMY_SEMTYPES = {
    "T017",  # Anatomical Structure
    "T021",  # Fully Formed Anatomical Structure
    "T023",  # Body Part, Organ, or Organ Component
    "T024",  # Tissue
    "T025",  # Cell
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction
}

PROCEDURE_SEMTYPES = {
    "T058",  # Health Care Activity
    "T059",  # Laboratory Procedure
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
}

DEVICE_SEMTYPES = {
    "T074",  # Medical Device
    "T075",  # Research Device
    "T168",  # Food (catches device-adjacent concepts)
    "T073",  # Manufactured Object
}

DISEASE_SEMTYPES = {
    "T019",  # Congenital Abnormality
    "T020",  # Acquired Abnormality
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T037",  # Injury or Poisoning
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T184",  # Sign or Symptom
    "T190",  # Anatomical Abnormality
    "T191",  # Neoplastic Process
}

PHARMA_SEMTYPES = {
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
}

ALL_ALLOWED_SEMTYPES = (
    ANATOMY_SEMTYPES
    | PROCEDURE_SEMTYPES
    | DEVICE_SEMTYPES
    | DISEASE_SEMTYPES
    | PHARMA_SEMTYPES
)

# ---------------------------------------------------------------------------
# Domain keyword patterns (case-insensitive, applied to concept names)
# ---------------------------------------------------------------------------

# NOTE: Patterns are applied ONLY to the UMLS preferred name (not aliases)
# to avoid cross-domain noise. Keep patterns specific to the thoracic/
# pulmonary domain — generic terms like "segment", "lobe", "stent"
# without a qualifier will match GI, cardiac, neuro, etc.

THORACIC_ANATOMY_PATTERNS = [
    # Airways (highly specific — these don't appear outside pulm)
    r"\btrachea\b", r"\btracheal\b",
    r"\bbronchus\b", r"\bbronchi\b", r"\bbronchial\b", r"\bbronchiole",
    r"\bcarina\b", r"\bsubglott", r"\bglottis\b",
    r"\blarynx\b", r"\blaryngeal\b", r"\blaryngo",
    r"\bairway\b", r"\bairways\b",
    # Lungs (require "lung" / "pulmonary" qualifier)
    r"\blung\b", r"\blungs\b", r"\bpulmonary\b",
    r"\bpulmonol", r"\bpneumon",
    r"\blobar\s+bronch", r"\blobectom",
    r"\balveol",  # alveolar, alveolus
    r"\blung\s+parenchym",
    r"\bhilum\s+of\s+(the\s+)?lung", r"\bpulmonary\s+hilum",
    # Pleura (specific)
    r"\bpleura\b", r"\bpleural\b", r"\bpleurodesis\b",
    r"\bpleuroscop", r"\bpleurisy\b", r"\bpleuro",
    # Mediastinum
    r"\bmediastin",
    # Lymph nodes (thoracic-specific stations)
    r"\bsubcarin", r"\bparatracheal\b", r"\baortopulm",
    r"\bprevascular\b", r"\bparaesoph",
    r"\bhilar\s+lymph", r"\bmediastinal\s+lymph",
    r"\blymph\s+node\s+station",
    # Chest wall (with qualifiers)
    r"\bchest\s+wall\b", r"\bthoracic\b", r"\bthorax\b",
    r"\bdiaphragm\b",
    r"\bintercostal\s+(muscle|space|nerve)",
    # Vasculature (pulmonary only)
    r"\bpulmonary\s+arter", r"\bpulmonary\s+vein",
    r"\bpulmonary\s+vascu",
]

RESPIRATORY_DISEASE_PATTERNS = [
    # Cancers (require pulm/lung qualifier)
    r"\blung\s+(cancer|neoplasm|carcinoma|tumor|tumour|mass|nodule|lesion)",
    r"\bmesothelioma\b", r"\bthymoma\b",
    r"\bsmall\s+cell\s+lung", r"\bnon.small\s+cell",
    r"\bnsclc\b", r"\bsclc\b", r"\bpancoast\b",
    r"\bbronchial\s+carcinoid", r"\bcarcinoid.*bronch",
    r"\bendobronchial\s+(tumor|tumour|lesion|mass|carcinoma)",
    # Obstructive pulmonary
    r"\bcopd\b", r"\bchronic\s+obstructive\s+pulmonary",
    r"\basthma\b",
    r"\bemphysema\b",
    r"\bbronchiect",
    # Pulmonary infections
    r"\bpneumonia\b",
    r"\bpulmonary\s+tuberculosis", r"\btb\s+.*lung",
    r"\bpulmonary\s+aspergill", r"\binvasive\s+pulmonary",
    r"\blung\s+abscess\b",
    r"\bbronchitis\b",
    # Interstitial / diffuse
    r"\binterstitial\s+lung",
    r"\bidiopathic\s+pulmonary\s+fibrosis\b",
    r"\bsarcoidosis\b",
    r"\bhypersensitivity\s+pneumonitis",
    r"\bpulmonary\s+fibrosis\b",
    # Pleural diseases
    r"\bpleural\s+effusion\b", r"\bpleural\s+empyema\b",
    r"\bpneumothorax\b", r"\bhemothorax\b", r"\bchylothorax\b",
    r"\bmalignant\s+pleural", r"\btrapped\s+lung\b",
    # Airway obstruction (specific)
    r"\bairway\s+(obstruct|stenos|compress|collapse|malaci)",
    r"\btracheomalaci", r"\bbronchomala",
    r"\bsubglottic\s+stenos", r"\btracheal\s+stenos",
    r"\bcentral\s+airway\s+obstruct",
    # Pulmonary vascular
    r"\bpulmonary\s+hypertens",
    r"\bpulmonary\s+embol",
    # Pulmonary symptoms (require pulm qualifier)
    r"\batelectas",
    r"\bhemoptysis\b",
    r"\bstridor\b",
    r"\brespiratory\s+fail",
    r"\bacute\s+respiratory\s+distress",
]

BRONCHOSCOPIC_EQUIPMENT_PATTERNS = [
    # Scopes (very IP-specific)
    r"\bbronchoscop",
    r"\bthoracoscop", r"\bpleuroscop",
    r"\bmediastinoscop",
    # EBUS
    r"\bendobronchial\s+ultrasound",
    r"\bconvex\s+probe\s+ebus",
    r"\bradial\s+probe\s+ebus",
    # Navigation
    r"\belectromagnetic\s+navigat.*bronch",
    r"\brobotic\s+bronch",
    # Biopsy (bronchoscopic)
    r"\btransbronchial\s+needle",
    r"\btransbronchial\s+biops",
    r"\bcryobiops",
    r"\bbronchial\s+brush",
    r"\bbronchoalveolar\s+lavage",
    r"\bendobronchial\s+biops",
    # Therapeutic devices (require airway/bronch qualifier)
    r"\bairway\s+stent", r"\bbronchial\s+stent", r"\btracheal\s+stent",
    r"\bendobronchial\s+stent",
    r"\bsilicone\s+stent", r"\bmetallic\s+stent",
    r"\bballoon\s+dilat.*\b(airway|bronch|trache)",
    r"\bargon\s+plasma\s+coagulat",
    r"\bcryotherap.*\b(airway|bronch|endobronch)",
    r"\bcryoablat",
    r"\bcryospray",
    r"\bphotodynamic\s+therap",
    r"\bendobronchial\s+(debulk|ablat|electrocaut)",
    r"\bbrachytherap.*\b(bronch|endobronch|lung)",
    # BLVR
    r"\bzephyr\s+valve", r"\bspiration\s+valve",
    r"\bchartis\b",
    r"\bcollateral\s+ventilat",
    r"\bbronchial\s+valve\b",
    r"\blung\s+volume\s+reduc",
    r"\bendobronchial\s+valve",
    # Pleural devices
    r"\bchest\s+tube\b", r"\bthoracostomy\s+tube",
    r"\bindwelling\s+pleural\s+catheter",
    r"\btalc\s+(pleurodesis|poudrage|slurry)",
    r"\bthoracentesis\b",
    # Airway management
    r"\bendotracheal\s+tube\b", r"\bendotracheal\s+intubat",
    r"\btracheostomy\b", r"\btracheotomy\b",
    # Sedation (specific agents for bronch)
    r"\bpropofol\b", r"\bmidazolam\b", r"\bfentanyl\b",
    r"\bmoderate\s+sedation\b", r"\bdeep\s+sedation\b",
    r"\bgeneral\s+anesthesia\b",
    # Imaging (bronch-associated)
    r"\bfluoroscop.*\b(bronch|chest|lung|thorac)",
    r"\bcone.beam\s+ct\b",
]

# Category-gated pattern groups: each pattern list is paired with the
# semantic types it's allowed to match. This prevents e.g. "\blung\b"
# in the anatomy list from pulling in 20k lung-related drugs/procedures.
CATEGORY_GATED_PATTERNS: list[tuple[list[str], set[str]]] = [
    (THORACIC_ANATOMY_PATTERNS, ANATOMY_SEMTYPES),
    (RESPIRATORY_DISEASE_PATTERNS, DISEASE_SEMTYPES),
    (BRONCHOSCOPIC_EQUIPMENT_PATTERNS, PROCEDURE_SEMTYPES | DEVICE_SEMTYPES | PHARMA_SEMTYPES),
]

# Flat list for simple boolean checks (used by expand-rels pre-scan)
ALL_PATTERNS = (
    THORACIC_ANATOMY_PATTERNS
    + RESPIRATORY_DISEASE_PATTERNS
    + BRONCHOSCOPIC_EQUIPMENT_PATTERNS
)

# ---------------------------------------------------------------------------
# Seed CUIs — always included regardless of pattern matching
# ---------------------------------------------------------------------------

SEED_CUIS: Set[str] = {
    # Anatomy
    "C0040578",  # Trachea
    "C0006255",  # Bronchus
    "C0024109",  # Lung
    "C0225713",  # Right lung
    "C0225730",  # Left lung
    "C0225756",  # Right upper lobe
    "C0225757",  # Right middle lobe
    "C0225758",  # Right lower lobe
    "C0225759",  # Left upper lobe
    "C0225760",  # Left lower lobe
    "C0024876",  # Lingula
    "C0006272",  # Bronchiole
    "C0225699",  # Carina
    "C0025066",  # Mediastinum
    "C0032225",  # Pleura
    "C0678482",  # Pleural cavity
    "C1183682",  # Subcarinal lymph node
    "C0729374",  # Hilum of lung
    "C0012369",  # Diaphragm
    "C0039492",  # Thoracic cavity
    # Procedures
    "C0006290",  # Bronchoscopy
    "C0189217",  # Rigid bronchoscopy
    "C1883418",  # EBUS-TBNA
    "C2711980",  # Endobronchial ultrasound of mediastinum
    "C0040048",  # Thoracentesis
    "C0040590",  # Tracheostomy
    "C0176643",  # Transbronchial needle aspiration
    "C0189384",  # Pleurodesis
    "C0159775",  # Chest tube insertion
    "C0521373",  # Bronchoalveolar lavage
    "C0184898",  # Biopsy
    "C0394903",  # Transbronchial biopsy
    "C0450277",  # Stent placement
    "C3714787",  # Endobronchial valve placement
    "C2959830",  # Cryobiopsy
    # Diseases
    "C0024117",  # COPD
    "C0013990",  # Emphysema
    "C0024121",  # Lung neoplasms
    "C0025500",  # Mesothelioma
    "C0032285",  # Pneumonia
    "C0032326",  # Pneumothorax
    "C0264464",  # Pleural effusion
    "C0206141",  # Idiopathic pulmonary fibrosis
    "C0036202",  # Sarcoidosis
    "C0004096",  # Asthma
    "C0236018",  # Hemoptysis
    "C0037854",  # Tracheal stenosis
    "C0238371",  # Airway obstruction
    # Devices
    "C0441364",  # Bronchoscope
    "C0038257",  # Stent device
    "C0180936",  # Biopsy forceps
}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_compiled_flat_patterns: list[re.Pattern] = []
_compiled_gated_patterns: list[tuple[list[re.Pattern], set[str]]] = []


def _get_flat_patterns() -> list[re.Pattern]:
    """Compiled flat pattern list (for expand-rels pre-scan only)."""
    global _compiled_flat_patterns
    if not _compiled_flat_patterns:
        _compiled_flat_patterns = [re.compile(p, re.IGNORECASE) for p in ALL_PATTERNS]
    return _compiled_flat_patterns


def _get_gated_patterns() -> list[tuple[list[re.Pattern], set[str]]]:
    """Compiled category-gated patterns: [(compiled_pats, allowed_semtypes)]."""
    global _compiled_gated_patterns
    if not _compiled_gated_patterns:
        _compiled_gated_patterns = [
            ([re.compile(p, re.IGNORECASE) for p in pats], semtypes)
            for pats, semtypes in CATEGORY_GATED_PATTERNS
        ]
    return _compiled_gated_patterns


def _matches_domain(name: str) -> bool:
    """Return True if the name matches any IP-domain pattern (flat, ungated)."""
    for pat in _get_flat_patterns():
        if pat.search(name):
            return True
    return False


def _matches_domain_gated(name: str, concept_semtypes: Set[str]) -> bool:
    """Return True if name matches a pattern whose semtype gate overlaps."""
    for compiled_pats, allowed_semtypes in _get_gated_patterns():
        if not concept_semtypes & allowed_semtypes:
            continue
        for pat in compiled_pats:
            if pat.search(name):
                return True
    return False


# ---------------------------------------------------------------------------
# RRF parsing
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _parse_rrf_line(line: str) -> List[str]:
    """Parse a single pipe-delimited RRF line (trailing pipe is normal)."""
    return line.rstrip("\n").rstrip("|").split("|")


def _load_semtypes(mrsty_path: Path) -> Dict[str, Set[str]]:
    """Load MRSTY.RRF → {CUI: {TUI, ...}}."""
    logger.info("Loading semantic types from %s ...", mrsty_path)
    cui_semtypes: Dict[str, Set[str]] = defaultdict(set)
    # MRSTY columns: CUI|TUI|STN|STY|ATUI|CVF
    with open(mrsty_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = _parse_rrf_line(line)
            cui, tui = fields[0], fields[1]
            cui_semtypes[cui].add(tui)
    logger.info("  Loaded semantic types for %d CUIs", len(cui_semtypes))
    return dict(cui_semtypes)


def _load_semtype_labels(mrsty_path: Path) -> Dict[str, str]:
    """Load MRSTY.RRF → {TUI: label} for human-readable output."""
    labels: Dict[str, str] = {}
    with open(mrsty_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = _parse_rrf_line(line)
            tui, sty = fields[1], fields[3]
            if tui not in labels:
                labels[tui] = sty
    return labels


def _identify_eligible_cuis(
    cui_semtypes: Dict[str, Set[str]],
) -> Set[str]:
    """Return the set of CUIs that have at least one allowed semantic type."""
    eligible = set()
    for cui, tuis in cui_semtypes.items():
        if tuis & ALL_ALLOWED_SEMTYPES:
            eligible.add(cui)
    logger.info(
        "  %d CUIs have at least one allowed semantic type (of %d total)",
        len(eligible),
        len(cui_semtypes),
    )
    return eligible


def _load_concepts(
    mrconso_path: Path,
    eligible_cuis: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Stream MRCONSO.RRF and collect names/aliases for eligible CUIs.

    MRCONSO columns (key ones):
        0: CUI  1: LAT  2: TS  6: ISPREF  11: SAB  12: TTY  14: STR
    We keep only ENG rows and prefer the preferred name (TS=P, ISPREF=Y).
    """
    logger.info("Streaming concepts from %s ...", mrconso_path)
    # Accumulate: CUI → { preferred_name, aliases set, source vocabs }
    raw: Dict[str, Dict[str, Any]] = {}
    lines_read = 0
    kept = 0

    target_cuis = eligible_cuis | SEED_CUIS

    with open(mrconso_path, "r", encoding="utf-8") as f:
        for line in f:
            lines_read += 1
            if lines_read % 2_000_000 == 0:
                logger.info("  ...read %dM MRCONSO lines", lines_read // 1_000_000)

            fields = _parse_rrf_line(line)
            cui = fields[0]

            # Fast reject: not in our target set
            if cui not in target_cuis:
                continue

            lang = fields[1]
            if lang != "ENG":
                continue

            ts = fields[2]        # P=preferred, S=non-preferred
            ispref = fields[6]    # Y=preferred form of the string
            sab = fields[11]      # source vocabulary
            name = fields[14]     # the actual term string

            if cui not in raw:
                raw[cui] = {
                    "preferred_name": None,
                    "aliases": set(),
                    "sources": set(),
                }

            entry = raw[cui]
            entry["sources"].add(sab)

            # Pick preferred name: TS=P and ISPREF=Y
            if ts == "P" and ispref == "Y" and entry["preferred_name"] is None:
                entry["preferred_name"] = name
            else:
                entry["aliases"].add(name)

            kept += 1

    # Fallback: if no preferred name was found, use first alias
    for cui, entry in raw.items():
        if entry["preferred_name"] is None and entry["aliases"]:
            entry["preferred_name"] = next(iter(entry["aliases"]))
            entry["aliases"].discard(entry["preferred_name"])

    logger.info(
        "  Parsed %d MRCONSO lines, collected %d ENG entries for %d CUIs",
        lines_read,
        kept,
        len(raw),
    )
    return raw


def _load_definitions(
    mrdef_path: Path,
    target_cuis: Set[str],
) -> Dict[str, str]:
    """Stream MRDEF.RRF and pick one definition per CUI.

    MRDEF columns: CUI|AUI|ATUI|SATUI|SAB|DEF|SUPPRESS|CVF
    Prefer MSH (MeSH) or NCI definitions.
    """
    logger.info("Loading definitions from %s ...", mrdef_path)
    defs: Dict[str, str] = {}
    preferred_sources = {"MSH", "NCI", "NCI_NCI-GLOSS", "SNOMEDCT_US", "HPO"}

    with open(mrdef_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = _parse_rrf_line(line)
            cui, sab, defn = fields[0], fields[4], fields[5]
            if cui not in target_cuis:
                continue
            # Keep first preferred-source def, or first def if none preferred
            if cui not in defs:
                defs[cui] = defn
            elif sab in preferred_sources and cui in defs:
                defs[cui] = defn  # upgrade to preferred source

    logger.info("  Loaded definitions for %d CUIs", len(defs))
    return defs


def _load_relationships(
    mrrel_path: Path,
    target_cuis: Set[str],
    eligible_cuis: Set[str],
) -> Set[str]:
    """Stream MRREL.RRF and find parent/child/sibling CUIs of target concepts.

    MRREL columns: CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF
    REL values: PAR (parent), CHD (child), SIB (sibling), RB (broader), RN (narrower)
    """
    logger.info("Scanning relationships from %s (this may take a minute)...", mrrel_path)
    expansion_rels = {"PAR", "CHD", "RN", "RB"}
    expanded: Set[str] = set()
    lines_read = 0

    with open(mrrel_path, "r", encoding="utf-8") as f:
        for line in f:
            lines_read += 1
            if lines_read % 10_000_000 == 0:
                logger.info("  ...read %dM MRREL lines", lines_read // 1_000_000)

            fields = _parse_rrf_line(line)
            cui1, rel, cui2 = fields[0], fields[3], fields[4]

            if rel not in expansion_rels:
                continue

            # If one side is in our target set, pull in the other side
            if cui1 in target_cuis and cui2 in eligible_cuis and cui2 not in target_cuis:
                expanded.add(cui2)
            elif cui2 in target_cuis and cui1 in eligible_cuis and cui1 not in target_cuis:
                expanded.add(cui1)

    logger.info(
        "  Scanned %d MRREL lines, found %d related CUIs to expand",
        lines_read,
        len(expanded),
    )
    return expanded


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _categorize_semtypes(semtypes: Set[str]) -> List[str]:
    cats = set()
    for st in semtypes:
        if st in ANATOMY_SEMTYPES:
            cats.add("anatomy")
        if st in PROCEDURE_SEMTYPES:
            cats.add("procedure")
        if st in DEVICE_SEMTYPES:
            cats.add("device")
        if st in DISEASE_SEMTYPES:
            cats.add("disease")
        if st in PHARMA_SEMTYPES:
            cats.add("pharmacology")
    return sorted(cats)


def _filter_and_assemble(
    raw_concepts: Dict[str, Dict[str, Any]],
    cui_semtypes: Dict[str, Set[str]],
    definitions: Dict[str, str] | None = None,
    expanded_cuis: Set[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Apply domain-pattern filter and build final concept dicts."""
    concepts: Dict[str, Dict[str, Any]] = {}
    stats = {"seed": 0, "pattern": 0, "expanded": 0, "skipped": 0}

    for cui, raw in raw_concepts.items():
        preferred = raw["preferred_name"]
        if not preferred:
            stats["skipped"] += 1
            continue

        semtypes = cui_semtypes.get(cui, set())
        is_seed = cui in SEED_CUIS
        is_expanded = expanded_cuis and cui in expanded_cuis

        # Pattern check: preferred name only, gated by semantic type category
        name_matches = _matches_domain_gated(preferred, semtypes)

        if not (is_seed or name_matches or is_expanded):
            stats["skipped"] += 1
            continue

        if is_seed:
            stats["seed"] += 1
        elif is_expanded:
            stats["expanded"] += 1
        else:
            stats["pattern"] += 1

        # Deduplicate aliases, remove preferred name, cap at 20
        alias_set = set()
        clean_aliases = []
        for a in raw["aliases"]:
            lower = a.strip().lower()
            if lower and lower != preferred.lower() and lower not in alias_set:
                alias_set.add(lower)
                clean_aliases.append(a.strip())
        clean_aliases = clean_aliases[:10]

        categories = _categorize_semtypes(semtypes)

        # Collect source vocabularies (useful for provenance)
        sources = sorted(raw.get("sources", set()))

        entry: Dict[str, Any] = {
            "name": preferred,
            "semtypes": sorted(semtypes),
            "categories": categories,
            "aliases": clean_aliases,
            "sources": sources[:5],  # cap to keep file small
        }

        if definitions and cui in definitions:
            entry["definition"] = definitions[cui][:500]  # truncate long defs

        if is_expanded:
            entry["expanded"] = True

        concepts[cui] = entry

    logger.info(
        "Assembly: %d concepts kept (seed=%d, pattern=%d, expanded=%d, skipped=%d)",
        len(concepts),
        stats["seed"],
        stats["pattern"],
        stats["expanded"],
        stats["skipped"],
    )
    return concepts


def _build_term_index(concepts: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build reverse index: lowercase term → [CUI, ...]."""
    index: Dict[str, List[str]] = defaultdict(list)
    for cui, data in concepts.items():
        key = data["name"].strip().lower()
        if key and cui not in index[key]:
            index[key].append(cui)
        for alias in data.get("aliases", []):
            key = alias.strip().lower()
            if key and cui not in index[key]:
                index[key].append(cui)
    return dict(index)


def _build_output(
    concepts: Dict[str, Dict[str, Any]],
    term_index: Dict[str, List[str]],
    umls_dir: Path,
    semtype_labels: Dict[str, str],
    include_defs: bool,
    expand_rels: bool,
) -> Dict[str, Any]:
    cat_counts: Dict[str, int] = defaultdict(int)
    for data in concepts.values():
        for cat in data.get("categories", []):
            cat_counts[cat] += 1

    # Build a semtype legend for the subset actually used
    used_tuis = set()
    for data in concepts.values():
        used_tuis.update(data.get("semtypes", []))
    semtype_legend = {tui: semtype_labels.get(tui, "?") for tui in sorted(used_tuis)}

    return {
        "_meta": {
            "version": "1.0",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "umls_source": str(umls_dir),
            "umls_release": "2025AA",
            "description": (
                "Lightweight UMLS concept map filtered for interventional "
                "pulmonology: thoracic anatomy, respiratory diseases, and "
                "bronchoscopic equipment/procedures."
            ),
            "concept_count": len(concepts),
            "term_index_count": len(term_index),
            "category_counts": dict(cat_counts),
            "options": {
                "include_definitions": include_defs,
                "expand_relationships": expand_rels,
                "semantic_type_filter": sorted(ALL_ALLOWED_SEMTYPES),
                "domain_pattern_count": len(ALL_PATTERNS),
                "seed_cui_count": len(SEED_CUIS),
            },
        },
        "semtype_legend": semtype_legend,
        "concepts": concepts,
        "term_index": term_index,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "data" / "knowledge" / "ip_umls_map.json"

    parser = argparse.ArgumentParser(
        description="Extract IP-relevant UMLS concepts from RRF files into a lightweight JSON map.",
    )
    parser.add_argument(
        "--umls-dir",
        type=Path,
        default=DEFAULT_UMLS_DIR,
        help=f"Path to UMLS META directory containing RRF files (default: {DEFAULT_UMLS_DIR})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=default_output,
        help=f"Output path (default: {default_output})",
    )
    parser.add_argument(
        "--expand-rels",
        action="store_true",
        help="Traverse MRREL.RRF parent/child/sibling relationships to expand coverage.",
    )
    parser.add_argument(
        "--include-defs",
        action="store_true",
        help="Include concept definitions from MRDEF.RRF (adds ~30%% file size).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate RRF files exist
    umls_dir: Path = args.umls_dir
    required_files = ["MRCONSO.RRF", "MRSTY.RRF"]
    if args.expand_rels:
        required_files.append("MRREL.RRF")
    if args.include_defs:
        required_files.append("MRDEF.RRF")

    for fname in required_files:
        fpath = umls_dir / fname
        if not fpath.exists():
            logger.error("Required file not found: %s", fpath)
            logger.error("Ensure --umls-dir points to the UMLS META directory.")
            sys.exit(1)

    logger.info("UMLS source: %s", umls_dir)

    # Step 1: Load semantic types
    cui_semtypes = _load_semtypes(umls_dir / "MRSTY.RRF")
    semtype_labels = _load_semtype_labels(umls_dir / "MRSTY.RRF")

    # Step 2: Find CUIs with at least one allowed semantic type
    eligible_cuis = _identify_eligible_cuis(cui_semtypes)

    # Step 3: Optionally expand via MRREL
    expanded_cuis: Set[str] | None = None
    target_cuis = eligible_cuis | SEED_CUIS
    if args.expand_rels:
        # First pass: get pattern-matched CUIs to use as seeds for expansion.
        # We need MRCONSO for that, so do a quick name scan first.
        logger.info("Pre-scanning MRCONSO for pattern-matched CUIs (for MRREL expansion)...")
        prescan_raw = _load_concepts(umls_dir / "MRCONSO.RRF", eligible_cuis)
        pattern_cuis = set()
        for cui, raw in prescan_raw.items():
            pn = raw["preferred_name"] or ""
            if _matches_domain(pn) or cui in SEED_CUIS:
                pattern_cuis.add(cui)
        logger.info("  Pre-scan found %d pattern-matched CUIs", len(pattern_cuis))

        expanded_cuis = _load_relationships(
            umls_dir / "MRREL.RRF",
            pattern_cuis | SEED_CUIS,
            eligible_cuis,
        )
        # Re-load MRCONSO with expanded target set
        target_cuis = eligible_cuis | SEED_CUIS | expanded_cuis
        raw_concepts = _load_concepts(umls_dir / "MRCONSO.RRF", target_cuis)
    else:
        raw_concepts = _load_concepts(umls_dir / "MRCONSO.RRF", target_cuis)

    # Step 4: Optionally load definitions
    definitions: Dict[str, str] | None = None
    if args.include_defs:
        all_matched_cuis = set(raw_concepts.keys())
        definitions = _load_definitions(umls_dir / "MRDEF.RRF", all_matched_cuis)

    # Step 5: Filter by domain patterns and assemble
    concepts = _filter_and_assemble(
        raw_concepts, cui_semtypes, definitions, expanded_cuis
    )

    # Step 6: Build term index
    term_index = _build_term_index(concepts)

    # Step 7: Assemble and write output
    output = _build_output(
        concepts, term_index, umls_dir, semtype_labels,
        args.include_defs, args.expand_rels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Use compact separators; _meta and semtype_legend are small enough
    # to keep readable in the file header, concepts/term_index stay dense.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=None, separators=(",", ":"), ensure_ascii=False)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    logger.info(
        "Done! Wrote %d concepts (%d index terms) to %s (%.2f MB)",
        len(concepts),
        len(term_index),
        args.output,
        size_mb,
    )
    logger.info("Category breakdown: %s", dict(output["_meta"]["category_counts"]))


if __name__ == "__main__":
    main()
