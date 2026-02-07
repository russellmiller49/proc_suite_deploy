#!/usr/bin/env python3
"""Generate a CPT->keyword mapping for keyword-guard gating.

The generator is deterministic and offline-only. Inputs:
- KB JSON: data/knowledge/ip_coding_billing_v3_0.json
- Optional YAML keyword seed files under data/keyword_mappings/

Output:
- data/keyword_mappings/cpt_keywords.generated.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_MAX_KEYWORDS_PER_CPT = 50

# Conservative synonym routing from KB synonym groups to CPT codes.
SYNONYM_GROUP_TO_CPTS: dict[str, set[str]] = {
    # Bronchoscopy diagnostics/interventions
    "bal_terms": {"31624"},
    "tblb_terms": {"31628", "31632"},
    "tbna_terms": {"31629", "31633", "31652", "31653"},
    "linear_ebus_terms": {"31652", "31653"},
    "ebus_station_terms": {"31652", "31653"},
    "radial_ebus_terms": {"31654"},
    "navigation_terms": {"31627"},
    "navigation_concept_terms": {"31627"},
    "navigation_status_terms": {"31627"},
    "aspiration_terms": {"31645", "31646"},
    "foreign_body_terms": {"31635"},
    "dilation_terms": {"31630", "31631"},
    "stent_terms": {"31636", "31637", "31638"},
    "ablation_terms": {"31641"},
    "blvr_terms": {"31647", "31648", "31649", "31651"},
    "valve_terms": {"31647", "31648", "31649", "31651"},
    "chartis_terms": {"31647", "31651"},
    "tracheostomy_terms": {"31600", "31603", "31605", "31610"},
    "pdt_terms": {"31600", "31603", "31605", "31610"},
    # Pleural/thoracic
    "thoracentesis_terms": {"32554", "32555"},
    "thoracentesis_imaging_terms": {"32555"},
    "chest_tube_terms": {"32551"},
    "tunneled_pleural_catheter_terms": {"32550"},
    "pleural_drainage_terms": {"32556", "32557"},
    "pleurodesis_terms": {"32560", "32650"},
    "thoracoscopy_terms": {"32601", "32604", "32606", "32607", "32608", "32609", "32650", "32653"},
    "thoracoscopy_bundled_drain_terms": {"32601", "32604", "32606", "32607", "32608", "32609", "32650", "32653"},
    "thoracoscopy_separate_drain_terms": {"32601", "32604", "32606", "32607", "32608", "32609", "32650", "32653"},
    "thoracoscopy_pleural_site_terms": {"32609", "32653"},
    "thoracoscopy_lung_site_terms": {"32609", "32653"},
    "thoracoscopy_mediastinal_site_terms": {"32601", "32604", "32606", "32607", "32608", "32609", "32653"},
    "thoracoscopy_pericardial_site_terms": {"32601", "32604", "32606", "32607", "32608", "32609", "32653"},
    # Documentation/support terms that can still aid keyword recall for supported CPTs
    "peripheral_lesion_terms": {"31628", "31629", "31632", "31633", "31654"},
}

ABBREV_EXPANSIONS: dict[str, str] = {
    "dx": "diagnostic",
    "bx": "biopsy",
    "bronch": "bronchoscopy",
    "bronchoscope": "bronchoscopy",
    "samplng": "sampling",
    "subseq": "subsequent",
    "initl": "initial",
}

TOKEN_STOPWORDS = {
    "and",
    "with",
    "without",
    "for",
    "the",
    "a",
    "an",
    "of",
    "to",
    "via",
    "or",
    "in",
    "on",
    "by",
    "from",
    "each",
    "other",
    "planned",
    "plan",
    "only",
    "none",
    "deferred",
    "not",
}

NEGATIVE_SYNONYM_BUCKET_HINTS = (
    "negative",
    "neg",
    "planned_only",
    "deferred",
    "absent",
    "not_performed",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_five_digit_cpt(code: str) -> bool:
    return bool(re.fullmatch(r"\d{5}", str(code or "").strip()))


def _normalize_phrase(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("×", "x")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", text)
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _descriptor_phrases(descriptor: str) -> list[str]:
    out: list[str] = []
    base = _normalize_phrase(descriptor)
    if not base:
        return out
    out.append(base)

    # Add an expanded variant for common CPT abbreviation fragments.
    tokens = base.split()
    expanded = " ".join(ABBREV_EXPANSIONS.get(tok, tok) for tok in tokens)
    if expanded and expanded != base:
        out.append(expanded)

    # Add selective single-token terms for robust matching.
    for tok in expanded.split():
        tok_clean = tok.strip(".,;:()[]{}")
        if len(tok_clean) < 4:
            continue
        if tok_clean.isdigit() or tok_clean in TOKEN_STOPWORDS:
            continue
        out.append(tok_clean)
    return out


def _extract_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str)]
    if isinstance(value, dict):
        out: list[str] = []
        for key, nested in value.items():
            key_norm = str(key).strip().lower()
            if any(hint in key_norm for hint in NEGATIVE_SYNONYM_BUCKET_HINTS):
                continue
            out.extend(_extract_string_list(nested))
        return out
    return []


def _load_kb(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"KB must be an object: {path}")
    return raw


def _load_yaml_seeds(directory: Path) -> dict[str, list[str]]:
    if yaml is None or not directory.exists() or not directory.is_dir():
        return {}

    seed_keywords: dict[str, list[str]] = defaultdict(list)
    for yaml_path in sorted(list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))):
        # Skip generated json and unrelated files by using schema checks below.
        try:
            loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(loaded, dict):
            continue
        code = str(loaded.get("code", "")).strip()
        if not _is_five_digit_cpt(code):
            continue

        for phrase in _extract_string_list(loaded.get("positive_phrases")):
            seed_keywords[code].append(phrase)

        description = loaded.get("description")
        if isinstance(description, str) and description.strip():
            seed_keywords[code].append(description)

    return dict(seed_keywords)


def generate_cpt_keywords(
    *,
    repo_root: Path | None = None,
    kb_path: Path | None = None,
    seed_dir: Path | None = None,
    max_keywords_per_cpt: int = DEFAULT_MAX_KEYWORDS_PER_CPT,
) -> dict[str, list[str]]:
    root = repo_root or _repo_root()
    kb_file = kb_path or (root / "data" / "knowledge" / "ip_coding_billing_v3_0.json")
    seeds_dir = seed_dir or (root / "data" / "keyword_mappings")

    kb = _load_kb(kb_file)
    master = kb.get("master_code_index") or {}
    if not isinstance(master, dict):
        raise ValueError("KB missing master_code_index")

    raw_map: dict[str, list[str]] = defaultdict(list)

    # 1) Base descriptors from master_code_index.
    for code in sorted(master.keys()):
        if not _is_five_digit_cpt(code):
            continue
        entry = master.get(code)
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type", "")).strip().lower() != "cpt":
            continue

        descriptor = entry.get("descriptor")
        if isinstance(descriptor, str):
            raw_map[code].extend(_descriptor_phrases(descriptor))

        family = entry.get("family")
        if isinstance(family, str) and family.strip():
            raw_map[code].append(family)

    # 2) Routed KB synonyms.
    synonyms = kb.get("synonyms") or {}
    if isinstance(synonyms, dict):
        for group_name, phrases_blob in sorted(synonyms.items()):
            targets = SYNONYM_GROUP_TO_CPTS.get(group_name, set())
            if not targets:
                continue
            phrases = _extract_string_list(phrases_blob)
            if not phrases:
                continue
            for code in sorted(targets):
                if code in raw_map:
                    raw_map[code].extend(phrases)

    # 3) Optional YAML positive-phrase seeds.
    for code, phrases in _load_yaml_seeds(seeds_dir).items():
        raw_map[code].extend(phrases)

    # 4) Normalize + dedupe + cap + sort.
    finalized: dict[str, list[str]] = {}
    for code in sorted(raw_map.keys()):
        seen: set[str] = set()
        ordered: list[str] = []
        for phrase in raw_map[code]:
            norm = _normalize_phrase(phrase)
            if len(norm) < 3:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            ordered.append(norm)

        if max_keywords_per_cpt > 0:
            ordered = ordered[:max_keywords_per_cpt]
        finalized[code] = sorted(ordered)

    return finalized


def serialize_mapping(mapping: dict[str, list[str]]) -> str:
    return json.dumps(mapping, indent=2, sort_keys=True) + "\n"


def write_mapping(mapping: dict[str, list[str]], output_path: Path) -> bytes:
    content = serialize_mapping(mapping).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return content


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CPT keyword mapping JSON for keyword guard")
    parser.add_argument(
        "--kb",
        type=Path,
        default=Path("data/knowledge/ip_coding_billing_v3_0.json"),
        help="Path to KB JSON file",
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("data/keyword_mappings"),
        help="Directory with optional YAML keyword maps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/keyword_mappings/cpt_keywords.generated.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-keywords-per-cpt",
        type=int,
        default=DEFAULT_MAX_KEYWORDS_PER_CPT,
        help="Max number of keywords retained per CPT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _repo_root()

    kb_path = args.kb if args.kb.is_absolute() else root / args.kb
    seed_dir = args.seed_dir if args.seed_dir.is_absolute() else root / args.seed_dir
    output_path = args.output if args.output.is_absolute() else root / args.output

    mapping = generate_cpt_keywords(
        repo_root=root,
        kb_path=kb_path,
        seed_dir=seed_dir,
        max_keywords_per_cpt=max(0, int(args.max_keywords_per_cpt)),
    )
    write_mapping(mapping, output_path)

    counts = sorted(((code, len(phrases)) for code, phrases in mapping.items()), key=lambda kv: (-kv[1], kv[0]))
    top = ", ".join(f"{code}:{count}" for code, count in counts[:8])
    print(f"Generated keywords for {len(mapping)} CPTs -> {output_path}")
    print(f"Top keyword counts: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
