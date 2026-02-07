from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Set, Tuple, Dict


# Matches:
#   - CPT: 5 digits (e.g., 31622)
#   - HCPCS Level II: Letter + 4 digits (e.g., C1601, J7665)
#   - Optional leading "+" used in this JSON to denote add-on codes (e.g., +31627)
CODE_RE = re.compile(r"^\+?(?:\d{5}|[A-Z]\d{4})$")


def normalize_code(code: str, *, keep_addon_plus: bool) -> str:
    code = code.strip().upper()
    return code if keep_addon_plus else code.lstrip("+")


def extract_codes_anywhere(obj: Any) -> Set[str]:
    """
    Recursively traverse a JSON-like structure and collect any strings/keys
    that look like CPT/HCPCS codes (including optional leading '+').
    """
    found: Set[str] = set()

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip().upper()
            if CODE_RE.match(s):
                found.add(s)
            return
        if isinstance(x, (int, float)):
            # Codes should be strings, but guard just in case
            s = str(int(x)).strip()
            if CODE_RE.match(s):
                found.add(s)
            return
        if isinstance(x, list):
            for item in x:
                walk(item)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                # Keys can be codes too
                if isinstance(k, str):
                    ks = k.strip().upper()
                    if CODE_RE.match(ks):
                        found.add(ks)
                walk(v)
            return

    walk(obj)
    return found


def _codes_from_dict_keys(d: Any) -> Set[str]:
    if not isinstance(d, dict):
        return set()
    out = set()
    for k in d.keys():
        if isinstance(k, str):
            ks = k.strip().upper()
            if CODE_RE.match(ks):
                out.add(ks)
    return out


def collect_billable_codes(data: dict) -> Set[str]:
    """
    Collect codes that are intended to be "real" billable / selectable codes
    in this knowledge base (vs codes that show up only in references/logic).
    """
    billable: Set[str] = set()

    # 1) code_lists (explicit curated lists)
    code_lists = data.get("code_lists", {})
    if isinstance(code_lists, dict):
        for _, codes in code_lists.items():
            if isinstance(codes, list):
                for c in codes:
                    if isinstance(c, str) and CODE_RE.match(c.strip().upper()):
                        billable.add(c.strip().upper())

    # 2) add_on_codes list (explicit)
    add_on_codes = data.get("add_on_codes", [])
    if isinstance(add_on_codes, list):
        for c in add_on_codes:
            if isinstance(c, str) and CODE_RE.match(c.strip().upper()):
                billable.add(c.strip().upper())

    # 3) fee_schedules.*.codes (keys are CPT/HCPCS; include + add-ons)
    fee_schedules = data.get("fee_schedules", {})
    if isinstance(fee_schedules, dict):
        for _, sched in fee_schedules.items():
            if isinstance(sched, dict):
                billable |= _codes_from_dict_keys(sched.get("codes", {}))

    # 4) cms_rvus sections (bronchoscopy/pleural/thoracoscopy/sedation/em/imaging)
    cms_rvus = data.get("cms_rvus", {})
    if isinstance(cms_rvus, dict):
        for section_name, section in cms_rvus.items():
            if isinstance(section_name, str) and section_name.startswith("_"):
                continue
            billable |= _codes_from_dict_keys(section)

    # 5) simplified RVU tables
    billable |= _codes_from_dict_keys(data.get("rvus", {}))
    billable |= _codes_from_dict_keys(data.get("rvus_pleural", {}))
    billable |= _codes_from_dict_keys(data.get("rvus_sedation_em", {}))

    # 6) hcpcs top-level keys (HCPCS Level II device/drug codes; alphanumeric only)
    billable |= _codes_from_dict_keys(data.get("hcpcs", {}))

    # 7) pleural cpt maps (redundant but harmless; helps if future versions omit other sections)
    pleural = data.get("pleural", {})
    if isinstance(pleural, dict):
        billable |= _codes_from_dict_keys(pleural.get("cpt_map", {}))
        billable |= _codes_from_dict_keys(pleural.get("thoracoscopy_cpt_map", {}))

    return billable


def build_valid_ip_codes(
    json_path: str | Path,
    *,
    keep_addon_plus: bool = False,
    include_reference_codes: bool = False,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Returns:
      - valid_codes: the final normalized code set
      - debug: a dict with useful intermediate sets
    """
    json_path = Path(json_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))

    codes_anywhere = extract_codes_anywhere(data)
    codes_billable = collect_billable_codes(data)

    # "reference-only" are codes referenced somewhere (e.g., NCCI pairs) but not in billable lists
    codes_reference_only = codes_anywhere - codes_billable

    selected = set(codes_billable)
    if include_reference_codes:
        selected |= codes_reference_only

    normalized = {normalize_code(c, keep_addon_plus=keep_addon_plus) for c in selected}

    debug = {
        "codes_anywhere_raw": codes_anywhere,
        "codes_billable_raw": codes_billable,
        "codes_reference_only_raw": codes_reference_only,
    }
    return normalized, debug


def format_as_python_set(codes: Iterable[str], cols: int = 10) -> str:
    codes_sorted = sorted(set(codes))
    chunks = [codes_sorted[i : i + cols] for i in range(0, len(codes_sorted), cols)]
    lines = []
    for chunk in chunks:
        lines.append("    " + ", ".join(f'"{c}"' for c in chunk) + ",")
    return "VALID_IP_CODES = {\n" + "\n".join(lines) + "\n}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build VALID_IP_CODES set from knowledge base")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=Path("data/knowledge/ip_coding_billing_v3_0.json"),
        help="Path to knowledge base JSON file (default: data/knowledge/ip_coding_billing_v3_0.json)",
    )
    parser.add_argument(
        "--keep-addon-plus",
        action="store_true",
        help="Keep '+' prefix on add-on codes",
    )
    parser.add_argument(
        "--include-reference-codes",
        action="store_true",
        help="Include reference-only codes (e.g., NCCI-only codes like 32100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: print to stdout)",
    )
    args = parser.parse_args()

    if not args.kb_path.exists():
        print(f"Error: Knowledge base file not found: {args.kb_path}")
        print(f"Please ensure the file exists or specify with --kb-path")
        exit(1)

    # Recommended: normalize add-on "+", and DO NOT include reference-only codes (e.g., 32100)
    valid_codes, dbg = build_valid_ip_codes(
        args.kb_path,
        keep_addon_plus=args.keep_addon_plus,
        include_reference_codes=args.include_reference_codes,
    )

    output_text = f"# Generated from {args.kb_path}\n"
    output_text += f"# Total codes: {len(valid_codes)}\n"
    output_text += format_as_python_set(valid_codes)

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"✓ Wrote {len(valid_codes)} codes to {args.output}")
    else:
        print(f"Built {len(valid_codes)} codes")
        print(output_text)

    # If you want to see logic-only references (e.g., NCCI-only codes like 32100):
    # valid_with_refs, dbg2 = build_valid_ip_codes(args.kb_path, keep_addon_plus=False, include_reference_codes=True)
    # print("Reference-only codes:", sorted(dbg2["codes_reference_only_raw"]))
