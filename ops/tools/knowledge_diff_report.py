#!/usr/bin/env python3
"""Generate a simple diff report between two knowledge base JSON files.

Focuses on high-signal changes for yearly updates:
- Codes added/removed in master_code_index
- Descriptor changes
- RVU changes (rvu_simplified + cms_pfs_2026 totals when available)
- add_on_codes changes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Path to the old KB JSON")
    ap.add_argument("--new", required=True, help="Path to the new KB JSON")
    ap.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max examples to print per section (default: 50)",
    )
    return ap.parse_args()


@dataclass(frozen=True)
class _KB:
    path: Path
    sha256: str
    version: str | None
    data: dict[str, Any]


def _load_kb(path: Path) -> _KB:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return _KB(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        version=str(data.get("version")) if data.get("version") is not None else None,
        data=data,
    )


def _master_index(kb: _KB) -> dict[str, Any]:
    master = kb.data.get("master_code_index")
    return master if isinstance(master, dict) else {}


def _add_on_codes(kb: _KB) -> set[str]:
    add_ons = kb.data.get("add_on_codes")
    if not isinstance(add_ons, list):
        return set()
    return {str(x) for x in add_ons if isinstance(x, str) and x.strip()}


def _descriptor(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("descriptor")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _rvu_simplified(entry: Any) -> tuple[float, float, float] | None:
    if not isinstance(entry, dict):
        return None
    simplified = entry.get("rvu_simplified")
    if not isinstance(simplified, dict):
        return None
    work = simplified.get("work")
    pe = simplified.get("pe")
    mp = simplified.get("mp")
    if work is None or pe is None or mp is None:
        return None
    try:
        return float(work), float(pe), float(mp)
    except (TypeError, ValueError):
        return None


def _cms_total_facility_rvu(entry: Any) -> float | None:
    if not isinstance(entry, dict):
        return None
    financials = entry.get("financials")
    if not isinstance(financials, dict):
        return None
    cms = financials.get("cms_pfs_2026")
    if not isinstance(cms, dict):
        return None
    total = cms.get("total_facility_rvu")
    if total is None:
        return None
    try:
        return float(total)
    except (TypeError, ValueError):
        return None


def _fmt_rvu(triple: tuple[float, float, float] | None) -> str:
    if not triple:
        return "n/a"
    work, pe, mp = triple
    total = work + pe + mp
    return f"work={work:.2f} pe={pe:.2f} mp={mp:.2f} total={total:.2f}"


def _print_header(old: _KB, new: _KB) -> None:
    print("Knowledge Diff Report")
    print(f"- old: {old.path} (version={old.version}, sha256={old.sha256[:12]}…)")
    print(f"- new: {new.path} (version={new.version}, sha256={new.sha256[:12]}…)")
    print()


def main() -> int:
    args = _parse_args()
    old = _load_kb(Path(args.old))
    new = _load_kb(Path(args.new))

    _print_header(old, new)

    old_master = _master_index(old)
    new_master = _master_index(new)

    old_codes = set(old_master.keys())
    new_codes = set(new_master.keys())
    added = sorted(new_codes - old_codes)
    removed = sorted(old_codes - new_codes)

    print("**Master Code Index**")
    print(f"- codes added: {len(added)}")
    for code in added[: args.limit]:
        print(f"  - + {code}")
    if len(added) > args.limit:
        print(f"  - … ({len(added) - args.limit} more)")
    print(f"- codes removed: {len(removed)}")
    for code in removed[: args.limit]:
        print(f"  - - {code}")
    if len(removed) > args.limit:
        print(f"  - … ({len(removed) - args.limit} more)")

    descriptor_changed: list[str] = []
    rvu_changed: list[str] = []
    cms_total_changed: list[str] = []

    for code in sorted(old_codes & new_codes):
        old_entry = old_master.get(code)
        new_entry = new_master.get(code)

        if _descriptor(old_entry) != _descriptor(new_entry):
            descriptor_changed.append(code)

        if _rvu_simplified(old_entry) != _rvu_simplified(new_entry):
            rvu_changed.append(code)

        if _cms_total_facility_rvu(old_entry) != _cms_total_facility_rvu(new_entry):
            cms_total_changed.append(code)

    print(f"- descriptor changes: {len(descriptor_changed)}")
    for code in descriptor_changed[: args.limit]:
        print(f"  - {code}: {(_descriptor(old_master.get(code)) or '')!r} -> {(_descriptor(new_master.get(code)) or '')!r}")
    if len(descriptor_changed) > args.limit:
        print(f"  - … ({len(descriptor_changed) - args.limit} more)")

    print(f"- rvu_simplified changes: {len(rvu_changed)}")
    for code in rvu_changed[: args.limit]:
        print(f"  - {code}: {_fmt_rvu(_rvu_simplified(old_master.get(code)))} -> {_fmt_rvu(_rvu_simplified(new_master.get(code)))}")
    if len(rvu_changed) > args.limit:
        print(f"  - … ({len(rvu_changed) - args.limit} more)")

    print(f"- cms_pfs_2026.total_facility_rvu changes: {len(cms_total_changed)}")
    for code in cms_total_changed[: args.limit]:
        o = _cms_total_facility_rvu(old_master.get(code))
        n = _cms_total_facility_rvu(new_master.get(code))
        print(f"  - {code}: {o if o is not None else 'n/a'} -> {n if n is not None else 'n/a'}")
    if len(cms_total_changed) > args.limit:
        print(f"  - … ({len(cms_total_changed) - args.limit} more)")
    print()

    old_add_ons = _add_on_codes(old)
    new_add_ons = _add_on_codes(new)
    added_add_ons = sorted(new_add_ons - old_add_ons)
    removed_add_ons = sorted(old_add_ons - new_add_ons)

    print("**Add-on Codes**")
    print(f"- added: {len(added_add_ons)}")
    for code in added_add_ons[: args.limit]:
        print(f"  - + {code}")
    if len(added_add_ons) > args.limit:
        print(f"  - … ({len(added_add_ons) - args.limit} more)")

    print(f"- removed: {len(removed_add_ons)}")
    for code in removed_add_ons[: args.limit]:
        print(f"  - - {code}")
    if len(removed_add_ons) > args.limit:
        print(f"  - … ({len(removed_add_ons) - args.limit} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

