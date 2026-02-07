#!/usr/bin/env python3
"""Immediate hotfix utilities for critical extraction issues.

Primary use: apply checkbox-negative corrections where some EMR templates encode
unchecked options as "0- Item" or "[ ] Item", which can be hallucinated as True.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def apply_checkbox_correction(text: str, record: Any) -> Any:
    """Apply checkbox-template negation corrections (shared with production)."""
    record_dict: dict[str, Any]
    if isinstance(record, dict):
        record_dict = dict(record)
    elif hasattr(record, "model_dump"):
        record_dict = dict(record.model_dump())  # type: ignore[no-any-return]
    else:
        raise TypeError("record must be a dict or a RegistryRecord-like object")

    _ensure_repo_on_path()
    from app.registry.postprocess.template_checkbox_negation import apply_template_checkbox_negation
    from app.registry.schema import RegistryRecord

    record_obj = RegistryRecord.model_validate(record_dict)
    updated, warnings = apply_template_checkbox_negation(text or "", record_obj)
    if not warnings:
        return record

    # Best-effort: return same type when possible.
    if isinstance(record, dict):
        return updated.model_dump()
    return updated


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply immediate extraction logic fixes to a record JSON.")
    parser.add_argument("--note", type=Path, required=True, help="Path to raw/masked note text file")
    parser.add_argument("--record", type=Path, required=True, help="Path to RegistryRecord JSON file")
    args = parser.parse_args()

    note_text = _load_text(args.note)
    record_json = _load_json(args.record)
    updated = apply_checkbox_correction(note_text, record_json)

    out = updated if isinstance(updated, dict) else getattr(updated, "model_dump", lambda: updated)()
    print(json.dumps(out, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
