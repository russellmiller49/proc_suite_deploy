#!/usr/bin/env python3
"""Summarize strict-render fallback reasons for two reporter seed-path evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.reporter_seed_eval import build_seed_path_fallback_reason_report, maybe_write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_seed_path_fallback_reason_report(
        left_report=_load(args.left_report),
        right_report=_load(args.right_report),
        left_path=str(args.left_report),
        right_path=str(args.right_report),
    )
    maybe_write_json(args.output, payload)
    print(f"Wrote fallback summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
