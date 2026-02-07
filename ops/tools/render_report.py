#!/usr/bin/env python3
"""Render a structured report from an extraction JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.reporting import compose_structured_report_from_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a structured IP report from extraction JSON.")
    parser.add_argument("--input", required=True, help="Path to extraction JSON payload.")
    parser.add_argument("--output", help="Path to write the rendered report. Prints to stdout if omitted.")
    parser.add_argument("--strict", action="store_true", help="Enable style strict mode validation.")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    report = compose_structured_report_from_extraction(payload, strict=args.strict)

    if args.output:
        Path(args.output).write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
