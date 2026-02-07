from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.reporting import build_procedure_bundle_from_extraction, get_coder_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump coder-friendly view of a structured bundle or extraction JSON.")
    parser.add_argument("--input", required=True, help="Path to extraction JSON payload.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    bundle = build_procedure_bundle_from_extraction(payload)
    view = get_coder_view(bundle)
    print(json.dumps(view, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
