#!/usr/bin/env python3
"""
Validate a JSON Schema file.

Supports draft-07 and 2020-12 (auto-detected via the schema's "$schema" field).

Usage:
    python ops/tools/validate_jsonschema.py --schema data/knowledge/IP_Registry.json
"""

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, Draft7Validator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        required=True,
        help="Path to the JSON schema file (e.g., data/knowledge/IP_Registry_Enhanced_v2.json)",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.is_file():
        print(f"ERROR: Schema file not found at {schema_path}", file=sys.stderr)
        return 1

    with schema_path.open() as f:
        schema = json.load(f)

    declared = str(schema.get("$schema") or "").lower()
    if "draft-07" in declared or "draft/07" in declared:
        Draft7Validator.check_schema(schema)
    else:
        # Default to 2020-12 when explicit or unknown. This keeps the script useful
        # for newer schemas without requiring flags.
        Draft202012Validator.check_schema(schema)
    print(f"Schema OK: {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
