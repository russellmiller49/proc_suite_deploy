#!/usr/bin/env python3
"""Validate a knowledge+schema release locally (no external network calls).

This script is intended to backstop knowledge/schema refactors:
- Loads the KB via both the lightweight JSON loader and the main KB adapter
- Validates the registry schema can build a RegistryRecord model
- Runs a no-op extraction in the **parallel_ner** pathway to ensure nothing crashes at import/runtime
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kb",
        default="data/knowledge/ip_coding_billing_v3_0.json",
        help="Path to knowledge base JSON (default: data/knowledge/ip_coding_billing_v3_0.json)",
    )
    ap.add_argument(
        "--schema",
        default="data/knowledge/IP_Registry.json",
        help="Path to registry JSON schema (default: data/knowledge/IP_Registry.json)",
    )
    ap.add_argument(
        "--no-op-note",
        default="",
        help="Note text for a no-op registry extraction run (default: empty string).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if KB filename semantic version mismatches internal version.",
    )
    return ap.parse_args()


def _extract_semver_from_filename(path: Path) -> tuple[int, int] | None:
    import re

    m = re.search(r"_v(\d+)[._](\d+)\.json$", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _extract_semver_from_kb_version(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().lstrip("v").split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def main() -> int:
    args = _parse_args()
    kb_path = Path(args.kb)
    schema_path = Path(args.schema)

    if not kb_path.is_file():
        print(f"ERROR: KB not found: {kb_path}", file=sys.stderr)
        return 2
    if not schema_path.is_file():
        print(f"ERROR: Schema not found: {schema_path}", file=sys.stderr)
        return 2

    # 1) Basic JSON parse
    try:
        kb_json = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: KB is not valid JSON: {kb_path} ({exc})", file=sys.stderr)
        return 2

    kb_version = kb_json.get("version")
    file_semver = _extract_semver_from_filename(kb_path.resolve())
    kb_semver = _extract_semver_from_kb_version(kb_version)
    if args.strict and file_semver and kb_semver and file_semver != kb_semver:
        print(
            f"ERROR: KB filename semver {file_semver} != internal version {kb_semver} ({kb_path})",
            file=sys.stderr,
        )
        return 2

    # 2) Validate KB schema using the main loader (Draft-07 schema)
    try:
        from app.common.knowledge import get_knowledge

        _ = get_knowledge(kb_path, force_reload=True)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: KB failed Procedure Suite knowledge schema validation: {exc}", file=sys.stderr)
        return 2

    # 2.5) Semantic Validation (Integrity & Logic)
    print("Running semantic validation...", file=sys.stderr)
    try:
        from app.domain.knowledge_base.validator import SemanticValidator

        validator = SemanticValidator(kb_json)
        issues = validator.validate()
        if issues:
            print(f"ERROR: Found {len(issues)} semantic issues in KB:", file=sys.stderr)
            for issue in issues:
                print(f"  - {issue}", file=sys.stderr)
            return 2  # Hard fail
        print("OK: Semantic validation passed", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Semantic validation crashed: {exc}", file=sys.stderr)
        return 2

    # 3) Validate the KB adapter loads and can resolve a representative code
    try:
        from app.coder.adapters.persistence.csv_kb_adapter import JsonKnowledgeBaseAdapter

        kb_repo = JsonKnowledgeBaseAdapter(kb_path)
        sample = kb_repo.get_procedure_info("31628") or kb_repo.get_procedure_info("+31628")
        if sample is None:
            print("ERROR: KB adapter could not resolve CPT 31628", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: KB adapter load failed: {exc}", file=sys.stderr)
        return 2

    # 4) Validate RegistryRecord model can be built from schema (dynamic model)
    try:
        from app.registry.schema import RegistryRecord

        _ = RegistryRecord.model_validate({})
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: RegistryRecord model build/validation failed: {exc}", file=sys.stderr)
        return 2

    # 5) No-op registry extraction (should not crash; no external calls required)
    try:
        from app.registry.application.registry_service import RegistryService

        # Ensure no-network behavior: use the parallel_ner pathway instead of the LLM RegistryEngine.
        previous_engine = os.environ.get("REGISTRY_EXTRACTION_ENGINE")
        os.environ["REGISTRY_EXTRACTION_ENGINE"] = "parallel_ner"
        try:
            record, _warnings, _meta = RegistryService(default_version="v3").extract_record(
                args.no_op_note or "",
                note_id="validate_knowledge_release",
            )
        finally:
            if previous_engine is None:
                os.environ.pop("REGISTRY_EXTRACTION_ENGINE", None)
            else:
                os.environ["REGISTRY_EXTRACTION_ENGINE"] = previous_engine
        if record is None:
            print("ERROR: registry extract_record returned None", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: registry no-op extraction failed: {exc}", file=sys.stderr)
        return 2

    # 6) Deterministic RegistryRecord→CPT should not crash
    try:
        from app.coder.domain_rules.registry_to_cpt.coding_rules import derive_all_codes_with_meta

        _codes, _rationales, _warnings = derive_all_codes_with_meta(record)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: deterministic Registry→CPT derivation failed: {exc}", file=sys.stderr)
        return 2

    print("OK: validate_knowledge_release passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
