#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


if not _truthy_env("PROCSUITE_SKIP_DOTENV"):
    load_dotenv(override=False)

from app.registry.application.registry_service import RegistryService  # noqa: E402
from app.registry.deterministic_extractors import run_deterministic_extractors  # noqa: E402
from app.registry.processing.masking import mask_offset_preserving  # noqa: E402
from app.registry.processing.navigation_fiducials import (  # noqa: E402
    apply_navigation_fiducials,
)
from app.registry.schema import RegistryRecord  # noqa: E402
from app.registry.self_correction.keyword_guard import scan_for_omissions  # noqa: E402


def _read_note_text(path: str | None, inline_text: str | None) -> str:
    if inline_text:
        return inline_text
    if not path:
        raise ValueError("Provide --note or --text.")
    note_path = Path(path)
    return note_path.read_text(encoding="utf-8")


def _collect_performed_flags(record_data: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    procs = record_data.get("procedures_performed")
    if isinstance(procs, dict):
        for name, payload in procs.items():
            if isinstance(payload, dict) and payload.get("performed") is True:
                flags.add(f"procedures_performed.{name}.performed")

    pleural = record_data.get("pleural_procedures")
    if isinstance(pleural, dict):
        for name, payload in pleural.items():
            if isinstance(payload, dict) and payload.get("performed") is True:
                flags.add(f"pleural_procedures.{name}.performed")

    if record_data.get("established_tracheostomy_route") is True:
        flags.add("established_tracheostomy_route")

    granular = record_data.get("granular_data")
    if isinstance(granular, dict):
        targets = granular.get("navigation_targets")
        if isinstance(targets, list):
            for target in targets:
                if isinstance(target, dict) and target.get("fiducial_marker_placed") is True:
                    flags.add("granular_data.navigation_targets[*].fiducial_marker_placed")
                    break

    return flags


def _apply_seed_uplift(
    record_data: dict[str, Any],
    seed: dict[str, Any],
    masked_note_text: str,
) -> tuple[dict[str, Any], list[str]]:
    uplifted: list[str] = []

    seed_procs = seed.get("procedures_performed")
    if isinstance(seed_procs, dict):
        record_procs = record_data.get("procedures_performed") or {}
        if not isinstance(record_procs, dict):
            record_procs = {}
        for name, payload in seed_procs.items():
            if not isinstance(payload, dict) or payload.get("performed") is not True:
                continue
            existing = record_procs.get(name) or {}
            if not isinstance(existing, dict):
                existing = {}
            if existing.get("performed") is not True:
                existing["performed"] = True
                uplifted.append(f"procedures_performed.{name}.performed")
            for key, value in payload.items():
                if key == "performed":
                    continue
                if existing.get(key) in (None, "", [], {}):
                    existing[key] = value
            record_procs[name] = existing
        if record_procs:
            record_data["procedures_performed"] = record_procs

    seed_pleural = seed.get("pleural_procedures")
    if isinstance(seed_pleural, dict):
        record_pleural = record_data.get("pleural_procedures") or {}
        if not isinstance(record_pleural, dict):
            record_pleural = {}
        for name, payload in seed_pleural.items():
            if not isinstance(payload, dict) or payload.get("performed") is not True:
                continue
            existing = record_pleural.get(name) or {}
            if not isinstance(existing, dict):
                existing = {}
            if existing.get("performed") is not True:
                existing["performed"] = True
                uplifted.append(f"pleural_procedures.{name}.performed")
            for key, value in payload.items():
                if key == "performed":
                    continue
                if existing.get(key) in (None, "", [], {}):
                    existing[key] = value
            record_pleural[name] = existing
        if record_pleural:
            record_data["pleural_procedures"] = record_pleural

    if seed.get("established_tracheostomy_route") is True:
        if record_data.get("established_tracheostomy_route") is not True:
            record_data["established_tracheostomy_route"] = True
            uplifted.append("established_tracheostomy_route")

    if apply_navigation_fiducials(record_data, masked_note_text):
        uplifted.append("granular_data.navigation_targets[*].fiducial_marker_placed")

    return record_data, uplifted


def _print_list(label: str, items: list[str] | set[str]) -> None:
    if not items:
        print(f"{label}: (none)")
        return
    if isinstance(items, set):
        items = sorted(items)
    print(f"{label}:")
    for item in items:
        print(f"  - {item}")


def _print_self_correction_diagnostics(result) -> None:
    audit_report = getattr(result, "audit_report", None)
    high_conf = getattr(audit_report, "high_conf_omissions", None) if audit_report is not None else None
    if not high_conf:
        print("Audit high-conf omissions: (none)")
    else:
        print("Audit high-conf omissions:")
        for pred in high_conf:
            cpt = getattr(pred, "cpt", None)
            prob = getattr(pred, "prob", None)
            bucket = getattr(pred, "bucket", None)
            try:
                prob_str = f"{float(prob):.2f}" if prob is not None else "?"
            except Exception:
                prob_str = "?"
            print(f"  - {cpt} (prob={prob_str}, bucket={bucket})")

    warnings = getattr(result, "warnings", None)
    if isinstance(warnings, list):
        self_correct_warnings = [w for w in warnings if isinstance(w, str) and "SELF_CORRECT" in w]
        auto_corrected = [w for w in warnings if isinstance(w, str) and "AUTO_CORRECTED" in w]
        diag = self_correct_warnings + auto_corrected
        if diag:
            print("Self-correction diagnostics:")
            for w in diag:
                print(f"  - {w}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test the registry extraction pipeline on a note."
    )
    parser.add_argument("--note", help="Path to a note text file.")
    parser.add_argument("--text", help="Inline note text.")
    parser.add_argument(
        "--self-correct",
        action="store_true",
        help="Attempt self-correction via extract_fields (requires raw-ML + LLM).",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Allow real LLM calls (disables stub/offline defaults).",
    )
    args = parser.parse_args()

    try:
        note_text = _read_note_text(args.note, args.text)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.real_llm:
        os.environ.setdefault("REGISTRY_USE_STUB_LLM", "0")
        os.environ.setdefault("GEMINI_OFFLINE", "0")
        os.environ.setdefault("OPENAI_OFFLINE", "0")
    else:
        if os.getenv("REGISTRY_USE_STUB_LLM") is None:
            os.environ["REGISTRY_USE_STUB_LLM"] = "1"
        if os.getenv("GEMINI_OFFLINE") is None:
            os.environ["GEMINI_OFFLINE"] = "1"

    masked = mask_offset_preserving(note_text)

    service = RegistryService()
    record, warnings, meta = service.extract_record(note_text)

    before_flags = _collect_performed_flags(record.model_dump())

    seed = run_deterministic_extractors(masked)
    record_data = record.model_dump()
    record_data, uplifted = _apply_seed_uplift(record_data, seed, masked)
    uplifted_flags = set(uplifted)
    after_flags = _collect_performed_flags(record_data)

    record_after = RegistryRecord(**record_data)
    omission_warnings = scan_for_omissions(masked, record_after)

    _print_list("Performed flags (extract_record)", before_flags)
    _print_list("Performed flags added by deterministic uplift", uplifted_flags)
    _print_list("Performed flags (after uplift)", after_flags)
    _print_list("Extract warnings", warnings)
    _print_list("Omission warnings", omission_warnings)

    if args.self_correct:
        os.environ.setdefault("REGISTRY_SELF_CORRECT_ENABLED", "1")
        try:
            result = service.extract_fields(note_text)
        except Exception as exc:
            print(f"SELF_CORRECT_ERROR: {exc}")
        else:
            _print_self_correction_diagnostics(result)
            if result.self_correction:
                print("Self-correction applied:")
                for item in result.self_correction:
                    applied_paths = getattr(item, "applied_paths", None)
                    if isinstance(applied_paths, list) and applied_paths:
                        print(f"  - {item.trigger.target_cpt}: applied {', '.join(applied_paths)}")
                    else:
                        print(f"  - {item.trigger.target_cpt}: applied (no paths recorded)")
            else:
                print("Self-correction applied: (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
