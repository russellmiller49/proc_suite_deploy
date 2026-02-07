#!/usr/bin/env python3
"""Batch smoke test for registry pipeline on random notes.

This script randomly selects N notes from a notes directory, runs the registry
pipeline smoke test on each, and saves all output to a text file.

Supported note formats in --notes-dir:
- *.json: dict of note_id->note_text (we pick the first non "_syn_" key)
- *.txt: one note per file
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
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


def _format_list(label: str, items: list[str] | set[str]) -> str:
    """Format a list of items for output."""
    if not items:
        return f"{label}: (none)\n"
    if isinstance(items, set):
        items = sorted(items)
    lines = [f"{label}:"]
    for item in items:
        lines.append(f"  - {item}")
    return "\n".join(lines) + "\n"


def _format_self_correction_diagnostics(result) -> str:
    lines: list[str] = []

    audit_report = getattr(result, "audit_report", None)
    high_conf = getattr(audit_report, "high_conf_omissions", None) if audit_report is not None else None
    if not high_conf:
        lines.append("Audit high-conf omissions: (none)\n")
    else:
        lines.append("Audit high-conf omissions:\n")
        for pred in high_conf:
            cpt = getattr(pred, "cpt", None)
            prob = getattr(pred, "prob", None)
            bucket = getattr(pred, "bucket", None)
            try:
                prob_str = f"{float(prob):.2f}" if prob is not None else "?"
            except Exception:
                prob_str = "?"
            lines.append(f"  - {cpt} (prob={prob_str}, bucket={bucket})\n")

    warnings = getattr(result, "warnings", None)
    if isinstance(warnings, list):
        self_correct_warnings = [w for w in warnings if isinstance(w, str) and "SELF_CORRECT" in w]
        auto_corrected = [w for w in warnings if isinstance(w, str) and "AUTO_CORRECTED" in w]
        diag = self_correct_warnings + auto_corrected
        if diag:
            lines.append("Self-correction diagnostics:\n")
            for w in diag:
                lines.append(f"  - {w}\n")

    return "".join(lines)


def _run_smoke_test(note_text: str, note_id: str, self_correct: bool = False) -> str:
    """Run smoke test on a single note and return formatted output."""
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"NOTE: {note_id}")
    output_lines.append("=" * 80)
    output_lines.append("")

    try:
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

        output_lines.append(_format_list("Performed flags (extract_record)", before_flags))
        output_lines.append(_format_list("Performed flags added by deterministic uplift", uplifted_flags))
        output_lines.append(_format_list("Performed flags (after uplift)", after_flags))
        output_lines.append(_format_list("Extract warnings", warnings))
        output_lines.append(_format_list("Omission warnings", omission_warnings))

        if self_correct:
            os.environ.setdefault("REGISTRY_SELF_CORRECT_ENABLED", "1")
            try:
                result = service.extract_fields(note_text)
            except Exception as exc:
                output_lines.append(f"SELF_CORRECT_ERROR: {exc}\n")
            else:
                output_lines.append(_format_self_correction_diagnostics(result))
                if result.self_correction:
                    output_lines.append("Self-correction applied:\n")
                    for item in result.self_correction:
                        applied_paths = getattr(item, "applied_paths", None)
                        if isinstance(applied_paths, list) and applied_paths:
                            output_lines.append(
                                f"  - {item.trigger.target_cpt}: applied {', '.join(applied_paths)}\n"
                            )
                        else:
                            output_lines.append(f"  - {item.trigger.target_cpt}: applied (no paths recorded)\n")
                else:
                    output_lines.append("Self-correction applied: (none)\n")

        output_lines.append("")
        output_lines.append("STATUS: SUCCESS")
        output_lines.append("")

    except Exception as exc:
        output_lines.append(f"ERROR: {exc}\n")
        output_lines.append("STATUS: FAILED")
        output_lines.append("")

    return "".join(output_lines)


def _load_notes_from_directory(notes_dir: Path) -> dict[str, str]:
    """Load notes from JSON and/or TXT files in a directory.

    Returns a dict mapping note_id to note_text.

    - For *.json files: expects a dict of key->text and selects the first key that
      does NOT contain "_syn_" (to avoid synthetic variants), matching prior behavior.
    - For *.txt files: note_id is the filename stem, and note_text is file contents.
    """
    notes: dict[str, str] = {}

    json_files = sorted(notes_dir.glob("*.json"))
    txt_files = sorted(notes_dir.glob("*.txt"))

    # Backwards-compatible: keep supporting JSON sources, but allow TXT-only dirs.
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Expected JSON object mapping note keys to text")

            # Find the main note (key without '_syn_' suffix)
            main_key = None
            for key in data.keys():
                if isinstance(key, str) and "_syn_" not in key:
                    main_key = key
                    break

            if main_key and isinstance(data.get(main_key), str):
                note_id = json_file.stem
                notes[note_id] = data[main_key]
        except Exception as exc:
            print(f"Warning: Failed to load {json_file}: {exc}", file=sys.stderr)

    for txt_file in txt_files:
        try:
            note_text = txt_file.read_text(encoding="utf-8")
            note_id = txt_file.stem
            # Prefer JSON if both exist for same stem, but warn so it's visible.
            if note_id in notes:
                print(
                    f"Warning: Duplicate note_id '{note_id}' from {txt_file}; keeping JSON version",
                    file=sys.stderr,
                )
                continue
            notes[note_id] = note_text
        except Exception as exc:
            print(f"Warning: Failed to load {txt_file}: {exc}", file=sys.stderr)

    return notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch smoke test the registry extraction pipeline on random notes."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of random notes to test (default: 30)",
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=ROOT / "data" / "knowledge" / "patient_note_texts",
        help=(
            "Directory containing note files (*.json or *.txt) "
            "(default: data/knowledge/patient_note_texts)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: registry_smoke_batch_<timestamp>.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
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

    # Set up environment
    if args.real_llm:
        os.environ.setdefault("REGISTRY_USE_STUB_LLM", "0")
        os.environ.setdefault("GEMINI_OFFLINE", "0")
    else:
        if os.getenv("REGISTRY_USE_STUB_LLM") is None:
            os.environ["REGISTRY_USE_STUB_LLM"] = "1"
        if os.getenv("GEMINI_OFFLINE") is None:
            os.environ["GEMINI_OFFLINE"] = "1"

    # Load notes
    if not args.notes_dir.exists():
        print(f"ERROR: Notes directory not found: {args.notes_dir}", file=sys.stderr)
        return 1

    print(f"Loading notes from {args.notes_dir}...", file=sys.stderr)
    all_notes = _load_notes_from_directory(args.notes_dir)
    
    if not all_notes:
        print(f"ERROR: No notes found in {args.notes_dir}", file=sys.stderr)
        return 1

    print(f"Loaded {len(all_notes)} notes", file=sys.stderr)

    # Select random notes
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}", file=sys.stderr)

    count = min(args.count, len(all_notes))
    selected_notes = random.sample(list(all_notes.items()), count)
    
    print(f"Selected {count} random notes for testing", file=sys.stderr)

    # Determine output file
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / f"registry_smoke_batch_{timestamp}.txt"

    # Run smoke tests
    print(f"Running smoke tests...", file=sys.stderr)
    print(f"Output will be saved to: {output_path}", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("REGISTRY PIPELINE BATCH SMOKE TEST\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Notes tested: {count}\n")
        f.write(f"Notes directory: {args.notes_dir}\n")
        if args.seed is not None:
            f.write(f"Random seed: {args.seed}\n")
        f.write(f"Self-correction: {args.self_correct}\n")
        f.write(f"Real LLM enabled: {args.real_llm}\n")
        f.write("=" * 80 + "\n")
        f.write("\n")

        # Run tests
        success_count = 0
        failed_count = 0

        for i, (note_id, note_text) in enumerate(selected_notes, 1):
            print(f"[{i}/{count}] Testing {note_id}...", file=sys.stderr)
            result = _run_smoke_test(note_text, note_id, args.self_correct)
            f.write(result)
            f.flush()

            if "STATUS: SUCCESS" in result:
                success_count += 1
            else:
                failed_count += 1

        # Write summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total notes tested: {count}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Success rate: {success_count/count*100:.1f}%\n")
        f.write("=" * 80 + "\n")

    print(f"\nCompleted! Results saved to: {output_path}", file=sys.stderr)
    print(f"Summary: {success_count} successful, {failed_count} failed", file=sys.stderr)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
