#!/usr/bin/env python3
"""Sample prompts from reporter-training JSONL files and run reporter pipeline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.services.qa_pipeline import ReportingStrategy, SimpleReporterStrategy
from app.registry.application.registry_service import RegistryService
from app.reporting.engine import ReporterEngine, _load_procedure_order, default_schema_registry, default_template_registry
from app.reporting.inference import InferenceEngine
from app.reporting.validation import ValidationEngine


DEFAULT_INPUT_DIR = Path("/home/rjm/projects/proc_suite_notes/reporter_training/reporter_training")
DEFAULT_OUTPUT = Path("reporter_tests.txt")


@dataclass(frozen=True)
class SeedPrompt:
    source_file: Path
    line_number: int
    record_id: str
    prompt: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--count", type=int, default=20, help="How many prompts to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output text file path.")
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Primary field name for prompt text (default: prompt).",
    )
    parser.add_argument(
        "--include-metadata-json",
        action="store_true",
        help="Also emit machine-readable JSON results.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional metadata JSON output path (default: --output with .json suffix).",
    )
    return parser.parse_args(argv)


def _extract_prompt(row: dict[str, Any], prompt_field: str) -> str:
    for field in (prompt_field, "prompt", "prompt_text", "text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def sample_prompts(input_dir: Path, *, count: int, seed: int, prompt_field: str) -> tuple[list[SeedPrompt], int]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in: {input_dir}")

    rng = random.Random(seed)
    reservoir: list[SeedPrompt] = []
    seen = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                prompt = _extract_prompt(row, prompt_field)
                if not prompt:
                    continue

                record_id = str(row.get("id") or f"{file_path.stem}:{line_num}")
                item = SeedPrompt(
                    source_file=file_path,
                    line_number=line_num,
                    record_id=record_id,
                    prompt=prompt,
                )

                seen += 1
                if len(reservoir) < count:
                    reservoir.append(item)
                else:
                    replacement_idx = rng.randrange(seen)
                    if replacement_idx < count:
                        reservoir[replacement_idx] = item

    if not reservoir:
        raise RuntimeError("No usable prompt rows found.")

    rng.shuffle(reservoir)
    return reservoir, seen


def build_reporter_strategy() -> tuple[ReportingStrategy, RegistryService]:
    templates = default_template_registry()
    schemas = default_schema_registry()
    reporter_engine = ReporterEngine(
        templates,
        schemas,
        procedure_order=_load_procedure_order(),
    )
    registry_engine = RegistryService()
    strategy = ReportingStrategy(
        reporter_engine=reporter_engine,
        inference_engine=InferenceEngine(),
        validation_engine=ValidationEngine(templates, schemas),
        registry_engine=registry_engine,
        simple_strategy=SimpleReporterStrategy(),
    )
    return strategy, registry_engine


def run_reporter_markdown(
    prompt_text: str,
    *,
    strategy: ReportingStrategy,
    registry_engine: RegistryService,
) -> tuple[str, list[str]]:
    extraction = registry_engine.extract_fields_extraction_first(prompt_text)
    record_data = extraction.record.model_dump(exclude_none=True)
    rendered = strategy.render(text=prompt_text, registry_data={"record": record_data})
    markdown = str(rendered.get("markdown") or "").strip()
    warnings = [str(w) for w in (rendered.get("warnings") or [])]
    return markdown, warnings


def write_results(
    output_path: Path,
    rows: list[SeedPrompt],
    *,
    strategy: ReportingStrategy,
    registry_engine: RegistryService,
) -> tuple[int, int, list[dict[str, Any]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    failure_count = 0
    case_results: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Reporter random seed run\n")
        handle.write(f"Cases: {len(rows)}\n\n")

        for idx, row in enumerate(rows, start=1):
            handle.write("=" * 100 + "\n")
            handle.write(f"CASE {idx}\n")
            handle.write(f"source: {row.source_file}:{row.line_number}\n")
            handle.write(f"id: {row.record_id}\n\n")
            handle.write("PROMPT\n")
            handle.write("-" * 100 + "\n")
            handle.write(row.prompt.rstrip() + "\n\n")
            handle.write("OUTPUT\n")
            handle.write("-" * 100 + "\n")

            case_payload: dict[str, Any] = {
                "case_index": idx,
                "source_file": str(row.source_file),
                "line_number": row.line_number,
                "id": row.record_id,
                "prompt": row.prompt,
                "markdown": None,
                "warnings": [],
                "error": None,
            }

            try:
                markdown, warnings = run_reporter_markdown(
                    row.prompt,
                    strategy=strategy,
                    registry_engine=registry_engine,
                )
                handle.write((markdown or "<empty markdown>").rstrip() + "\n")
                if warnings:
                    handle.write("\nWARNINGS\n")
                    handle.write("-" * 100 + "\n")
                    for warning in warnings:
                        handle.write(f"- {warning}\n")
                case_payload["markdown"] = markdown
                case_payload["warnings"] = warnings
                success_count += 1
            except Exception as exc:  # noqa: BLE001
                handle.write(f"<ERROR: {type(exc).__name__}: {exc}>\n")
                case_payload["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                failure_count += 1
            handle.write("\n")
            case_results.append(case_payload)

    return success_count, failure_count, case_results


def write_metadata_json(
    metadata_path: Path,
    *,
    input_dir: Path,
    output_txt_path: Path,
    prompt_field: str,
    sample_count: int,
    seed: int,
    total_candidates: int,
    success_count: int,
    failure_count: int,
    cases: list[dict[str, Any]],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_txt_path": str(output_txt_path),
        "prompt_field": prompt_field,
        "sample_count": sample_count,
        "seed": seed,
        "total_candidates": total_candidates,
        "success_count": success_count,
        "failure_count": failure_count,
        "cases": cases,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.count <= 0:
        raise ValueError("--count must be > 0")

    sampled_rows, total_candidates = sample_prompts(
        args.input_dir,
        count=int(args.count),
        seed=int(args.seed),
        prompt_field=str(args.prompt_field),
    )

    strategy, registry_engine = build_reporter_strategy()
    success_count, failure_count, case_results = write_results(
        args.output,
        sampled_rows,
        strategy=strategy,
        registry_engine=registry_engine,
    )

    metadata_written = None
    if bool(args.include_metadata_json):
        metadata_path = args.metadata_output or args.output.with_suffix(".json")
        write_metadata_json(
            metadata_path,
            input_dir=args.input_dir,
            output_txt_path=args.output,
            prompt_field=str(args.prompt_field),
            sample_count=len(sampled_rows),
            seed=int(args.seed),
            total_candidates=total_candidates,
            success_count=success_count,
            failure_count=failure_count,
            cases=case_results,
        )
        metadata_written = metadata_path

    print(f"Sampled {len(sampled_rows)} prompts from {total_candidates} candidates.")
    print(f"Wrote: {args.output}")
    if metadata_written is not None:
        print(f"Wrote metadata: {metadata_written}")
    print(f"Succeeded: {success_count} | Failed: {failure_count}")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
