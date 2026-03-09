#!/usr/bin/env python3
"""Evaluate reporter generation via the registry_extract_fields seed path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def configure_eval_env() -> dict[str, str]:
    if _truthy_env("PROCSUITE_ALLOW_ONLINE"):
        return {}

    forced = {
        "PROCSUITE_SKIP_DOTENV": "1",
        "PROCSUITE_PIPELINE_MODE": "extraction_first",
        "REGISTRY_EXTRACTION_ENGINE": "parallel_ner",
        "REGISTRY_SELF_CORRECT_ENABLED": "0",
        "REGISTRY_LLM_FALLBACK_ON_COVERAGE_FAIL": "0",
        "REGISTRY_AUDITOR_SOURCE": "disabled",
        "REGISTRY_USE_STUB_LLM": "1",
        "GEMINI_OFFLINE": "1",
        "OPENAI_OFFLINE": "1",
        "REPORTER_DISABLE_LLM": "1",
        "QA_REPORTER_ALLOW_SIMPLE_FALLBACK": "0",
        "PROCSUITE_FAST_MODE": "1",
        "PROCSUITE_SKIP_WARMUP": "1",
    }

    applied: dict[str, str] = {}
    for key, value in forced.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            applied[key] = value
    return applied


_APPLIED_ENV_DEFAULTS = configure_eval_env()

from app.common.reporter_seed_eval import (
    ReporterEvalCaseOutput,
    ReporterEvalRow,
    evaluate_seed_path,
    load_eval_rows,
    maybe_subsample,
    maybe_write_json,
)
from app.registry.application.registry_service import RegistryService
from app.reporting.seed_pipeline import run_reporter_seed_pipeline, seed_outcome_from_registry_result

DEFAULT_INPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_test.jsonl")
DEFAULT_OUTPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_baseline_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-field", default="prompt_text")
    parser.add_argument("--strict", action="store_true", help="Render in strict mode for fallback-rate measurement.")
    return parser.parse_args(argv)


def build_case_runner(*, registry_service: RegistryService, strict: bool):
    def _run(row: ReporterEvalRow) -> ReporterEvalCaseOutput:
        extraction_result = registry_service.extract_fields(row.prompt_text)
        seed_outcome = seed_outcome_from_registry_result(
            extraction_result,
            masked_seed_text=row.prompt_text,
        )
        pipeline_result = run_reporter_seed_pipeline(
            seed_outcome,
            note_text=row.prompt_text,
            strict=strict,
            debug_enabled=False,
        )
        return ReporterEvalCaseOutput(
            markdown=pipeline_result.markdown,
            warnings=list(seed_outcome.warnings or []),
            quality_flags=list(seed_outcome.quality_flags or []),
            needs_review=bool(seed_outcome.needs_review),
            render_fallback_used=bool(pipeline_result.render_fallback_used),
            render_fallback_reason=pipeline_result.render_fallback_reason,
            render_fallback_category=pipeline_result.render_fallback_category,
            render_fallback_details=dict(pipeline_result.render_fallback_details or {}),
        )

    return _run


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    rows = maybe_subsample(
        load_eval_rows(Path(args.input), prompt_field=str(args.prompt_field or "prompt_text")),
        int(args.max_cases),
        int(args.seed),
    )
    registry_service = RegistryService()
    payload = evaluate_seed_path(
        rows=rows,
        seed_path="registry_extract_fields",
        run_case=build_case_runner(registry_service=registry_service, strict=bool(args.strict)),
        registry_service=registry_service,
        input_path=str(args.input),
        output_path=str(args.output),
        prompt_field=str(args.prompt_field or "prompt_text"),
        environment_defaults_applied=_APPLIED_ENV_DEFAULTS,
        metadata={
            "production_default": True,
            "challenger_only": False,
            "strict_requested": bool(args.strict),
        },
    )
    maybe_write_json(Path(args.output), payload)
    print("Seed path: registry_extract_fields")
    print(f"Wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
