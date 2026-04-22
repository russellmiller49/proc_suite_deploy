#!/usr/bin/env python3
"""Evaluate reporter generation via the llm_findings seed path."""

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
    allow_online = _truthy_env("PROCSUITE_ALLOW_ONLINE")

    forced = {
        "PROCSUITE_SKIP_DOTENV": "1",
        "PROCSUITE_PIPELINE_MODE": "extraction_first",
        "REGISTRY_EXTRACTION_ENGINE": "parallel_ner",
        "REGISTRY_SELF_CORRECT_ENABLED": "0",
        "REGISTRY_LLM_FALLBACK_ON_COVERAGE_FAIL": "0",
        "REGISTRY_AUDITOR_SOURCE": "disabled",
        "REGISTRY_USE_STUB_LLM": "1",
        "GEMINI_OFFLINE": "1",
        "QA_REPORTER_ALLOW_SIMPLE_FALLBACK": "0",
        "PROCSUITE_FAST_MODE": "1",
        "PROCSUITE_SKIP_WARMUP": "1",
    }

    if allow_online:
        forced["OPENAI_OFFLINE"] = "0"
        forced["REPORTER_DISABLE_LLM"] = "0"
    else:
        forced["OPENAI_OFFLINE"] = "1"
        forced["REPORTER_DISABLE_LLM"] = "1"

    applied: dict[str, str] = {}
    for key, value in forced.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            applied[key] = value
    return applied


_APPLIED_ENV_DEFAULTS = configure_eval_env()

from app.common.exceptions import LLMError
from app.common.reporter_seed_eval import (
    ReporterEvalCaseOutput,
    ReporterEvalRow,
    drop_reason_counts,
    evaluate_seed_path,
    load_eval_rows,
    load_seed_fixture,
    maybe_subsample,
    maybe_write_json,
)
from app.registry.application.registry_service import RegistryService
from app.registry.schema import RegistryRecord
from app.reporting.llm_findings import (
    ClinicalContextV1,
    LLMFindingsSeedResult,
    build_record_payload_for_reporting,
    seed_registry_record_from_llm_findings,
)
from app.reporting.seed_pipeline import run_reporter_seed_pipeline, seed_outcome_from_llm_findings_seed

DEFAULT_INPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_test.jsonl")
DEFAULT_OUTPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_llm_findings_eval_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-field", default="prompt_text")
    parser.add_argument("--strict", action="store_true", help="Render in strict mode for fallback-rate measurement.")
    parser.add_argument(
        "--seed-fixture",
        type=Path,
        default=None,
        help="Optional offline fixture for canned llm_findings seeds keyed by row id.",
    )
    return parser.parse_args(argv)


def categorize_llm_error(exc: Exception) -> str | None:
    if not isinstance(exc, LLMError):
        return None
    msg = str(exc).strip().lower()
    if not msg:
        return "other"
    if any(token in msg for token in ("unauthorized", "invalid api key", "api_key", "authentication", "status=401")):
        return "auth"
    if any(token in msg for token in ("model", "not found", "does not exist", "you do not have access")):
        return "model_access"
    if any(token in msg for token in ("rate limit", "status=429", "too many requests")):
        return "rate_limit"
    if "timeout" in msg:
        return "timeout"
    if any(token in msg for token in ("network error", "connection", "dns", "ssl")):
        return "network"
    if any(token in msg for token in ("status=400", "bad request", "unsupported", "invalid", "parameter")):
        return "bad_request"
    return "other"


def _seed_from_fixture_row(row: ReporterEvalRow, fixture_map: dict[str, dict[str, object]]) -> LLMFindingsSeedResult:
    payload = fixture_map.get(row.id)
    if payload is None:
        raise KeyError(f"No llm seed fixture case found for {row.id}")

    context_payload = payload.get("context")
    context = ClinicalContextV1.model_validate(context_payload) if isinstance(context_payload, dict) else None

    return LLMFindingsSeedResult(
        record=RegistryRecord.model_validate(payload.get("record") or {}),
        masked_prompt_text=str(payload.get("masked_prompt_text") or row.prompt_text),
        cpt_codes=[str(code) for code in list(payload.get("cpt_codes") or [])],
        warnings=[str(item) for item in list(payload.get("warnings") or []) if str(item)],
        needs_review=bool(payload.get("needs_review")),
        context=context,
        accepted_items=[],
        accepted_findings=int(payload.get("accepted_findings") or 0),
        dropped_findings=int(payload.get("dropped_findings") or 0),
    )


def build_case_runner(
    *,
    strict: bool,
    fixture_map: dict[str, dict[str, object]] | None,
):
    def _run(row: ReporterEvalRow) -> ReporterEvalCaseOutput:
        try:
            if fixture_map is not None:
                seed = _seed_from_fixture_row(row, fixture_map)
            else:
                seed = seed_registry_record_from_llm_findings(row.prompt_text)
        except Exception as exc:  # noqa: BLE001
            error_code = categorize_llm_error(exc)
            raise RuntimeError(error_code or str(type(exc).__name__)) from exc

        seed_outcome = seed_outcome_from_llm_findings_seed(
            seed,
            reporting_payload=build_record_payload_for_reporting(seed),
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
            quality_flags=list(pipeline_result.quality_flags or []),
            needs_review=bool(pipeline_result.needs_manual_review),
            render_fallback_used=bool(pipeline_result.render_fallback_used),
            render_fallback_reason=pipeline_result.render_fallback_reason,
            render_fallback_category=pipeline_result.render_fallback_category,
            render_fallback_details=dict(pipeline_result.render_fallback_details or {}),
            accepted_findings=int(getattr(seed, "accepted_findings", 0) or 0),
            dropped_findings=int(getattr(seed, "dropped_findings", 0) or 0),
            drop_reason_counts=drop_reason_counts(list(seed_outcome.warnings or [])),
        )

    return _run


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    if args.seed_fixture is None and not _truthy_env("PROCSUITE_ALLOW_ONLINE"):
        print("This evaluation requires real GPT calls unless --seed-fixture is provided.")
        print("Set PROCSUITE_ALLOW_ONLINE=1 or pass --seed-fixture for offline challenger evaluation.")
        return 2

    rows = maybe_subsample(
        load_eval_rows(Path(args.input), prompt_field=str(args.prompt_field or "prompt_text")),
        int(args.max_cases),
        int(args.seed),
    )
    fixture_map = load_seed_fixture(Path(args.seed_fixture)) if args.seed_fixture else None
    registry_service = RegistryService()
    payload = evaluate_seed_path(
        rows=rows,
        seed_path="llm_findings",
        run_case=build_case_runner(strict=bool(args.strict), fixture_map=fixture_map),
        registry_service=registry_service,
        input_path=str(args.input),
        output_path=str(args.output),
        prompt_field=str(args.prompt_field or "prompt_text"),
        environment_defaults_applied=_APPLIED_ENV_DEFAULTS,
        metadata={
            "production_default": False,
            "challenger_only": True,
            "strict_requested": bool(args.strict),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "openai_primary_api": os.getenv("OPENAI_PRIMARY_API"),
            "openai_model_structurer": os.getenv("OPENAI_MODEL_STRUCTURER"),
            "seed_fixture_used": bool(args.seed_fixture),
            "seed_fixture_path": str(args.seed_fixture) if args.seed_fixture else None,
        },
    )
    maybe_write_json(Path(args.output), payload)
    print("Seed path: llm_findings")
    print(f"Wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
