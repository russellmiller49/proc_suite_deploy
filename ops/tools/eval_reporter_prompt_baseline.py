#!/usr/bin/env python3
"""Evaluate baseline reporter performance on reporter_prompt split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def configure_eval_env() -> dict[str, str]:
    """Force deterministic/offline settings unless explicitly allowed online."""
    if _truthy_env("PROCSUITE_ALLOW_ONLINE"):
        return {}

    forced = {
        # Avoid loading repo-local dotenv (which may contain API keys).
        "PROCSUITE_SKIP_DOTENV": "1",
        # Required runtime invariant (keep scripts aligned with service behavior).
        "PROCSUITE_PIPELINE_MODE": "extraction_first",
        "REGISTRY_EXTRACTION_ENGINE": "parallel_ner",
        # No LLM self-correct/fallback during eval scoring.
        "REGISTRY_SELF_CORRECT_ENABLED": "0",
        "REGISTRY_LLM_FALLBACK_ON_COVERAGE_FAIL": "0",
        # Disable RAW-ML auditing (may call LLM depending on config).
        "REGISTRY_AUDITOR_SOURCE": "disabled",
        # Force stub/offline LLMs even if keys are present in the environment.
        "REGISTRY_USE_STUB_LLM": "1",
        "GEMINI_OFFLINE": "1",
        "OPENAI_OFFLINE": "1",
        # Reporter-specific offline guardrails.
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


# Apply env before importing app modules so any LLM clients are stubbed.
_APPLIED_ENV_DEFAULTS = configure_eval_env()

from app.api.services.qa_pipeline import ReportingStrategy, SimpleReporterStrategy
from app.registry.application.registry_service import RegistryService
from app.reporting.engine import (
    ReporterEngine,
    _load_procedure_order,
    compose_report_from_text,
    default_schema_registry,
    default_template_registry,
)
from app.reporting.inference import InferenceEngine
from app.reporting.validation import ValidationEngine
from ml.scripts.generate_reporter_gold_dataset import (
    CRITICAL_FLAG_EXACT,
    CRITICAL_FLAG_PREFIXES,
    collect_performed_flags,
)

DEFAULT_INPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_test.jsonl")
DEFAULT_OUTPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_baseline_report.json")

REQUIRED_SECTION_HEADERS = [
    "INTERVENTIONAL PULMONOLOGY OPERATIVE REPORT",
    "INDICATION FOR OPERATION",
    "CONSENT",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
    "PROCEDURE",
    "ANESTHESIA",
    "MONITORING",
    "COMPLICATIONS",
    "PROCEDURE IN DETAIL",
    "IMPRESSION / PLAN",
]


@dataclass(frozen=True)
class BaselineRow:
    id: str
    prompt_text: str
    completion_canonical: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines)


def similarity_ratio(reference: str, candidate: str) -> float:
    return SequenceMatcher(None, normalize_text(reference), normalize_text(candidate)).ratio()


def missing_sections(report_text: str) -> list[str]:
    upper = (report_text or "").upper()
    return [header for header in REQUIRED_SECTION_HEADERS if header.upper() not in upper]


def to_eval_rows(rows: list[dict[str, Any]]) -> list[BaselineRow]:
    out: list[BaselineRow] = []
    for idx, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt_text") or "").strip()
        completion = str(row.get("completion_canonical") or "").strip()
        row_id = str(row.get("id") or f"row_{idx}")
        if not prompt or not completion:
            continue
        out.append(BaselineRow(id=row_id, prompt_text=prompt, completion_canonical=completion))
    return out


def maybe_subsample(rows: list[BaselineRow], max_cases: int, seed: int) -> list[BaselineRow]:
    if max_cases <= 0 or len(rows) <= max_cases:
        return rows
    rng = random.Random(seed)
    picked = rng.sample(rows, max_cases)
    picked.sort(key=lambda r: r.id)
    return picked


def _is_critical_flag(path: str) -> bool:
    return path in CRITICAL_FLAG_EXACT or path.startswith(CRITICAL_FLAG_PREFIXES)


def extract_flags_and_cpt(note_text: str, registry_service: RegistryService) -> tuple[set[str], set[str]]:
    result = registry_service.extract_fields_extraction_first(note_text)
    record_data = result.record.model_dump(exclude_none=True)
    flags = collect_performed_flags(record_data)
    cpt = {str(code) for code in (result.cpt_codes or [])}
    return flags, cpt


def _cpt_jaccard(gold: set[str], pred: set[str]) -> float:
    union = gold | pred
    if not union:
        return 1.0
    return float(len(gold & pred) / len(union))


def _flag_f1(gold: set[str], pred: set[str]) -> tuple[float, int, int]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    if prec + rec == 0:
        return 0.0, fp, fn
    return (2 * prec * rec / (prec + rec)), fp, fn


def build_structured_renderer() -> Callable[[str], str]:
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

    def _render(prompt_text: str) -> str:
        # Avoid the "lightweight registry extraction" pathway (can fail and/or call LLMs
        # depending on environment). Use extraction-first record as the structured seed.
        extraction = registry_engine.extract_fields_extraction_first(prompt_text)
        record_data = extraction.record.model_dump(exclude_none=True)
        payload = strategy.render(text=prompt_text, registry_data={"record": record_data})
        return str(payload.get("markdown") or "")

    return _render


def build_dictation_renderer() -> Callable[[str], str]:
    def _render(prompt_text: str) -> str:
        _report, markdown = compose_report_from_text(prompt_text)
        return markdown

    return _render


def evaluate_renderer(
    rows: list[BaselineRow],
    *,
    renderer: Callable[[str], str],
    registry_service: RegistryService,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    sim_scores: list[float] = []
    cpt_scores: list[float] = []
    f1_scores: list[float] = []
    full_shell_count = 0
    failures = 0
    critical_extra_cases = 0

    for row in rows:
        try:
            generated = renderer(row.prompt_text)
            sim = similarity_ratio(row.completion_canonical, generated)
            sim_scores.append(sim)

            miss = missing_sections(generated)
            if not miss:
                full_shell_count += 1

            gold_flags, gold_cpt = extract_flags_and_cpt(row.completion_canonical, registry_service)
            pred_flags, pred_cpt = extract_flags_and_cpt(generated, registry_service)

            cpt_j = _cpt_jaccard(gold_cpt, pred_cpt)
            flag_f1, fp, fn = _flag_f1(gold_flags, pred_flags)
            critical_extra = sorted([flag for flag in (pred_flags - gold_flags) if _is_critical_flag(flag)])

            if critical_extra:
                critical_extra_cases += 1

            cpt_scores.append(cpt_j)
            f1_scores.append(flag_f1)

            per_case.append(
                {
                    "id": row.id,
                    "text_similarity": round(sim, 4),
                    "missing_sections": miss,
                    "cpt_jaccard": round(cpt_j, 4),
                    "performed_flag_f1": round(flag_f1, 4),
                    "critical_extra_flags": critical_extra,
                    "flag_false_positive_count": fp,
                    "flag_false_negative_count": fn,
                    "error": None,
                }
            )
        except Exception as exc:
            failures += 1
            per_case.append(
                {
                    "id": row.id,
                    "text_similarity": 0.0,
                    "missing_sections": REQUIRED_SECTION_HEADERS,
                    "cpt_jaccard": 0.0,
                    "performed_flag_f1": 0.0,
                    "critical_extra_flags": [],
                    "flag_false_positive_count": 0,
                    "flag_false_negative_count": 0,
                    "error": str(exc),
                }
            )

    total = len(rows)
    successful = total - failures

    def _avg(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    summary = {
        "total_cases": total,
        "successful_cases": successful,
        "failed_cases": failures,
        "avg_text_similarity": round(_avg(sim_scores), 4),
        "required_section_coverage": round((full_shell_count / successful) if successful else 0.0, 4),
        "avg_cpt_jaccard": round(_avg(cpt_scores), 4),
        "avg_performed_flag_f1": round(_avg(f1_scores), 4),
        "critical_extra_flag_rate": round((critical_extra_cases / successful) if successful else 0.0, 4),
    }

    return {"summary": summary, "per_case": per_case}


def _clinical_score(summary: dict[str, Any]) -> float:
    return float(summary.get("avg_cpt_jaccard", 0.0)) + float(summary.get("avg_performed_flag_f1", 0.0))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    raw_rows = load_jsonl(args.input)
    rows = maybe_subsample(to_eval_rows(raw_rows), int(args.max_cases), int(args.seed))

    registry_service = RegistryService()

    dictation_report = evaluate_renderer(
        rows,
        renderer=build_dictation_renderer(),
        registry_service=registry_service,
    )
    structured_report = evaluate_renderer(
        rows,
        renderer=build_structured_renderer(),
        registry_service=registry_service,
    )

    dict_score = _clinical_score(dictation_report["summary"])
    structured_score = _clinical_score(structured_report["summary"])
    comparator = "structured" if structured_score >= dict_score else "dictation"

    payload = {
        "created_at": datetime_now_iso(),
        "input_path": str(args.input),
        "row_count": len(rows),
        "environment_defaults_applied": _APPLIED_ENV_DEFAULTS,
        "baselines": {
            "compose_report_from_text": dictation_report,
            "structured_reporting_strategy": structured_report,
        },
        "comparator_baseline": comparator,
        "comparator_reason": "higher avg_cpt_jaccard + avg_performed_flag_f1",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Baseline comparator: {comparator}")
    print(f"Wrote report: {args.output}")
    return 0


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
