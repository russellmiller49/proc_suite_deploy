#!/usr/bin/env python3
"""Evaluate reporter quality against a reporter-gold JSONL dataset."""

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

DEFAULT_INPUT = Path("data/ml_training/reporter_golden/v1/reporter_gold_accepted.jsonl")
DEFAULT_OUTPUT = Path("data/ml_training/reporter_golden/v1/reporter_gold_eval_report.json")


@dataclass(frozen=True)
class EvaluationRow:
    id: str
    input_text: str
    ideal_output: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Gold dataset JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Evaluation report JSON path.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional max case count (0 means all).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic subsampling.")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
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


def to_eval_rows(rows: list[dict[str, Any]]) -> list[EvaluationRow]:
    out: list[EvaluationRow] = []
    for idx, row in enumerate(rows, start=1):
        input_text = str(row.get("input_text") or "").strip()
        ideal_output = str(row.get("ideal_output") or row.get("ideal_output_candidate") or "").strip()
        row_id = str(row.get("id") or f"row_{idx}")
        if not input_text or not ideal_output:
            continue
        out.append(EvaluationRow(id=row_id, input_text=input_text, ideal_output=ideal_output))
    return out


def maybe_subsample(rows: list[EvaluationRow], max_cases: int, seed: int) -> list[EvaluationRow]:
    if max_cases <= 0 or len(rows) <= max_cases:
        return rows
    rng = random.Random(seed)
    picked = rng.sample(rows, max_cases)
    picked.sort(key=lambda r: r.id)
    return picked


def evaluate_rows(
    rows: list[EvaluationRow],
    *,
    render_report: Callable[[str], str],
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    similarities: list[float] = []
    generated_full_shell_count = 0
    failures = 0

    for row in rows:
        try:
            generated = render_report(row.input_text)
            missing = missing_sections(generated)
            score = similarity_ratio(row.ideal_output, generated)
            similarities.append(score)
            if not missing:
                generated_full_shell_count += 1

            per_case.append(
                {
                    "id": row.id,
                    "similarity": round(score, 4),
                    "missing_sections_generated": missing,
                    "missing_sections_ideal": missing_sections(row.ideal_output),
                    "generated_length": len(generated),
                    "ideal_length": len(row.ideal_output),
                    "error": None,
                }
            )
        except Exception as exc:
            failures += 1
            per_case.append(
                {
                    "id": row.id,
                    "similarity": 0.0,
                    "missing_sections_generated": REQUIRED_SECTION_HEADERS,
                    "missing_sections_ideal": [],
                    "generated_length": 0,
                    "ideal_length": len(row.ideal_output),
                    "error": str(exc),
                }
            )

    avg_similarity = float(sum(similarities) / len(similarities)) if similarities else 0.0
    min_similarity = float(min(similarities)) if similarities else 0.0
    full_shell_rate = float(generated_full_shell_count / len(rows)) if rows else 0.0

    summary = {
        "total_cases": len(rows),
        "successful_cases": len(rows) - failures,
        "failed_cases": failures,
        "avg_similarity": round(avg_similarity, 4),
        "min_similarity": round(min_similarity, 4),
        "generated_full_shell_rate": round(full_shell_rate, 4),
    }
    return {
        "summary": summary,
        "per_case": per_case,
    }


def _build_renderer() -> Callable[[str], str]:
    # Keep evaluation deterministic and offline-friendly by default.
    os.environ.setdefault("PROCSUITE_SKIP_DOTENV", "1")
    os.environ.setdefault("PROCSUITE_SKIP_WARMUP", "1")
    os.environ.setdefault("REPORTER_DISABLE_LLM", "1")
    os.environ.setdefault("QA_REPORTER_ALLOW_SIMPLE_FALLBACK", "0")
    os.environ.setdefault("REGISTRY_SELF_CORRECT_ENABLED", "0")
    os.environ.setdefault("REGISTRY_EXTRACTION_ENGINE", "parallel_ner")

    from app.api.services.qa_pipeline import ReportingStrategy, SimpleReporterStrategy
    from app.registry.application.registry_service import RegistryService
    from app.reporting.engine import (
        ReporterEngine,
        _load_procedure_order,
        default_schema_registry,
        default_template_registry,
    )
    from app.reporting.inference import InferenceEngine
    from app.reporting.validation import ValidationEngine

    templates = default_template_registry()
    schemas = default_schema_registry()
    reporter_engine = ReporterEngine(
        templates,
        schemas,
        procedure_order=_load_procedure_order(),
    )
    inference_engine = InferenceEngine()
    validation_engine = ValidationEngine(templates, schemas)
    registry_service = RegistryService()

    class _RegistryAdapter:
        def __init__(self, svc: RegistryService):
            self._svc = svc

        def run(self, text: str, explain: bool = False):  # noqa: ARG002
            result = self._svc.extract_fields_extraction_first(text)
            return result.record

    strategy = ReportingStrategy(
        reporter_engine=reporter_engine,
        inference_engine=inference_engine,
        validation_engine=validation_engine,
        registry_engine=_RegistryAdapter(registry_service),
        simple_strategy=SimpleReporterStrategy(),
    )

    def _render(note_text: str) -> str:
        extraction = registry_service.extract_fields_extraction_first(note_text)
        record_dict = extraction.record.model_dump(exclude_none=True)
        payload = strategy.render(text=note_text, registry_data={"record": record_dict})
        return str(payload.get("markdown") or "")

    return _render


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    raw_rows = load_jsonl(args.input)
    eval_rows = to_eval_rows(raw_rows)
    eval_rows = maybe_subsample(eval_rows, int(args.max_cases), int(args.seed))

    renderer = _build_renderer()
    report = evaluate_rows(eval_rows, render_report=renderer)
    report["input_path"] = str(args.input)
    report["output_path"] = str(args.output)
    report["created_at"] = datetime_now_iso()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(
        "Reporter gold eval: "
        f"cases={summary['total_cases']} "
        f"avg_similarity={summary['avg_similarity']} "
        f"min_similarity={summary['min_similarity']} "
        f"full_shell_rate={summary['generated_full_shell_rate']}"
    )
    print(f"Wrote report: {args.output}")
    return 0


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

