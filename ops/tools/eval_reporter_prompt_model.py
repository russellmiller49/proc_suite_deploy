#!/usr/bin/env python3
"""Evaluate trained prompt->bundle reporter model end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.registry.application.registry_service import RegistryService
from app.reporting.engine import compose_structured_report_with_meta
from ml.lib.reporter_json_parse import parse_and_validate_bundle
from ml.scripts.generate_reporter_gold_dataset import (
    CRITICAL_FLAG_EXACT,
    CRITICAL_FLAG_PREFIXES,
    collect_performed_flags,
)

DEFAULT_INPUT = Path("data/ml_training/reporter_prompt/v1/prompt_to_bundle_test.jsonl")
DEFAULT_MODEL_DIR = Path("artifacts/reporter_prompt_bundle_v1")
DEFAULT_OUTPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_model_eval_report.json")

PROMOTION_GATES = {
    "bundle_parse_success_rate": 0.95,
    "render_success_rate": 0.99,
    "avg_cpt_jaccard": 0.70,
    "avg_performed_flag_f1": 0.75,
    "critical_extra_flag_rate": 0.03,
    "avg_text_similarity": 0.25,
}

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
class EvalRow:
    id: str
    prompt_text: str
    completion_canonical: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args(argv)


def configure_eval_env() -> dict[str, str]:
    defaults = {
        "PROCSUITE_SKIP_DOTENV": "1",
        "PROCSUITE_SKIP_WARMUP": "1",
        "REPORTER_DISABLE_LLM": "1",
        "QA_REPORTER_ALLOW_SIMPLE_FALLBACK": "0",
        "REGISTRY_SELF_CORRECT_ENABLED": "0",
        "REGISTRY_LLM_FALLBACK_ON_COVERAGE_FAIL": "0",
        "PROCSUITE_FAST_MODE": "1",
        "LLM_PROVIDER": "openai_compat",
        "OPENAI_OFFLINE": "1",
    }
    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def format_source_prompt(prompt_text: str) -> str:
    return (
        "Generate a valid ProcedureBundle JSON object from this clinical prompt. "
        "Return JSON only.\n\n"
        f"PROMPT:\n{prompt_text.strip()}"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_rows(raw: list[dict[str, Any]]) -> list[EvalRow]:
    out: list[EvalRow] = []
    for idx, row in enumerate(raw, start=1):
        row_id = str(row.get("id") or f"row_{idx}")
        prompt = str(row.get("prompt_text") or "").strip()
        completion = str(row.get("completion_canonical") or "").strip()
        if not prompt or not completion:
            continue
        out.append(EvalRow(id=row_id, prompt_text=prompt, completion_canonical=completion))
    return out


def maybe_subsample(rows: list[EvalRow], max_cases: int, seed: int) -> list[EvalRow]:
    if max_cases <= 0 or len(rows) <= max_cases:
        return rows
    rng = random.Random(seed)
    picked = rng.sample(rows, max_cases)
    picked.sort(key=lambda item: item.id)
    return picked


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines)


def similarity_ratio(reference: str, candidate: str) -> float:
    return SequenceMatcher(None, normalize_text(reference), normalize_text(candidate)).ratio()


def missing_sections(report_text: str) -> list[str]:
    upper = (report_text or "").upper()
    return [header for header in REQUIRED_SECTION_HEADERS if header.upper() not in upper]


def _is_critical_flag(path: str) -> bool:
    return path in CRITICAL_FLAG_EXACT or path.startswith(CRITICAL_FLAG_PREFIXES)


def extract_flags_and_cpt(note_text: str, registry_service: RegistryService) -> tuple[set[str], set[str]]:
    result = registry_service.extract_fields_extraction_first(note_text)
    record_data = result.record.model_dump(exclude_none=True)
    flags = collect_performed_flags(record_data)
    cpt = {str(code) for code in (result.cpt_codes or [])}
    return flags, cpt


def cpt_jaccard(gold: set[str], pred: set[str]) -> float:
    union = gold | pred
    if not union:
        return 1.0
    return float(len(gold & pred) / len(union))


def flag_f1(gold: set[str], pred: set[str]) -> tuple[float, int, int]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    if prec + rec == 0:
        return 0.0, fp, fn
    return (2 * prec * rec / (prec + rec)), fp, fn


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_gates(summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    primary_pass = True

    for key, threshold in PROMOTION_GATES.items():
        observed = float(summary.get(key, 0.0))
        # lower-is-better metric
        if key == "critical_extra_flag_rate":
            passed = observed <= threshold
        else:
            passed = observed >= threshold

        checks[key] = {
            "observed": round(observed, 4),
            "threshold": threshold,
            "passed": passed,
        }

        if key != "avg_text_similarity" and not passed:
            primary_pass = False

    return {
        "primary_gates_passed": primary_pass,
        "all_checks": checks,
        "deployment_recommendation": "allow_optional_qa_integration" if primary_pass else "do_not_integrate",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    applied_env = configure_eval_env()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    raw_rows = load_jsonl(args.input)
    rows = maybe_subsample(to_rows(raw_rows), int(args.max_cases), int(args.seed))

    registry_service = RegistryService()

    parse_success = 0
    render_success = 0
    failures = 0
    full_shell_count = 0
    critical_extra_cases = 0

    sim_scores: list[float] = []
    cpt_scores: list[float] = []
    f1_scores: list[float] = []

    per_case: list[dict[str, Any]] = []

    for row in rows:
        parse_error = None
        render_error = None
        parse_notes: list[str] = []
        generated_text = ""
        rendered_markdown = ""

        prompt = format_source_prompt(row.prompt_text)
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=int(args.max_source_length),
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(args.max_new_tokens),
                num_beams=int(args.num_beams),
                do_sample=False,
            )

        # Keep special tokens so T5 brace placeholders (`<extra_id_0>` / `<extra_id_1>`) survive.
        # `ml.lib.reporter_json_parse` normalizes them back to `{` / `}`.
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)

        bundle = None
        try:
            bundle, _payload, parse_notes = parse_and_validate_bundle(generated_text, decode_codec=True)
            parse_success += 1
        except Exception as exc:
            parse_error = str(exc)

        if bundle is not None:
            try:
                structured = compose_structured_report_with_meta(bundle, strict=False, embed_metadata=False)
                rendered_markdown = structured.text
                render_success += 1
            except Exception as exc:
                render_error = str(exc)

        if rendered_markdown:
            sim = similarity_ratio(row.completion_canonical, rendered_markdown)
            sim_scores.append(sim)

            miss = missing_sections(rendered_markdown)
            if not miss:
                full_shell_count += 1

            gold_flags, gold_cpt = extract_flags_and_cpt(row.completion_canonical, registry_service)
            pred_flags, pred_cpt = extract_flags_and_cpt(rendered_markdown, registry_service)

            cpt_score = cpt_jaccard(gold_cpt, pred_cpt)
            f1, fp, fn = flag_f1(gold_flags, pred_flags)
            critical_extra = sorted([flag for flag in (pred_flags - gold_flags) if _is_critical_flag(flag)])
            if critical_extra:
                critical_extra_cases += 1

            cpt_scores.append(cpt_score)
            f1_scores.append(f1)

            per_case.append(
                {
                    "id": row.id,
                    "parse_success": parse_error is None,
                    "render_success": render_error is None,
                    "parse_notes": parse_notes,
                    "text_similarity": round(sim, 4),
                    "missing_sections": miss,
                    "cpt_jaccard": round(cpt_score, 4),
                    "performed_flag_f1": round(f1, 4),
                    "critical_extra_flags": critical_extra,
                    "flag_false_positive_count": fp,
                    "flag_false_negative_count": fn,
                    "parse_error": parse_error,
                    "render_error": render_error,
                }
            )
        else:
            failures += 1
            per_case.append(
                {
                    "id": row.id,
                    "parse_success": parse_error is None,
                    "render_success": render_error is None,
                    "parse_notes": parse_notes,
                    "text_similarity": 0.0,
                    "missing_sections": REQUIRED_SECTION_HEADERS,
                    "cpt_jaccard": 0.0,
                    "performed_flag_f1": 0.0,
                    "critical_extra_flags": [],
                    "flag_false_positive_count": 0,
                    "flag_false_negative_count": 0,
                    "parse_error": parse_error,
                    "render_error": render_error,
                }
            )

    total = len(rows)
    parse_rate = float(parse_success / total) if total else 0.0
    render_rate = float(render_success / total) if total else 0.0

    summary = {
        "total_cases": total,
        "failed_cases": failures,
        "bundle_parse_success_rate": round(parse_rate, 4),
        "render_success_rate": round(render_rate, 4),
        "avg_text_similarity": round(_avg(sim_scores), 4),
        "required_section_coverage": round((full_shell_count / render_success) if render_success else 0.0, 4),
        "avg_cpt_jaccard": round(_avg(cpt_scores), 4),
        "avg_performed_flag_f1": round(_avg(f1_scores), 4),
        "critical_extra_flag_rate": round((critical_extra_cases / render_success) if render_success else 0.0, 4),
    }

    gate_report = evaluate_gates(summary)

    payload = {
        "created_at": datetime_now_iso(),
        "input_path": str(args.input),
        "model_dir": str(args.model_dir),
        "environment_defaults_applied": applied_env,
        "summary": summary,
        "promotion_gate_report": gate_report,
        "per_case": per_case,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Primary gates passed: {gate_report['primary_gates_passed']}")
    print(f"Wrote report: {args.output}")
    return 0


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
