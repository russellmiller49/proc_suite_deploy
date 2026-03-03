#!/usr/bin/env python3
"""Compare reporter_prompt llm_findings eval results across two OpenAI models.

This runs the existing `eval_reporter_prompt_llm_findings.py` twice (Model A vs Model B)
on the same dataset split and produces a PHI-safe diff report (metrics only).

Offline-by-default: the underlying eval script requires real GPT calls; set:
  PROCSUITE_ALLOW_ONLINE=1
  LLM_PROVIDER=openai_compat
  OPENAI_API_KEY=...
  OPENAI_PRIMARY_API=responses

Example:
  .venv/bin/python ops/tools/compare_reporter_prompt_llm_findings_models.py \\
    --max-cases 25 \\
    --model-a gpt-5-mini \\
    --model-b gpt-5.2 \\
    --output-dir data/ml_training/reporter_prompt/v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = Path("data/ml_training/reporter_prompt/v1/reporter_prompt_test.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/ml_training/reporter_prompt/v1")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_model_tag(model: str) -> str:
    raw = (model or "").strip() or "model"
    return "".join(ch if ch.isalnum() else "_" for ch in raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt-field", default="prompt_text")
    parser.add_argument("--max-cases", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-a", default="gpt-5-mini")
    parser.add_argument("--model-b", default="gpt-5.2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="")
    parser.add_argument("--diff-only", action="store_true", help="Skip running eval; only diff existing --report-a/--report-b.")
    parser.add_argument("--report-a", type=Path, default=None)
    parser.add_argument("--report-b", type=Path, default=None)
    parser.add_argument(
        "--diff-path",
        type=Path,
        default=None,
        help="Optional explicit path for the diff JSON (default: derived from --output-dir + timestamp).",
    )
    parser.add_argument(
        "--min-delta-avg-cpt-jaccard",
        type=float,
        default=None,
        help="Optional test assertion: require (model_b - model_a) avg_cpt_jaccard delta >= this value.",
    )
    return parser.parse_args(argv)


def _run_eval(*, model: str, input_path: Path, output_path: Path, prompt_field: str, max_cases: int, seed: int) -> None:
    eval_script = ROOT / "ops" / "tools" / "eval_reporter_prompt_llm_findings.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    env = dict(os.environ)
    env["OPENAI_MODEL_STRUCTURER"] = str(model)

    cmd = [
        sys.executable,
        str(eval_script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--prompt-field",
        str(prompt_field),
        "--max-cases",
        str(int(max_cases)),
        "--seed",
        str(int(seed)),
    ]
    subprocess.run(cmd, env=env, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _diff_summaries(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        "required_section_coverage",
        "avg_cpt_jaccard",
        "avg_performed_flag_f1",
        "critical_extra_flag_rate",
        "avg_accepted_findings",
        "avg_dropped_findings",
    )
    out: dict[str, Any] = {}
    for metric in metrics:
        a = _as_float(summary_a.get(metric))
        b = _as_float(summary_b.get(metric))
        if a is None or b is None:
            out[metric] = {"a": a, "b": b, "delta": None}
        else:
            out[metric] = {"a": a, "b": b, "delta": round(b - a, 6)}
    return out


def _diff_per_case(per_case_a: list[dict[str, Any]], per_case_b: list[dict[str, Any]]) -> dict[str, Any]:
    by_id_a = {str(item.get("id") or ""): item for item in per_case_a if isinstance(item, dict) and item.get("id")}
    by_id_b = {str(item.get("id") or ""): item for item in per_case_b if isinstance(item, dict) and item.get("id")}
    all_ids = sorted(set(by_id_a) | set(by_id_b))

    def _num(item: dict[str, Any] | None, key: str) -> float | None:
        if not item:
            return None
        return _as_float(item.get(key))

    rows: list[dict[str, Any]] = []
    improved_cpt = 0
    worsened_cpt = 0
    ties_cpt = 0
    for case_id in all_ids:
        a = by_id_a.get(case_id)
        b = by_id_b.get(case_id)
        a_cpt = _num(a, "cpt_jaccard")
        b_cpt = _num(b, "cpt_jaccard")
        delta_cpt = None if a_cpt is None or b_cpt is None else round(b_cpt - a_cpt, 6)
        if delta_cpt is None:
            pass
        elif delta_cpt > 0:
            improved_cpt += 1
        elif delta_cpt < 0:
            worsened_cpt += 1
        else:
            ties_cpt += 1

        rows.append(
            {
                "id": case_id,
                "a": {
                    "cpt_jaccard": a_cpt,
                    "performed_flag_f1": _num(a, "performed_flag_f1"),
                    "accepted_findings": int(a.get("accepted_findings") or 0) if isinstance(a, dict) else None,
                    "dropped_findings": int(a.get("dropped_findings") or 0) if isinstance(a, dict) else None,
                    "critical_extra_flags": list(a.get("critical_extra_flags") or []) if isinstance(a, dict) else None,
                    "error": a.get("error") if isinstance(a, dict) else None,
                },
                "b": {
                    "cpt_jaccard": b_cpt,
                    "performed_flag_f1": _num(b, "performed_flag_f1"),
                    "accepted_findings": int(b.get("accepted_findings") or 0) if isinstance(b, dict) else None,
                    "dropped_findings": int(b.get("dropped_findings") or 0) if isinstance(b, dict) else None,
                    "critical_extra_flags": list(b.get("critical_extra_flags") or []) if isinstance(b, dict) else None,
                    "error": b.get("error") if isinstance(b, dict) else None,
                },
                "delta": {
                    "cpt_jaccard": delta_cpt,
                    "performed_flag_f1": (
                        None
                        if _num(a, "performed_flag_f1") is None or _num(b, "performed_flag_f1") is None
                        else round(_num(b, "performed_flag_f1") - _num(a, "performed_flag_f1"), 6)
                    ),
                    "accepted_findings": (
                        None
                        if not isinstance(a, dict) or not isinstance(b, dict)
                        else int(b.get("accepted_findings") or 0) - int(a.get("accepted_findings") or 0)
                    ),
                },
            }
        )

    return {
        "counts": {
            "total_cases": len(all_ids),
            "cpt_improved_cases": improved_cpt,
            "cpt_worsened_cases": worsened_cpt,
            "cpt_tied_cases": ties_cpt,
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.diff_only and not _truthy_env("PROCSUITE_ALLOW_ONLINE"):
        print("This comparison requires real GPT calls.")
        print("Set PROCSUITE_ALLOW_ONLINE=1 and configure OpenAI env vars to run.")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = str(args.tag or "").strip()
    if tag:
        tag = f"_{tag}"
    now = _now_tag()

    model_a_tag = _safe_model_tag(str(args.model_a))
    model_b_tag = _safe_model_tag(str(args.model_b))

    if args.diff_only and (args.report_a is None or args.report_b is None):
        print("--diff-only requires both --report-a and --report-b.")
        return 2

    report_a = Path(args.report_a) if args.report_a else output_dir / f"reporter_prompt_llm_findings_eval_{model_a_tag}_{now}{tag}.json"
    report_b = Path(args.report_b) if args.report_b else output_dir / f"reporter_prompt_llm_findings_eval_{model_b_tag}_{now}{tag}.json"
    diff_path = (
        Path(args.diff_path)
        if args.diff_path
        else output_dir / f"reporter_prompt_llm_findings_model_compare_{model_a_tag}_vs_{model_b_tag}_{now}{tag}.json"
    )

    if not args.diff_only:
        _run_eval(
            model=str(args.model_a),
            input_path=Path(args.input),
            output_path=report_a,
            prompt_field=str(args.prompt_field),
            max_cases=int(args.max_cases),
            seed=int(args.seed),
        )
        _run_eval(
            model=str(args.model_b),
            input_path=Path(args.input),
            output_path=report_b,
            prompt_field=str(args.prompt_field),
            max_cases=int(args.max_cases),
            seed=int(args.seed),
        )

    if not report_a.exists():
        print(f"Report A not found: {report_a}")
        return 2
    if not report_b.exists():
        print(f"Report B not found: {report_b}")
        return 2

    data_a = _load_json(report_a)
    data_b = _load_json(report_b)

    summary_a = data_a.get("summary") or {}
    summary_b = data_b.get("summary") or {}

    summary_diff = _diff_summaries(summary_a, summary_b)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "prompt_field": str(args.prompt_field),
        "max_cases": int(args.max_cases),
        "seed": int(args.seed),
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "report_a_path": str(report_a),
        "report_b_path": str(report_b),
        "summary_a": summary_a,
        "summary_b": summary_b,
        "summary_diff": summary_diff,
        "per_case_diff": _diff_per_case(list(data_a.get("per_case") or []), list(data_b.get("per_case") or [])),
    }

    diff_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote model-compare diff: {diff_path}")
    print("Summary diff:")
    print(json.dumps(summary_diff, indent=2))

    min_delta_cpt = args.min_delta_avg_cpt_jaccard
    if min_delta_cpt is not None:
        observed_delta = summary_diff.get("avg_cpt_jaccard", {}).get("delta")
        if observed_delta is None or float(observed_delta) < float(min_delta_cpt):
            print(
                "FAILED assertion: avg_cpt_jaccard delta "
                f"{observed_delta} < min required {float(min_delta_cpt)}"
            )
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
