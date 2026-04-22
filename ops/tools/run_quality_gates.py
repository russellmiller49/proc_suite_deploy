#!/usr/bin/env python3
"""Run fast PR or nightly extraction quality gates and emit artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.quality_gate_reports import build_report_delta, render_delta_markdown, write_json, write_text


PR_PYTEST_TARGETS = [
    "tests/registry/test_regression_pack.py",
    "tests/registry/test_header_evidence_integrity.py",
    "tests/common/test_path_redaction.py",
    "tests/common/test_quality_gate_reports.py",
    "tests/reporting/test_reporter_clinical_fidelity.py",
    "tests/quality/test_reporter_seed_dual_path_matrix.py",
    "tests/scripts/test_eval_golden.py",
    "tests/scripts/test_reporter_seed_eval_tools.py",
    "tests/scripts/test_run_quality_gates.py",
]

PR_EXTRACTION_INPUT = ROOT / "tests" / "fixtures" / "unified_quality_corpus.json"
PR_EXTRACTION_BASELINE = ROOT / "reports" / "unified_quality_corpus_extraction_baseline.json"


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    command: list[str]
    stdout_path: str | None = None
    stderr_path: str | None = None
    output_path: str | None = None
    returncode: int | None = None
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["pr", "nightly"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _run_command(
    *,
    name: str,
    command: list[str],
    output_dir: Path,
    env: dict[str, str] | None = None,
) -> StepResult:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path = output_dir / f"{name}.stdout.txt"
    stderr_path = output_dir / f"{name}.stderr.txt"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return StepResult(
        name=name,
        status="passed" if proc.returncode == 0 else "failed",
        command=command,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        returncode=proc.returncode,
        error=None if proc.returncode == 0 else f"command failed with exit code {proc.returncode}",
    )


def _static_step_result(
    *,
    name: str,
    status: str,
    output_dir: Path,
    command: list[str] | None = None,
    output_path: Path | None = None,
    error: str | None = None,
    note: str | None = None,
) -> StepResult:
    stdout_path = output_dir / f"{name}.stdout.txt"
    stderr_path = output_dir / f"{name}.stderr.txt"
    stdout_path.write_text((note or "") + ("\n" if note else ""), encoding="utf-8")
    stderr_path.write_text((error or "") + ("\n" if error else ""), encoding="utf-8")
    return StepResult(
        name=name,
        status=status,
        command=command or [],
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        output_path=str(output_path) if output_path else None,
        returncode=0 if status == "passed" else 1,
        error=error,
    )


def _mark_step_failed(step: StepResult, *, output_dir: Path, error: str) -> StepResult:
    stderr_path = Path(step.stderr_path) if step.stderr_path else output_dir / f"{step.name}.stderr.txt"
    existing = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    stderr_path.write_text(
        (existing + ("\n" if existing and not existing.endswith("\n") else "")) + error + "\n",
        encoding="utf-8",
    )
    return replace(
        step,
        status="failed",
        stderr_path=str(stderr_path),
        returncode=step.returncode if step.returncode not in (None, 0) else 1,
        error=error,
    )


def _run_checked_step(
    *,
    name: str,
    command: list[str],
    output_dir: Path,
    output_path: Path | None = None,
    validator: Any | None = None,
    env: dict[str, str] | None = None,
) -> StepResult:
    step = _run_command(name=name, command=command, output_dir=output_dir, env=env)
    if output_path is not None:
        step = replace(step, output_path=str(output_path))
    if step.status != "passed" or validator is None:
        return step
    try:
        validator(output_path)
    except Exception as exc:  # noqa: BLE001
        return _mark_step_failed(step, output_dir=output_dir, error=f"{type(exc).__name__}: {exc}")
    return step


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_extraction_gate(path: Path | None) -> None:
    if path is None:
        raise RuntimeError("Extraction fixture gate did not produce an output artifact")
    payload = _load_json(path)
    summary = dict(payload.get("summary") or {})
    pass_rate = float(summary.get("pass_rate", 0.0))
    failed_cases = int(summary.get("failed_cases", 0))
    if failed_cases != 0 or pass_rate < 1.0:
        raise RuntimeError(f"Extraction fixture gate failed: failed_cases={failed_cases}, pass_rate={pass_rate}")


def _render_diff_sections(diff_jobs: list[tuple[str, Path, Path]]) -> list[str]:
    lines: list[str] = []
    for title, current_path, baseline_path in diff_jobs:
        try:
            if not current_path.exists():
                lines.append(f"### {title}")
                lines.append(f"- Current report missing: `{current_path}`")
                continue
            if not baseline_path.exists():
                lines.append(f"### {title}")
                lines.append(f"- Baseline missing: `{baseline_path}`")
                continue
            delta = build_report_delta(current_path=current_path, baseline_path=baseline_path)
            delta_path = current_path.with_name(f"{current_path.stem}.delta.json")
            write_json(delta_path, delta)
            lines.append(render_delta_markdown(delta, title=title))
        except Exception as exc:  # noqa: BLE001
            lines.append(f"### {title}")
            lines.append(f"- Delta generation failed: `{type(exc).__name__}: {exc}`")
    return lines


def _create_summary(
    *,
    tier: str,
    steps: list[StepResult],
    diff_sections: list[str],
    output_dir: Path,
) -> None:
    failed_steps = [step for step in steps if step.status == "failed"]
    lines = [
        f"## Quality Gates ({tier})",
        f"- Output directory: `{output_dir}`",
        f"- Passed steps: `{sum(1 for step in steps if step.status == 'passed')}`",
        f"- Failed steps: `{len(failed_steps)}`",
    ]
    for step in steps:
        lines.append(
            f"- `{step.name}`: `{step.status}`"
            + (f" (exit `{step.returncode}`)" if step.returncode is not None else "")
        )
    if diff_sections:
        lines.append("")
        lines.extend(diff_sections)

    write_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": "procedure_suite.quality_gate.run.v1",
            "tier": tier,
            "output_dir": str(output_dir),
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "command": step.command,
                    "stdout_path": step.stdout_path,
                    "stderr_path": step.stderr_path,
                    "output_path": step.output_path,
                    "returncode": step.returncode,
                    "error": step.error,
                }
                for step in steps
            ],
        },
    )


def _run_gate(*, tier: str, output_dir: Path, extraction_output_name: str) -> int:
    steps: list[StepResult] = []
    diff_sections: list[str] = []
    try:
        pytest_command = [sys.executable, "-m", "pytest", "-q", *PR_PYTEST_TARGETS]
        steps.append(_run_command(name="focused_pytest", command=pytest_command, output_dir=output_dir))

        extraction_output = output_dir / extraction_output_name
        steps.append(
            _run_checked_step(
                name="extraction_eval",
                command=[
                    sys.executable,
                    str(ROOT / "ml" / "scripts" / "eval_golden.py"),
                    "--input",
                    str(PR_EXTRACTION_INPUT),
                    "--output",
                    str(extraction_output),
                    "--fail-under",
                    "100",
                ],
                output_dir=output_dir,
                output_path=extraction_output,
                validator=_assert_extraction_gate,
            )
        )

        diff_sections = _render_diff_sections(
            [
                ("Extraction Delta", extraction_output, PR_EXTRACTION_BASELINE),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        steps.append(
            _static_step_result(
                name=f"{tier}_gate_unhandled",
                status="failed",
                output_dir=output_dir,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
    finally:
        _create_summary(tier=tier, steps=steps, diff_sections=diff_sections, output_dir=output_dir)

    return 0 if all(step.status == "passed" for step in steps) else 1


def run_pr(output_dir: Path) -> int:
    return _run_gate(tier="pr", output_dir=output_dir, extraction_output_name="unified_quality_corpus_extraction.json")


def run_nightly(output_dir: Path) -> int:
    return _run_gate(
        tier="nightly",
        output_dir=output_dir,
        extraction_output_name="unified_quality_corpus_extraction_nightly.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tier == "pr":
        return run_pr(output_dir)
    return run_nightly(output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
