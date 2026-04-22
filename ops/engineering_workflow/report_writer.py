"""Deterministic session report generation."""

from __future__ import annotations

from pathlib import Path

from .artifacts import write_json, write_text
from .contracts import PlanPacket, SessionReport, ValidationResult


def build_session_handoff(
    *,
    report: SessionReport,
    plan: PlanPacket,
    final_validation: ValidationResult | None,
) -> str:
    lines = [
        f"# Session Handoff - {report.session_id}",
        "",
        f"- Goal: `{report.goal}`",
        f"- Status: `{report.status}`",
        f"- Branch: `{report.branch}`",
        f"- Head SHA: `{report.head_sha}`",
        f"- Base ref: `{report.base_ref}`",
        f"- Risk level: `{report.risk_level.value}`",
        f"- Requires human review: `{report.requires_human_review}`",
        "",
        "## Summary",
        report.summary,
        "",
        "## Completed Slices",
    ]
    for task_id in report.completed_slices:
        lines.append(f"- `{task_id}`")
    if report.blocked_slice:
        lines.extend(["", "## Blocked Slice", f"- `{report.blocked_slice}`"])
    if report.open_questions:
        lines.extend(["", "## Open Questions", *[f"- {item}" for item in report.open_questions]])
    if plan.final_gate_commands:
        lines.extend(["", "## Final Gate Commands"])
        for command in plan.final_gate_commands:
            lines.append(f"- `{command.name}`: `{' '.join(command.command)}`")
    if final_validation is not None:
        lines.extend(
            [
                "",
                "## Final Validation",
                f"- Passed: `{final_validation.passed}`",
                f"- Classification: `{final_validation.failure_classification.category.value}`",
                f"- Rationale: {final_validation.failure_classification.rationale}",
            ]
        )
    if report.rollback_hint:
        lines.extend(["", "## Rollback Hint", report.rollback_hint])
    return "\n".join(lines) + "\n"


def write_session_outputs(
    *,
    artifact_dir: Path,
    report: SessionReport,
    plan: PlanPacket,
    final_validation: ValidationResult | None,
) -> tuple[Path, Path]:
    report_path = write_json(artifact_dir / "session_report.json", report)
    handoff_path = write_text(
        artifact_dir / "session_handoff.md",
        build_session_handoff(report=report, plan=plan, final_validation=final_validation),
    )
    return report_path, handoff_path

