"""Deterministic Codex prompt rendering."""

from __future__ import annotations

import json

from .contracts import PlanPacket, TaskSlice, ValidationResult


def render_codex_prompt(*, plan: PlanPacket, task: TaskSlice, repair_context: ValidationResult | None = None) -> str:
    sections = [
        "Return strict JSON only.",
        "Schema:",
        json.dumps(
            {
                "schema_version": plan.schema_version,
                "status": "implemented|blocked",
                "summary": "short summary",
                "changes_made": ["..."],
                "commands_run": ["..."],
                "tests_run": ["..."],
                "files_changed": ["repo/relative/path.py"],
                "open_questions": ["..."],
                "next_recommended_step": "..."
            },
            indent=2,
            sort_keys=True,
        ),
        f"Goal: {plan.goal}",
        f"Task ID: {task.id}",
        f"Task Title: {task.title}",
        f"Objective:\n{task.objective}",
        f"Allowed paths:\n{json.dumps(task.allowed_paths, indent=2)}",
        f"Protected paths:\n{json.dumps(task.protected_paths, indent=2)}",
        f"Done criteria:\n{json.dumps(task.done_criteria, indent=2)}",
        f"Validation commands:\n{json.dumps([cmd.model_dump(mode='json') for cmd in task.validation_commands], indent=2)}",
        f"Escalation triggers:\n{json.dumps(task.escalation_triggers, indent=2)}",
        f"Diff budget:\n{json.dumps(task.diff_budget.model_dump(mode='json'), indent=2)}",
    ]
    if repair_context is not None:
        sections.extend(
            [
                "This is a repair attempt. Fix only the localized validation issues below.",
                json.dumps(repair_context.model_dump(mode="json"), indent=2, sort_keys=True),
            ]
        )
    return "\n\n".join(sections) + "\n"

