"""Top-level workflow orchestration."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    append_event,
    ensure_session_dir,
    load_manifest,
    resolve_artifact_root,
    save_manifest,
    write_json,
    write_text,
)
from .codex_executor import CodexCliExecutor, CodexExecutor
from .contracts import PlanPacket, SessionReport, ValidationFailureCategory, WorkflowState
from .planner_agent import PlannerAgent
from .preflight import ensure_session_branch, run_preflight
from .prompt_renderer import render_codex_prompt
from .report_writer import write_session_outputs
from .scope_guard import capture_workspace_snapshot, evaluate_scope
from .state import ensure_record, initialize_manifest, set_state
from .validator import build_validation_result, run_commands


@dataclass
class WorkflowDependencies:
    planner: PlannerAgent | None = None
    codex_executor: CodexExecutor | None = None
    failure_classifier: object | None = None
    reporter: object | None = None


@dataclass
class WorkflowRunResult:
    session_dir: Path
    manifest_path: Path
    report_path: Path
    handoff_path: Path
    status: str


def _default_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:10]}"


def _dedupe_commands(plan: PlanPacket) -> list:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    commands = []
    for task in plan.task_slices:
        for command in task.validation_commands:
            key = (command.name, tuple(command.command))
            if key in seen:
                continue
            seen.add(key)
            commands.append(command)
    for command in plan.final_gate_commands:
        key = (command.name, tuple(command.command))
        if key in seen:
            continue
        seen.add(key)
        commands.append(command)
    return commands


class EngineeringWorkflowRunner:
    """Runs the deterministic engineering workflow."""

    def __init__(self, repo_root: Path, dependencies: WorkflowDependencies | None = None) -> None:
        self.repo_root = repo_root
        self.dependencies = dependencies or WorkflowDependencies()
        self.dependencies.planner = self.dependencies.planner or PlannerAgent()
        self.dependencies.codex_executor = self.dependencies.codex_executor or CodexCliExecutor()

    def run(
        self,
        *,
        goal: str,
        session_id: str | None = None,
        allow_dirty: bool = False,
        enable_tracing: bool = False,
        artifact_root: Path | None = None,
        resume: bool = False,
    ) -> WorkflowRunResult:
        if not enable_tracing:
            os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

        session_id = session_id or _default_session_id()
        root = resolve_artifact_root(artifact_root)
        session_dir = ensure_session_dir(root, session_id)

        if resume:
            manifest = load_manifest(session_dir)
        else:
            manifest = initialize_manifest(
                session_id=session_id,
                goal=goal,
                artifact_dir=session_dir,
                allow_dirty=allow_dirty,
                enable_tracing=enable_tracing,
            )
            save_manifest(session_dir, manifest)
            append_event(session_dir, "session_created", {"goal": goal})

        preflight = run_preflight(
            repo_root=self.repo_root,
            session_id=session_id,
            allow_dirty=allow_dirty,
        )
        write_json(session_dir / "preflight.json", preflight)
        manifest.head_sha = preflight.head_sha
        manifest.base_ref = preflight.base_ref
        manifest.branch = preflight.branch
        manifest.repo_state_fingerprint = preflight.repo_state_fingerprint
        set_state(session_dir, manifest, WorkflowState.PREFLIGHTED)
        append_event(session_dir, "preflight_complete", preflight.model_dump(mode="json"))

        if not preflight.execution_allowed:
            return self._finish_blocked(
                session_dir=session_dir,
                manifest=manifest,
                plan=PlanPacket(goal=goal),
                summary="Preflight blocked execution.",
                blocked_slice=None,
                open_questions=preflight.reasons,
                final_validation=None,
            )

        if not preflight.dirty:
            session_branch = f"codex/{session_id}"
            ensure_session_branch(self.repo_root, session_branch)
            manifest.branch = session_branch
            save_manifest(session_dir, manifest)

        plan = self.dependencies.planner.plan(goal=goal, preflight=preflight, repo_root=self.repo_root)
        write_json(session_dir / "plan.json", plan)
        manifest.requires_human_review = plan.requires_human_review
        set_state(session_dir, manifest, WorkflowState.PLANNED)
        append_event(session_dir, "plan_created", plan.model_dump(mode="json"))

        baseline_commands = _dedupe_commands(plan)
        baseline_results = run_commands(
            commands=baseline_commands,
            repo_root=self.repo_root,
            artifact_dir=session_dir,
            phase="baseline",
        )
        baseline_validation = build_validation_result(
            phase="baseline",
            baseline_results=[],
            current_results=baseline_results,
            changed_files=[],
        )
        write_json(session_dir / "baseline_validation.json", baseline_validation)
        if not baseline_validation.passed:
            set_state(session_dir, manifest, WorkflowState.BASELINE_BLOCKED)
            return self._finish_blocked(
                session_dir=session_dir,
                manifest=manifest,
                plan=plan,
                summary="Baseline validation was already red before Codex execution.",
                blocked_slice=None,
                open_questions=[baseline_validation.failure_classification.rationale],
                final_validation=baseline_validation,
            )

        final_validation = None
        completed_slices: list[str] = []
        blocked_slice: str | None = None
        open_questions: list[str] = []

        for task in plan.task_slices:
            record = ensure_record(manifest, task.id)
            set_state(session_dir, manifest, WorkflowState.RUNNING_SLICE, current_slice_id=task.id)
            before_snapshot = capture_workspace_snapshot(self.repo_root)
            prompt = render_codex_prompt(plan=plan, task=task)
            prompt_path = write_text(session_dir / f"slice_{task.id}_prompt.md", prompt)
            append_event(session_dir, "slice_started", {"task_id": task.id, "prompt_path": str(prompt_path)})

            codex_result = self.dependencies.codex_executor.run(
                prompt=prompt,
                repo_root=self.repo_root,
                artifact_dir=session_dir,
                task_id=f"slice_{task.id}",
            )
            codex_result_path = write_json(session_dir / f"slice_{task.id}_codex_result.json", codex_result)
            record.codex_result_path = str(codex_result_path)

            if codex_result.payload.status == "blocked":
                blocked_slice = task.id
                open_questions.extend(codex_result.payload.open_questions)
                break

            after_snapshot = capture_workspace_snapshot(self.repo_root)
            scope_check = evaluate_scope(repo_root=self.repo_root, task=task, before=before_snapshot, after=after_snapshot)
            task.actual_changed_files = scope_check.actual_changed_files
            task.out_of_scope_changes = scope_check.out_of_scope_changes
            scope_path = write_json(session_dir / f"slice_{task.id}_scope_check.json", scope_check)
            record.scope_check_path = str(scope_path)
            if not scope_check.allowed:
                blocked_slice = task.id
                open_questions.extend(scope_check.failure_reasons)
                break

            validation_results = run_commands(
                commands=task.validation_commands,
                repo_root=self.repo_root,
                artifact_dir=session_dir,
                phase=f"slice_{task.id}",
            )
            validation = build_validation_result(
                phase=f"slice_{task.id}",
                baseline_results=baseline_results,
                current_results=validation_results,
                changed_files=scope_check.actual_changed_files,
            )
            if (
                not validation.passed
                and validation.failure_classification.category == ValidationFailureCategory.UNKNOWN
                and self.dependencies.failure_classifier is not None
            ):
                validation.failure_classification = self.dependencies.failure_classifier.classify(validation)
            validation_path = write_json(session_dir / f"slice_{task.id}_validation.json", validation)
            record.validation_path = str(validation_path)

            if validation.passed:
                record.status = "passed"
                completed_slices.append(task.id)
                save_manifest(session_dir, manifest)
                continue

            if validation.failure_classification.repair_eligible and task.repair_attempts < 1:
                task.repair_attempts += 1
                record.repair_attempts = task.repair_attempts
                set_state(session_dir, manifest, WorkflowState.NEEDS_REPAIR, current_slice_id=task.id)
                repair_before = capture_workspace_snapshot(self.repo_root)
                repair_prompt = render_codex_prompt(plan=plan, task=task, repair_context=validation)
                repair_prompt_path = write_text(session_dir / f"slice_{task.id}_repair_prompt.md", repair_prompt)
                append_event(session_dir, "repair_started", {"task_id": task.id, "prompt_path": str(repair_prompt_path)})
                repair_result = self.dependencies.codex_executor.run(
                    prompt=repair_prompt,
                    repo_root=self.repo_root,
                    artifact_dir=session_dir,
                    task_id=f"slice_{task.id}_repair",
                )
                write_json(session_dir / f"slice_{task.id}_repair_codex_result.json", repair_result)
                repair_after = capture_workspace_snapshot(self.repo_root)
                repair_scope = evaluate_scope(
                    repo_root=self.repo_root,
                    task=task,
                    before=repair_before,
                    after=repair_after,
                )
                write_json(session_dir / f"slice_{task.id}_repair_scope_check.json", repair_scope)
                if not repair_scope.allowed:
                    blocked_slice = task.id
                    open_questions.extend(repair_scope.failure_reasons)
                    break
                repair_command_results = run_commands(
                    commands=task.validation_commands,
                    repo_root=self.repo_root,
                    artifact_dir=session_dir,
                    phase=f"slice_{task.id}_repair",
                )
                repair_validation = build_validation_result(
                    phase=f"slice_{task.id}_repair",
                    baseline_results=baseline_results,
                    current_results=repair_command_results,
                    changed_files=repair_scope.actual_changed_files,
                )
                write_json(session_dir / f"slice_{task.id}_repair_validation.json", repair_validation)
                if repair_validation.passed:
                    record.status = "passed"
                    completed_slices.append(task.id)
                    save_manifest(session_dir, manifest)
                    continue

            blocked_slice = task.id
            open_questions.append(validation.failure_classification.rationale)
            break

        if blocked_slice is None:
            final_results = run_commands(
                commands=plan.final_gate_commands,
                repo_root=self.repo_root,
                artifact_dir=session_dir,
                phase="final_gate",
            )
            final_validation = build_validation_result(
                phase="final_gate",
                baseline_results=baseline_results,
                current_results=final_results,
                changed_files=[],
            )
            write_json(session_dir / "final_gate_validation.json", final_validation)
            if final_validation.passed:
                return self._finish_completed(
                    session_dir=session_dir,
                    manifest=manifest,
                    plan=plan,
                    completed_slices=completed_slices,
                    final_validation=final_validation,
                )
            open_questions.append(final_validation.failure_classification.rationale)

        return self._finish_blocked(
            session_dir=session_dir,
            manifest=manifest,
            plan=plan,
            summary="Workflow blocked before completing all slices.",
            blocked_slice=blocked_slice,
            open_questions=open_questions,
            final_validation=final_validation,
            completed_slices=completed_slices,
        )

    def _finish_completed(
        self,
        *,
        session_dir: Path,
        manifest,
        plan: PlanPacket,
        completed_slices: list[str],
        final_validation,
    ) -> WorkflowRunResult:
        report = SessionReport(
            session_id=manifest.session_id,
            goal=manifest.goal,
            status="completed",
            head_sha=manifest.head_sha or "",
            base_ref=manifest.base_ref or "",
            branch=manifest.branch or "",
            risk_level=plan.risk_level,
            requires_human_review=plan.requires_human_review,
            summary="All planned slices completed and final gates passed.",
            rollback_hint=plan.rollback_hint,
            completed_slices=completed_slices,
            artifact_dir=str(session_dir),
        )
        report_path, handoff_path = write_session_outputs(
            artifact_dir=session_dir,
            report=report,
            plan=plan,
            final_validation=final_validation,
        )
        manifest.final_report_path = str(report_path)
        manifest.handoff_path = str(handoff_path)
        set_state(session_dir, manifest, WorkflowState.COMPLETED)
        append_event(session_dir, "session_completed", report.model_dump(mode="json"))
        return WorkflowRunResult(
            session_dir=session_dir,
            manifest_path=session_dir / "session_manifest.json",
            report_path=report_path,
            handoff_path=handoff_path,
            status="completed",
        )

    def _finish_blocked(
        self,
        *,
        session_dir: Path,
        manifest,
        plan: PlanPacket,
        summary: str,
        blocked_slice: str | None,
        open_questions: list[str],
        final_validation,
        completed_slices: list[str] | None = None,
    ) -> WorkflowRunResult:
        report = SessionReport(
            session_id=manifest.session_id,
            goal=manifest.goal,
            status="blocked",
            head_sha=manifest.head_sha or "",
            base_ref=manifest.base_ref or "",
            branch=manifest.branch or "",
            risk_level=plan.risk_level,
            requires_human_review=True,
            summary=summary,
            rollback_hint=plan.rollback_hint,
            completed_slices=completed_slices or [],
            blocked_slice=blocked_slice,
            open_questions=open_questions,
            artifact_dir=str(session_dir),
        )
        report_path, handoff_path = write_session_outputs(
            artifact_dir=session_dir,
            report=report,
            plan=plan,
            final_validation=final_validation,
        )
        manifest.final_report_path = str(report_path)
        manifest.handoff_path = str(handoff_path)
        set_state(session_dir, manifest, WorkflowState.BLOCKED, current_slice_id=blocked_slice)
        append_event(session_dir, "session_blocked", report.model_dump(mode="json"))
        return WorkflowRunResult(
            session_dir=session_dir,
            manifest_path=session_dir / "session_manifest.json",
            report_path=report_path,
            handoff_path=handoff_path,
            status="blocked",
        )

