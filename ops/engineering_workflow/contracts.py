"""Typed contracts for the deterministic engineering workflow."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "proc_suite.engineering_workflow.v1"


class WorkflowState(str, Enum):
    CREATED = "created"
    PREFLIGHTED = "preflighted"
    PLANNED = "planned"
    BASELINE_BLOCKED = "baseline_blocked"
    RUNNING_SLICE = "running_slice"
    NEEDS_REPAIR = "needs_repair"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ValidationFailureCategory(str, Enum):
    NONE = "none"
    TOUCHED_FILE_SYNTAX_IMPORT_FAILURE = "touched_file_syntax_import_failure"
    LOCALIZED_REGRESSION = "localized_regression"
    ENVIRONMENTAL_FAILURE = "environmental_failure"
    FLAKY_FAILURE = "flaky_failure"
    PREEXISTING_UNRELATED_FAILURE = "preexisting_unrelated_failure"
    UNKNOWN = "unknown"


class CommandSpec(BaseModel):
    name: str
    command: list[str]


class DiffBudget(BaseModel):
    max_changed_files: int = 5
    max_added_lines: int = 400
    max_deleted_lines: int = 400


class TaskSlice(BaseModel):
    id: str
    title: str
    objective: str
    allowed_paths: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=list)
    done_criteria: list[str] = Field(default_factory=list)
    validation_commands: list[CommandSpec] = Field(default_factory=list)
    escalation_triggers: list[str] = Field(default_factory=list)
    diff_budget: DiffBudget = Field(default_factory=DiffBudget)
    actual_changed_files: list[str] = Field(default_factory=list)
    out_of_scope_changes: list[str] = Field(default_factory=list)
    repair_attempts: int = 0


class PlanPacket(BaseModel):
    schema_version: str = SCHEMA_VERSION
    goal: str
    non_goals: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_human_review: bool = False
    rollback_hint: str | None = None
    summary: str = ""
    task_slices: list[TaskSlice] = Field(default_factory=list)
    final_gate_commands: list[CommandSpec] = Field(default_factory=list)


class PreflightReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    head_sha: str
    base_ref: str
    branch: str
    repo_state_fingerprint: str
    dirty: bool
    dirty_files: list[str] = Field(default_factory=list)
    discovered_tools: dict[str, str] = Field(default_factory=dict)
    execution_allowed: bool = True
    reasons: list[str] = Field(default_factory=list)


class WorkspaceSnapshot(BaseModel):
    file_hashes: dict[str, str | None] = Field(default_factory=dict)


class CodexReportPayload(BaseModel):
    schema_version: str = SCHEMA_VERSION
    status: Literal["implemented", "blocked"] = "implemented"
    summary: str
    changes_made: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    tests_run: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_recommended_step: str | None = None


class CodexRunResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    payload: CodexReportPayload
    executor_command: list[str] = Field(default_factory=list)
    returncode: int
    stdout_path: str
    stderr_path: str
    parsed_json_path: str


class ScopeCheckResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    task_id: str
    actual_changed_files: list[str] = Field(default_factory=list)
    out_of_scope_changes: list[str] = Field(default_factory=list)
    protected_path_changes: list[str] = Field(default_factory=list)
    added_lines: int = 0
    deleted_lines: int = 0
    diff_budget_passed: bool = True
    allowed: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


class CommandResult(BaseModel):
    name: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str
    phase: str


class FailureClassification(BaseModel):
    category: ValidationFailureCategory = ValidationFailureCategory.NONE
    repair_eligible: bool = False
    rationale: str = ""


class ValidationResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    phase: str
    passed: bool
    command_results: list[CommandResult] = Field(default_factory=list)
    failure_classification: FailureClassification = Field(default_factory=FailureClassification)


class SliceExecutionRecord(BaseModel):
    task_id: str
    status: Literal["pending", "passed", "blocked"] = "pending"
    codex_result_path: str | None = None
    scope_check_path: str | None = None
    validation_path: str | None = None
    repair_attempts: int = 0


class SessionManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    goal: str
    artifact_dir: str
    current_state: WorkflowState = WorkflowState.CREATED
    head_sha: str | None = None
    base_ref: str | None = None
    branch: str | None = None
    repo_state_fingerprint: str | None = None
    allow_dirty: bool = False
    enable_tracing: bool = False
    requires_human_review: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    current_slice_id: str | None = None
    records: list[SliceExecutionRecord] = Field(default_factory=list)
    final_report_path: str | None = None
    handoff_path: str | None = None


class SessionReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    goal: str
    status: Literal["completed", "blocked"]
    head_sha: str
    base_ref: str
    branch: str
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_human_review: bool = False
    summary: str
    rollback_hint: str | None = None
    completed_slices: list[str] = Field(default_factory=list)
    blocked_slice: str | None = None
    open_questions: list[str] = Field(default_factory=list)
    artifact_dir: str

