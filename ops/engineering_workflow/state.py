"""Workflow state helpers."""

from __future__ import annotations

from pathlib import Path

from .artifacts import save_manifest
from .contracts import SessionManifest, SliceExecutionRecord, WorkflowState


def initialize_manifest(
    *,
    session_id: str,
    goal: str,
    artifact_dir: Path,
    allow_dirty: bool,
    enable_tracing: bool,
) -> SessionManifest:
    return SessionManifest(
        session_id=session_id,
        goal=goal,
        artifact_dir=str(artifact_dir),
        allow_dirty=allow_dirty,
        enable_tracing=enable_tracing,
    )


def set_state(
    session_dir: Path,
    manifest: SessionManifest,
    state: WorkflowState,
    *,
    current_slice_id: str | None = None,
) -> SessionManifest:
    manifest.current_state = state
    manifest.current_slice_id = current_slice_id
    save_manifest(session_dir, manifest)
    return manifest


def ensure_record(manifest: SessionManifest, task_id: str) -> SliceExecutionRecord:
    for record in manifest.records:
        if record.task_id == task_id:
            return record
    record = SliceExecutionRecord(task_id=task_id)
    manifest.records.append(record)
    return record

