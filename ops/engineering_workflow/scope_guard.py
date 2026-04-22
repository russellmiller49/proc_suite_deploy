"""Workspace snapshotting and scope enforcement."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .contracts import ScopeCheckResult, TaskSlice, WorkspaceSnapshot


def _git_status(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    files: list[str] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if "->" in path:
            path = path.split("->", maxsplit=1)[1].strip()
        else:
            path = path.strip()
        absolute_path = repo_root / path
        if absolute_path.is_dir():
            for child in sorted(absolute_path.rglob("*")):
                if child.is_file():
                    files.append(str(child.relative_to(repo_root)))
            continue
        files.append(path)
    return sorted(dict.fromkeys(files))


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def capture_workspace_snapshot(repo_root: Path) -> WorkspaceSnapshot:
    file_hashes = {}
    for relative_path in _git_status(repo_root):
        file_hashes[relative_path] = _hash_file(repo_root / relative_path)
    return WorkspaceSnapshot(file_hashes=file_hashes)


def _matches_prefix(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def _diff_numstat(repo_root: Path) -> tuple[int, int]:
    proc = subprocess.run(
        ["git", "diff", "--numstat"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    added = 0
    deleted = 0
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            added += 0 if parts[0] == "-" else int(parts[0])
            deleted += 0 if parts[1] == "-" else int(parts[1])
        except ValueError:
            continue
    return added, deleted


def evaluate_scope(*, repo_root: Path, task: TaskSlice, before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> ScopeCheckResult:
    all_paths = sorted(set(before.file_hashes) | set(after.file_hashes))
    actual_changed_files = [
        path for path in all_paths if before.file_hashes.get(path) != after.file_hashes.get(path)
    ]
    out_of_scope = [path for path in actual_changed_files if task.allowed_paths and not _matches_prefix(path, task.allowed_paths)]
    protected = [path for path in actual_changed_files if _matches_prefix(path, task.protected_paths)]
    added_lines, deleted_lines = _diff_numstat(repo_root)

    failure_reasons: list[str] = []
    if out_of_scope:
        failure_reasons.append("Out-of-scope file changes detected.")
    if protected:
        failure_reasons.append("Protected file changes detected.")
    if len(actual_changed_files) > task.diff_budget.max_changed_files:
        failure_reasons.append("Changed file budget exceeded.")
    if added_lines > task.diff_budget.max_added_lines:
        failure_reasons.append("Added line budget exceeded.")
    if deleted_lines > task.diff_budget.max_deleted_lines:
        failure_reasons.append("Deleted line budget exceeded.")

    return ScopeCheckResult(
        task_id=task.id,
        actual_changed_files=actual_changed_files,
        out_of_scope_changes=out_of_scope,
        protected_path_changes=protected,
        added_lines=added_lines,
        deleted_lines=deleted_lines,
        diff_budget_passed=not any("budget" in reason.lower() for reason in failure_reasons),
        allowed=not failure_reasons,
        failure_reasons=failure_reasons,
    )
