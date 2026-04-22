"""Preflight checks for the engineering workflow."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from .contracts import PreflightReport


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def parse_dirty_files(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[str] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        parts = line[3:]
        if "->" in parts:
            files.append(parts.split("->", maxsplit=1)[1].strip())
        else:
            files.append(parts.strip())
    return sorted(dict.fromkeys(files))


def build_repo_state_fingerprint(repo_root: Path, head_sha: str, branch: str, dirty_files: list[str]) -> str:
    raw = "\n".join([str(repo_root), head_sha, branch, *dirty_files])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def discover_tools(codex_executable: str | None = None) -> dict[str, str]:
    repo_wrapper = Path(__file__).resolve().parents[1] / "tools" / "codex_exec_json_wrapper.sh"
    tools = {
        "git": shutil.which("git") or "",
        "make": shutil.which("make") or "",
        "pytest": shutil.which("pytest") or "",
        "python": shutil.which("python3") or shutil.which("python") or "",
        "codex": (
            codex_executable
            or os.environ.get("CODEX_EXECUTABLE")
            or (str(repo_wrapper) if repo_wrapper.is_file() and os.access(repo_wrapper, os.X_OK) else "")
            or shutil.which("codex")
            or ""
        ),
    }
    return {name: path for name, path in tools.items() if path}


def choose_base_ref(repo_root: Path) -> str:
    for candidate in ("origin/main", "main", "origin/master", "master"):
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", candidate],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return candidate
    return "HEAD"


def run_preflight(
    *,
    repo_root: Path,
    session_id: str,
    allow_dirty: bool,
    codex_executable: str | None = None,
) -> PreflightReport:
    head_sha = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    base_ref = choose_base_ref(repo_root)
    dirty_files = parse_dirty_files(repo_root)
    dirty = bool(dirty_files)
    discovered_tools = discover_tools(codex_executable)
    reasons: list[str] = []
    execution_allowed = True

    if dirty and not allow_dirty:
        execution_allowed = False
        reasons.append("Repository is dirty. Re-run with --allow-dirty to override.")
    if "git" not in discovered_tools:
        execution_allowed = False
        reasons.append("git executable is not available.")
    if "python" not in discovered_tools:
        execution_allowed = False
        reasons.append("python executable is not available.")
    if "codex" not in discovered_tools:
        reasons.append("Codex executable was not auto-discovered. Set CODEX_EXECUTABLE before execution.")

    return PreflightReport(
        session_id=session_id,
        head_sha=head_sha,
        base_ref=base_ref,
        branch=branch,
        repo_state_fingerprint=build_repo_state_fingerprint(repo_root, head_sha, branch, dirty_files),
        dirty=dirty,
        dirty_files=dirty_files,
        discovered_tools=discovered_tools,
        execution_allowed=execution_allowed,
        reasons=reasons,
    )


def ensure_session_branch(repo_root: Path, session_branch: str) -> str:
    current_branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if current_branch == session_branch:
        return current_branch

    subprocess.run(
        ["git", "checkout", "-b", session_branch],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return session_branch
