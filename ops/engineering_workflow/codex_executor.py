"""Codex CLI execution adapter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from .artifacts import write_text
from .contracts import CodexReportPayload, CodexRunResult


class CodexExecutor(Protocol):
    def run(self, *, prompt: str, repo_root: Path, artifact_dir: Path, task_id: str) -> CodexRunResult:
        """Run Codex for a task."""


class CodexCliExecutor:
    """Runs the local Codex CLI and expects strict JSON stdout."""

    def __init__(self, executable: str | None = None, extra_args: list[str] | None = None) -> None:
        repo_wrapper = Path(__file__).resolve().parents[1] / "tools" / "codex_exec_json_wrapper.sh"
        self.executable = (
            executable
            or os.environ.get("CODEX_EXECUTABLE")
            or (str(repo_wrapper) if repo_wrapper.is_file() and os.access(repo_wrapper, os.X_OK) else None)
            or shutil.which("codex")
        )
        self.extra_args = extra_args or []

    def _build_command(self) -> list[str]:
        if not self.executable:
            raise RuntimeError("Codex executable not found. Set CODEX_EXECUTABLE to the local Codex CLI path.")
        return [self.executable, *self.extra_args]

    def run(self, *, prompt: str, repo_root: Path, artifact_dir: Path, task_id: str) -> CodexRunResult:
        prompt_path = write_text(artifact_dir / f"{task_id}_stdin_prompt.txt", prompt)
        stdout_path = artifact_dir / f"{task_id}_stdout.json"
        stderr_path = artifact_dir / f"{task_id}_stderr.txt"
        command = self._build_command()
        proc = subprocess.run(
            command,
            cwd=repo_root,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Codex command failed for task {task_id} with exit code {proc.returncode}. See {stderr_path}."
            )
        try:
            payload = CodexReportPayload.model_validate(json.loads(proc.stdout))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Codex output for task {task_id} was not valid JSON matching the expected schema. Prompt stored at {prompt_path}."
            ) from exc
        return CodexRunResult(
            payload=payload,
            executor_command=command,
            returncode=proc.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            parsed_json_path=str(stdout_path),
        )
