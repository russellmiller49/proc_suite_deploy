"""Deterministic validation runner and failure classifier."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .contracts import (
    CommandResult,
    CommandSpec,
    FailureClassification,
    ValidationFailureCategory,
    ValidationResult,
)

ENVIRONMENT_MARKERS = (
    "command not found",
    "No such file or directory",
    "ConnectionError",
    "timed out",
    "network is unreachable",
)
FLAKY_MARKERS = ("flaky", "timeout while waiting", "resource temporarily unavailable")
SYNTAX_MARKERS = ("SyntaxError", "ImportError", "ModuleNotFoundError", "NameError")


def is_allowed_command(command: list[str]) -> bool:
    if not command:
        return False
    executable = os.path.basename(command[0])
    if executable in {"pytest", "make"}:
        return True
    if len(command) >= 2 and executable.startswith("python") and command[1].startswith("ops/tools/"):
        return True
    return False


def run_commands(
    *,
    commands: list[CommandSpec],
    repo_root: Path,
    artifact_dir: Path,
    phase: str,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    for command_spec in commands:
        if not is_allowed_command(command_spec.command):
            raise RuntimeError(f"Validation command is not allowed: {command_spec.command}")
        proc = subprocess.run(
            command_spec.command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", command_spec.name)
        stdout_path = artifact_dir / f"{phase}_{safe_name}.stdout.txt"
        stderr_path = artifact_dir / f"{phase}_{safe_name}.stderr.txt"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        results.append(
            CommandResult(
                name=command_spec.name,
                command=command_spec.command,
                returncode=proc.returncode,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                phase=phase,
            )
        )
    return results


def classify_failure(
    *,
    baseline_results: list[CommandResult],
    current_results: list[CommandResult],
    changed_files: list[str],
) -> FailureClassification:
    failing_results = [result for result in current_results if result.returncode != 0]
    if not failing_results:
        return FailureClassification(
            category=ValidationFailureCategory.NONE,
            repair_eligible=False,
            rationale="All commands passed.",
        )

    baseline_failures = {result.name for result in baseline_results if result.returncode != 0}
    stdout_stderr_text = []
    for result in failing_results:
        stdout_stderr_text.append(Path(result.stdout_path).read_text(encoding="utf-8"))
        stdout_stderr_text.append(Path(result.stderr_path).read_text(encoding="utf-8"))
    combined_text = "\n".join(stdout_stderr_text)

    if any(result.name in baseline_failures for result in failing_results):
        return FailureClassification(
            category=ValidationFailureCategory.PREEXISTING_UNRELATED_FAILURE,
            repair_eligible=False,
            rationale="The same validation command was already failing before Codex made changes.",
        )

    if any(marker.lower() in combined_text.lower() for marker in ENVIRONMENT_MARKERS):
        return FailureClassification(
            category=ValidationFailureCategory.ENVIRONMENTAL_FAILURE,
            repair_eligible=False,
            rationale="Validation failed because the execution environment was unavailable or misconfigured.",
        )

    if any(marker.lower() in combined_text.lower() for marker in FLAKY_MARKERS):
        return FailureClassification(
            category=ValidationFailureCategory.FLAKY_FAILURE,
            repair_eligible=False,
            rationale="Validation appears flaky or resource-constrained.",
        )

    if any(marker in combined_text for marker in SYNTAX_MARKERS):
        return FailureClassification(
            category=ValidationFailureCategory.TOUCHED_FILE_SYNTAX_IMPORT_FAILURE,
            repair_eligible=True,
            rationale="Validation indicates a syntax or import failure in the touched slice.",
        )

    changed_names = {Path(path).name for path in changed_files}
    if any(name and name in combined_text for name in changed_names):
        return FailureClassification(
            category=ValidationFailureCategory.TOUCHED_FILE_SYNTAX_IMPORT_FAILURE,
            repair_eligible=True,
            rationale="Validation output references a touched file directly.",
        )

    return FailureClassification(
        category=ValidationFailureCategory.LOCALIZED_REGRESSION,
        repair_eligible=True,
        rationale="Validation failed after the slice and does not match a pre-existing, flaky, or environmental failure.",
    )


def build_validation_result(
    *,
    phase: str,
    baseline_results: list[CommandResult],
    current_results: list[CommandResult],
    changed_files: list[str],
) -> ValidationResult:
    failure_classification = classify_failure(
        baseline_results=baseline_results,
        current_results=current_results,
        changed_files=changed_files,
    )
    return ValidationResult(
        phase=phase,
        passed=all(result.returncode == 0 for result in current_results),
        command_results=current_results,
        failure_classification=failure_classification,
    )
