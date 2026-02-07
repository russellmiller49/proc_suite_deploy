#!/usr/bin/env python3
"""
Create a slim Git branch for external review, excluding large files.

This script creates an orphan branch with only essential files for code review,
excluding ML models, training data, archives, and other large artifacts.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Directories to exclude
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "data/knowledge",
    "data",
    "data/models",
    "data/ml_training",
    "ops/tools/phi_test_node",
    "artifacts",
    "archive",
    "dist",
    "proc_suite.egg-info",
    "reports",
    "validation_results",
}

# File extensions to exclude
EXCLUDED_EXTENSIONS = {
    ".bin",
    ".db",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".h5",
    ".ckpt",
    ".pkl",
    ".joblib",
    ".pb",
    ".tflite",
    ".mlmodel",
    ".tar",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".zip",
    ".rar",
    ".7z",
    ".gz",
    ".bz2",
    ".xz",
    ".jsonl",  # Training data files
    ".pyc",
    ".pyo",
}

# File patterns to exclude (by filename)
EXCLUDED_PATTERNS = {
    "classifier.pt",
    "classifier.pkl",
    "mlb.pkl",
    "tokenizer.json",
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "model.pth",
}

# Directories to keep (even if parent is excluded)
KEEP_DIRS = {
    "docs",
    "modules",
    "scripts",
    "tests",
    "schemas",
    "configs",
}


def is_archive_file(filename: str) -> bool:
    """Check if a file is an archive by name."""
    archive_names = {
        "archive",
        "backup",
        "old",
        "temp",
        "tmp",
    }
    name_lower = filename.lower()
    return any(archive_name in name_lower for archive_name in archive_names)


def should_exclude_path(path: Path, repo_root: Path) -> bool:
    """Check if a path should be excluded from the slim branch."""
    try:
        rel_path = path.relative_to(repo_root)
    except ValueError:
        return True

    # Check if any part of the path matches excluded dirs
    parts = rel_path.parts
    for part in parts:
        if part in EXCLUDED_DIRS:
            # Check if this is a keep directory
            if any(keep_dir in parts for keep_dir in KEEP_DIRS):
                # If it's a keep dir, only exclude if the excluded part comes after
                keep_indices = [i for i, p in enumerate(parts) if p in KEEP_DIRS]
                exclude_index = next((i for i, p in enumerate(parts) if p in EXCLUDED_DIRS), None)
                if exclude_index is not None and keep_indices:
                    if min(keep_indices) < exclude_index:
                        continue  # Keep this path
            return True

    # Check file extension
    if path.is_file():
        for ext in EXCLUDED_EXTENSIONS:
            if str(path).endswith(ext):
                return True

        # Check filename patterns
        if path.name in EXCLUDED_PATTERNS:
            return True

        # Check for archive files by name
        if is_archive_file(path.name):
            return True

        # Exclude large JSON files (>5MB)
        if path.suffix == ".json":
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > 5:
                    return True
            except (OSError, ValueError):
                pass

    return False


def run_git_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    result = subprocess.run(
        ["git"] + cmd,
        capture_output=True,
        text=True,
        check=check,
    )
    return result


def create_slim_branch(source_branch: str, target_branch: str, force: bool = False) -> None:
    """Create a slim branch from the source branch."""
    repo_root = Path.cwd()

    # Check if we're in a git repo
    if not (repo_root / ".git").exists():
        print("Error: Not in a git repository")
        sys.exit(1)

    # Check if target branch exists
    result = run_git_command(["branch", "--list", target_branch], check=False)
    if result.stdout.strip() and not force:
        print(f"Error: Branch '{target_branch}' already exists. Use --force to overwrite.")
        sys.exit(1)

    # Get current branch
    current_branch_result = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    current_branch = current_branch_result.stdout.strip()

    print(f"Creating slim branch '{target_branch}' from '{source_branch}'...")

    # Checkout source branch first
    print(f"Checking out source branch '{source_branch}'...")
    run_git_command(["checkout", source_branch])

    # Delete target branch if it exists and force is set
    if force:
        run_git_command(["branch", "-D", target_branch], check=False)

    # Create orphan branch
    print(f"Creating orphan branch '{target_branch}'...")
    run_git_command(["checkout", "--orphan", target_branch])

    # Remove all files from staging
    print("Removing all files from staging...")
    run_git_command(["rm", "-rf", "--cached", "."], check=False)

    # Add files that should be included
    print("Adding files to slim branch...")
    added_count = 0
    skipped_count = 0

    # Get all files from source branch
    result = run_git_command(["ls-tree", "-r", "--name-only", source_branch])
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    for file_path_str in files:
        if not file_path_str:
            continue

        file_path = repo_root / file_path_str

        if should_exclude_path(file_path, repo_root):
            skipped_count += 1
            continue

        # Check if file exists (it should, since we're on the source branch)
        if file_path.exists():
            try:
                run_git_command(["add", file_path_str])
                added_count += 1
            except subprocess.CalledProcessError:
                skipped_count += 1

    print(f"\nAdded {added_count} files, skipped {skipped_count} files")

    # Commit
    print("Committing slim branch...")
    run_git_command(
        ["commit", "-m", f"Create slim branch from {source_branch} for external review"]
    )

    print(f"\n✓ Slim branch '{target_branch}' created successfully!")
    print(f"  To push to remote: git push -f origin {target_branch}")
    print(f"  To return to previous branch: git checkout {current_branch}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a slim Git branch for external review"
    )
    parser.add_argument(
        "--source",
        default="v19",
        help="Source branch to create slim branch from (default: v19)",
    )
    parser.add_argument(
        "--target",
        default="slim-review",
        help="Target branch name (default: slim-review)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if target branch exists",
    )
    args = parser.parse_args()

    create_slim_branch(args.source, args.target, args.force)


if __name__ == "__main__":
    main()
