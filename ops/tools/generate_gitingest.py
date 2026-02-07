#!/usr/bin/env python3
"""
Generate gitingest.md - A token-budget friendly snapshot of the repo structure
and curated important files for LLM/context ingestion.
"""

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set


# Configuration
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "coverage",
    "data",
    "dist",
    "distilled",
    "node_modules",
    "proc_suite.egg-info",
    "reports",
    "validation_results",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}

EXCLUDED_FILE_EXTENSIONS = {
    ".bin",
    ".db",
    ".gif",
    ".jpeg",
    ".jpg",
    ".map",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".pickle",
    ".webp",
    ".xlsx",
    ".xls",
    ".tar.gz",
    ".zip",
    ".pyc",
    ".pyo",
}

IMPORTANT_DIRS = [
    "app/",
    "ml/",
    "ops/",
    "ui/",
    "proc_report/",
    "proc_autocode/",
    "proc_nlp/",
    "proc_registry/",
    "proc_schemas/",
    "schemas/",
    "configs/",
    "tests/",
]

IMPORTANT_FILES = [
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    "runtime.txt",
    "app/api/fastapi_app.py",
    "app/coder/application/coding_service.py",
    "app/registry/application/registry_service.py",
    "app/agents/contracts.py",
    "app/agents/run_pipeline.py",
    "docs/AGENTS.md",
    "docs/DEVELOPMENT.md",
    "docs/ARCHITECTURE.md",
    "docs/INSTALLATION.md",
    "docs/USER_GUIDE.md",
    ".claude/commands/phi-redactor.md",
    ".claude/commands/registry-data-prep.md",
]

DETAIL_DEFAULT_INCLUDE_DIRS = [
    "app/",
    "ml/",
    "ops/",
    "ui/",
    "proc_schemas/",
    "config/",
    "tests/",
    "docs/",
]

DETAIL_DEFAULT_INCLUDE_EXTENSIONS = {
    ".j2",
    ".jinja",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}


def get_git_info() -> tuple[str, str]:
    """Get current git branch and commit hash."""
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return branch, commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", "unknown"


def should_exclude_path(path: Path, repo_root: Path) -> bool:
    """Check if a path should be excluded."""
    try:
        rel_path = path.relative_to(repo_root)
    except ValueError:
        return True
    
    # Check if any part of the path matches excluded dirs
    parts = rel_path.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return True

    # Check file extension
    if path.is_file():
        for ext in EXCLUDED_FILE_EXTENSIONS:
            if str(path).endswith(ext):
                return True

    return False


def build_tree(root: Path, repo_root: Path, depth: int = 0) -> list[str]:
    """Build a directory tree structure matching the existing format."""
    lines = []
    indent = "  " * depth

    # Get all items in current directory
    try:
        items = sorted(
            [p for p in root.iterdir() if not should_exclude_path(p, repo_root)],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        return lines

    for item in items:
        # Get relative path from repo root
        rel_path = item.relative_to(repo_root)
        path_str = str(rel_path).replace("\\", "/")
        
        lines.append(f"{indent}- {path_str}/" if item.is_dir() else f"{indent}- {path_str}")

        if item.is_dir():
            lines.extend(build_tree(item, repo_root, depth + 1))

    return lines


def get_repo_tree(repo_root: Path) -> str:
    """Generate the repository tree structure."""
    # Start with the root directory name
    root_name = repo_root.name if repo_root.name else "."
    tree_lines = [f"- {root_name}/"]
    
    # Build the rest of the tree
    tree_lines.extend(build_tree(repo_root, repo_root, depth=1))
    
    return "\n".join(tree_lines)


def read_file_content(file_path: Path) -> str:
    """Read file content, handling encoding issues."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"# Error reading file: {e}"


def is_probably_text_file(file_path: Path, probe_bytes: int = 8192) -> bool:
    """
    Heuristic to avoid including binary/unreadable files.
    - Reject if NUL bytes are present in a small prefix.
    - Reject if UTF-8 strict decode fails on that prefix.
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(probe_bytes)
        if b"\x00" in chunk:
            return False
        try:
            chunk.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
        return True
    except Exception:
        return False


def looks_minified_text(text: str) -> bool:
    """Best-effort detection of minified bundles (very long lines / low newline density)."""
    if not text:
        return False
    max_line_len = max((len(line) for line in text.splitlines()), default=0)
    if max_line_len >= 2000:
        return True
    if len(text) >= 100_000:
        newline_ratio = text.count("\n") / max(1, len(text))
        if newline_ratio < 0.0005:
            return True
    return False


def iter_detail_candidate_files(
    repo_root: Path, include_dirs: Iterable[str], include_exts: Set[str]
) -> list[Path]:
    """Return repo-relative, filtered candidate files under include_dirs."""
    candidates: list[Path] = []
    for dir_str in include_dirs:
        dir_path = (repo_root / dir_str).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            continue

        for root, dirnames, filenames in os.walk(dir_path):
            root_path = Path(root)
            # Prune excluded dirs early
            dirnames[:] = [
                d for d in dirnames if not should_exclude_path(root_path / d, repo_root)
            ]

            for name in filenames:
                p = root_path / name
                if should_exclude_path(p, repo_root):
                    continue
                if include_exts and p.suffix.lower() not in include_exts:
                    continue
                candidates.append(p)

    # Stable ordering: prioritize active script trees and python, then smaller files.
    def sort_key(p: Path) -> tuple[int, int, int, str]:
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        in_scripts = 0 if (rel.startswith("ml/scripts/") or rel.startswith("ops/tools/")) else 1
        is_py = 0 if p.suffix.lower() == ".py" else 1
        try:
            size = p.stat().st_size
        except Exception:
            size = 10**9
        return (in_scripts, is_py, size, rel.lower())

    candidates.sort(key=sort_key)
    return candidates


def generate_gitingest_details(
    repo_root: Path,
    output_path: Path,
    include_dirs: Iterable[str],
    include_exts: Set[str],
    max_bytes: int,
    max_files: int,
    inline_mode: str,  # "none" | "curated" | "all"
) -> None:
    """Generate a more granular, text-only companion document for deeper LLM context."""
    print(f"Generating gitingest_details.md from {repo_root}...")

    branch, commit = get_git_info()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    candidates = iter_detail_candidate_files(repo_root, include_dirs, include_exts)

    included_manifest: list[tuple[str, int]] = []
    inlined: list[str] = []
    inlined_files = 0
    skipped: list[tuple[str, str]] = []

    def should_inline(rel_path: str, suffix: str) -> bool:
        if inline_mode == "none":
            return False
        if inline_mode == "all":
            return True
        # curated
        if rel_path.startswith("ml/scripts/") and suffix in {".py", ".md"}:
            return True
        if rel_path.startswith("ops/tools/") and suffix in {".py", ".md"}:
            return True
        if rel_path.startswith("app/") and suffix == ".py":
            return True
        if rel_path.startswith("proc_nlp/") and suffix == ".py":
            return True
        if rel_path.startswith("proc_schemas/") and suffix == ".py":
            return True
        return False

    for p in candidates:
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        try:
            size = p.stat().st_size
        except Exception:
            skipped.append((rel, "stat_failed"))
            continue

        if size > max_bytes:
            skipped.append((rel, f"too_large>{max_bytes}B"))
            continue
        if not p.is_file():
            skipped.append((rel, "not_a_file"))
            continue
        if not is_probably_text_file(p):
            skipped.append((rel, "binary_or_non_utf8"))
            continue

        included_manifest.append((rel, size))

        suffix = p.suffix.lower()
        if should_inline(rel, suffix):
            if inlined_files >= max_files:
                skipped.append((rel, f"inline_cap_reached>{max_files}"))
                continue

            text = read_file_content(p)
            if suffix in {".js", ".ts"} and looks_minified_text(text):
                skipped.append((rel, "minified_bundle"))
                continue

            inlined_files += 1
            inlined.extend(
                [
                    "---",
                    f"### `{rel}`",
                    f"- Size: `{size}` bytes",
                    "```",
                    text,
                    "```",
                    "",
                ]
            )

    content_parts = [
        "# Procedure Suite — gitingest (details)",
        "",
        f"Generated: `{timestamp}`",
        f"Git: `{branch}` @ `{commit}`",
        "",
        "## What this file is",
        "- A **second** document you can provide to an LLM when more detail is needed.",
        "- Focuses on **text-readable** code/docs and skips binaries, oversized files, and (best-effort) minified bundles.",
        "",
        "## Selection settings",
        f"- Include dirs: `{', '.join(include_dirs)}`",
        f"- Include extensions: `{'`, `'.join(sorted(include_exts))}`",
        f"- Max file size: `{max_bytes}` bytes",
        f"- Inline mode: `{inline_mode}`",
        f"- Inline cap (files): `{max_files}`",
        "",
        "## Manifest (filtered candidates)",
        "",
    ]

    if included_manifest:
        content_parts.append("```")
        for rel, size in included_manifest:
            content_parts.append(f"{size:>9}  {rel}")
        content_parts.append("```")
    else:
        content_parts.append("_No matching text files found under the provided include dirs._")

    content_parts.extend(["", "## Skipped (reason)", ""])
    if skipped:
        content_parts.append("```")
        for rel, reason in skipped[:500]:
            content_parts.append(f"{reason:>22}  {rel}")
        if len(skipped) > 500:
            content_parts.append(f"... and {len(skipped) - 500} more")
        content_parts.append("```")
    else:
        content_parts.append("_Nothing skipped._")

    content_parts.extend(["", "## Inlined file contents", ""])
    if inlined:
        content_parts.extend(inlined)
    else:
        content_parts.append("_Inline mode was `none`, or no files met the inline criteria._")

    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content_parts))
    print(f"✅ Successfully generated {output_path}")


def generate_gitingest(repo_root: Path, output_path: Path) -> None:
    """Generate the gitingest.md file."""
    print(f"Generating gitingest.md from {repo_root}...")

    # Get git info
    branch, commit = get_git_info()
    # Format timestamp with timezone (matching original format)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    # Generate repo tree
    print("Building repository tree...")
    repo_tree = get_repo_tree(repo_root)

    # Build the markdown content
    content_parts = [
        "# Procedure Suite — gitingest (curated)",
        "",
        f"Generated: `{timestamp}`",
        f"Git: `{branch}` @ `{commit}`",
        "",
        "## What this file is",
        "- A **token-budget friendly** snapshot of the repo **structure** + a curated set of **important files**.",
        "- Intended for LLM/context ingestion; excludes large artifacts (models, datasets, caches).",
        "",
        "## Exclusions (high level)",
        f"- Directories: `{', '.join(sorted(EXCLUDED_DIRS))}`",
        f"- File types: `{'`, `'.join(sorted(EXCLUDED_FILE_EXTENSIONS))}`",
        "",
        "## Repo tree (pruned)",
        "```",
        repo_tree,
        "```",
        "",
        "## Important directories (not inlined)",
    ]

    # Add important directories
    for dir_name in IMPORTANT_DIRS:
        content_parts.append(f"- `{dir_name}`")

    content_parts.extend([
        "",
        "## Important files (inlined)",
        "",
    ])

    # Add important files
    print("Inlining important files...")
    for file_path_str in IMPORTANT_FILES:
        file_path = repo_root / file_path_str
        if file_path.exists():
            print(f"  Reading {file_path_str}...")
            file_content = read_file_content(file_path)
            content_parts.extend([
                "---",
                f"### `{file_path_str}`",
                "```",
                file_content,
                "```",
                "",
            ])
        else:
            print(f"  Warning: {file_path_str} not found, skipping...")

    # Write the file
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content_parts))

    print(f"✅ Successfully generated {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate curated gitingest markdown documents for LLM ingestion."
    )
    parser.add_argument(
        "--output",
        default="gitingest.md",
        help="Path (relative to repo root) for the base curated gitingest output.",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Skip generating the base gitingest.md file.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Also generate a second, more granular details document.",
    )
    parser.add_argument(
        "--details-output",
        default="gitingest_details.md",
        help="Path (relative to repo root) for the details output.",
    )
    parser.add_argument(
        "--details-include",
        action="append",
        default=[],
        help="Directory (relative to repo root) to include in details scan. Can be repeated.",
    )
    parser.add_argument(
        "--details-max-bytes",
        type=int,
        default=200_000,
        help="Max size per manifested/inlined file in details doc (bytes).",
    )
    parser.add_argument(
        "--details-max-files",
        type=int,
        default=75,
        help="Max number of files to inline into details doc.",
    )
    parser.add_argument(
        "--details-inline",
        choices=["none", "curated", "all"],
        default="curated",
        help="Inline file contents mode for details doc.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    if not args.no_base:
        output_path = repo_root / args.output
        generate_gitingest(repo_root, output_path)

    if args.details:
        include_dirs = args.details_include or DETAIL_DEFAULT_INCLUDE_DIRS
        details_output = repo_root / args.details_output
        generate_gitingest_details(
            repo_root=repo_root,
            output_path=details_output,
            include_dirs=include_dirs,
            include_exts=DETAIL_DEFAULT_INCLUDE_EXTENSIONS,
            max_bytes=args.details_max_bytes,
            max_files=args.details_max_files,
            inline_mode=args.details_inline,
        )


if __name__ == "__main__":
    main()
