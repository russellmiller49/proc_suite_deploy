#!/usr/bin/env python3
"""
Index the Procedure Suite repository into a JSONL file.

Why this exists:
- `docs/REPO_INDEX.md` is a curated "high-signal map" of key entrypoints.
- This script produces a *mechanical, full-repo* index (file inventory + metadata)
  for tooling / search / diff / inspection.

Default behavior:
- Indexes **tracked files** + **unignored untracked files** (via git), which is
  typically what you want when you say "full repo" (and avoids `.gitignore` noise).
- Skips hashing/reading for large files by default (configurable).

Usage:
  python ops/tools/index_repo.py
  python ops/tools/index_repo.py --out repo_index_all.jsonl
  python ops/tools/index_repo.py --include-content --max-content-bytes 262144
  python ops/tools/index_repo.py --mode walk   # ignore git, walk filesystem
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


DEFAULT_SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".idea",
    ".vscode",
}

# These are *repo-relative* path prefixes to exclude from indexing by default.
# Rationale: these folders tend to contain large corpora / training artifacts and
# can overwhelm indexes (and are often not useful for code navigation).
DEFAULT_SKIP_PREFIXES = [
    "data/knowledge/golden_extractions",
    "data/knowledge/golden_extractions_final",
    "data/knowledge/golden_extractions_scrubbed",
    "data/knowledge/golden_registry_v3",
    "data/knowledge/patient_note_texts",
    "data/knowledge/patient_note_texts_complete",
    "data/granular annotations/Additional_notes",
    "data/granular annotations/Empty_python_scripts_updated",
    "data/granular annotations/notes_text",
    "data/granular annotations/phase0_excels",
    "data/granular annotations/python scripts",
    "data/granular annotations/Python_update_scripts",
    "data/granular annotations/python_update_scripts_complete",
    "example",
    "format example",
]


@dataclass(frozen=True)
class GitInfo:
    branch: Optional[str]
    commit: Optional[str]


def _run_git(repo_root: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _get_git_root(cwd: Path) -> Optional[Path]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    root = proc.stdout.strip()
    if not root:
        return None
    return Path(root)


def _get_git_info(repo_root: Path) -> GitInfo:
    branch = None
    commit = None
    rc, out, _ = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0:
        branch = out.strip() or None
    rc, out, _ = _run_git(repo_root, ["rev-parse", "--short", "HEAD"])
    if rc == 0:
        commit = out.strip() or None
    return GitInfo(branch=branch, commit=commit)


def _iter_files_git(repo_root: Path) -> Iterator[Path]:
    """
    Yield repo-relative paths for:
    - tracked files
    - unignored untracked files
    """
    def _git_ls(args: list[str]) -> list[str]:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", *args],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
        raw = proc.stdout
        if not raw:
            return []
        return [p for p in raw.decode("utf-8", errors="replace").split("\x00") if p]

    paths = set(_git_ls([]))
    paths.update(_git_ls(["--others", "--exclude-standard"]))

    for rel in sorted(paths):
        # Normalize as Path (posix separator from git)
        yield Path(rel)


def _iter_files_walk(repo_root: Path, skip_dir_names: set[str]) -> Iterator[Path]:
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip_dir_names]
        for name in files:
            full = Path(root) / name
            try:
                rel = full.relative_to(repo_root)
            except Exception:
                continue
            yield rel


def _normalize_skip_prefixes(prefixes: Iterable[str]) -> list[str]:
    """
    Normalize skip prefixes to posix paths without leading './' and without
    trailing slashes (comparison is done with exact match or prefix + '/').
    """
    out: list[str] = []
    for p in prefixes:
        s = p.strip().replace("\\", "/")
        if not s:
            continue
        if s.startswith("./"):
            s = s[2:]
        s = s.rstrip("/")
        if s:
            out.append(s)
    # Ensure deterministic behavior if caller passes duplicates
    return sorted(set(out))


def _should_skip_relpath(rel_posix: str, skip_prefixes: list[str]) -> bool:
    rel_posix = rel_posix.lstrip("./")
    for prefix in skip_prefixes:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return True
    return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_head_bytes(path: Path, n: int) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def _looks_text(sample: bytes) -> bool:
    # Fast heuristic: NUL bytes usually indicate binary.
    if b"\x00" in sample:
        return False
    # If it decodes as UTF-8 (or mostly), treat as text.
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _count_lines_utf8(path: Path, max_bytes: int) -> Optional[int]:
    """
    Count lines for reasonably sized text files only.
    Returns None when file is too large or non-utf8-ish.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return None
    if size > max_bytes:
        return None
    try:
        # Stream decode to avoid reading huge files (still bounded by max_bytes).
        count = 0
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                if not chunk:
                    break
                count += chunk.count(b"\n")
        return count + 1 if size > 0 else 0
    except Exception:
        return None


def _safe_rel(path: Path) -> str:
    return path.as_posix()


def build_index(
    repo_root: Path,
    out_path: Path,
    *,
    mode: str,
    max_hash_bytes: int,
    include_content: bool,
    max_content_bytes: int,
    max_line_count_bytes: int,
    skip_dir_names: set[str],
    skip_prefixes: list[str],
    max_files: Optional[int],
) -> None:
    git_info = _get_git_info(repo_root) if mode == "git" else GitInfo(None, None)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    if mode == "git":
        files_iter: Iterable[Path] = _iter_files_git(repo_root)
    elif mode == "walk":
        files_iter = _iter_files_walk(repo_root, skip_dir_names=skip_dir_names)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    hashed = 0
    with out_path.open("w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "type": "repo_meta",
                    "repo_root": str(repo_root),
                    "mode": mode,
                    "git_branch": git_info.branch,
                    "git_commit": git_info.commit,
                    "started_at": started_at,
                    "max_hash_bytes": max_hash_bytes,
                    "include_content": include_content,
                    "max_content_bytes": max_content_bytes,
                    "max_line_count_bytes": max_line_count_bytes,
                    "skip_prefixes": skip_prefixes,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        for rel in files_iter:
            if max_files is not None and total >= max_files:
                break

            full = repo_root / rel
            rel_str = _safe_rel(rel)

            if _should_skip_relpath(rel_str, skip_prefixes):
                continue

            try:
                st = full.lstat()
            except FileNotFoundError:
                continue

            mime, _ = mimetypes.guess_type(str(full))
            rec: dict[str, object] = {
                "type": "file",
                "path": rel_str,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "is_symlink": full.is_symlink(),
                "mime": mime or "application/octet-stream",
            }

            # Text/binary heuristic (cheap sample)
            is_text = False
            try:
                sample = _read_head_bytes(full, 8192) if st.st_size > 0 else b""
                is_text = _looks_text(sample)
            except Exception:
                is_text = False
            rec["is_text"] = is_text

            if is_text:
                rec["line_count"] = _count_lines_utf8(full, max_bytes=max_line_count_bytes)

            if st.st_size <= max_hash_bytes and not full.is_symlink():
                try:
                    rec["sha256"] = _sha256_file(full)
                    hashed += 1
                except Exception:
                    rec["sha256"] = None

            if include_content and is_text and st.st_size <= max_content_bytes:
                try:
                    content = full.read_text(encoding="utf-8", errors="replace")
                    rec["content"] = content
                except Exception:
                    rec["content"] = None

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

        out.write(
            json.dumps(
                {
                    "type": "repo_summary",
                    "total_files_written": total,
                    "hashed_files": hashed,
                    "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"Wrote {total:,} file records to {out_path}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Index the proc_suite repo into JSONL.")
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root (default: detected git root from current working directory).",
    )
    parser.add_argument(
        "--out",
        default="repo_index_all.jsonl",
        help="Output JSONL path (default: repo_root/repo_index_all.jsonl).",
    )
    parser.add_argument(
        "--mode",
        choices=["git", "walk"],
        default="git",
        help="File discovery mode: 'git' (tracked+unignored) or 'walk' (filesystem).",
    )
    parser.add_argument(
        "--max-hash-bytes",
        type=int,
        default=10_000_000,
        help="Only hash files <= this size in bytes (default: 10,000,000).",
    )
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Include full UTF-8 content for small text files (off by default).",
    )
    parser.add_argument(
        "--max-content-bytes",
        type=int,
        default=128_000,
        help="Only include content for text files <= this size (default: 128,000).",
    )
    parser.add_argument(
        "--max-line-count-bytes",
        type=int,
        default=2_000_000,
        help="Only compute line counts for text files <= this size (default: 2,000,000).",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        help="Directory name to skip (can be repeated). Only applies to --mode walk.",
    )
    parser.add_argument(
        "--skip-prefix",
        action="append",
        default=[],
        help="Repo-relative path prefix to skip (can be repeated). Applies to all modes.",
    )
    parser.add_argument(
        "--no-default-skip-prefixes",
        action="store_true",
        help="Disable the built-in skip prefixes list.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for quick runs/debugging.",
    )

    args = parser.parse_args(argv)

    repo_root: Optional[Path]
    if args.root:
        repo_root = Path(args.root).resolve()
    else:
        repo_root = _get_git_root(Path.cwd())

    if not repo_root or not repo_root.exists():
        print("ERROR: Could not determine repo root. Pass --root explicitly.", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    skip_dir_names = set(DEFAULT_SKIP_DIR_NAMES)
    skip_dir_names.update(args.skip_dir)

    skip_prefixes: list[str] = []
    if not args.no_default_skip_prefixes:
        skip_prefixes.extend(DEFAULT_SKIP_PREFIXES)
    skip_prefixes.extend(args.skip_prefix)
    skip_prefixes = _normalize_skip_prefixes(skip_prefixes)

    try:
        build_index(
            repo_root=repo_root,
            out_path=out_path,
            mode=args.mode,
            max_hash_bytes=args.max_hash_bytes,
            include_content=args.include_content,
            max_content_bytes=args.max_content_bytes,
            max_line_count_bytes=args.max_line_count_bytes,
            skip_dir_names=skip_dir_names,
            skip_prefixes=skip_prefixes,
            max_files=args.max_files,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
