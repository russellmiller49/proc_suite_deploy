from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all python scripts in data/granular annotations/Python_update_scripts/"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failing script (default: continue).",
    )
    parser.add_argument(
        "--pattern",
        default="note_*.py",
        help="Glob pattern to select scripts (default: note_*.py).",
    )
    parser.add_argument(
        "--failure-report",
        type=Path,
        default=Path("reports/python_update_scripts_failures.json"),
        help="Where to write failure report JSON (default: reports/python_update_scripts_failures.json).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "data" / "granular annotations" / "Python_update_scripts"

    if not scripts_dir.exists():
        raise SystemExit(f"Scripts directory not found: {scripts_dir}")

    scripts = sorted(scripts_dir.glob(args.pattern))
    if not scripts:
        print(f"No scripts matched {args.pattern!r} in {scripts_dir}")
        return 0

    print(f"Found {len(scripts)} scripts in: {scripts_dir}")
    print(f"Python: {sys.executable}")
    print("-" * 60)

    failures: list[Path] = []
    failure_details: list[dict] = []
    start_all = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    for idx, script_path in enumerate(scripts, start=1):
        rel = script_path.relative_to(repo_root)
        print(f"[{idx}/{len(scripts)}] Running {rel} ...", flush=True)

        # Run each script in repo root so relative-path scripts behave consistently.
        start_one = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
        )
        elapsed_one = time.time() - start_one

        # Preserve script output (but printed after the script finishes).
        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stderr.write("\n")

        if result.returncode != 0:
            failures.append(script_path)
            failure_details.append(
                {
                    "script": str(rel),
                    "returncode": int(result.returncode),
                    "elapsed_seconds": round(elapsed_one, 6),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            print(f"  -> FAILED (exit {result.returncode})")
            if args.fail_fast:
                break

    elapsed = time.time() - start_all
    print("-" * 60)
    print(f"Done in {elapsed:.2f}s")
    print(f"Failed: {len(failures)}")
    if failures:
        for p in failures:
            print(f" - {p.relative_to(repo_root)}")

        # Write a structured failure report for debugging.
        report_path = args.failure_report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": started_at,
            "elapsed_seconds": round(elapsed, 6),
            "python": sys.executable,
            "scripts_dir": str(scripts_dir.relative_to(repo_root)),
            "pattern": args.pattern,
            "total_scripts": len(scripts),
            "failed_scripts": len(failure_details),
            "failures": failure_details,
        }
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Failure report written to: {report_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

