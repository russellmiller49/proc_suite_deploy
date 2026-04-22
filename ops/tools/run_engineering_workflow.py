#!/usr/bin/env python3
"""Run the deterministic engineering workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.engineering_workflow import EngineeringWorkflowRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True, help="High-level engineering task for the workflow.")
    parser.add_argument("--session-id", help="Resume or force a specific session id.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing session manifest.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow execution even if the repo is already dirty.")
    parser.add_argument("--enable-tracing", action="store_true", help="Enable Agents SDK tracing.")
    parser.add_argument("--artifact-root", type=Path, help="Override the default artifact root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = EngineeringWorkflowRunner(repo_root=ROOT)
    result = runner.run(
        goal=args.goal,
        session_id=args.session_id,
        allow_dirty=args.allow_dirty,
        enable_tracing=args.enable_tracing,
        artifact_root=args.artifact_root,
        resume=args.resume,
    )
    print(f"Workflow status: {result.status}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Report: {result.report_path}")
    print(f"Handoff: {result.handoff_path}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

