#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


if not _truthy_env("PROCSUITE_SKIP_DOTENV"):
    load_dotenv(override=False)

from app.registry.model_runtime import (  # noqa: E402
    get_registry_runtime_dir,
    resolve_model_backend,
    verify_registry_runtime_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify registry runtime model bundle artifacts."
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "auto"],
        default=None,
        help="Backend to validate (defaults to MODEL_BACKEND resolution).",
    )
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help="Override registry runtime directory.",
    )
    args = parser.parse_args()

    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else get_registry_runtime_dir()
    backend = args.backend or resolve_model_backend()

    try:
        warnings = verify_registry_runtime_bundle(
            backend=backend,
            runtime_dir=runtime_dir,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OK: registry runtime bundle valid (backend={backend}, runtime_dir={runtime_dir})")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
