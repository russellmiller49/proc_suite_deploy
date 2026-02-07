#!/usr/bin/env python3
"""Check ONNX graph inputs for transformers.js compatibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REQUIRED = ("input_ids", "attention_mask")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Validate that an ONNX model exposes required graph inputs (e.g., input_ids, attention_mask)."
    )
    ap.add_argument("onnx_path", type=Path, help="Path to an ONNX model file (e.g., onnx/model.onnx)")
    ap.add_argument(
        "--require",
        action="append",
        default=list(DEFAULT_REQUIRED),
        help="Input name to require (repeatable). Default: input_ids + attention_mask",
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    onnx_path: Path = args.onnx_path
    required: list[str] = args.require

    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        return 2

    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError as exc:
        print("ERROR: `onnx` is not installed. Install with: pip install onnx", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    m = onnx.load(str(onnx_path))
    input_names = [i.name for i in m.graph.input]

    print(f"ONNX: {onnx_path}")
    print(f"Inputs ({len(input_names)}): {', '.join(input_names) if input_names else '(none)'}")

    missing = [name for name in required if name not in input_names]
    if missing:
        print(
            f"FAIL: Missing required inputs: {missing}. Found: {input_names}",
            file=sys.stderr,
        )
        return 1

    print("OK: Required inputs present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

