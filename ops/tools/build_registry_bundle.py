#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _git_short_sha(repo_dir: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_dir))
            .decode()
            .strip()
        )
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _copytree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _is_zone_identifier(path: Path) -> bool:
    return path.name.endswith(":Zone.Identifier")


def build_bundle(src_dir: Path, out_dir: Path, version: str, backend: str) -> tuple[Path, Path]:
    src_dir = src_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "config.json",
        "thresholds.json",
    ]
    for rel in required_files:
        if not (src_dir / rel).exists():
            raise FileNotFoundError(f"Missing {rel} in {src_dir}")

    tokenizer_dir = src_dir / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Missing tokenizer/ in {src_dir}")

    weights = None
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        if (src_dir / candidate).exists():
            weights = src_dir / candidate
            break
    if not weights:
        raise FileNotFoundError("Missing model weights (model.safetensors or pytorch_model.bin)")

    # Labels: prefer label_order.json, otherwise registry_label_fields.json
    label_order_path = src_dir / "label_order.json"
    registry_label_fields_path = src_dir / "registry_label_fields.json"
    if label_order_path.exists():
        labels = _read_json(label_order_path)
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise ValueError("label_order.json must be a JSON string list")
    elif registry_label_fields_path.exists():
        labels = _read_json(registry_label_fields_path)
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise ValueError("registry_label_fields.json must be a JSON string list")
    else:
        raise FileNotFoundError("Missing label_order.json or registry_label_fields.json")

    manifest: dict[str, Any] = {
        "model_version": version,
        "model_backend": backend,
        "created_at": _utc_now_iso(),
        "label_count": len(labels),
        "source_dir": str(src_dir),
    }

    repo_root = Path(__file__).resolve().parents[2]
    sha = _git_short_sha(repo_root)
    if sha:
        manifest["repo_commit_sha"] = sha

    with tempfile.TemporaryDirectory(prefix="registry_bundle_stage_") as td:
        stage = Path(td)

        # Copy core files
        shutil.copy2(src_dir / "config.json", stage / "config.json")
        shutil.copy2(weights, stage / weights.name)
        shutil.copy2(src_dir / "thresholds.json", stage / "thresholds.json")

        # Copy labels (always include both for convenience)
        (stage / "label_order.json").write_text(json.dumps(labels, indent=2, sort_keys=False) + "\n")
        (stage / "registry_label_fields.json").write_text(
            json.dumps(labels, indent=2, sort_keys=False) + "\n"
        )

        # Copy tokenizer directory
        _copytree(tokenizer_dir, stage / "tokenizer")

        # Strip Windows ADS marker files if they exist
        for p in list(stage.rglob("*")):
            if _is_zone_identifier(p):
                p.unlink(missing_ok=True)

        # Write manifest
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        # Build tarball
        tar_path = out_dir / "bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for p in sorted(stage.rglob("*")):
                if p.is_dir():
                    continue
                rel = p.relative_to(stage)
                tf.add(p, arcname=str(rel))

        # Also write manifest.json next to tarball
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return tar_path, out_dir / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deployable registry model bundle tarball.")
    parser.add_argument("--src", required=True, help="Source directory containing model artifacts")
    parser.add_argument("--out-dir", default="dist/registry_bundle", help="Output directory")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx"], help="Bundle backend")
    parser.add_argument("--version", default="", help="Model version string (used in manifest and S3 path)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out_dir)
    version = args.version.strip() or f"local-{_git_short_sha(Path(__file__).resolve().parents[2]) or 'unknown'}"
    backend = args.backend

    tar_path, manifest_path = build_bundle(src_dir=src_dir, out_dir=out_dir, version=version, backend=backend)

    print(f"Built: {tar_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Version: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

