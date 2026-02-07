#!/usr/bin/env python3
"""Bootstrap the PHI redactor vendor bundle into the UI vendor directory.

Intended for Railway deployment where large model assets live in S3.

Env vars:
- PHI_REDACTOR_VENDOR_BUNDLE_S3_URI: s3://<bucket>/<key>/bundle.tar.gz
  - Fallback: PHI_REDACTOR_VENDOR_BUNDLE_S3_URI_ONNX
- PHI_REDACTOR_VENDOR_DIR: destination directory
  (default: ui/static/phi_redactor/vendor/phi_distilbert_ner_quant)
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOOTSTRAP_STATE_FILENAME = ".bootstrap_state.json"


@dataclass(frozen=True)
class BootstrapResult:
    vendor_dir: Path
    downloaded: bool
    manifest: dict[str, Any]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri}")
    no_scheme = uri[len("s3://") :]
    bucket, _, key = no_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3:// URI: {uri}")
    return bucket, key


def _download_s3_key(bucket: str, key: str, dest: Path) -> None:
    import boto3  # type: ignore

    client = boto3.client("s3")
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=dest.name, dir=str(dest.parent), delete=False) as tf:
            tmp_path = Path(tf.name)
        client.download_file(bucket, key, str(tmp_path))
        os.replace(str(tmp_path), str(dest))
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _extract_tarball_to_dir(tar_gz_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz_path, "r:gz") as tf:
        tf.extractall(path=dest_dir)


def _flatten_single_root(extracted_dir: Path) -> Path:
    children = [p for p in extracted_dir.iterdir() if p.name not in (".DS_Store",)]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extracted_dir


def _replace_tree(src_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir)


def _normalize_model_layout(root: Path) -> None:
    """Ensure transformers.js-compatible layout (onnx/model.onnx)."""
    onnx_dir = root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Preferred destination
    dest = onnx_dir / "model.onnx"
    if dest.exists():
        return

    candidates = [
        onnx_dir / "model_quantized.onnx",
        root / "model.onnx",
        root / "model_quantized.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, dest)
            return

    raise FileNotFoundError("Unable to locate model.onnx or model_quantized.onnx in bundle")


def _read_bootstrap_state(dest_dir: Path) -> dict[str, Any]:
    path = dest_dir / BOOTSTRAP_STATE_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_bootstrap_state(dest_dir: Path, state: dict[str, Any]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / BOOTSTRAP_STATE_FILENAME
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _configured_uri() -> str | None:
    return os.getenv("PHI_REDACTOR_VENDOR_BUNDLE_S3_URI") or os.getenv(
        "PHI_REDACTOR_VENDOR_BUNDLE_S3_URI_ONNX"
    )


def _vendor_dir() -> Path:
    return Path(
        os.getenv(
            "PHI_REDACTOR_VENDOR_DIR",
            "ui/static/phi_redactor/vendor/phi_distilbert_ner_quant",
        )
    )


def ensure_phi_redactor_vendor_bundle() -> BootstrapResult:
    # Local dev convenience: load .env if present.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    uri = (_configured_uri() or "").strip()
    vendor_dir = _vendor_dir()

    if not uri:
        return BootstrapResult(vendor_dir=vendor_dir, downloaded=False, manifest={})

    if not uri.endswith(".tar.gz"):
        raise ValueError(
            "PHI_REDACTOR_VENDOR_BUNDLE_S3_URI must point to a .tar.gz bundle "
            f"(got: {uri})."
        )

    state = _read_bootstrap_state(vendor_dir)
    state_uri = (state.get("configured_source_uri") or "").strip()

    existing_config = vendor_dir / "config.json"
    if existing_config.exists() and state_uri == uri:
        _normalize_model_layout(vendor_dir)
        try:
            manifest = json.loads((vendor_dir / "manifest.json").read_text())
        except Exception:
            manifest = {}
        return BootstrapResult(vendor_dir=vendor_dir, downloaded=False, manifest=manifest)

    bucket, key = _parse_s3_uri(uri)

    with tempfile.TemporaryDirectory(prefix="phi_redactor_vendor_") as td:
        td_path = Path(td)
        tar_path = td_path / "bundle.tar.gz"
        _download_s3_key(bucket=bucket, key=key, dest=tar_path)

        extracted = td_path / "extracted"
        _extract_tarball_to_dir(tar_path, extracted)
        root = _flatten_single_root(extracted)
        _normalize_model_layout(root)

        manifest: dict[str, Any] = {
            "bundle_type": "phi_redactor_vendor",
            "model_backend": "onnx",
            "source_uri": uri,
            "source_type": "s3_tarball",
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

        _replace_tree(root, vendor_dir)

    _write_bootstrap_state(
        vendor_dir,
        {
            "configured_source_uri": uri,
        },
    )
    return BootstrapResult(vendor_dir=vendor_dir, downloaded=True, manifest=manifest)


def main() -> int:
    result = ensure_phi_redactor_vendor_bundle()
    uri = _configured_uri()
    if uri:
        action = "downloaded" if result.downloaded else "cached"
        print(
            f"[bootstrap_phi_redactor_vendor_bundle] {action} PHI vendor bundle into {result.vendor_dir} "
            f"(source={uri})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
