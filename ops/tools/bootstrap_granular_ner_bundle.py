#!/usr/bin/env python3
"""Bootstrap the granular NER ONNX bundle into a local runtime directory.

This is intended for Railway deployment where large ONNX artifacts are stored in S3
and fetched at container start.

Env vars:
- GRANULAR_NER_BUNDLE_S3_URI_ONNX: s3://<bucket>/<key>/bundle.tar.gz
  - Fallback: GRANULAR_NER_BUNDLE_S3_URI
- GRANULAR_NER_RUNTIME_DIR: where to unpack the bundle (default: data/models/granular_ner_runtime)

On success, sets:
- GRANULAR_NER_MODEL_DIR=<GRANULAR_NER_RUNTIME_DIR>
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
    runtime_dir: Path
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
    # Lazy import so local dev can run without boto3 unless bootstrap is enabled.
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


def _read_bootstrap_state(runtime_dir: Path) -> dict[str, Any]:
    path = runtime_dir / BOOTSTRAP_STATE_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_bootstrap_state(runtime_dir: Path, state: dict[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / BOOTSTRAP_STATE_FILENAME
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _configured_uri() -> str | None:
    return os.getenv("GRANULAR_NER_BUNDLE_S3_URI_ONNX") or os.getenv("GRANULAR_NER_BUNDLE_S3_URI")


def _runtime_dir() -> Path:
    return Path(os.getenv("GRANULAR_NER_RUNTIME_DIR", "data/models/granular_ner_runtime"))


def ensure_granular_ner_bundle() -> BootstrapResult:
    uri = (_configured_uri() or "").strip()
    runtime_dir = _runtime_dir()

    if not uri:
        # Nothing to do; still expose resolved model dir if it's already present.
        if runtime_dir.exists():
            os.environ.setdefault("GRANULAR_NER_MODEL_DIR", str(runtime_dir))
        return BootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest={})

    if not uri.endswith(".tar.gz"):
        raise ValueError(
            "GRANULAR_NER_BUNDLE_S3_URI_ONNX must point to a .tar.gz bundle "
            f"(got: {uri})."
        )

    existing_manifest_path = runtime_dir / "manifest.json"
    state = _read_bootstrap_state(runtime_dir)
    state_uri = (state.get("configured_source_uri") or "").strip()

    if existing_manifest_path.exists() and state_uri == uri:
        # Bundle URIs are immutable; matching state is sufficient.
        os.environ["GRANULAR_NER_MODEL_DIR"] = str(runtime_dir)
        try:
            manifest = json.loads(existing_manifest_path.read_text())
        except Exception:
            manifest = {}
        return BootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest=manifest)

    bucket, key = _parse_s3_uri(uri)

    with tempfile.TemporaryDirectory(prefix="granular_ner_bundle_") as td:
        td_path = Path(td)
        tar_path = td_path / "bundle.tar.gz"
        _download_s3_key(bucket=bucket, key=key, dest=tar_path)

        extracted = td_path / "extracted"
        _extract_tarball_to_dir(tar_path, extracted)
        root = _flatten_single_root(extracted)

        # Add/override minimal manifest for provenance.
        manifest: dict[str, Any] = {
            "bundle_type": "granular_ner",
            "model_backend": "onnx",
            "source_uri": uri,
            "source_type": "s3_tarball",
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

        _replace_tree(root, runtime_dir)

    _write_bootstrap_state(
        runtime_dir,
        {
            "configured_source_uri": uri,
        },
    )
    os.environ["GRANULAR_NER_MODEL_DIR"] = str(runtime_dir)
    return BootstrapResult(runtime_dir=runtime_dir, downloaded=True, manifest=manifest)


def main() -> int:
    result = ensure_granular_ner_bundle()
    uri = _configured_uri()
    if uri:
        action = "downloaded" if result.downloaded else "cached"
        print(
            f"[bootstrap_granular_ner_bundle] {action} granular NER bundle into {result.runtime_dir} "
            f"(source={uri})"
        )
        print(f"[bootstrap_granular_ner_bundle] GRANULAR_NER_MODEL_DIR={os.environ.get('GRANULAR_NER_MODEL_DIR')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

