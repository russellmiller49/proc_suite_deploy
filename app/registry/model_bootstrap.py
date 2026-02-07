from __future__ import annotations

import os
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.common.logger import get_logger
from app.registry.model_runtime import get_registry_runtime_dir, read_registry_manifest, resolve_model_backend

logger = get_logger("registry.model_bootstrap")


@dataclass(frozen=True)
class BundleBootstrapResult:
    runtime_dir: Path
    downloaded: bool
    manifest: dict[str, Any]


BOOTSTRAP_STATE_FILENAME = ".bootstrap_state.json"


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


def _download_s3_to_path(uri: str, dest: Path) -> None:
    bucket, key = _parse_s3_uri(uri)
    _download_s3_key(bucket=bucket, key=key, dest=dest)


def _download_s3_prefix(bucket: str, prefix: str, dest_dir: Path) -> None:
    """Download all objects under an S3 prefix into dest_dir, preserving relative paths."""
    # Lazy import so local dev can run without boto3 unless bootstrap is enabled.
    import boto3  # type: ignore

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key or key.endswith("/"):
                continue
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            if not rel:
                continue
            _download_s3_key(bucket=bucket, key=key, dest=dest_dir / rel)


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


def _s3_uri_for_backend(backend: str) -> str | None:
    if backend == "pytorch":
        return os.getenv("MODEL_BUNDLE_S3_URI_PYTORCH") or os.getenv("MODEL_BUNDLE_S3_URI")
    if backend == "onnx":
        return os.getenv("MODEL_BUNDLE_S3_URI_ONNX") or os.getenv("MODEL_BUNDLE_S3_URI")
    return None


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


def _resolve_latest_onnx_source_uri(configured_uri: str) -> str | None:
    """Resolve a broad S3 prefix (e.g. bucket root) to the newest ONNX export prefix."""
    bucket, key = _parse_s3_uri(configured_uri)

    # If a specific file URI is provided, resolve to its parent directory.
    if key.endswith(".onnx"):
        key = key.rsplit("/", 1)[0] + "/"

    prefix = key if key.endswith("/") else f"{key}/"

    # Lazy import so local dev can run without boto3 unless bootstrap is enabled.
    import boto3  # type: ignore

    client = boto3.client("s3")

    # Fast-path: ONNX model exists directly under the provided prefix.
    for candidate in (
        f"{prefix}registry_model_int8.onnx",
        f"{prefix}model_int8.onnx",
        f"{prefix}model.onnx",
    ):
        try:
            client.head_object(Bucket=bucket, Key=candidate)
            return f"s3://{bucket}/{prefix}"
        except Exception:
            continue

    # Otherwise, search for the newest model_int8 export under the prefix.
    paginator = client.get_paginator("list_objects_v2")
    newest_key: str | None = None
    newest_ts = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            obj_key = obj.get("Key")
            if not obj_key:
                continue
            if not (obj_key.endswith("/model_int8.onnx") or obj_key.endswith("/registry_model_int8.onnx")):
                continue
            ts = obj.get("LastModified")
            if newest_ts is None or (ts is not None and ts > newest_ts):
                newest_ts = ts
                newest_key = obj_key
    if not newest_key:
        return None
    resolved_prefix = newest_key.rsplit("/", 1)[0] + "/"
    return f"s3://{bucket}/{resolved_prefix}"


def apply_resolved_registry_paths(backend: str, runtime_dir: Path | None = None) -> None:
    runtime_dir = runtime_dir or get_registry_runtime_dir()
    os.environ["REGISTRY_RUNTIME_DIR"] = str(runtime_dir)

    tokenizer_path = runtime_dir / "tokenizer"
    os.environ["REGISTRY_TOKENIZER_PATH"] = str(tokenizer_path)
    os.environ["REGISTRY_THRESHOLDS_PATH"] = str(runtime_dir / "thresholds.json")
    os.environ["REGISTRY_LABEL_FIELDS_PATH"] = str(runtime_dir / "registry_label_fields.json")

    if backend == "pytorch":
        os.environ["REGISTRY_MODEL_DIR"] = str(runtime_dir)
    elif backend == "onnx":
        os.environ["REGISTRY_ONNX_MODEL_PATH"] = str(runtime_dir / "registry_model_int8.onnx")


def _ensure_registry_bundle_from_s3_prefix(uri: str, backend: str, runtime_dir: Path) -> dict[str, Any]:
    """Build a runtime bundle by downloading required artifacts from an S3 prefix."""
    configured_uri = uri
    bucket, key = _parse_s3_uri(uri)
    # If a specific file URI is provided, treat its parent directory as the prefix.
    if key.endswith(".onnx"):
        key = key.rsplit("/", 1)[0] + "/"
    prefix = key if key.endswith("/") else f"{key}/"

    # Lazy import so local dev can run without boto3 unless bootstrap is enabled.
    import boto3  # type: ignore

    client = boto3.client("s3")

    def _pick_key(candidates: list[str]) -> str | None:
        """Prefer the first candidate (file or prefix) that exists."""
        for candidate in candidates:
            try:
                if candidate.endswith("/"):
                    resp = client.list_objects_v2(Bucket=bucket, Prefix=candidate, MaxKeys=1)
                    if resp.get("Contents"):
                        return candidate
                else:
                    client.head_object(Bucket=bucket, Key=candidate)
                    return candidate
            except Exception:
                continue
        return None

    def _resolve_run_prefix_for_onnx(search_prefix: str) -> str | None:
        """If the provided prefix is too broad, pick the newest ONNX export under it."""
        paginator = client.get_paginator("list_objects_v2")
        newest_key: str | None = None
        newest_ts = None
        for page in paginator.paginate(Bucket=bucket, Prefix=search_prefix):
            for obj in page.get("Contents", []):
                obj_key = obj.get("Key")
                if not obj_key:
                    continue
                if not (obj_key.endswith("/model_int8.onnx") or obj_key.endswith("/registry_model_int8.onnx")):
                    continue
                ts = obj.get("LastModified")
                if newest_ts is None or (ts is not None and ts > newest_ts):
                    newest_ts = ts
                    newest_key = obj_key
        if not newest_key:
            return None
        return newest_key.rsplit("/", 1)[0] + "/"

    # Layout heuristics:
    # - release bundles may store tokenizer/ and label_order.json at the root.
    # - training run exports in `classifiers/` often store them under `model/`.
    if backend == "onnx":
        # Support pointing MODEL_BUNDLE_S3_URI_ONNX at a higher-level prefix such as
        # `s3://procedure-suite-models/classifiers/` by discovering the latest run folder.
        direct_model_key = _pick_key(
            [
                f"{prefix}registry_model_int8.onnx",
                f"{prefix}model_int8.onnx",
                f"{prefix}model.onnx",
            ]
        )
        if not direct_model_key:
            resolved = _resolve_run_prefix_for_onnx(prefix)
            if resolved:
                prefix = resolved

    resolved_uri = f"s3://{bucket}/{prefix}"

    tokenizer_prefix = _pick_key([f"{prefix}tokenizer/", f"{prefix}model/tokenizer/"])
    label_key = _pick_key(
        [
            f"{prefix}label_order.json",
            f"{prefix}registry_label_fields.json",
            f"{prefix}model/label_order.json",
        ]
    )

    if backend == "onnx":
        model_key = _pick_key(
            [
                f"{prefix}registry_model_int8.onnx",
                f"{prefix}model_int8.onnx",
                f"{prefix}model.onnx",
            ]
        )
        thresholds_key = _pick_key(
            [
                f"{prefix}thresholds_int8.json",
                f"{prefix}thresholds.json",
                f"{prefix}model/thresholds.json",
            ]
        )
        if not model_key:
            raise FileNotFoundError(f"No ONNX model found under {uri}")
        if not tokenizer_prefix:
            raise FileNotFoundError(f"No tokenizer directory found under {uri}")
        if not label_key:
            raise FileNotFoundError(f"No label order file found under {uri}")

        # Stage into temp dir then swap into place.
        with tempfile.TemporaryDirectory(prefix="registry_prefix_bundle_") as td:
            td_path = Path(td)
            bundle_root = td_path / "bundle"
            bundle_root.mkdir(parents=True, exist_ok=True)

            _download_s3_key(bucket=bucket, key=model_key, dest=bundle_root / "registry_model_int8.onnx")
            _download_s3_key(bucket=bucket, key=label_key, dest=bundle_root / "label_order.json")

            # Provide both filenames used by runtime code.
            shutil.copyfile(bundle_root / "label_order.json", bundle_root / "registry_label_fields.json")

            if thresholds_key:
                _download_s3_key(bucket=bucket, key=thresholds_key, dest=bundle_root / "thresholds.json")
            else:
                # Allow the ONNX predictor to fall back to 0.5 thresholds.
                (bundle_root / "thresholds.json").write_text("{}")

            _download_s3_prefix(bucket=bucket, prefix=tokenizer_prefix, dest_dir=bundle_root / "tokenizer")

            # Minimal manifest for provenance.
            model_version = prefix.strip("/").split("/")[-1] if prefix.strip("/") else None
            manifest: dict[str, Any] = {
                "model_backend": "onnx",
                "model_version": model_version,
                "source_uri": resolved_uri,
                "configured_source_uri": configured_uri,
                "source_type": "s3_prefix",
            }
            (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

            _replace_tree(bundle_root, runtime_dir)

        return read_registry_manifest()

    if backend == "pytorch":
        # Selectively download the minimal torch bundle from either root or model/ subtree.
        bundle_prefix = _pick_key([f"{prefix}model/", prefix])
        if not bundle_prefix:
            raise FileNotFoundError(f"No bundle directory found under {uri}")

        # If label_key came from model/, keep it; otherwise we will still download it.
        thresholds_key = _pick_key([f"{bundle_prefix}thresholds.json", f"{prefix}thresholds.json"])
        config_key = _pick_key([f"{bundle_prefix}config.json"])
        weights_key = _pick_key([f"{bundle_prefix}model.safetensors", f"{bundle_prefix}pytorch_model.bin"])
        classifier_key = _pick_key([f"{bundle_prefix}classifier.pt"])
        tokenizer_prefix = _pick_key([f"{bundle_prefix}tokenizer/"]) or tokenizer_prefix

        if not label_key:
            raise FileNotFoundError(f"No label order file found under {uri}")
        if not thresholds_key:
            raise FileNotFoundError(f"No thresholds.json found under {uri}")
        if not config_key or not weights_key or not classifier_key:
            raise FileNotFoundError(
                f"Missing required torch artifacts under {uri} (need config.json, model weights, classifier.pt)"
            )
        if not tokenizer_prefix:
            raise FileNotFoundError(f"No tokenizer directory found under {uri}")

        with tempfile.TemporaryDirectory(prefix="registry_prefix_bundle_") as td:
            td_path = Path(td)
            bundle_root = td_path / "bundle"
            bundle_root.mkdir(parents=True, exist_ok=True)

            _download_s3_key(bucket=bucket, key=config_key, dest=bundle_root / "config.json")
            # Keep the filename expected by HF AutoModel.
            dest_weights_name = "model.safetensors" if weights_key.endswith(".safetensors") else "pytorch_model.bin"
            _download_s3_key(bucket=bucket, key=weights_key, dest=bundle_root / dest_weights_name)
            _download_s3_key(bucket=bucket, key=classifier_key, dest=bundle_root / "classifier.pt")
            _download_s3_key(bucket=bucket, key=thresholds_key, dest=bundle_root / "thresholds.json")
            _download_s3_key(bucket=bucket, key=label_key, dest=bundle_root / "label_order.json")
            shutil.copyfile(bundle_root / "label_order.json", bundle_root / "registry_label_fields.json")
            _download_s3_prefix(bucket=bucket, prefix=tokenizer_prefix, dest_dir=bundle_root / "tokenizer")

            model_version = prefix.strip("/").split("/")[-1] if prefix.strip("/") else None
            manifest: dict[str, Any] = {
                "model_backend": "pytorch",
                "model_version": model_version,
                "source_uri": resolved_uri,
                "configured_source_uri": configured_uri,
                "source_type": "s3_prefix",
            }
            (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
            _replace_tree(bundle_root, runtime_dir)

        return read_registry_manifest()

    return {}


def ensure_registry_model_bundle(backend: str | None = None) -> BundleBootstrapResult:
    """Download + extract the configured registry model bundle into the runtime dir.

    No-op if:
    - no S3 URI is configured for the resolved backend, or
    - a manifest.json already exists in the runtime directory.
    """
    backend = (backend or resolve_model_backend()).strip().lower()
    if backend not in ("pytorch", "onnx"):
        backend = "auto"

    runtime_dir = get_registry_runtime_dir()
    existing_manifest = read_registry_manifest()

    # Default "auto" behavior: don't bootstrap unless explicitly configured.
    if backend == "auto":
        configured = os.getenv("MODEL_BUNDLE_S3_URI_PYTORCH") or os.getenv("MODEL_BUNDLE_S3_URI_ONNX") or os.getenv("MODEL_BUNDLE_S3_URI")
        if not configured:
            if existing_manifest:
                apply_resolved_registry_paths(backend=backend, runtime_dir=runtime_dir)
                return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest=existing_manifest)
            return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest={})
        # If only one URI is configured, prefer pytorch for local/dev ergonomics.
        backend = "pytorch" if os.getenv("MODEL_BUNDLE_S3_URI_PYTORCH") or os.getenv("MODEL_BUNDLE_S3_URI") else "onnx"

    uri = _s3_uri_for_backend(backend)
    if not uri:
        if existing_manifest:
            apply_resolved_registry_paths(backend=backend, runtime_dir=runtime_dir)
            return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest=existing_manifest)
        return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest={})

    # If we already have a manifest, only skip when it matches the configured source.
    # This avoids getting stuck on a stale local bundle when env vars point to a new model.
    state = _read_bootstrap_state(runtime_dir)
    state_configured_uri = state.get("configured_source_uri") or state.get("source_uri")
    state_resolved_uri = state.get("resolved_source_uri")
    if existing_manifest and state_configured_uri == uri and state.get("backend") == backend:
        # Tarball bundles are immutable by URI, so matching config is sufficient.
        if uri.endswith(".tar.gz"):
            apply_resolved_registry_paths(backend=backend, runtime_dir=runtime_dir)
            return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest=existing_manifest)

        # Prefix-based config may point at a moving target (e.g. `.../classifiers/`).
        # Resolve the latest ONNX prefix and only skip when unchanged.
        if backend == "onnx" and state_resolved_uri:
            try:
                latest_resolved = _resolve_latest_onnx_source_uri(uri)
            except Exception:
                latest_resolved = None
            if latest_resolved and latest_resolved == state_resolved_uri:
                apply_resolved_registry_paths(backend=backend, runtime_dir=runtime_dir)
                return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest=existing_manifest)
            if latest_resolved is None:
                # If we can't resolve (e.g. creds/network issue), trust the local bundle.
                apply_resolved_registry_paths(backend=backend, runtime_dir=runtime_dir)
                return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest=existing_manifest)

        # For non-ONNX prefix bundles, skip when we have a resolved source recorded.
        if backend != "onnx" and state_resolved_uri:
            apply_resolved_registry_paths(backend=backend, runtime_dir=runtime_dir)
            return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=False, manifest=existing_manifest)

    logger.info("Downloading registry model bundle", extra={"backend": backend, "s3_uri": uri})

    downloaded_manifest: dict[str, Any] = {}
    if uri.endswith(".tar.gz"):
        with tempfile.TemporaryDirectory(prefix="registry_bundle_") as td:
            td_path = Path(td)
            tar_path = td_path / "bundle.tar.gz"
            _download_s3_to_path(uri, tar_path)

            extracted = td_path / "extracted"
            _extract_tarball_to_dir(tar_path, extracted)
            root = _flatten_single_root(extracted)

            # Swap into place
            _replace_tree(root, runtime_dir)

        downloaded_manifest = read_registry_manifest()
    else:
        downloaded_manifest = _ensure_registry_bundle_from_s3_prefix(uri=uri, backend=backend, runtime_dir=runtime_dir)

    manifest = downloaded_manifest or read_registry_manifest()
    resolved_source_uri = uri if uri.endswith(".tar.gz") else manifest.get("source_uri")
    _write_bootstrap_state(
        runtime_dir,
        {
            "backend": backend,
            "configured_source_uri": uri,
            "resolved_source_uri": resolved_source_uri,
        },
    )
    apply_resolved_registry_paths(backend=backend, runtime_dir=runtime_dir)
    return BundleBootstrapResult(runtime_dir=runtime_dir, downloaded=True, manifest=manifest)
