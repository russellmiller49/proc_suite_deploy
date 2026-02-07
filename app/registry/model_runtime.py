from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_RUNTIME_DIR = Path("data/models/registry_runtime")


@dataclass(frozen=True)
class RegistryModelProvenance:
    backend: str | None
    version: str | None


def get_registry_runtime_dir() -> Path:
    override = os.getenv("REGISTRY_RUNTIME_DIR")
    return Path(override) if override else DEFAULT_REGISTRY_RUNTIME_DIR


def get_registry_manifest_path() -> Path:
    return get_registry_runtime_dir() / "manifest.json"


def read_registry_manifest() -> dict[str, Any]:
    path = get_registry_manifest_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def resolve_model_backend() -> str:
    value = os.getenv("MODEL_BACKEND", "").strip().lower()
    if value in ("pytorch", "onnx", "auto"):
        return value
    # Default to PyTorch for local/dev; production can set MODEL_BACKEND=onnx explicitly.
    return "pytorch"


def get_registry_model_provenance() -> RegistryModelProvenance:
    manifest = read_registry_manifest()
    backend = os.getenv("MODEL_BACKEND")
    backend = backend.strip().lower() if backend else None
    if backend == "":
        backend = None

    version = None
    if isinstance(manifest, dict):
        version_val = manifest.get("model_version") or manifest.get("version")
        if isinstance(version_val, str) and version_val.strip():
            version = version_val.strip()

        manifest_backend = manifest.get("model_backend") or manifest.get("backend")
        if backend is None and isinstance(manifest_backend, str) and manifest_backend.strip():
            backend = manifest_backend.strip().lower()

    # If MODEL_BACKEND isn't set, still expose the resolved mode ("auto") for logging.
    if backend is None:
        backend = resolve_model_backend()

    return RegistryModelProvenance(backend=backend, version=version)


def verify_registry_runtime_bundle(
    *,
    backend: str | None = None,
    runtime_dir: Path | None = None,
) -> list[str]:
    """Validate registry model artifacts for the configured backend.

    Returns a list of warnings. Raises RuntimeError if required artifacts are missing.
    """
    resolved_backend = (backend or resolve_model_backend()).strip().lower()
    runtime_dir = runtime_dir or get_registry_runtime_dir()

    warnings: list[str] = []
    errors: list[str] = []

    def _first_existing(paths: list[Path]) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None

    def _require(path: Path, label: str) -> None:
        if not path.exists():
            errors.append(f"Missing {label} at {path}")

    def _check_pytorch() -> tuple[list[str], list[str]]:
        local_errors: list[str] = []
        local_warnings: list[str] = []

        config_path = runtime_dir / "config.json"
        tokenizer_dir = runtime_dir / "tokenizer"
        thresholds_path = runtime_dir / "thresholds.json"
        label_path = _first_existing(
            [runtime_dir / "label_order.json", runtime_dir / "registry_label_fields.json"]
        )
        weights_path = _first_existing(
            [runtime_dir / "model.safetensors", runtime_dir / "pytorch_model.bin"]
        )
        classifier_path = runtime_dir / "classifier.pt"

        for path, label in [
            (config_path, "config.json"),
            (tokenizer_dir, "tokenizer/"),
            (thresholds_path, "thresholds.json"),
            (classifier_path, "classifier.pt"),
        ]:
            if not path.exists():
                local_errors.append(f"Missing {label} at {path}")

        if label_path is None:
            local_errors.append(
                "Missing label_order.json or registry_label_fields.json in runtime bundle"
            )

        if weights_path is None:
            local_errors.append(
                "Missing model weights (model.safetensors or pytorch_model.bin) in runtime bundle"
            )

        return local_errors, local_warnings

    def _check_onnx() -> tuple[list[str], list[str]]:
        local_errors: list[str] = []
        local_warnings: list[str] = []

        model_path = _first_existing(
            [
                runtime_dir / "registry_model_int8.onnx",
                runtime_dir / "registry_model.onnx",
            ]
        )
        tokenizer_dir = _first_existing(
            [runtime_dir / "tokenizer", runtime_dir / "roberta_registry_tokenizer"]
        )
        thresholds_path = _first_existing(
            [
                runtime_dir / "thresholds.json",
                runtime_dir / "registry_thresholds.json",
                runtime_dir / "roberta_registry_thresholds.json",
            ]
        )
        label_path = _first_existing(
            [runtime_dir / "label_order.json", runtime_dir / "registry_label_fields.json"]
        )

        if model_path is None:
            local_errors.append("Missing ONNX model (registry_model_int8.onnx or registry_model.onnx)")
        else:
            data_path = model_path.with_suffix(model_path.suffix + ".data")
            if not data_path.exists():
                try:
                    size_bytes = model_path.stat().st_size
                except OSError:
                    size_bytes = 0
                if size_bytes < 50_000_000:
                    local_errors.append(
                        f"Missing ONNX external data file at {data_path}"
                    )
                else:
                    local_warnings.append(
                        f"ONNX external data file not found at {data_path}; "
                        "assuming weights are embedded."
                    )

        if tokenizer_dir is None:
            local_errors.append("Missing tokenizer directory in runtime bundle")
        if thresholds_path is None:
            local_errors.append("Missing thresholds.json in runtime bundle")
        if label_path is None:
            local_errors.append(
                "Missing label_order.json or registry_label_fields.json in runtime bundle"
            )

        return local_errors, local_warnings

    if resolved_backend == "auto":
        pytorch_errors, pytorch_warnings = _check_pytorch()
        warnings.extend(pytorch_warnings)
        if not pytorch_errors:
            return warnings
        onnx_errors, onnx_warnings = _check_onnx()
        warnings.extend(onnx_warnings)
        if not onnx_errors:
            return warnings
        errors.append("No usable registry runtime bundle found for backend=auto")
        errors.extend([f"pytorch: {msg}" for msg in pytorch_errors])
        errors.extend([f"onnx: {msg}" for msg in onnx_errors])
    elif resolved_backend == "pytorch":
        pytorch_errors, pytorch_warnings = _check_pytorch()
        errors.extend(pytorch_errors)
        warnings.extend(pytorch_warnings)
    elif resolved_backend == "onnx":
        onnx_errors, onnx_warnings = _check_onnx()
        errors.extend(onnx_errors)
        warnings.extend(onnx_warnings)
    else:
        errors.append(f"Unknown backend '{resolved_backend}' for registry runtime validation")

    if errors:
        raise RuntimeError("; ".join(errors))

    return warnings


__all__ = [
    "RegistryModelProvenance",
    "get_registry_manifest_path",
    "get_registry_model_provenance",
    "get_registry_runtime_dir",
    "read_registry_manifest",
    "resolve_model_backend",
    "verify_registry_runtime_bundle",
]
