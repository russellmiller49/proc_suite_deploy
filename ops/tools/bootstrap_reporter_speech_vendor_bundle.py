#!/usr/bin/env python3
"""Bootstrap the local reporter speech vendor bundles into the UI vendor directories.

This script downloads the same-origin assets required by
`ui/static/phi_redactor/speech.worker.js`:

- A local Transformers.js-compatible Whisper base bundle under:
  `ui/static/phi_redactor/vendor/speech_whisper_base_en/`
- A local Transformers.js-compatible Whisper tiny bundle under:
  `ui/static/phi_redactor/vendor/speech_whisper_tiny_en/`
- The ONNX Runtime WASM backend files under:
  `ui/static/phi_redactor/vendor/transformers/`

By default it mirrors the same assets into the classic UI variant as well.

Environment variables:
- REPORTER_SPEECH_VENDOR_MODELS (default: base,tiny)
- REPORTER_SPEECH_VENDOR_REPO_ID
  (default base-model override: Xenova/whisper-base.en)
- REPORTER_SPEECH_VENDOR_REVISION (default: main)
- REPORTER_SPEECH_VENDOR_DIR
  (default: ui/static/phi_redactor/vendor/speech_whisper_base_en)
- REPORTER_SPEECH_VENDOR_CLASSIC_DIR
  (default: ui/static/phi_redactor_classic/vendor/speech_whisper_base_en)
- REPORTER_SPEECH_VENDOR_INCLUDE_CLASSIC (default: true)
- REPORTER_SPEECH_VENDOR_FORCE (default: false)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOOTSTRAP_STATE_FILENAME = ".bootstrap_state.json"
TRANSFORMERS_JS_VERSION = "2.17.2"
PRIMARY_MODEL_VARIANT = "base"
DEFAULT_MODEL_VARIANTS: tuple[str, ...] = ("base", "tiny")

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "base": {
        "label": "Base",
        "repo_id": "Xenova/whisper-base.en",
        "bundle_dir_name": "speech_whisper_base_en",
    },
    "tiny": {
        "label": "Tiny",
        "repo_id": "Xenova/whisper-tiny.en",
        "bundle_dir_name": "speech_whisper_tiny_en",
    },
}

MODEL_TEXT_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

MODEL_ONNX_FILES: tuple[str, ...] = (
    "onnx/encoder_model_quantized.onnx",
    "onnx/decoder_model_merged_quantized.onnx",
)

TRANSFORMERS_WASM_FILES: tuple[str, ...] = (
    "ort-wasm.wasm",
    "ort-wasm-threaded.wasm",
    "ort-wasm-simd.wasm",
    "ort-wasm-simd-threaded.wasm",
)


@dataclass(frozen=True)
class ModelBundleResult:
    key: str
    label: str
    repo_id: str
    atlas_vendor_dir: Path
    classic_vendor_dir: Path | None
    downloaded: bool


@dataclass(frozen=True)
class BootstrapResult:
    atlas_vendor_dir: Path
    atlas_runtime_dir: Path
    classic_vendor_dir: Path | None
    classic_runtime_dir: Path | None
    downloaded_model: bool
    downloaded_runtime: bool
    manifest: dict[str, Any]
    models: dict[str, ModelBundleResult]


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


def _bootstrap_state_path(dest_dir: Path) -> Path:
    return dest_dir / BOOTSTRAP_STATE_FILENAME


def _read_bootstrap_state(dest_dir: Path) -> dict[str, Any]:
    path = _bootstrap_state_path(dest_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_bootstrap_state(dest_dir: Path, state: dict[str, Any]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    _bootstrap_state_path(dest_dir).write_text(json.dumps(state, indent=2, sort_keys=True))


def _replace_tree(src_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir)


def _repo_file_url(repo_id: str, revision: str, relative_path: str) -> str:
    repo = urllib.parse.quote(repo_id, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    rel = urllib.parse.quote(relative_path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{rev}/{rel}"


def _wasm_file_url(filename: str) -> str:
    return (
        f"https://cdn.jsdelivr.net/npm/@xenova/transformers@{TRANSFORMERS_JS_VERSION}/dist/{filename}"
    )


def _download_to_path(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=dest.name, dir=str(dest.parent), delete=False) as tf:
            tmp_path = Path(tf.name)
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(str(tmp_path), str(dest))
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _all_files_present(root: Path, required_files: tuple[str, ...]) -> bool:
    return all((root / rel).is_file() for rel in required_files)


def _default_atlas_vendor_dir() -> Path:
    return Path(
        os.getenv(
            "REPORTER_SPEECH_VENDOR_DIR",
            "ui/static/phi_redactor/vendor/speech_whisper_base_en",
        )
    )


def _default_classic_vendor_dir() -> Path:
    return Path(
        os.getenv(
            "REPORTER_SPEECH_VENDOR_CLASSIC_DIR",
            "ui/static/phi_redactor_classic/vendor/speech_whisper_base_en",
        )
    )


def _runtime_dir_for_vendor_dir(vendor_dir: Path) -> Path:
    return vendor_dir.parent / "transformers"


def _normalize_model_variants(models: tuple[str, ...] | None) -> tuple[str, ...]:
    raw_models = models
    if raw_models is None:
        env_models = os.getenv("REPORTER_SPEECH_VENDOR_MODELS", ",".join(DEFAULT_MODEL_VARIANTS))
        raw_models = tuple(part.strip() for part in env_models.split(","))

    resolved: list[str] = []
    for raw_model in raw_models:
        model_key = raw_model.strip().lower()
        if not model_key:
            continue
        if model_key not in MODEL_VARIANTS:
            valid = ", ".join(sorted(MODEL_VARIANTS))
            raise ValueError(f"Unsupported reporter speech model '{raw_model}'. Expected one of: {valid}")
        if model_key not in resolved:
            resolved.append(model_key)

    if not resolved:
        raise ValueError("At least one reporter speech model must be selected")
    return tuple(resolved)


def _variant_vendor_dir(primary_vendor_dir: Path, variant: str) -> Path:
    spec = MODEL_VARIANTS[variant]
    if variant == PRIMARY_MODEL_VARIANT:
        return primary_vendor_dir
    return primary_vendor_dir.parent / spec["bundle_dir_name"]


def _build_manifest(
    *,
    models: tuple[str, ...],
    revision: str,
    repo_ids: dict[str, str],
) -> dict[str, Any]:
    return {
        "bundle_type": "reporter_speech_vendor",
        "primary_model_variant": PRIMARY_MODEL_VARIANT,
        "revision": revision,
        "transformers_js_version": TRANSFORMERS_JS_VERSION,
        "required_model_files": list(MODEL_TEXT_FILES + MODEL_ONNX_FILES),
        "required_runtime_files": list(TRANSFORMERS_WASM_FILES),
        "model_variants": {
            model_key: {
                "label": MODEL_VARIANTS[model_key]["label"],
                "repo_id": repo_ids[model_key],
                "bundle_dir_name": MODEL_VARIANTS[model_key]["bundle_dir_name"],
            }
            for model_key in models
        },
    }


def ensure_reporter_speech_vendor_bundle(
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    models: tuple[str, ...] | None = None,
    atlas_vendor_dir: Path | None = None,
    classic_vendor_dir: Path | None = None,
    include_classic: bool | None = None,
    force: bool | None = None,
    dry_run: bool = False,
) -> BootstrapResult:
    _load_dotenv()

    resolved_models = _normalize_model_variants(models)
    resolved_primary_repo_id = (
        repo_id or os.getenv("REPORTER_SPEECH_VENDOR_REPO_ID") or MODEL_VARIANTS[PRIMARY_MODEL_VARIANT]["repo_id"]
    ).strip()
    resolved_revision = (revision or os.getenv("REPORTER_SPEECH_VENDOR_REVISION") or "main").strip()
    resolved_atlas_vendor_dir = Path(atlas_vendor_dir or _default_atlas_vendor_dir())
    resolved_classic_vendor_dir = Path(classic_vendor_dir or _default_classic_vendor_dir())
    resolved_include_classic = (
        include_classic
        if include_classic is not None
        else _truthy_env("REPORTER_SPEECH_VENDOR_INCLUDE_CLASSIC", default=True)
    )
    resolved_force = force if force is not None else _truthy_env("REPORTER_SPEECH_VENDOR_FORCE")

    atlas_runtime_dir = _runtime_dir_for_vendor_dir(resolved_atlas_vendor_dir)
    classic_runtime_dir = (
        _runtime_dir_for_vendor_dir(resolved_classic_vendor_dir) if resolved_include_classic else None
    )

    repo_ids = {
        model_key: (
            resolved_primary_repo_id
            if model_key == PRIMARY_MODEL_VARIANT
            else MODEL_VARIANTS[model_key]["repo_id"]
        )
        for model_key in resolved_models
    }
    manifest = _build_manifest(models=resolved_models, revision=resolved_revision, repo_ids=repo_ids)

    required_model_files = MODEL_TEXT_FILES + MODEL_ONNX_FILES
    required_runtime_files = TRANSFORMERS_WASM_FILES
    model_results: dict[str, ModelBundleResult] = {}
    downloaded_any_model = False

    for model_key in resolved_models:
        spec = MODEL_VARIANTS[model_key]
        atlas_variant_dir = _variant_vendor_dir(resolved_atlas_vendor_dir, model_key)
        classic_variant_dir = (
            _variant_vendor_dir(resolved_classic_vendor_dir, model_key) if resolved_include_classic else None
        )
        state = _read_bootstrap_state(atlas_variant_dir)
        model_cached = (
            not resolved_force
            and state.get("variant") == model_key
            and state.get("repo_id") == repo_ids[model_key]
            and state.get("revision") == resolved_revision
            and _all_files_present(atlas_variant_dir, required_model_files)
        )

        variant_manifest = {
            **manifest,
            "active_variant": model_key,
            "active_variant_label": spec["label"],
            "active_variant_repo_id": repo_ids[model_key],
        }

        if not dry_run and not model_cached:
            with tempfile.TemporaryDirectory(prefix=f"reporter_speech_vendor_{model_key}_") as td:
                stage_dir = Path(td) / atlas_variant_dir.name
                for relative_path in required_model_files:
                    _download_to_path(
                        _repo_file_url(repo_ids[model_key], resolved_revision, relative_path),
                        stage_dir / relative_path,
                    )
                (stage_dir / "manifest.json").write_text(
                    json.dumps(variant_manifest, indent=2, sort_keys=True)
                )
                _write_bootstrap_state(
                    stage_dir,
                    {
                        "variant": model_key,
                        "repo_id": repo_ids[model_key],
                        "revision": resolved_revision,
                    },
                )
                _replace_tree(stage_dir, atlas_variant_dir)

        if resolved_include_classic and not dry_run and classic_variant_dir is not None:
            _replace_tree(atlas_variant_dir, classic_variant_dir)

        model_results[model_key] = ModelBundleResult(
            key=model_key,
            label=spec["label"],
            repo_id=repo_ids[model_key],
            atlas_vendor_dir=atlas_variant_dir,
            classic_vendor_dir=classic_variant_dir,
            downloaded=not model_cached,
        )
        downloaded_any_model = downloaded_any_model or (not model_cached)

    runtime_cached = not resolved_force and _all_files_present(atlas_runtime_dir, required_runtime_files)
    if not dry_run and not runtime_cached:
        atlas_runtime_dir.mkdir(parents=True, exist_ok=True)
        for filename in required_runtime_files:
            _download_to_path(_wasm_file_url(filename), atlas_runtime_dir / filename)

    if resolved_include_classic and not dry_run:
        _replace_tree(
            atlas_runtime_dir,
            classic_runtime_dir or resolved_classic_vendor_dir.parent / "transformers",
        )

    return BootstrapResult(
        atlas_vendor_dir=_variant_vendor_dir(resolved_atlas_vendor_dir, PRIMARY_MODEL_VARIANT),
        atlas_runtime_dir=atlas_runtime_dir,
        classic_vendor_dir=(
            _variant_vendor_dir(resolved_classic_vendor_dir, PRIMARY_MODEL_VARIANT)
            if resolved_include_classic
            else None
        ),
        classic_runtime_dir=classic_runtime_dir,
        downloaded_model=downloaded_any_model,
        downloaded_runtime=not runtime_cached,
        manifest=manifest,
        models=model_results,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap the local reporter speech Whisper bundles into the UI vendor directories.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override the Hugging Face repo id for the base model bundle (default: Xenova/whisper-base.en)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face revision to download (default: main)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model variants to download (default: base,tiny)",
    )
    parser.add_argument(
        "--atlas-vendor-dir",
        default=None,
        help="Destination for the atlas base speech model bundle",
    )
    parser.add_argument(
        "--classic-vendor-dir",
        default=None,
        help="Destination for the classic base speech model bundle",
    )
    parser.add_argument(
        "--skip-classic",
        action="store_true",
        help="Do not mirror assets into the classic UI variant",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if the configured bundles already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without downloading files",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    models = tuple(part.strip() for part in args.models.split(",")) if args.models else None
    result = ensure_reporter_speech_vendor_bundle(
        repo_id=args.repo_id,
        revision=args.revision,
        models=models,
        atlas_vendor_dir=Path(args.atlas_vendor_dir) if args.atlas_vendor_dir else None,
        classic_vendor_dir=Path(args.classic_vendor_dir) if args.classic_vendor_dir else None,
        include_classic=not args.skip_classic,
        force=args.force,
        dry_run=args.dry_run,
    )

    mode = "dry-run" if args.dry_run else "applied"
    model_summary = ", ".join(
        f"{model_key}:{'downloaded' if model_result.downloaded else 'cached'}"
        for model_key, model_result in result.models.items()
    )
    print(
        f"[bootstrap_reporter_speech_vendor_bundle] {mode} "
        f"models={model_summary} "
        f"runtime={'downloaded' if result.downloaded_runtime else 'cached'} "
        f"atlas={result.atlas_vendor_dir}"
    )
    print(f"  atlas runtime: {result.atlas_runtime_dir}")
    for model_key, model_result in result.models.items():
        print(
            f"  atlas {model_key} bundle: {model_result.atlas_vendor_dir} ({model_result.repo_id})"
        )
    if result.classic_runtime_dir:
        print(f"  classic runtime: {result.classic_runtime_dir}")
    for model_key, model_result in result.models.items():
        if model_result.classic_vendor_dir:
            print(f"  classic {model_key} bundle: {model_result.classic_vendor_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
