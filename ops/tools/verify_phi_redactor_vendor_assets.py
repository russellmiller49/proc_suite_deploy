#!/usr/bin/env python3
"""Verify PHI redactor vendor assets are present for the IU bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

VENDOR_DIR = Path("ui/static/phi_redactor/vendor/phi_distilbert_ner_quant")


def _error(message: str, errors: list[str]) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []

    if not VENDOR_DIR.exists():
        _error(f"Missing vendor directory: {VENDOR_DIR}", errors)
    elif not VENDOR_DIR.is_dir():
        _error(f"Vendor path is not a directory: {VENDOR_DIR}", errors)

    config_path = VENDOR_DIR / "config.json"
    if not config_path.exists():
        _error(f"Missing config.json: {config_path}", errors)

    onnx_dir = VENDOR_DIR / "onnx"
    onnx_model = onnx_dir / "model.onnx"
    if not onnx_model.exists():
        _error(f"Missing model.onnx: {onnx_model}", errors)

    tokenizer_json = VENDOR_DIR / "tokenizer.json"
    tokenizer_config = VENDOR_DIR / "tokenizer_config.json"
    vocab_txt = VENDOR_DIR / "vocab.txt"
    vocab_json = VENDOR_DIR / "vocab.json"
    merges_txt = VENDOR_DIR / "merges.txt"

    has_tokenizer_bundle = tokenizer_json.exists()
    has_tokenizer_parts = tokenizer_config.exists() and (vocab_txt.exists() or vocab_json.exists() or merges_txt.exists())
    if not (has_tokenizer_bundle or has_tokenizer_parts):
        _error(
            "Missing tokenizer files: expected tokenizer.json or tokenizer_config.json + vocab files",
            errors,
        )

    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if "id2label" not in config and "label2id" not in config:
                _error("config.json is missing label map keys (id2label/label2id)", errors)
        except Exception as exc:  # pragma: no cover - guard for malformed config
            _error(f"Failed to parse config.json: {exc}", errors)

    if errors:
        for message in errors:
            print(f"[verify_phi_redactor_vendor_assets] ERROR: {message}", file=sys.stderr)
        return 1

    print("[verify_phi_redactor_vendor_assets] OK: PHI redactor vendor assets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
