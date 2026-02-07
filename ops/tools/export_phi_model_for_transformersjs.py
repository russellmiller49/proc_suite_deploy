#!/usr/bin/env python3
"""Export PHI DistilBERT model to a transformers.js-compatible ONNX bundle.

Target layout (Xenova/transformers.js friendly):
  <out_dir>/
    config.json
    tokenizer.json
    ...
    protected_terms.json
    onnx/model.onnx
    onnx/model_quantized.onnx (optional, opt-in)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

# Ensure repo root is on sys.path (so `import app.*` works when running as a script).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.phi.adapters.phi_redactor_hybrid import (
    ANATOMICAL_TERMS,
    DEVICE_MANUFACTURERS,
    PROTECTED_DEVICE_NAMES,
)

MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "label_map.json",
]

REQUIRED_ONNX_INPUTS = ("input_ids", "attention_mask")


def parse_bool(value: str | bool | None) -> bool:
    """Parse bool-ish CLI args while supporting `--flag false` and `--flag`."""
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="artifacts/phi_distilbert_ner")
    ap.add_argument(
        "--out-dir",
        default="ui/static/phi_redactor/vendor/phi_distilbert_ner",
    )
    ap.add_argument(
        "--quantize",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Also export `onnx/model_quantized.onnx` (opt-in; WASM INT8 may misbehave).",
    )
    ap.add_argument(
        "--static-quantize",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Use static quantization instead of dynamic (smaller model, ~40-50%% size reduction).",
    )
    ap.add_argument(
        "--clean",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Remove prior export artifacts from --out-dir before exporting.",
    )
    return ap


def run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        tool = cmd[0]
        raise RuntimeError(f"{tool} not found. Install with: pip install 'optimum[onnxruntime]'") from exc


def locate_exported_onnx(export_dir: Path) -> Path:
    """Locate the exported ONNX model within an Optimum output directory."""
    direct = [
        export_dir / "model.onnx",
        export_dir / "onnx" / "model.onnx",
    ]
    for candidate in direct:
        if candidate.exists():
            return candidate

    candidates: list[Path] = []
    candidates.extend(export_dir.glob("*.onnx"))
    onnx_subdir = export_dir / "onnx"
    if onnx_subdir.exists():
        candidates.extend(onnx_subdir.glob("*.onnx"))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No ONNX files found in Optimum export dir: {export_dir}")

    # Prefer the largest ONNX file if multiple exist.
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def write_protected_terms(out_dir: Path) -> None:
    terms = {
        "anatomy_terms": sorted({t.lower() for t in ANATOMICAL_TERMS}),
        "device_manufacturers": sorted({t.lower() for t in DEVICE_MANUFACTURERS}),
        "protected_device_names": sorted({t.lower() for t in PROTECTED_DEVICE_NAMES}),
        "ln_station_regex": r"^\\d{1,2}[LRlr](?:[is])?$",
        "segment_regex": r"^[LRlr][Bb]\\d{1,2}(?:\\+\\d{1,2})?$",
        "address_markers": [
            "street",
            "st",
            "rd",
            "road",
            "ave",
            "avenue",
            "dr",
            "drive",
            "blvd",
            "boulevard",
            "lane",
            "ln",
            "zip",
            "zipcode",
            "address",
            "city",
            "state",
            "ste",
            "suite",
            "apt",
            "unit",
        ],
        "code_markers": [
            "cpt",
            "code",
            "codes",
            "billing",
            "submitted",
            "justification",
            "rvu",
            "coding",
            "radiology",
            "guidance",
            "ct",
            "modifier",
            "billed",
            "cbct",
        ],
        "station_markers": ["station", "stations", "nodes", "sampled", "ebus", "tbna", "ln"],
    }
    with open(out_dir / "protected_terms.json", "w") as f:
        json.dump(terms, f, indent=2)


def format_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def find_optimum_cli() -> str:
    """Find optimum-cli in the current Python environment."""
    # First try to find it in the same Python environment's bin directory
    python_bin = Path(sys.executable).parent
    optimum_cli = python_bin / "optimum-cli"
    if optimum_cli.exists():
        return str(optimum_cli)
    
    # Fall back to which (may find system version, but better than nothing)
    which_cli = shutil.which("optimum-cli")
    if which_cli:
        return which_cli

    raise RuntimeError("optimum-cli not found. Install with: pip install 'optimum[onnxruntime]'")


def validate_onnx_inputs(model_onnx: Path, required: Iterable[str] = REQUIRED_ONNX_INPUTS) -> None:
    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "onnx is required to validate exported model signatures. Install with: pip install onnx"
        ) from exc

    m = onnx.load(str(model_onnx))
    input_names = [i.name for i in m.graph.input]
    missing = [name for name in required if name not in input_names]
    if missing:
        raise RuntimeError(
            "Re-export failed: token-classification ONNX must have attention_mask.\n"
            f"ONNX missing required inputs: {missing}. Found: {input_names}"
        )


def quantize_onnx_model(model_onnx: Path, quantized_onnx: Path, use_static: bool = False) -> None:
    """Quantize ONNX model using dynamic or static quantization.
    
    Args:
        model_onnx: Path to input ONNX model
        quantized_onnx: Path for quantized output
        use_static: If True, use static quantization (smaller but requires calibration data)
    """
    try:
        from onnxruntime.quantization import (
            QuantType,
            quantize_dynamic,
            quantize_static,
            CalibrationDataReader,
        )
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for quantization. Install with: pip install onnxruntime"
        ) from exc

    quantized_onnx.parent.mkdir(parents=True, exist_ok=True)
    
    if use_static:
        # Static quantization is more aggressive but requires calibration data
        # Create a simple calibration data reader with representative tokenized inputs
        class TokenCalibrationDataReader(CalibrationDataReader):
            def __init__(self, model_path: Path):
                self.model_path = model_path
                self.data_iter = None
                # Generate representative calibration samples
                # These are dummy tokenized sequences that represent typical input
                self.calibration_samples = []
                for _ in range(10):  # Use 10 calibration samples
                    # Create varied length sequences (typical for token classification)
                    seq_len = np.random.randint(128, 512)
                    self.calibration_samples.append({
                        "input_ids": np.random.randint(0, 30000, (1, seq_len), dtype=np.int64),
                        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                    })
            
            def get_next(self):
                if self.data_iter is None:
                    self.data_iter = iter(self.calibration_samples)
                return next(self.data_iter, None)
        
        try:
            print("Performing static quantization (this may take a few minutes)...")
            quantize_static(
                str(model_onnx),
                str(quantized_onnx),
                calibration_data_reader=TokenCalibrationDataReader(model_onnx),
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                optimize_model=True,
            )
            print("Static quantization completed successfully.")
        except Exception as e:
            # Fallback to dynamic if static fails
            print(f"Static quantization failed ({e}), falling back to dynamic...")
            quantize_dynamic(str(model_onnx), str(quantized_onnx), weight_type=QuantType.QInt8)
    else:
        # Dynamic quantization (default - faster, no calibration needed)
        print("Performing dynamic quantization...")
        quantize_dynamic(str(model_onnx), str(quantized_onnx), weight_type=QuantType.QInt8)


def clean_export_dir(out_dir: Path) -> None:
    """Remove generated artifacts so output layout is deterministic."""
    onnx_dir = out_dir / "onnx"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    for filename in [*MODEL_FILES, "protected_terms.json", "model.onnx", "model_quantized.onnx"]:
        path = out_dir / filename
        if path.exists():
            path.unlink()


def main() -> None:
    args = build_arg_parser().parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"--model-dir not found: {model_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_export_dir(out_dir)

    optimum_cli = find_optimum_cli()

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model_path = onnx_dir / "model.onnx"

    with tempfile.TemporaryDirectory(prefix="phi_onnx_export_") as tmp:
        tmp_dir = Path(tmp)
        run(
            [
                optimum_cli,
                "export",
                "onnx",
                "--model",
                str(model_dir),
                "--task",
                "token-classification",
                str(tmp_dir),
            ]
        )
        exported_onnx = locate_exported_onnx(tmp_dir)
        shutil.copy2(exported_onnx, model_path)

    if not model_path.exists():
        raise RuntimeError("Export failed: model.onnx not found.")
    model_size = model_path.stat().st_size
    if model_size < 5_000_000:
        raise RuntimeError("Export failed: model.onnx is unexpectedly small.")
    validate_onnx_inputs(model_path)
    print(f"Exported model.onnx size: {format_size(model_size)}")

    if args.quantize:
        quantized_path = onnx_dir / "model_quantized.onnx"
        quantize_onnx_model(model_path, quantized_path, use_static=args.static_quantize)
        if not quantized_path.exists() or quantized_path.stat().st_size < 5_000_000:
            raise RuntimeError("Quantization failed: model_quantized.onnx missing or too small.")
        validate_onnx_inputs(quantized_path)
        quantized_size = quantized_path.stat().st_size
        reduction_pct = (1 - quantized_size / model_size) * 100
        print(f"Quantized model_quantized.onnx size: {format_size(quantized_size)} ({reduction_pct:.1f}% reduction)")

    for name in MODEL_FILES:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    write_protected_terms(out_dir)


if __name__ == "__main__":
    main()
