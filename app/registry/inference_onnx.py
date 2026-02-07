"""ONNX-based inference service for registry procedure flags.

Provides a lightweight, CPU-only inference option for production deployment
that matches the RegistryMLPredictor interface from ml_coder.registry_predictor.

Key features:
- INT8 quantized model for minimal RAM usage (~110MB)
- CPU-only inference with ONNX Runtime
- Per-class threshold application from thresholds.json
- Head + Tail tokenization for clinical notes

Usage:
    from app.registry.inference_onnx import ONNXRegistryPredictor

    predictor = ONNXRegistryPredictor()
    predictions = predictor.predict_proba(note_text)
    positive_fields = predictor.predict(note_text)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.common.logger import get_logger

logger = get_logger("registry.inference_onnx")

# Default paths for ONNX model artifacts
MODELS_DIR = Path("models")
ONNX_MODEL_PATH = MODELS_DIR / "registry_model_int8.onnx"
TOKENIZER_PATH = MODELS_DIR / "roberta_registry_tokenizer"
THRESHOLDS_PATH = Path("data/models") / "roberta_registry_thresholds.json"
LABEL_FIELDS_PATH = Path("data/ml_training/registry_label_fields.json")


@dataclass
class RegistryFieldPrediction:
    """Single registry field prediction with probability and threshold info.

    Matches the interface from ml_coder.registry_predictor.
    """

    field: str
    probability: float
    threshold: float
    is_positive: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field": self.field,
            "probability": self.probability,
            "threshold": self.threshold,
            "is_positive": self.is_positive,
        }


@dataclass
class RegistryCaseClassification:
    """Full case classification result for registry procedure flags.

    Matches the interface from ml_coder.registry_predictor.
    """

    note_text: str
    predictions: list[RegistryFieldPrediction]
    positive_fields: list[str]
    difficulty: str  # "HIGH_CONF" or "LOW_CONF"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "note_text": self.note_text[:200] + "..." if len(self.note_text) > 200 else self.note_text,
            "predictions": [p.to_dict() for p in self.predictions],
            "positive_fields": self.positive_fields,
            "difficulty": self.difficulty,
        }


class HeadTailTokenizer:
    """Tokenizer with Head + Tail truncation for clinical notes.

    Keeps first 382 tokens + last 128 tokens to preserve both
    procedure information (top) and complications/plan (bottom).
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        head_tokens: int = 382,
        tail_tokens: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens

    def __call__(self, text: str) -> dict[str, np.ndarray]:
        """Tokenize with Head + Tail truncation.

        Args:
            text: Input clinical note text

        Returns:
            Dict with input_ids and attention_mask as numpy arrays
        """
        # Tokenize without truncation first
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"][0]

        # Apply Head + Tail if too long
        content_max = self.max_length - 2  # Reserve for [CLS] and [SEP]

        if len(input_ids) > content_max:
            head_ids = input_ids[: self.head_tokens]
            tail_ids = input_ids[-self.tail_tokens :]
            input_ids = np.concatenate([head_ids, tail_ids])

        # Add special tokens
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # Build sequence: [CLS] + tokens + [SEP] + padding
        full_ids = np.concatenate([
            np.array([cls_id]),
            input_ids,
            np.array([sep_id]),
        ])

        # Pad to max_length
        pad_length = self.max_length - len(full_ids)
        if pad_length > 0:
            full_ids = np.concatenate([full_ids, np.full(pad_length, pad_id)])

        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = (full_ids != pad_id).astype(np.int64)
        input_ids = full_ids.astype(np.int64)

        return {
            "input_ids": input_ids[np.newaxis, :],  # Add batch dimension
            "attention_mask": attention_mask[np.newaxis, :],
        }


class ONNXRegistryPredictor:
    """Lightweight ONNX-based registry prediction.

    Designed for production deployment with:
    - CPU-only inference (no GPU required)
    - Low memory footprint (~110MB for INT8 model)
    - Fast inference (<100ms typical)
    - Per-class threshold application

    Implements the same interface as RegistryMLPredictor.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        thresholds_path: str | Path | None = None,
        label_fields_path: str | Path | None = None,
        max_length: int = 512,
    ) -> None:
        """Initialize ONNX predictor.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer directory
            thresholds_path: Path to thresholds JSON
            label_fields_path: Path to label fields JSON
            max_length: Maximum sequence length
        """
        self.available = False
        self._max_length = max_length
        self._session = None
        self._tokenizer = None
        self._head_tail_tokenizer = None
        self._label_names: list[str] = []
        self._thresholds: dict[str, float] = {}

        model_path = Path(model_path) if model_path else ONNX_MODEL_PATH
        tokenizer_path = Path(tokenizer_path) if tokenizer_path else TOKENIZER_PATH
        thresholds_path = Path(thresholds_path) if thresholds_path else THRESHOLDS_PATH
        label_fields_path = Path(label_fields_path) if label_fields_path else LABEL_FIELDS_PATH

        try:
            self._load_artifacts(
                model_path,
                tokenizer_path,
                thresholds_path,
                label_fields_path,
            )
            self.available = True
            logger.info(
                "ONNXRegistryPredictor initialized with %d labels",
                len(self._label_names),
            )
        except Exception as e:
            logger.warning("Failed to initialize ONNXRegistryPredictor: %s", e)

    def _load_artifacts(
        self,
        model_path: Path,
        tokenizer_path: Path,
        thresholds_path: Path,
        label_fields_path: Path,
    ) -> None:
        """Load ONNX model, tokenizer, thresholds, and label names."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Check paths exist
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        if not label_fields_path.exists():
            raise FileNotFoundError(f"Label fields not found: {label_fields_path}")

        # Load ONNX model with CPU provider
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Create Head + Tail tokenizer wrapper
        self._head_tail_tokenizer = HeadTailTokenizer(
            self._tokenizer,
            max_length=self._max_length,
            head_tokens=382,
            tail_tokens=128,
        )

        # Load label fields
        with open(label_fields_path) as f:
            self._label_names = json.load(f)

        # Load thresholds
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                self._thresholds = json.load(f)
        else:
            # Default to 0.5 if no thresholds file
            self._thresholds = {name: 0.5 for name in self._label_names}
            logger.warning(
                "Thresholds file not found at %s, using default 0.5",
                thresholds_path,
            )

        # Validate label count vs model output dimension.
        # If they disagree, prefer a safe fallback that matches the model.
        try:
            outputs = self._session.get_outputs()
            out_shape = outputs[0].shape if outputs else None
            # Common: [batch, n_labels]
            output_dim = out_shape[-1] if isinstance(out_shape, list) and out_shape else None
            if isinstance(output_dim, int) and output_dim > 0 and len(self._label_names) != output_dim:
                # Best fallback: ACTIVE_LABELS (drops known dormant labels like pleural_biopsy).
                try:
                    from ml.lib.ml_coder.registry_label_schema import ACTIVE_LABELS

                    if len(ACTIVE_LABELS) == output_dim:
                        logger.warning(
                            "ONNX label mismatch: label_fields=%d but model outputs=%d. "
                            "Falling back to ACTIVE_LABELS (%d). "
                            "Fix by regenerating registry_runtime/registry_label_fields.json to match the model bundle.",
                            len(self._label_names),
                            output_dim,
                            len(ACTIVE_LABELS),
                        )
                        self._label_names = list(ACTIVE_LABELS)
                    else:
                        logger.warning(
                            "ONNX label mismatch: label_fields=%d but model outputs=%d. "
                            "Truncating label list to %d (may reduce correctness). "
                            "Fix by regenerating registry_runtime/registry_label_fields.json to match the model bundle.",
                            len(self._label_names),
                            output_dim,
                            output_dim,
                        )
                        self._label_names = list(self._label_names)[:output_dim]
                except Exception:
                    logger.warning(
                        "ONNX label mismatch: label_fields=%d but model outputs=%d. "
                        "Truncating label list to %d. "
                        "Fix by regenerating registry_runtime/registry_label_fields.json to match the model bundle.",
                        len(self._label_names),
                        output_dim,
                        output_dim,
                    )
                    self._label_names = list(self._label_names)[:output_dim]

                # Ensure thresholds line up with label names after fallback.
                if isinstance(self._thresholds, dict):
                    self._thresholds = {k: float(v) for k, v in self._thresholds.items() if k in set(self._label_names)}
                for name in self._label_names:
                    self._thresholds.setdefault(name, 0.5)
        except Exception:
            # Never fail predictor init due to introspection; predict_proba() will handle errors.
            pass

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation to logits."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @property
    def labels(self) -> list[str]:
        """Return list of registry field names."""
        return self._label_names.copy()

    @property
    def thresholds(self) -> dict[str, float]:
        """Return per-field thresholds."""
        return self._thresholds.copy()

    def threshold_for(self, field: str) -> float:
        """Get threshold for a specific field."""
        return self._thresholds.get(field, 0.5)

    def predict_proba(self, note_text: str) -> list[RegistryFieldPrediction]:
        """Return per-label probabilities for the given note text.

        Args:
            note_text: Clinical procedure note text

        Returns:
            List of RegistryFieldPrediction sorted by probability (descending)
        """
        if not self.available or self._session is None:
            return [
                RegistryFieldPrediction(
                    field=name,
                    probability=0.0,
                    threshold=self._thresholds.get(name, 0.5),
                    is_positive=False,
                )
                for name in self._label_names
            ]

        text = note_text.strip() if note_text else ""
        if not text:
            return [
                RegistryFieldPrediction(
                    field=name,
                    probability=0.0,
                    threshold=self._thresholds.get(name, 0.5),
                    is_positive=False,
                )
                for name in self._label_names
            ]

        try:
            # Tokenize with Head + Tail strategy
            inputs = self._head_tail_tokenizer(text)

            # Run inference
            logits = self._session.run(None, inputs)[0]
            probs = self._sigmoid(logits[0])

        except Exception as e:
            logger.exception("ONNX inference failed: %s", e)
            return [
                RegistryFieldPrediction(
                    field=name,
                    probability=0.0,
                    threshold=self._thresholds.get(name, 0.5),
                    is_positive=False,
                )
                for name in self._label_names
            ]

        # Safety: if the model output length doesn't match label names, don't crash.
        if len(probs) != len(self._label_names):
            logger.warning(
                "ONNX output/label mismatch at runtime: probs=%d labels=%d. Returning empty predictions. "
                "Fix your model bundle (registry_runtime) to align label_fields.json with ONNX head size.",
                len(probs),
                len(self._label_names),
            )
            return []

        # Build predictions with per-class thresholds
        predictions = []
        for idx, field in enumerate(self._label_names):
            p = float(probs[idx])
            thresh = float(self._thresholds.get(field, 0.5))
            predictions.append(
                RegistryFieldPrediction(
                    field=field,
                    probability=p,
                    threshold=thresh,
                    is_positive=p >= thresh,
                )
            )

        # Sort by probability (descending)
        predictions.sort(key=lambda x: x.probability, reverse=True)
        return predictions

    def predict(self, note_text: str) -> list[str]:
        """Get predicted registry fields above their thresholds.

        Args:
            note_text: Clinical procedure note text

        Returns:
            List of field names classified as positive
        """
        preds = self.predict_proba(note_text)
        return [p.field for p in preds if p.is_positive]

    def predict_with_probs(self, note_text: str) -> dict[str, float]:
        """Get all predictions as a dict mapping field names to probabilities.

        Args:
            note_text: Clinical procedure note text

        Returns:
            Dict mapping field names to probability values
        """
        preds = self.predict_proba(note_text)
        return {p.field: p.probability for p in preds}

    def classify_case(self, note_text: str) -> RegistryCaseClassification:
        """Classify a case and determine confidence level.

        Args:
            note_text: Clinical procedure note text

        Returns:
            RegistryCaseClassification with predictions and difficulty
        """
        preds = self.predict_proba(note_text)
        positive_fields = [p.field for p in preds if p.is_positive]

        # Determine difficulty based on prediction confidence
        # HIGH_CONF if at least one positive with high probability
        high_conf_positives = [
            p for p in preds
            if p.is_positive and p.probability >= 0.8
        ]
        difficulty = "HIGH_CONF" if high_conf_positives else "LOW_CONF"

        return RegistryCaseClassification(
            note_text=note_text,
            predictions=preds,
            positive_fields=positive_fields,
            difficulty=difficulty,
        )

    def classify_batch(
        self,
        note_texts: list[str],
    ) -> list[RegistryCaseClassification]:
        """Classify multiple cases.

        Args:
            note_texts: List of clinical procedure note texts

        Returns:
            List of RegistryCaseClassification objects
        """
        return [self.classify_case(text) for text in note_texts]

    def get_registry_flags(self, note_text: str) -> dict[str, bool]:
        """Get registry boolean flags from prediction.

        Args:
            note_text: Clinical procedure note text

        Returns:
            Dict mapping registry field names to boolean values
        """
        preds = self.predict_proba(note_text)
        return {p.field: p.is_positive for p in preds}


# Factory function for easy instantiation
def get_onnx_predictor(
    model_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    thresholds_path: str | Path | None = None,
) -> ONNXRegistryPredictor | None:
    """Get an ONNX predictor instance if available.

    Returns None if ONNX model is not available.

    Args:
        model_path: Optional path to ONNX model
        tokenizer_path: Optional path to tokenizer
        thresholds_path: Optional path to thresholds JSON

    Returns:
        ONNXRegistryPredictor instance or None
    """
    try:
        predictor = ONNXRegistryPredictor(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            thresholds_path=thresholds_path,
        )
        if predictor.available:
            return predictor
    except Exception as e:
        logger.debug("ONNX predictor not available: %s", e)

    return None


__all__ = [
    "ONNXRegistryPredictor",
    "RegistryFieldPrediction",
    "RegistryCaseClassification",
    "get_onnx_predictor",
]
