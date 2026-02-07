"""PyTorch-based inference service for registry procedure flags.

This mirrors the shape of the ONNX predictor but uses local PyTorch/HF artifacts.

Bundle contract (directory):
- config.json
- model.safetensors and/or pytorch_model.bin
- tokenizer/ (HuggingFace tokenizer files)
- thresholds.json
- label_order.json (preferred) or registry_label_fields.json

If the bundle does not contain a trained classification head, this predictor
will mark itself unavailable (so callers can fall back without changing behavior).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.common.logger import get_logger

logger = get_logger("registry.inference_pytorch")


@dataclass
class RegistryFieldPrediction:
    field: str
    probability: float
    threshold: float
    is_positive: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "probability": self.probability,
            "threshold": self.threshold,
            "is_positive": self.is_positive,
        }


@dataclass
class RegistryCaseClassification:
    note_text: str
    predictions: list[RegistryFieldPrediction]
    positive_fields: list[str]
    difficulty: str  # "HIGH_CONF" or "LOW_CONF"

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_text": self.note_text[:200] + "..." if len(self.note_text) > 200 else self.note_text,
            "predictions": [p.to_dict() for p in self.predictions],
            "positive_fields": list(self.positive_fields),
            "difficulty": self.difficulty,
        }


class TorchRegistryPredictor:
    """Registry multi-label predictor using a PyTorch model bundle."""

    def __init__(self, bundle_dir: str | Path, max_length: int = 512) -> None:
        self.available = False
        self._bundle_dir = Path(bundle_dir)
        self._max_length = max_length

        self._labels: list[str] = []
        self._thresholds: dict[str, float] = {}
        self._tokenizer = None
        self._model = None
        self._classifier = None

        try:
            self._load_bundle()
            self.available = True
        except Exception as exc:
            logger.warning("TorchRegistryPredictor unavailable: %s", exc)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    def _load_labels(self) -> list[str]:
        label_order = self._bundle_dir / "label_order.json"
        if label_order.exists():
            data = json.loads(label_order.read_text())
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return list(data)

        label_fields = self._bundle_dir / "registry_label_fields.json"
        if label_fields.exists():
            data = json.loads(label_fields.read_text())
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return list(data)

        raise FileNotFoundError(
            f"Missing label order file (expected {label_order} or {label_fields})"
        )

    def _load_thresholds(self) -> dict[str, float]:
        thresholds_path = self._bundle_dir / "thresholds.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Missing thresholds.json at {thresholds_path}")
        data = json.loads(thresholds_path.read_text())
        if not isinstance(data, dict):
            raise ValueError("thresholds.json must be a JSON object")
        thresholds: dict[str, float] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                thresholds[k] = float(v)
        return thresholds

    def _load_bundle(self) -> None:
        if not self._bundle_dir.exists():
            raise FileNotFoundError(f"Bundle dir not found: {self._bundle_dir}")

        self._labels = self._load_labels()
        self._thresholds = self._load_thresholds()

        # Lazy imports so local dev can run without torch/transformers if desired.
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer_dir = self._bundle_dir / "tokenizer"
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"Missing tokenizer/ directory at {tokenizer_dir}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

        # Load base encoder (expects state_dict shaped like *Model, not *ForSequenceClassification).
        self._model = AutoModel.from_pretrained(str(self._bundle_dir), local_files_only=True)
        self._model.eval()

        # Optional: trained linear head saved as classifier.pt (state_dict with weight/bias).
        classifier_path = self._bundle_dir / "classifier.pt"
        if classifier_path.exists():
            state = torch.load(str(classifier_path), map_location="cpu")
            if not isinstance(state, dict) or "weight" not in state or "bias" not in state:
                raise ValueError(f"Unexpected classifier.pt format at {classifier_path}")

            hidden_size = int(getattr(self._model.config, "hidden_size", 0) or 0)
            if hidden_size <= 0:
                raise ValueError("Model config missing hidden_size")

            num_labels = len(self._labels)
            classifier = torch.nn.Linear(hidden_size, num_labels)
            classifier.load_state_dict(state)
            classifier.eval()
            self._classifier = classifier
        else:
            raise FileNotFoundError(
                "Bundle missing classifier.pt (trained classification head); "
                "cannot run PyTorch registry predictions."
            )

    def predict_proba(self, note_text: str) -> list[RegistryFieldPrediction]:
        if not self.available:
            return [
                RegistryFieldPrediction(
                    field=name,
                    probability=0.0,
                    threshold=self._thresholds.get(name, 0.5),
                    is_positive=False,
                )
                for name in self._labels
            ]

        import torch

        text = note_text.strip() if note_text else ""
        if not text:
            return [
                RegistryFieldPrediction(
                    field=name,
                    probability=0.0,
                    threshold=self._thresholds.get(name, 0.5),
                    is_positive=False,
                )
                for name in self._labels
            ]

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._classifier is not None

        tokens = self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**tokens)
            # Use [CLS] embedding as pooled representation (works for BERT/RoBERTa-style encoders).
            cls = outputs.last_hidden_state[:, 0, :]
            logits = self._classifier(cls)
            probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()

        preds: list[RegistryFieldPrediction] = []
        for field, p in zip(self._labels, probs):
            threshold = float(self._thresholds.get(field, 0.5))
            probability = float(p)
            preds.append(
                RegistryFieldPrediction(
                    field=field,
                    probability=probability,
                    threshold=threshold,
                    is_positive=probability >= threshold,
                )
            )

        preds.sort(key=lambda x: x.probability, reverse=True)
        return preds

    def predict(self, note_text: str) -> list[str]:
        return [p.field for p in self.predict_proba(note_text) if p.is_positive]

    def classify_case(self, note_text: str) -> RegistryCaseClassification:
        preds = self.predict_proba(note_text)
        positive = [p.field for p in preds if p.is_positive]
        difficulty = "HIGH_CONF" if positive else "LOW_CONF"
        return RegistryCaseClassification(
            note_text=note_text,
            predictions=preds,
            positive_fields=positive,
            difficulty=difficulty,
        )
