"""Prediction service for registry procedure flags (multi-label classification).

Provides:
- RegistryMLPredictor: ML predictor for registry boolean procedure flags
- RegistryFieldPrediction: Single field prediction with probability
- RegistryCaseClassification: Full case classification result
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np

from app.common.logger import get_logger
from ml.lib.ml_coder.data_prep import REGISTRY_TARGET_FIELDS

logger = get_logger("ml_coder.registry_predictor")

# Default paths for registry model artifacts
MODELS_DIR = Path("data/models")
REGISTRY_PIPELINE_PATH = MODELS_DIR / "registry_classifier.pkl"
REGISTRY_MLB_PATH = MODELS_DIR / "registry_mlb.pkl"
REGISTRY_THRESHOLDS_PATH = MODELS_DIR / "registry_thresholds.json"
REGISTRY_LABEL_FIELDS_PATH = Path("data/ml_training/registry_label_fields.json")


@dataclass
class RegistryFieldPrediction:
    """Single registry field prediction with probability and threshold info."""

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

    Attributes:
        note_text: The input note text (truncated for display)
        predictions: All field predictions sorted by probability (descending)
        positive_fields: Fields classified as positive (above threshold)
        difficulty: Overall difficulty ("HIGH_CONF" if any positives, else "LOW_CONF")
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


def load_registry_thresholds(path: str | Path | None = None) -> dict[str, float]:
    """Load per-field thresholds from JSON file.

    Args:
        path: Path to thresholds JSON. If None, uses default path.
              If file doesn't exist, returns default 0.5 for all fields.

    Returns:
        Dict mapping field names to threshold values.
    """
    path = Path(path) if path else REGISTRY_THRESHOLDS_PATH
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Support both flat dict and nested format
        if "per_field" in data:
            return data["per_field"]
        return data
    # Default: 0.5 threshold for all fields
    return {field: 0.5 for field in REGISTRY_TARGET_FIELDS}


class RegistryMLPredictor:
    """ML predictor for registry procedure flags (multi-label classification).

    This predictor wraps a trained multi-label classifier and provides:
    - predict_proba: Get probability predictions for all registry fields
    - classify_case: Classify case as HIGH_CONF or LOW_CONF based on thresholds

    The model expects clinical note text as input and outputs predictions
    for each procedure flag in the registry schema.
    """

    def __init__(
        self,
        model: Any | None = None,
        label_names: Iterable[str] | None = None,
        thresholds: Mapping[str, float] | None = None,
        model_path: str | Path | None = None,
        mlb_path: str | Path | None = None,
        thresholds_path: str | Path | None = None,
    ) -> None:
        """Initialize the registry predictor.

        Args:
            model: Pre-loaded model (for testing/injection)
            label_names: List of label names in model order
            thresholds: Dict of per-field thresholds
            model_path: Path to trained pipeline pickle
            mlb_path: Path to MultiLabelBinarizer pickle
            thresholds_path: Path to thresholds JSON
        """
        self.available = False

        # Allow injection for testing
        if model is not None and thresholds is not None and label_names is not None:
            self._model = model
            self._label_names = list(label_names)
            self._thresholds = dict(thresholds)
            self.available = True
            logger.info("RegistryMLPredictor initialized with injected model")
        else:
            self._load_artifacts_from_disk(
                model_path=model_path,
                mlb_path=mlb_path,
                thresholds_path=thresholds_path,
            )

    def _load_artifacts_from_disk(
        self,
        model_path: str | Path | None = None,
        mlb_path: str | Path | None = None,
        thresholds_path: str | Path | None = None,
    ) -> None:
        """Load trained model, label binarizer metadata, and per-label thresholds."""
        model_path = Path(model_path) if model_path else REGISTRY_PIPELINE_PATH
        mlb_path = Path(mlb_path) if mlb_path else REGISTRY_MLB_PATH

        try:
            logger.info("Loading registry model from %s", model_path)
            self._model = joblib.load(model_path)

            # Load label names from MLB or fallback to label fields JSON
            if mlb_path.exists():
                mlb = joblib.load(mlb_path)
                self._label_names = list(mlb.classes_)
                logger.info("Loaded %d labels from MLB", len(self._label_names))
            elif REGISTRY_LABEL_FIELDS_PATH.exists():
                with open(REGISTRY_LABEL_FIELDS_PATH) as f:
                    self._label_names = json.load(f)
                logger.info("Loaded %d labels from label_fields.json", len(self._label_names))
            else:
                # Fallback to full REGISTRY_TARGET_FIELDS
                self._label_names = list(REGISTRY_TARGET_FIELDS)
                logger.warning(
                    "No MLB or label_fields.json found, using REGISTRY_TARGET_FIELDS (%d labels)",
                    len(self._label_names),
                )

            # Load thresholds
            self._thresholds = load_registry_thresholds(thresholds_path)

            self.available = True
            logger.info(
                "RegistryMLPredictor initialized with %d labels",
                len(self._label_names),
            )

        except FileNotFoundError as e:
            logger.warning(
                "Registry ML artifacts not found: %s. Predictions disabled.", e
            )
            self._model = None
            self._label_names = []
            self._thresholds = {}
            self.available = False

        except Exception as exc:
            logger.exception("Failed to load registry ML artifacts: %s", exc)
            self._model = None
            self._label_names = []
            self._thresholds = {}
            self.available = False

    @property
    def labels(self) -> list[str]:
        """Return list of registry field names the model can predict."""
        return self._label_names.copy()

    @property
    def thresholds(self) -> dict[str, float]:
        """Return per-field thresholds."""
        return self._thresholds.copy()

    def threshold_for(self, field: str) -> float:
        """Get threshold for a specific field (defaults to 0.5)."""
        return self._thresholds.get(field, 0.5)

    def predict_proba(self, note_text: str) -> list[RegistryFieldPrediction]:
        """Return per-label probabilities for the given note text.

        Args:
            note_text: Clinical procedure note text

        Returns:
            List of RegistryFieldPrediction objects sorted by probability (descending)
        """
        if not self.available or not self._model:
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
            proba = self._model.predict_proba([text])
        except Exception as exc:
            logger.exception("Registry ML prediction failed: %s", exc)
            return [
                RegistryFieldPrediction(
                    field=name,
                    probability=0.0,
                    threshold=self._thresholds.get(name, 0.5),
                    is_positive=False,
                )
                for name in self._label_names
            ]

        # Handle different predict_proba output formats
        # OneVsRestClassifier with probability calibration returns (n_samples, n_labels)
        # Some estimators return list of (n_samples, 2) arrays
        probs_for_labels: np.ndarray
        if isinstance(proba, list):
            # proba is a list of (n_samples, 2) arrays, one per label
            # Take column 1 (positive class probability) for each
            probs_for_labels = np.array([p[0, 1] if p.shape[1] > 1 else p[0, 0] for p in proba])
        else:
            # proba is (n_samples, n_labels)
            probs_for_labels = proba[0]

        predictions: list[RegistryFieldPrediction] = []
        for idx, field in enumerate(self._label_names):
            p = float(probs_for_labels[idx])
            thresh = float(self._thresholds.get(field, 0.5))
            predictions.append(
                RegistryFieldPrediction(
                    field=field,
                    probability=p,
                    threshold=thresh,
                    is_positive=p >= thresh,
                )
            )

        # Sort by probability descending
        predictions.sort(key=lambda x: x.probability, reverse=True)
        return predictions

    def predict(self, note_text: str) -> list[str]:
        """Get predicted registry fields above their thresholds.

        Args:
            note_text: Clinical procedure note text

        Returns:
            List of field names predicted as positive
        """
        preds = self.predict_proba(note_text)
        return [p.field for p in preds if p.is_positive]

    def classify_case(self, note_text: str) -> RegistryCaseClassification:
        """Classify a case into HIGH_CONF or LOW_CONF.

        HIGH_CONF: At least one field predicted positive above threshold.
        LOW_CONF: No fields predicted positive.

        Args:
            note_text: Clinical procedure note text

        Returns:
            RegistryCaseClassification with predictions and difficulty level
        """
        preds = self.predict_proba(note_text)
        positive_fields = [p.field for p in preds if p.is_positive]
        difficulty = "HIGH_CONF" if positive_fields else "LOW_CONF"

        return RegistryCaseClassification(
            note_text=note_text,
            predictions=preds,
            positive_fields=positive_fields,
            difficulty=difficulty,
        )

    def classify_batch(self, note_texts: list[str]) -> list[RegistryCaseClassification]:
        """Classify multiple cases.

        Args:
            note_texts: List of clinical note texts

        Returns:
            List of RegistryCaseClassification objects
        """
        return [self.classify_case(text) for text in note_texts]


__all__ = [
    "RegistryMLPredictor",
    "RegistryFieldPrediction",
    "RegistryCaseClassification",
    "load_registry_thresholds",
    "REGISTRY_PIPELINE_PATH",
    "REGISTRY_MLB_PATH",
    "REGISTRY_THRESHOLDS_PATH",
]


if __name__ == "__main__":
    # Sanity check: load model and run prediction on a synthetic note
    print("=" * 60)
    print("Registry ML Predictor Sanity Check")
    print("=" * 60)

    try:
        predictor = RegistryMLPredictor()

        if not predictor.available:
            print("\nModel not available. Please train the registry classifier first.")
            print("Run: python -m ml.lib.ml_coder.registry_training")
        else:
            print(f"\nModel loaded with {len(predictor.labels)} labels:")
            for label in predictor.labels:
                print(f"  - {label}")

            # Test with a synthetic note
            test_note = """
            Procedure: Bronchoscopy with EBUS-TBNA
            Patient underwent flexible bronchoscopy under moderate sedation.
            Linear EBUS performed with sampling of stations 4R, 7, and 11R.
            TBNA performed at each station with 22-gauge needle.
            Good specimen quality noted. No complications.
            """

            print("\n" + "-" * 60)
            print("Test Note (truncated):")
            print(test_note[:200] + "...")

            print("\n" + "-" * 60)
            print("Top 10 Predictions (by probability):")

            preds = predictor.predict_proba(test_note)
            for pred in preds[:10]:
                status = "POSITIVE" if pred.is_positive else ""
                print(
                    f"  {pred.field:30s} | prob={pred.probability:.3f} "
                    f"| thresh={pred.threshold:.2f} | {status}"
                )

            print("\n" + "-" * 60)
            classification = predictor.classify_case(test_note)
            print(f"Case Difficulty: {classification.difficulty}")
            print(f"Positive Fields: {classification.positive_fields}")

    except Exception as e:
        print(f"\nError during sanity check: {e}")
        import traceback
        traceback.print_exc()
