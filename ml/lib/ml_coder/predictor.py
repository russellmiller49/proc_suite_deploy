"""Prediction service that wraps the trained CPT classifier.

Includes:
- MLCoderService: Simple prediction service (legacy)
- MLCoderPredictor: Ternary case difficulty classification (HIGH_CONF/GRAY_ZONE/LOW_CONF)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from app.common.logger import get_logger
from app.infra.cache import get_ml_memory_cache
from app.infra.settings import get_infra_settings
from ml.lib.ml_coder.thresholds import CaseDifficulty, Thresholds, load_thresholds
from ml.lib.ml_coder.training import MLB_PATH, PIPELINE_PATH

logger = get_logger("ml_coder.predictor")

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_cache(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _ml_cache_key(prefix: str, text: str) -> str:
    normalized = _normalize_for_cache(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


class MLCoderService:
    """Thin wrapper around the trained classifier pipeline and binarizer."""

    def __init__(self, models_dir: str | Path | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path("data/models")
        self.pipeline_path = self.models_dir / "cpt_classifier.pkl"
        self.mlb_path = self.models_dir / "mlb.pkl"
        self.pipeline: Pipeline | None = None
        self.mlb: MultiLabelBinarizer | None = None
        self.available = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        try:
            self.pipeline = joblib.load(self.pipeline_path)
            self.mlb = joblib.load(self.mlb_path)
            self.available = True
            logger.info("Loaded ML artifacts from %s", self.models_dir)
        except FileNotFoundError:
            logger.warning(
                "ML artifacts not found at %s. Machine learning predictions disabled.",
                self.models_dir,
            )
            self.available = False
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to load ML artifacts: %s", exc)
            self.available = False

    def predict(self, text: str, threshold: float = 0.5) -> List[Dict[str, float | str]]:
        """Return ML predictions for the supplied text above a probability threshold."""

        if not self.available or not self.pipeline or not self.mlb:
            return []

        normalized = text.strip()
        if not normalized:
            return []

        try:
            probabilities = self.pipeline.predict_proba([normalized])
        except AttributeError:  # pragma: no cover - should not happen with the configured model
            logger.warning("ML pipeline does not support probability predictions.")
            return []
        except Exception as exc:  # pragma: no cover - inference safety
            logger.exception("ML prediction failed: %s", exc)
            return []

        prob_array = probabilities[0]

        results: List[Dict[str, float | str]] = []
        for code, score in zip(self.mlb.classes_, prob_array):
            confidence = float(score)
            if confidence >= threshold:
                results.append(
                    {
                        "cpt": str(code),
                        "confidence": confidence,
                        "source": "ml_model",
                    }
                )

        results.sort(key=lambda item: item["confidence"], reverse=True)
        return results


@dataclass
class CodePrediction:
    """Single CPT code prediction with probability."""

    cpt: str
    prob: float

    def to_dict(self) -> dict[str, Any]:
        return {"cpt": self.cpt, "prob": self.prob}


@dataclass
class CaseClassification:
    """
    Full case classification result.

    Attributes:
        predictions: All predictions sorted by probability (descending)
        high_conf: Predictions above upper threshold
        gray_zone: Predictions between lower and upper thresholds
        difficulty: Overall case difficulty classification
    """

    predictions: list[CodePrediction]
    high_conf: list[CodePrediction]
    gray_zone: list[CodePrediction]
    difficulty: CaseDifficulty

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "high_conf": [p.to_dict() for p in self.high_conf],
            "gray_zone": [p.to_dict() for p in self.gray_zone],
            "difficulty": self.difficulty.value,
        }


class MLCoderPredictor:
    """
    ML-based CPT code predictor with ternary case difficulty classification.

    Classification policy:
    - HIGH_CONF: At least one prediction above upper threshold.
      → Handled by ML+Rules pipeline only.
    - GRAY_ZONE: No high-conf predictions, but at least one above lower threshold.
      → ML suggestions go to LLM as hints; LLM is final judge.
    - LOW_CONF: All predictions below lower threshold.
      → LLM acts as primary coder; ML opinion is weak context only.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        mlb_path: str | Path | None = None,
        thresholds: Thresholds | None = None,
        thresholds_path: str | Path | None = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained pipeline pickle (default: PIPELINE_PATH)
            mlb_path: Path to MultiLabelBinarizer pickle (default: MLB_PATH)
            thresholds: Pre-loaded Thresholds object
            thresholds_path: Path to thresholds JSON (if thresholds not provided)
        """
        model_path = Path(model_path) if model_path else PIPELINE_PATH
        mlb_path = Path(mlb_path) if mlb_path else MLB_PATH

        logger.info("Loading model from %s", model_path)
        self._pipeline = joblib.load(model_path)
        self._mlb = joblib.load(mlb_path)
        self._labels: list[str] = list(self._mlb.classes_)

        if thresholds:
            self._thresholds = thresholds
        else:
            self._thresholds = load_thresholds(thresholds_path)

        logger.info(
            "Predictor initialized with %d labels, upper=%.2f, lower=%.2f",
            len(self._labels),
            self._thresholds.upper,
            self._thresholds.lower,
        )

    @property
    def labels(self) -> list[str]:
        """Return list of CPT codes the model can predict."""
        return self._labels.copy()

    @property
    def thresholds(self) -> Thresholds:
        """Return the thresholds configuration."""
        return self._thresholds

    def predict_proba(self, note_text: str) -> list[CodePrediction]:
        """
        Get probability predictions for all codes.

        Args:
            note_text: Clinical note text

        Returns:
            List of CodePrediction objects sorted by probability (descending)
        """
        proba = self._pipeline.predict_proba([note_text])[0]  # shape: (n_labels,)

        predictions = []
        for cpt, p in zip(self._labels, proba):
            predictions.append(CodePrediction(cpt=cpt, prob=float(p)))

        predictions.sort(key=lambda x: x.prob, reverse=True)
        return predictions

    def predict(self, note_text: str, threshold: float = 0.5) -> list[str]:
        """
        Get predicted CPT codes above threshold.

        Args:
            note_text: Clinical note text
            threshold: Probability threshold for positive prediction

        Returns:
            List of predicted CPT codes
        """
        proba = self._pipeline.predict_proba([note_text])[0]
        return [cpt for cpt, p in zip(self._labels, proba) if p >= threshold]

    def classify_case(self, note_text: str) -> CaseClassification:
        """
        Classify a case into HIGH_CONF, GRAY_ZONE, or LOW_CONF.

        This is the main entry point for the hybrid ML/LLM pipeline.

        Args:
            note_text: Clinical note text

        Returns:
            CaseClassification with predictions and difficulty level
        """
        settings = get_infra_settings()
        cache_key: str | None = None
        if settings.enable_ml_cache:
            cache_key = _ml_cache_key("mlcoder.case", note_text)
            cached = get_ml_memory_cache().get(cache_key)
            if isinstance(cached, CaseClassification):
                return cached

        predictions = self.predict_proba(note_text)

        high_conf: list[CodePrediction] = []
        gray_zone: list[CodePrediction] = []

        for pred in predictions:
            upper = self._thresholds.upper_for(pred.cpt)
            lower = self._thresholds.lower_for(pred.cpt)

            if pred.prob >= upper:
                high_conf.append(pred)
            elif pred.prob >= lower:
                gray_zone.append(pred)

        # Determine overall difficulty
        if high_conf:
            difficulty = CaseDifficulty.HIGH_CONF
        elif gray_zone:
            difficulty = CaseDifficulty.GRAY_ZONE
        else:
            difficulty = CaseDifficulty.LOW_CONF

        result = CaseClassification(
            predictions=predictions,
            high_conf=high_conf,
            gray_zone=gray_zone,
            difficulty=difficulty,
        )

        if cache_key is not None:
            get_ml_memory_cache().set(cache_key, result, ttl_s=3600)

        return result

    def classify_batch(self, note_texts: list[str]) -> list[CaseClassification]:
        """
        Classify multiple cases.

        Args:
            note_texts: List of clinical note texts

        Returns:
            List of CaseClassification objects
        """
        return [self.classify_case(text) for text in note_texts]


__all__ = [
    "MLCoderService",
    "MLCoderPredictor",
    "CodePrediction",
    "CaseClassification",
]
