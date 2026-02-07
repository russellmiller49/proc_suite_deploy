"""ML Ranker adapter for CPT code prediction.

Provides ML-based code ranking/prediction to augment the rule engine
in the coding pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from observability.logging_config import get_logger

logger = get_logger("ml_ranker")


@dataclass
class MLCodePrediction:
    """A code prediction from the ML ranker."""

    code: str
    probability: float
    source: str = "ml_ranker"


@dataclass
class MLRankingResult:
    """Result from ML ranking of codes.

    Attributes:
        predictions: All predictions above the minimum threshold.
        confidence_map: Mapping of code -> probability for lookup.
        high_conf_codes: Codes above the high confidence threshold.
        gray_zone_codes: Codes in the gray zone (between lower and upper thresholds).
        difficulty: Case difficulty classification (HIGH_CONF, GRAY_ZONE, LOW_CONF).
    """

    predictions: list[MLCodePrediction]
    confidence_map: dict[str, float]
    high_conf_codes: list[str]
    gray_zone_codes: list[str]
    difficulty: str


class MLRankerPort(ABC):
    """Abstract port for ML-based code ranking.

    This port allows the CodingService to optionally integrate ML predictions
    alongside rule-based and LLM-based coding.
    """

    @abstractmethod
    def rank_codes(
        self,
        note_text: str,
        candidate_codes: list[str] | None = None,
    ) -> MLRankingResult:
        """Rank/predict CPT codes for the given note.

        Args:
            note_text: The procedure note text to analyze.
            candidate_codes: Optional list of candidate codes to rank.
                           If provided, only these codes will be ranked.
                           If None, all known codes will be scored.

        Returns:
            MLRankingResult with predictions and confidence information.
        """
        ...

    @abstractmethod
    def get_confidence(self, code: str, note_text: str) -> float:
        """Get ML confidence for a specific code.

        Args:
            code: The CPT code to check.
            note_text: The procedure note text.

        Returns:
            Probability (0.0-1.0) that this code applies.
        """
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        """Return True if the ML ranker is available (models loaded)."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the model version identifier."""
        ...


class MLCoderPredictorAdapter(MLRankerPort):
    """Adapter wrapping MLCoderPredictor for the CodingService pipeline.

    This adapter bridges the standalone ML predictor to the CodingService's
    port interface, allowing seamless integration of ML predictions.
    """

    def __init__(
        self,
        predictor: Any = None,  # Type hint is Any to avoid circular import
        model_path: str | Path | None = None,
        mlb_path: str | Path | None = None,
    ):
        """Initialize the adapter.

        Args:
            predictor: Pre-loaded MLCoderPredictor instance.
            model_path: Path to model file (if predictor not provided).
            mlb_path: Path to MLB file (if predictor not provided).
        """
        self._predictor = predictor
        self._available = False
        self._version = "unknown"

        if predictor is None and (model_path or mlb_path):
            self._load_predictor(model_path, mlb_path)
        elif predictor is not None:
            self._available = True
            self._version = getattr(predictor, "version", "ml_predictor_v1")

    def _load_predictor(
        self,
        model_path: str | Path | None,
        mlb_path: str | Path | None,
    ) -> None:
        """Lazily load the predictor from disk."""
        try:
            from ml.lib.ml_coder.predictor import MLCoderPredictor

            self._predictor = MLCoderPredictor(
                model_path=model_path,
                mlb_path=mlb_path,
            )
            self._available = True
            self._version = "ml_predictor_v1"
            logger.info("ML ranker loaded successfully")
        except FileNotFoundError:
            logger.warning("ML model artifacts not found - ML ranker disabled")
            self._available = False
        except Exception as exc:
            logger.exception("Failed to load ML ranker: %s", exc)
            self._available = False

    def rank_codes(
        self,
        note_text: str,
        candidate_codes: list[str] | None = None,
    ) -> MLRankingResult:
        """Rank codes using the ML predictor."""
        if not self._available or self._predictor is None:
            return MLRankingResult(
                predictions=[],
                confidence_map={},
                high_conf_codes=[],
                gray_zone_codes=[],
                difficulty="UNAVAILABLE",
            )

        # Get case classification from predictor
        classification = self._predictor.classify_case(note_text)

        # Build predictions list
        predictions: list[MLCodePrediction] = []
        for pred in classification.predictions:
            # Filter to candidate codes if provided
            if candidate_codes is not None and pred.cpt not in candidate_codes:
                continue
            predictions.append(
                MLCodePrediction(
                    code=pred.cpt,
                    probability=pred.prob,
                )
            )

        # Build confidence map
        confidence_map = {pred.code: pred.probability for pred in predictions}

        # Extract high-conf and gray-zone codes
        high_conf_codes = [p.cpt for p in classification.high_conf]
        gray_zone_codes = [p.cpt for p in classification.gray_zone]

        # Filter if candidate_codes provided
        if candidate_codes is not None:
            high_conf_codes = [c for c in high_conf_codes if c in candidate_codes]
            gray_zone_codes = [c for c in gray_zone_codes if c in candidate_codes]

        return MLRankingResult(
            predictions=predictions,
            confidence_map=confidence_map,
            high_conf_codes=high_conf_codes,
            gray_zone_codes=gray_zone_codes,
            difficulty=classification.difficulty.value,
        )

    def get_confidence(self, code: str, note_text: str) -> float:
        """Get ML confidence for a specific code."""
        if not self._available or self._predictor is None:
            return 0.0

        predictions = self._predictor.predict_proba(note_text)
        for pred in predictions:
            if pred.cpt == code:
                return pred.prob
        return 0.0

    @property
    def available(self) -> bool:
        """Return True if the ML ranker is available."""
        return self._available

    @property
    def version(self) -> str:
        """Return the model version identifier."""
        return self._version


def build_ml_ranker(
    model_path: str | Path | None = None,
    mlb_path: str | Path | None = None,
) -> Optional[MLRankerPort]:
    """Factory function to build an ML ranker if models are available.

    Returns None if models cannot be loaded (graceful degradation).
    """
    try:
        adapter = MLCoderPredictorAdapter(
            model_path=model_path,
            mlb_path=mlb_path,
        )
        if adapter.available:
            return adapter
        return None
    except Exception as exc:
        logger.warning("Could not initialize ML ranker: %s", exc)
        return None
