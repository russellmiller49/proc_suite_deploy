"""Confidence combination for parallel pathway results.

Combines deterministic (NER+Rules) and probabilistic (ML Classification)
signals into final confidence scores with evidence-gated review logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ConfidenceFactors:
    """Factors that contribute to final code confidence."""

    deterministic_found: bool
    """Path A (NER+Rules) derived this code."""

    ml_probability: float
    """Path B (ML Classification) probability (0-1)."""

    entity_confidence: float
    """Average NER confidence for supporting entities."""

    agreement: bool
    """Both paths agree on this code."""


@dataclass
class CodeConfidence:
    """Final confidence and explanation for a code."""

    code: str
    confidence: float
    explanation: str
    needs_review: bool
    review_reason: str | None


class ConfidenceCombiner:
    """Combine deterministic and probabilistic confidence scores.

    Strategy:
    - NER evidence auto-confirms unless ML strongly negates
    - ML-only high probability triggers review (possible NER blind spot)
    - Otherwise rely on ML probability without averaging
    """

    AUTO_CODE_CONFIDENCE = 0.95
    REVIEW_CONFIDENCE_FLOOR = 0.5
    ML_HIGH_CONF_THRESHOLD = 0.90
    ML_LOW_CONF_THRESHOLD = 0.10
    REVIEW_ON_LOW_ML_PROB_CODES = {
        # High-risk "history/presence" mentions frequently trigger false positives.
        "31636",
        "31637",
        "31638",
        "32550",
        "32551",
        "32552",
        "32556",
        "32557",
    }

    def combine(
        self,
        code: str,
        deterministic_found: bool,
        ml_probability: float,
        entity_confidence: float = 0.5,
    ) -> CodeConfidence:
        """
        Calculate final confidence for a code.

        Args:
            code: The CPT code
            deterministic_found: Whether Path A derived this code
            ml_probability: Path B probability for this code
            entity_confidence: Average confidence of NER entities (default 0.5)

        Returns:
            CodeConfidence with score, explanation, and review flags
        """
        if deterministic_found:
            if ml_probability < self.ML_LOW_CONF_THRESHOLD:
                if code in self.REVIEW_ON_LOW_ML_PROB_CODES:
                    confidence = max(self.REVIEW_CONFIDENCE_FLOOR, entity_confidence)
                    explanation = (
                        "FLAG_FOR_REVIEW: NER evidence found but ML probability is very low "
                        f"(ML prob: {ml_probability:.2f})"
                    )
                    needs_review = True
                    review_reason = (
                        "NER evidence found, but ML probability is very low; verify context (history/negation/aborted)."
                    )
                else:
                    confidence = min(0.99, max(self.AUTO_CODE_CONFIDENCE, entity_confidence))
                    explanation = f"AUTO_CODE: NER evidence found (ML prob: {ml_probability:.2f})"
                    needs_review = False
                    review_reason = None
            else:
                confidence = min(0.99, max(self.AUTO_CODE_CONFIDENCE, entity_confidence))
                explanation = (
                    f"AUTO_CODE: NER evidence found (ML prob: {ml_probability:.2f})"
                )
                needs_review = False
                review_reason = None
        elif ml_probability > self.ML_HIGH_CONF_THRESHOLD:
            confidence = max(self.REVIEW_CONFIDENCE_FLOOR, ml_probability)
            explanation = (
                "FLAG_FOR_REVIEW: ML detected procedure, but specific "
                f"device/anatomy was not found in text (ML prob: {ml_probability:.2f})"
            )
            needs_review = True
            review_reason = (
                "ML detected procedure, but specific device/anatomy was not found in text."
            )
        else:
            confidence = ml_probability
            explanation = (
                "No NER evidence and ML probability below review threshold "
                f"(ML prob: {ml_probability:.2f})"
            )
            needs_review = False
            review_reason = None

        return CodeConfidence(
            code=code,
            confidence=confidence,
            explanation=explanation,
            needs_review=needs_review,
            review_reason=review_reason,
        )

    def combine_all(
        self,
        path_a_codes: List[str],
        path_b_probabilities: Dict[str, float],
        entity_confidences: Dict[str, float] | None = None,
    ) -> Tuple[List[CodeConfidence], List[str]]:
        """
        Combine results for all codes from both pathways.

        Args:
            path_a_codes: Codes derived by NER+Rules
            path_b_probabilities: Code -> probability from ML classification
            entity_confidences: Code -> average NER entity confidence

        Returns:
            (list of CodeConfidence, list of review reasons)
        """
        if entity_confidences is None:
            entity_confidences = {}

        # Get all unique codes
        all_codes = set(path_a_codes) | set(path_b_probabilities.keys())

        results: List[CodeConfidence] = []
        review_reasons: List[str] = []

        for code in sorted(all_codes):
            deterministic_found = code in path_a_codes
            ml_prob = path_b_probabilities.get(code, 0.0)
            entity_conf = entity_confidences.get(code, 0.5)

            code_conf = self.combine(
                code=code,
                deterministic_found=deterministic_found,
                ml_probability=ml_prob,
                entity_confidence=entity_conf,
            )

            results.append(code_conf)

            if code_conf.needs_review and code_conf.review_reason:
                review_reasons.append(f"{code}: {code_conf.review_reason}")

        # Sort by confidence (highest first)
        results.sort(key=lambda c: c.confidence, reverse=True)

        return results, review_reasons
