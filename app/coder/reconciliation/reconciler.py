"""Code Reconciler - Compares extraction-derived codes with ML-predicted codes.

This module implements the double-check architecture where:
- Path A (Extraction): Structured data → Deterministic rules → CPT codes
- Path B (Prediction): Raw text → ML/LLM → CPT codes

The reconciler identifies discrepancies and provides recommendations:
- auto_approve: Both paths agree, safe to proceed
- review_needed: Minor discrepancies, human should verify
- flag_for_audit: Major discrepancies, requires investigation

Usage:
    reconciler = CodeReconciler()
    result = reconciler.reconcile(
        derived_codes=["31653", "31624"],
        predicted_codes=["31653", "31624", "31625"],
    )

    if result.recommendation == "auto_approve":
        # Safe to auto-code
        pass
    elif result.recommendation == "review_needed":
        # Present to coder for review
        pass
    else:
        # Flag for audit/investigation
        pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from app.common.logger import get_logger


logger = get_logger("coder.reconciliation")


class DiscrepancyType(Enum):
    """Types of discrepancies between extraction and prediction."""

    NONE = "none"
    EXTRACTION_ONLY = "extraction_only"  # Extraction found code, ML missed
    PREDICTION_ONLY = "prediction_only"  # ML predicted code, extraction missed
    BOTH = "both"  # Discrepancies in both directions


@dataclass
class CodeDiscrepancy:
    """Details about a specific code discrepancy.

    Attributes:
        code: The CPT code in question
        source: Which path found this code ("extraction", "prediction", "both")
        reason: Explanation for the discrepancy
        confidence: ML confidence for predicted codes (None for extraction-only)
        severity: How serious is this discrepancy ("low", "medium", "high")
    """

    code: str
    source: Literal["extraction", "prediction", "both"]
    reason: str
    confidence: float | None = None
    severity: Literal["low", "medium", "high"] = "medium"


@dataclass
class ReconciliationResult:
    """Result of reconciling extraction-derived vs ML-predicted codes.

    Attributes:
        matched: Codes found by both extraction and prediction
        extraction_only: Codes derived from extraction but not predicted
        prediction_only: Codes predicted but not derived from extraction
        discrepancies: Detailed discrepancy objects
        discrepancy_type: Overall discrepancy classification
        recommendation: Action recommendation
        confidence_score: Overall confidence in the reconciliation
        review_reasons: List of reasons if review is recommended
    """

    matched: list[str] = field(default_factory=list)
    extraction_only: list[str] = field(default_factory=list)
    prediction_only: list[str] = field(default_factory=list)
    discrepancies: list[CodeDiscrepancy] = field(default_factory=list)
    discrepancy_type: DiscrepancyType = DiscrepancyType.NONE
    recommendation: Literal["auto_approve", "review_needed", "flag_for_audit"] = "auto_approve"
    confidence_score: float = 1.0
    review_reasons: list[str] = field(default_factory=list)

    @property
    def has_discrepancies(self) -> bool:
        """Check if there are any discrepancies."""
        return bool(self.extraction_only or self.prediction_only)

    @property
    def total_codes(self) -> int:
        """Total unique codes across both paths."""
        return len(set(self.matched) | set(self.extraction_only) | set(self.prediction_only))

    @property
    def agreement_rate(self) -> float:
        """Percentage of codes that matched."""
        if self.total_codes == 0:
            return 1.0
        return len(self.matched) / self.total_codes


# High-value codes that warrant extra scrutiny
HIGH_VALUE_CODES = {
    "31653",  # EBUS 3+ stations (higher RVU than 31652)
    "31628",  # Transbronchial biopsy
    "31641",  # Tumor destruction
    "32550",  # IPC placement
    "32650",  # Thoracoscopy with pleurodesis
}

# Add-on codes (must have primary)
ADD_ON_CODES = {
    "31627",  # Navigation add-on
    "+31632",  # Additional lobe biopsy
    "+31633",  # Additional lobe TBNA
    "+31637",  # Additional bronchial stent
    "+31651",  # BLVR valve
    "+31654",  # Transendoscopic US during EBUS
}

# Code families - similar codes that may indicate minor discrepancies
CODE_FAMILIES = {
    "ebus": {"31652", "31653"},
    "biopsy": {"31625", "31628", "31629"},
    "sampling": {"31622", "31623", "31624"},
    "thoracentesis": {"32554", "32555"},
    "chest_drainage": {"32556", "32557"},
}


class CodeReconciler:
    """Reconciles extraction-derived codes with ML-predicted codes.

    This class implements the double-check mechanism for the extraction-first
    architecture, comparing deterministic derivation results with probabilistic
    ML predictions.

    Configuration:
        auto_approve_threshold: Agreement rate above which to auto-approve
        prediction_confidence_threshold: ML confidence below which to ignore
        flag_high_value_discrepancies: Whether high-value codes get extra scrutiny
    """

    def __init__(
        self,
        auto_approve_threshold: float = 1.0,
        prediction_confidence_threshold: float = 0.5,
        flag_high_value_discrepancies: bool = True,
    ) -> None:
        """Initialize the reconciler.

        Args:
            auto_approve_threshold: Agreement rate for auto-approval (default 1.0 = perfect match)
            prediction_confidence_threshold: Ignore ML predictions below this confidence
            flag_high_value_discrepancies: Extra scrutiny for high-RVU codes
        """
        self.auto_approve_threshold = auto_approve_threshold
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.flag_high_value_discrepancies = flag_high_value_discrepancies

    def reconcile(
        self,
        derived_codes: list[str],
        predicted_codes: list[str],
        prediction_confidences: dict[str, float] | None = None,
    ) -> ReconciliationResult:
        """Reconcile extraction-derived codes with ML-predicted codes.

        Args:
            derived_codes: CPT codes derived from registry extraction
            predicted_codes: CPT codes predicted by ML/LLM
            prediction_confidences: Optional confidence scores for predictions

        Returns:
            ReconciliationResult with matched/unmatched codes and recommendation
        """
        prediction_confidences = prediction_confidences or {}

        # Normalize codes (remove + prefix for comparison)
        derived_set = self._normalize_codes(derived_codes)
        predicted_set = self._normalize_codes(predicted_codes)

        # Filter low-confidence predictions
        if prediction_confidences:
            predicted_set = {
                code
                for code in predicted_set
                if prediction_confidences.get(code, 1.0) >= self.prediction_confidence_threshold
            }

        # Calculate sets
        matched = list(derived_set & predicted_set)
        extraction_only = list(derived_set - predicted_set)
        prediction_only = list(predicted_set - derived_set)

        # Build discrepancy objects
        discrepancies = self._build_discrepancies(
            extraction_only, prediction_only, prediction_confidences
        )

        # Determine discrepancy type
        discrepancy_type = self._classify_discrepancy_type(extraction_only, prediction_only)

        # Calculate confidence and recommendation
        confidence_score = self._calculate_confidence(
            matched, extraction_only, prediction_only, prediction_confidences
        )
        recommendation, review_reasons = self._determine_recommendation(
            matched,
            extraction_only,
            prediction_only,
            prediction_confidences,
            confidence_score,
        )

        result = ReconciliationResult(
            matched=sorted(matched),
            extraction_only=sorted(extraction_only),
            prediction_only=sorted(prediction_only),
            discrepancies=discrepancies,
            discrepancy_type=discrepancy_type,
            recommendation=recommendation,
            confidence_score=confidence_score,
            review_reasons=review_reasons,
        )

        self._log_reconciliation(result)
        return result

    def _normalize_codes(self, codes: list[str]) -> set[str]:
        """Normalize codes for comparison (strip + prefix)."""
        return {code.lstrip("+") for code in codes}

    def _build_discrepancies(
        self,
        extraction_only: list[str],
        prediction_only: list[str],
        confidences: dict[str, float],
    ) -> list[CodeDiscrepancy]:
        """Build detailed discrepancy objects."""
        discrepancies = []

        for code in extraction_only:
            severity = self._assess_severity(code, "extraction", confidences)
            discrepancies.append(
                CodeDiscrepancy(
                    code=code,
                    source="extraction",
                    reason=f"Extraction derived {code} but ML did not predict it",
                    confidence=None,
                    severity=severity,
                )
            )

        for code in prediction_only:
            conf = confidences.get(code)
            severity = self._assess_severity(code, "prediction", confidences)
            conf_str = f"{conf:.2f}" if conf is not None else "N/A"
            discrepancies.append(
                CodeDiscrepancy(
                    code=code,
                    source="prediction",
                    reason=f"ML predicted {code} (conf={conf_str}) but extraction did not derive it",
                    confidence=conf,
                    severity=severity,
                )
            )

        return discrepancies

    def _assess_severity(
        self, code: str, source: str, confidences: dict[str, float]
    ) -> Literal["low", "medium", "high"]:
        """Assess severity of a discrepancy."""
        # High-value codes are always medium or high severity
        if code in HIGH_VALUE_CODES:
            return "high" if source == "prediction" else "medium"

        # Add-on codes are lower severity (often context-dependent)
        if code in ADD_ON_CODES or code.startswith("+"):
            return "low"

        # Check if code is in a family with a matched code
        for family_codes in CODE_FAMILIES.values():
            if code in family_codes:
                return "low"  # Same-family discrepancy is low severity

        # Default based on prediction confidence
        if source == "prediction":
            conf = confidences.get(code, 0.5)
            if conf >= 0.9:
                return "high"
            elif conf >= 0.7:
                return "medium"
            return "low"

        return "medium"

    def _classify_discrepancy_type(
        self, extraction_only: list[str], prediction_only: list[str]
    ) -> DiscrepancyType:
        """Classify the overall discrepancy type."""
        has_extraction = bool(extraction_only)
        has_prediction = bool(prediction_only)

        if not has_extraction and not has_prediction:
            return DiscrepancyType.NONE
        elif has_extraction and has_prediction:
            return DiscrepancyType.BOTH
        elif has_extraction:
            return DiscrepancyType.EXTRACTION_ONLY
        else:
            return DiscrepancyType.PREDICTION_ONLY

    def _calculate_confidence(
        self,
        matched: list[str],
        extraction_only: list[str],
        prediction_only: list[str],
        confidences: dict[str, float],
    ) -> float:
        """Calculate overall confidence score for the reconciliation."""
        total = len(matched) + len(extraction_only) + len(prediction_only)
        if total == 0:
            return 1.0

        # Base confidence from agreement rate
        agreement_rate = len(matched) / total

        # Penalize high-confidence prediction misses
        penalty = 0.0
        for code in prediction_only:
            conf = confidences.get(code, 0.5)
            if conf >= 0.9:
                penalty += 0.1
            elif conf >= 0.7:
                penalty += 0.05

        # Slight penalty for extraction-only codes (potential overcoding)
        penalty += len(extraction_only) * 0.02

        return max(0.0, agreement_rate - penalty)

    def _determine_recommendation(
        self,
        matched: list[str],
        extraction_only: list[str],
        prediction_only: list[str],
        confidences: dict[str, float],
        confidence_score: float,
    ) -> tuple[Literal["auto_approve", "review_needed", "flag_for_audit"], list[str]]:
        """Determine the recommendation based on reconciliation results."""
        reasons = []

        # Perfect match → auto_approve
        if not extraction_only and not prediction_only:
            return "auto_approve", []

        # Check for high-value discrepancies
        high_value_extraction = set(extraction_only) & HIGH_VALUE_CODES
        high_value_prediction = set(prediction_only) & HIGH_VALUE_CODES

        if self.flag_high_value_discrepancies and (high_value_extraction or high_value_prediction):
            if high_value_prediction:
                reasons.append(
                    f"High-value codes predicted but not extracted: {sorted(high_value_prediction)}"
                )
            if high_value_extraction:
                reasons.append(
                    f"High-value codes extracted but not predicted: {sorted(high_value_extraction)}"
                )
            return "flag_for_audit", reasons

        # Check for high-confidence prediction misses
        high_conf_misses = [
            code
            for code in prediction_only
            if confidences.get(code, 0) >= 0.9
        ]
        if high_conf_misses:
            reasons.append(
                f"High-confidence predictions not found in extraction: {sorted(high_conf_misses)}"
            )
            return "flag_for_audit", reasons

        # Multiple discrepancies → review_needed
        total_discrepancies = len(extraction_only) + len(prediction_only)
        if total_discrepancies >= 3:
            reasons.append(f"Multiple discrepancies detected ({total_discrepancies} total)")
            return "review_needed", reasons

        # Single discrepancy with reasonable confidence → review_needed
        if prediction_only:
            reasons.append(f"ML predicted codes not found in extraction: {sorted(prediction_only)}")
        if extraction_only:
            reasons.append(f"Extraction found codes not predicted by ML: {sorted(extraction_only)}")

        # Check if discrepancies are within same family (lower concern)
        if self._are_same_family_discrepancies(extraction_only, prediction_only):
            reasons.append("Discrepancies are within same code family (e.g., 31652 vs 31653)")
            return "review_needed", reasons

        # Default to review_needed for any discrepancy
        return "review_needed", reasons

    def _are_same_family_discrepancies(
        self, extraction_only: list[str], prediction_only: list[str]
    ) -> bool:
        """Check if discrepancies are within the same code family."""
        for family_codes in CODE_FAMILIES.values():
            extraction_in_family = set(extraction_only) & family_codes
            prediction_in_family = set(prediction_only) & family_codes
            if extraction_in_family and prediction_in_family:
                return True
        return False

    def _log_reconciliation(self, result: ReconciliationResult) -> None:
        """Log reconciliation results."""
        if result.recommendation == "auto_approve":
            logger.info(
                f"Reconciliation: auto_approve, {len(result.matched)} codes matched"
            )
        elif result.recommendation == "review_needed":
            logger.info(
                f"Reconciliation: review_needed, "
                f"matched={len(result.matched)}, "
                f"extraction_only={len(result.extraction_only)}, "
                f"prediction_only={len(result.prediction_only)}"
            )
        else:
            logger.warning(
                f"Reconciliation: flag_for_audit, "
                f"reasons={result.review_reasons}"
            )


def reconcile_codes(
    derived_codes: list[str],
    predicted_codes: list[str],
    prediction_confidences: dict[str, float] | None = None,
) -> ReconciliationResult:
    """Convenience function to reconcile codes.

    Args:
        derived_codes: CPT codes derived from registry extraction
        predicted_codes: CPT codes predicted by ML/LLM
        prediction_confidences: Optional confidence scores for predictions

    Returns:
        ReconciliationResult with matched/unmatched codes and recommendation
    """
    reconciler = CodeReconciler()
    return reconciler.reconcile(derived_codes, predicted_codes, prediction_confidences)


__all__ = [
    "CodeReconciler",
    "ReconciliationResult",
    "CodeDiscrepancy",
    "DiscrepancyType",
    "reconcile_codes",
]
