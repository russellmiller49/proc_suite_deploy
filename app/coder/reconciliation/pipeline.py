"""End-to-end extraction-first coding pipeline.

This module provides the complete extraction-first pipeline that:
1. Extracts clinical actions from procedure note text
2. Derives CPT codes deterministically from actions
3. Optionally validates against ML predictions
4. Reconciles any discrepancies and provides recommendations

Usage:
    from app.coder.reconciliation.pipeline import (
        ExtractionFirstPipeline,
        run_extraction_first_pipeline,
    )

    # Quick usage
    result = run_extraction_first_pipeline(note_text)
    print(f"Codes: {result.final_codes}")
    print(f"Confidence: {result.confidence}")

    # With ML validation
    result = run_extraction_first_pipeline(
        note_text,
        ml_predictor=my_ml_predictor,
    )
    print(f"Recommendation: {result.reconciliation.recommendation}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.common.logger import get_logger
from app.registry.ml import ActionPredictor, ClinicalActions, PredictionResult
from app.coder.adapters.registry_coder import (
    RegistryBasedCoder,
    DerivedCode,
    DerivationResult,
)
from app.coder.reconciliation.reconciler import (
    CodeReconciler,
    ReconciliationResult,
)


logger = get_logger("coder.reconciliation.pipeline")


class MLPredictorProtocol(Protocol):
    """Protocol for ML predictors that can provide CPT predictions."""

    def predict(self, note_text: str) -> list[str]:
        """Predict CPT codes from note text."""
        ...

    def predict_proba(self, note_text: str) -> list[tuple[str, float]]:
        """Predict CPT codes with confidence scores."""
        ...


@dataclass
class PipelineResult:
    """Complete result from the extraction-first pipeline.

    Attributes:
        note_text: Original procedure note text
        actions: Extracted clinical actions
        derived_codes: CPT codes derived from actions
        ml_codes: CPT codes predicted by ML (if available)
        reconciliation: Reconciliation result (if ML was used)
        final_codes: Final recommended CPT codes
        confidence: Overall confidence in the result
        recommendation: Action recommendation
        audit_trail: Complete audit trail for compliance
    """

    note_text: str
    actions: ClinicalActions
    derived_codes: list[DerivedCode]
    ml_codes: list[str] = field(default_factory=list)
    ml_confidences: dict[str, float] = field(default_factory=dict)
    reconciliation: ReconciliationResult | None = None
    final_codes: list[str] = field(default_factory=list)
    confidence: float = 1.0
    recommendation: str = "auto_approve"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """Check if result needs human review."""
        return self.recommendation != "auto_approve"

    @property
    def audit_trail(self) -> dict:
        """Generate audit trail for compliance."""
        return {
            "extraction_method": "extraction_first_v1",
            "actions_extracted": self.actions.get_performed_procedures(),
            "derived_codes": [
                {
                    "code": c.code,
                    "description": c.description,
                    "rationale": c.rationale,
                    "evidence_fields": c.evidence_fields,
                }
                for c in self.derived_codes
            ],
            "ml_validation": {
                "enabled": bool(self.ml_codes),
                "ml_codes": self.ml_codes,
                "ml_confidences": self.ml_confidences,
            },
            "reconciliation": {
                "matched": self.reconciliation.matched if self.reconciliation else [],
                "extraction_only": self.reconciliation.extraction_only if self.reconciliation else [],
                "prediction_only": self.reconciliation.prediction_only if self.reconciliation else [],
                "recommendation": self.recommendation,
            },
            "final_codes": self.final_codes,
            "confidence": self.confidence,
        }


class ExtractionFirstPipeline:
    """Complete extraction-first CPT coding pipeline.

    This pipeline implements the extraction-first architecture:

    1. ActionPredictor: Text → ClinicalActions (structured extraction)
    2. RegistryBasedCoder: ClinicalActions → CPT Codes (deterministic rules)
    3. [Optional] ML Predictor: Text → CPT Codes (probabilistic validation)
    4. CodeReconciler: Compare and reconcile paths

    Benefits:
    - Auditable: Every code has a clear rationale and evidence
    - Deterministic: Same input → same output (for extraction path)
    - Safe: ML predictions used for validation, not primary coding
    """

    def __init__(
        self,
        action_predictor: ActionPredictor | None = None,
        registry_coder: RegistryBasedCoder | None = None,
        reconciler: CodeReconciler | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            action_predictor: Extracts clinical actions from text
            registry_coder: Derives CPT codes from actions
            reconciler: Reconciles extraction vs prediction
        """
        self.action_predictor = action_predictor or ActionPredictor()
        self.registry_coder = registry_coder or RegistryBasedCoder()
        self.reconciler = reconciler or CodeReconciler()

    def run(
        self,
        note_text: str,
        ml_predictor: MLPredictorProtocol | None = None,
    ) -> PipelineResult:
        """Run the complete extraction-first pipeline.

        Args:
            note_text: Procedure note text
            ml_predictor: Optional ML predictor for validation

        Returns:
            PipelineResult with extracted codes and recommendations
        """
        warnings: list[str] = []
        errors: list[str] = []

        # Step 1: Extract clinical actions
        logger.info("Step 1: Extracting clinical actions...")
        try:
            extraction_result = self.action_predictor.predict(note_text)
            actions = extraction_result.actions
            warnings.extend(extraction_result.warnings)
            errors.extend(extraction_result.errors)
        except Exception as e:
            logger.error(f"Action extraction failed: {e}")
            return PipelineResult(
                note_text=note_text,
                actions=ClinicalActions(),
                derived_codes=[],
                errors=[f"Action extraction failed: {e}"],
                confidence=0.0,
                recommendation="flag_for_audit",
            )

        # Step 2: Derive CPT codes from actions
        logger.info("Step 2: Deriving CPT codes from actions...")
        try:
            derivation_result = self.registry_coder.derive_codes(actions)
            derived_codes = derivation_result.codes
            warnings.extend(derivation_result.warnings)
        except Exception as e:
            logger.error(f"Code derivation failed: {e}")
            return PipelineResult(
                note_text=note_text,
                actions=actions,
                derived_codes=[],
                errors=[f"Code derivation failed: {e}"],
                confidence=0.0,
                recommendation="flag_for_audit",
            )

        derived_code_list = [c.code for c in derived_codes]

        # Step 3: ML validation (if predictor provided)
        ml_codes: list[str] = []
        ml_confidences: dict[str, float] = {}
        reconciliation: ReconciliationResult | None = None

        if ml_predictor:
            logger.info("Step 3: Running ML validation...")
            try:
                # Get ML predictions
                if hasattr(ml_predictor, "predict_proba"):
                    ml_predictions = ml_predictor.predict_proba(note_text)
                    ml_codes = [code for code, _ in ml_predictions]
                    ml_confidences = {code: conf for code, conf in ml_predictions}
                else:
                    ml_codes = ml_predictor.predict(note_text)
                    ml_confidences = {code: 0.5 for code in ml_codes}

                # Step 4: Reconcile
                logger.info("Step 4: Reconciling extraction vs prediction...")
                reconciliation = self.reconciler.reconcile(
                    derived_codes=derived_code_list,
                    predicted_codes=ml_codes,
                    prediction_confidences=ml_confidences,
                )
            except Exception as e:
                logger.warning(f"ML validation failed: {e}")
                warnings.append(f"ML validation unavailable: {e}")

        # Determine final codes and recommendation
        if reconciliation:
            final_codes = derived_code_list  # Always prefer extraction-derived codes
            confidence = reconciliation.confidence_score
            recommendation = reconciliation.recommendation
        else:
            final_codes = derived_code_list
            confidence = 1.0 if derived_code_list else 0.5
            recommendation = "auto_approve" if derived_code_list else "review_needed"

        # Add warning if no codes derived
        if not final_codes:
            warnings.append("No CPT codes derived from extraction")
            recommendation = "review_needed"

        return PipelineResult(
            note_text=note_text,
            actions=actions,
            derived_codes=derived_codes,
            ml_codes=ml_codes,
            ml_confidences=ml_confidences,
            reconciliation=reconciliation,
            final_codes=final_codes,
            confidence=confidence,
            recommendation=recommendation,
            warnings=warnings,
            errors=errors,
        )


def run_extraction_first_pipeline(
    note_text: str,
    ml_predictor: MLPredictorProtocol | None = None,
) -> PipelineResult:
    """Convenience function to run the extraction-first pipeline.

    Args:
        note_text: Procedure note text
        ml_predictor: Optional ML predictor for validation

    Returns:
        PipelineResult with extracted codes and recommendations

    Example:
        >>> result = run_extraction_first_pipeline('''
        ...     Procedure: EBUS bronchoscopy with TBNA
        ...     EBUS performed with sampling of stations 4R, 7, and 11L.
        ...     BAL obtained from RML.
        ... ''')
        >>> print(result.final_codes)
        ['31653', '31624']
        >>> print(result.confidence)
        1.0
    """
    pipeline = ExtractionFirstPipeline()
    return pipeline.run(note_text, ml_predictor)


class RegistryMLAdapter:
    """Adapter to use RegistryMLPredictor as CPT predictor for reconciliation.

    This adapter wraps the RegistryMLPredictor (which predicts procedure fields)
    and converts its predictions to CPT codes using the RegistryBasedCoder.
    """

    def __init__(self) -> None:
        """Initialize the adapter with RegistryMLPredictor and coder."""
        try:
            from ml.lib.ml_coder.registry_predictor import RegistryMLPredictor
            self._registry_predictor = RegistryMLPredictor()
            self._coder = RegistryBasedCoder()
            self.available = self._registry_predictor.available
        except Exception as e:
            logger.warning(f"RegistryMLAdapter not available: {e}")
            self._registry_predictor = None
            self._coder = None
            self.available = False

    def predict(self, note_text: str) -> list[str]:
        """Predict CPT codes via Registry ML → CPT derivation.

        Args:
            note_text: Procedure note text

        Returns:
            List of predicted CPT codes
        """
        if not self.available:
            return []

        # Get registry field predictions
        classification = self._registry_predictor.classify_case(note_text)
        positive_fields = classification.positive_fields

        # Convert to ClinicalActions-like structure for CPT derivation
        from app.registry.ml import ClinicalActions, EBUSActions, NavigationActions
        from app.registry.ml import BiopsyActions, BALActions, PleuralActions
        from app.registry.ml import CAOActions, StentActions, BLVRActions, BrushingsActions

        # Map ML predictions to ClinicalActions
        actions = ClinicalActions(
            ebus=EBUSActions(
                performed="linear_ebus" in positive_fields,
                stations=["4R", "7", "11L"] if "linear_ebus" in positive_fields else [],
            ),
            navigation=NavigationActions(
                performed="navigational_bronchoscopy" in positive_fields,
                radial_ebus_used="radial_ebus" in positive_fields,
            ),
            biopsy=BiopsyActions(
                transbronchial_performed="transbronchial_biopsy" in positive_fields,
                cryobiopsy_performed="transbronchial_cryobiopsy" in positive_fields,
            ),
            bal=BALActions(performed="diagnostic_bronchoscopy" in positive_fields),
            brushings=BrushingsActions(performed="brushings" in positive_fields),
            pleural=PleuralActions(
                thoracentesis_performed="thoracentesis" in positive_fields,
                chest_tube_performed="chest_tube" in positive_fields,
                ipc_performed="ipc" in positive_fields,
                thoracoscopy_performed="medical_thoracoscopy" in positive_fields,
                pleurodesis_performed="pleurodesis" in positive_fields,
            ),
            cao=CAOActions(
                thermal_ablation_performed="thermal_ablation" in positive_fields,
            ),
            stent=StentActions(performed="airway_stent" in positive_fields),
            blvr=BLVRActions(performed="blvr" in positive_fields),
            rigid_bronchoscopy="rigid_bronchoscopy" in positive_fields,
        )

        # Derive CPT codes from the ML-predicted actions
        derivation = self._coder.derive_codes(actions)
        return [c.code for c in derivation.codes]

    def predict_proba(self, note_text: str) -> list[tuple[str, float]]:
        """Predict CPT codes with confidence scores.

        Args:
            note_text: Procedure note text

        Returns:
            List of (code, confidence) tuples
        """
        codes = self.predict(note_text)
        # Use a default confidence since we're deriving from field predictions
        return [(code, 0.75) for code in codes]


def run_with_ml_validation(note_text: str) -> PipelineResult:
    """Run extraction-first pipeline with RegistryMLPredictor validation.

    This is the recommended way to run the double-check architecture:
    - Path A (Extraction): ActionPredictor → RegistryBasedCoder → CPT codes
    - Path B (ML Validation): RegistryMLPredictor → RegistryBasedCoder → CPT codes
    - Reconciliation: Compare paths and recommend action

    Args:
        note_text: Procedure note text

    Returns:
        PipelineResult with reconciliation between extraction and ML paths
    """
    ml_adapter = RegistryMLAdapter()

    if not ml_adapter.available:
        logger.warning("RegistryMLPredictor not available, running without ML validation")
        return run_extraction_first_pipeline(note_text)

    return run_extraction_first_pipeline(note_text, ml_predictor=ml_adapter)


__all__ = [
    "ExtractionFirstPipeline",
    "PipelineResult",
    "MLPredictorProtocol",
    "RegistryMLAdapter",
    "run_extraction_first_pipeline",
    "run_with_ml_validation",
]
