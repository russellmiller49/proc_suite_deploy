"""Parallel pathway orchestrator for CPT coding.

Runs both Path A (NER+Rules) and Path B (ML Classification) and combines
results with reconciliation and review flagging.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from app.common.logger import get_logger
from app.ner import GranularNERPredictor, NERExtractionResult
from app.registry.ner_mapping import NERToRegistryMapper, RegistryMappingResult
from app.coder.domain_rules.registry_to_cpt.coding_rules import derive_all_codes_with_meta
from app.coder.parallel_pathway.confidence_combiner import (
    ConfidenceCombiner,
    CodeConfidence,
)
from app.coder.parallel_pathway.reconciler import CodeReconciler, ReconciledCode
from app.common.spans import Span

logger = get_logger("coder.parallel_pathway")


@dataclass
class PathwayResult:
    """Result from a single pathway (A or B)."""

    codes: List[str]
    """CPT codes derived/predicted."""

    confidences: Dict[str, float]
    """Code -> confidence score."""

    rationales: Dict[str, str]
    """Code -> explanation."""

    source: Literal["ner_rules", "ml_classification"]
    """Which pathway produced this result."""

    processing_time_ms: float
    """Time taken in milliseconds."""

    # Additional details
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelPathwayResult:
    """Combined result from parallel pathway execution."""

    # Final outputs
    final_codes: List[str]
    """Combined codes with confidence above threshold."""

    final_confidences: Dict[str, float]
    """Code -> final combined confidence."""

    # Per-pathway results
    path_a_result: PathwayResult
    """NER + Rules result."""

    path_b_result: PathwayResult
    """ML Classification result."""

    # Confidence details
    code_confidences: List[CodeConfidence]
    """Detailed confidence for each code."""

    # Review flags
    needs_review: bool
    """True if any code needs human review."""

    review_reasons: List[str]
    """Reasons why review is needed."""

    # Explanations
    explanations: Dict[str, str]
    """Human-readable per-code explanations."""

    # Timing
    total_time_ms: float
    """Total processing time."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "final_codes": self.final_codes,
            "final_confidences": self.final_confidences,
            "pathway_results": {
                "path_a": {
                    "source": self.path_a_result.source,
                    "codes": self.path_a_result.codes,
                    "confidences": self.path_a_result.confidences,
                    "rationales": self.path_a_result.rationales,
                    "time_ms": self.path_a_result.processing_time_ms,
                },
                "path_b": {
                    "source": self.path_b_result.source,
                    "codes": self.path_b_result.codes,
                    "confidences": self.path_b_result.confidences,
                    "time_ms": self.path_b_result.processing_time_ms,
                },
            },
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "explanations": self.explanations,
            "total_time_ms": self.total_time_ms,
        }


class ParallelPathwayOrchestrator:
    """Orchestrates parallel NER+Rules and ML Classification pathways.

    Architecture:
        Text -> [Path A] -> NER -> Registry -> Rules -> Codes
             -> [Path B] -> ML Classifier -> Probabilities
                        |
                        v
               [Reconciler/Combiner]
                        |
                        v
               Final Codes + Confidence + Review Flags
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        ner_predictor: Optional[GranularNERPredictor] = None,
        ner_mapper: Optional[NERToRegistryMapper] = None,
        ml_predictor: Optional[Any] = None,  # TorchRegistryPredictor
        reconciler: Optional[CodeReconciler] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            ner_predictor: NER model for Path A (created if None)
            ner_mapper: NER-to-Registry mapper (created if None)
            ml_predictor: ML classifier for Path B (optional)
            confidence_threshold: Minimum confidence to include code
        """
        self.ner_predictor = ner_predictor or GranularNERPredictor(
            confidence_threshold=0.1  # Lower threshold for NER
        )
        self.ner_mapper = ner_mapper or NERToRegistryMapper()
        self.ml_predictor = ml_predictor
        self.confidence_threshold = confidence_threshold
        self.confidence_combiner = ConfidenceCombiner()
        self.reconciler = reconciler or CodeReconciler()

        logger.info(
            "ParallelPathwayOrchestrator initialized: NER=%s, ML=%s",
            "available" if self.ner_predictor.available else "unavailable",
            "available" if self.ml_predictor else "unavailable",
        )

    def process(
        self,
        note_text: str,
        ml_predictor: Optional[Any] = None,
    ) -> ParallelPathwayResult:
        """
        Run both pathways and combine results.

        Args:
            note_text: The procedure note text

        Returns:
            ParallelPathwayResult with combined codes and review flags
        """
        start_time = time.time()

        # Run Path A: NER -> Registry -> Rules
        path_a_result = self._run_path_a(note_text)

        # Run Path B: ML Classification
        path_b_result = self._run_path_b(note_text, ml_predictor=ml_predictor)

        # Combine results
        code_confidences, review_reasons = self.confidence_combiner.combine_all(
            path_a_codes=path_a_result.codes,
            path_b_probabilities=path_b_result.confidences,
        )

        # Filter by confidence threshold
        final_codes = [
            cc.code for cc in code_confidences
            if cc.confidence >= self.confidence_threshold
        ]

        final_confidences = {
            cc.code: cc.confidence for cc in code_confidences
            if cc.confidence >= self.confidence_threshold
        }

        # Build explanations
        explanations = {cc.code: cc.explanation for cc in code_confidences}

        # Determine if review is needed
        needs_review = any(cc.needs_review for cc in code_confidences)

        total_time = (time.time() - start_time) * 1000

        return ParallelPathwayResult(
            final_codes=final_codes,
            final_confidences=final_confidences,
            path_a_result=path_a_result,
            path_b_result=path_b_result,
            code_confidences=code_confidences,
            needs_review=needs_review,
            review_reasons=review_reasons,
            explanations=explanations,
            total_time_ms=total_time,
        )

    def run_parallel_process(
        self,
        note_text: str,
        ml_predictor: Optional[Any] = None,
    ):
        """Run parallel NER + ML reconciliation and return a RegistryExtractionResult."""
        parallel_result = self.process(note_text, ml_predictor=ml_predictor)
        path_a_result = parallel_result.path_a_result
        record = path_a_result.details.get("record")

        if record is None:
            from app.registry.schema import RegistryRecord

            record = RegistryRecord()

        ner_evidence = self._build_ner_evidence(path_a_result.details.get("ner_entities"))
        if ner_evidence:
            record_evidence = getattr(record, "evidence", None)
            if not isinstance(record_evidence, dict):
                record_evidence = {}
            for key, spans in ner_evidence.items():
                record_evidence.setdefault(key, []).extend(spans)
            record.evidence = record_evidence

        from app.registry.application.cpt_registry_mapping import aggregate_registry_fields
        from app.registry.application.registry_service import RegistryExtractionResult

        mapped_fields = (
            aggregate_registry_fields(parallel_result.final_codes, version="v3")
            if parallel_result.final_codes
            else {}
        )

        warnings = list(path_a_result.details.get("mapping_warnings", []))
        derivation_warnings = list(path_a_result.details.get("rules_warnings", []))

        return RegistryExtractionResult(
            record=record,
            cpt_codes=sorted(parallel_result.final_codes),
            coder_difficulty="unknown",
            coder_source="parallel_ner",
            mapped_fields=mapped_fields,
            code_rationales=path_a_result.rationales,
            derivation_warnings=derivation_warnings,
            warnings=warnings,
            needs_manual_review=parallel_result.needs_review,
            audit_warnings=list(parallel_result.review_reasons or []),
        )

    def _ensure_ml_predictor(self) -> Any | None:
        if self.ml_predictor is not None:
            return self.ml_predictor
        try:
            from ml.lib.ml_coder.registry_predictor import RegistryMLPredictor

            predictor = RegistryMLPredictor()
            if predictor.available:
                self.ml_predictor = predictor
                return predictor
        except Exception as exc:
            logger.debug("RegistryMLPredictor unavailable: %s", exc)
        return None

    def _build_ner_evidence(self, entities: list[Any] | None) -> dict[str, list[Span]]:
        spans: list[Span] = []
        for ent in entities or []:
            start = getattr(ent, "start_char", None)
            end = getattr(ent, "end_char", None)
            text = getattr(ent, "text", None)
            confidence = getattr(ent, "confidence", None)
            if start is None or end is None or text is None:
                continue
            spans.append(
                Span(
                    text=str(text),
                    start=int(start),
                    end=int(end),
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
        if not spans:
            return {}
        return {"ner_spans": spans}

    def _predict_ml_probabilities(
        self,
        note_text: str,
        ml_predictor: Optional[Any] = None,
    ) -> Dict[str, float]:
        predictor = ml_predictor or self._ensure_ml_predictor()
        if predictor is None:
            return {}
        if hasattr(predictor, "available") and not predictor.available:
            return {}
        if not hasattr(predictor, "predict_proba"):
            return {}

        try:
            preds = predictor.predict_proba(note_text)
        except Exception as exc:
            logger.debug("ML probability prediction failed: %s", exc)
            return {}

        from app.registry.audit.raw_ml_auditor import FLAG_TO_CPT_MAP

        ml_probabilities: Dict[str, float] = {}
        for pred in preds:
            if isinstance(pred, dict):
                field = pred.get("field") or pred.get("flag_name")
                prob = pred.get("probability") or pred.get("prob")
            else:
                field = getattr(pred, "field", None) or getattr(pred, "flag_name", None)
                prob = getattr(pred, "probability", None)
            if field is None or prob is None:
                continue
            for code in FLAG_TO_CPT_MAP.get(str(field), []):
                ml_probabilities[code] = max(ml_probabilities.get(code, 0.0), float(prob))

        return ml_probabilities

    def _run_path_a(self, note_text: str) -> PathwayResult:
        """Run Path A: NER -> Registry -> Rules."""
        start_time = time.time()

        codes: List[str] = []
        confidences: Dict[str, float] = {}
        rationales: Dict[str, str] = {}
        details: Dict[str, Any] = {}

        try:
            # 1. Run NER
            ner_result = self.ner_predictor.predict(note_text)
            details["ner_entity_count"] = len(ner_result.entities)
            details["ner_time_ms"] = ner_result.inference_time_ms
            details["ner_entities"] = ner_result.entities

            # 2. Map to Registry
            mapping_result = self.ner_mapper.map_entities(ner_result)
            record = mapping_result.record
            details["record"] = record  # Store for service integration
            details["mapping_result"] = mapping_result  # Full mapping result
            details["stations_sampled_count"] = mapping_result.stations_sampled_count
            details["mapping_warnings"] = mapping_result.warnings

            # 3. Run Rules
            codes_list, rules_rationales, rules_warnings = derive_all_codes_with_meta(record)
            codes = codes_list
            rationales = rules_rationales
            details["rules_warnings"] = rules_warnings

            # Set confidence based on entity confidence (simplified)
            for code in codes:
                # Higher confidence if we have strong NER evidence
                entity_conf = sum(e.confidence for e in ner_result.entities) / max(len(ner_result.entities), 1)
                confidences[code] = min(0.95, 0.5 + entity_conf * 0.5)

        except Exception as e:
            logger.warning("Path A failed: %s", e)
            details["error"] = str(e)

        processing_time = (time.time() - start_time) * 1000

        return PathwayResult(
            codes=codes,
            confidences=confidences,
            rationales=rationales,
            source="ner_rules",
            processing_time_ms=processing_time,
            details=details,
        )

    def _run_path_b(
        self,
        note_text: str,
        ml_predictor: Optional[Any] = None,
    ) -> PathwayResult:
        """Run Path B: ML Classification."""
        start_time = time.time()

        codes: List[str] = []
        confidences: Dict[str, float] = {}
        rationales: Dict[str, str] = {}
        details: Dict[str, Any] = {}

        predictor = ml_predictor or self.ml_predictor
        try:
            if predictor and getattr(predictor, "available", False):
                classify_case = getattr(predictor, "classify_case", None)
                predict_proba = getattr(predictor, "predict_proba", None)

                if callable(classify_case):
                    result = classify_case(note_text)
                elif callable(predict_proba):
                    preds = predict_proba(note_text)
                    positive_fields = [p.field for p in preds if getattr(p, "is_positive", False)]
                    difficulty = "HIGH_CONF" if positive_fields else "LOW_CONF"
                    result = {
                        "predictions": preds,
                        "positive_fields": positive_fields,
                        "difficulty": difficulty,
                    }
                else:
                    predicted_fields = list(getattr(predictor, "predict", lambda _: [])(note_text) or [])
                    result = {
                        "predictions": [
                            {"field": field, "probability": 0.5, "threshold": 0.0, "is_positive": True}
                            for field in predicted_fields
                        ],
                        "positive_fields": predicted_fields,
                        "difficulty": "HIGH_CONF" if predicted_fields else "LOW_CONF",
                    }

                # Convert predictions to codes using FLAG_TO_CPT_MAP
                from app.registry.audit.raw_ml_auditor import FLAG_TO_CPT_MAP

                predictions = getattr(result, "predictions", None)
                if predictions is None and isinstance(result, dict):
                    predictions = result.get("predictions")

                for pred in predictions or []:
                    is_positive = getattr(pred, "is_positive", None)
                    if is_positive is None and isinstance(pred, dict):
                        is_positive = pred.get("is_positive")
                    if not is_positive:
                        continue

                    field_name = getattr(pred, "field", None)
                    if field_name is None and isinstance(pred, dict):
                        field_name = pred.get("field")
                    if not field_name:
                        continue

                    probability = getattr(pred, "probability", None)
                    if probability is None and isinstance(pred, dict):
                        probability = pred.get("probability")
                    try:
                        prob_val = float(probability) if probability is not None else 0.5
                    except (TypeError, ValueError):
                        prob_val = 0.5

                    if field_name in FLAG_TO_CPT_MAP:
                        for cpt_code in FLAG_TO_CPT_MAP[field_name]:
                            codes.append(cpt_code)
                            confidences[cpt_code] = prob_val
                            rationales[cpt_code] = f"ML predicted {field_name}={prob_val:.2f}"

                positive_fields = getattr(result, "positive_fields", None)
                difficulty = getattr(result, "difficulty", None)
                if isinstance(result, dict):
                    positive_fields = result.get("positive_fields", positive_fields)
                    difficulty = result.get("difficulty", difficulty)

                details["ml_positive_fields"] = list(positive_fields or [])
                details["ml_difficulty"] = str(difficulty or "")
            else:
                details["ml_available"] = False

        except Exception as e:
            logger.warning("Path B failed: %s", e)
            details["error"] = str(e)

        processing_time = (time.time() - start_time) * 1000

        return PathwayResult(
            codes=codes,
            confidences=confidences,
            rationales=rationales,
            source="ml_classification",
            processing_time_ms=processing_time,
            details=details,
        )

    async def process_async(
        self,
        note_text: str,
        ml_predictor: Optional[Any] = None,
    ) -> ParallelPathwayResult:
        """
        Async version that runs both pathways concurrently.

        Args:
            note_text: The procedure note text

        Returns:
            ParallelPathwayResult with combined codes and review flags
        """
        start_time = time.time()

        # Run both pathways concurrently
        loop = asyncio.get_event_loop()
        path_a_task = loop.run_in_executor(None, self._run_path_a, note_text)
        path_b_task = loop.run_in_executor(None, self._run_path_b, note_text, ml_predictor)

        path_a_result, path_b_result = await asyncio.gather(path_a_task, path_b_task)

        # Combine results (same as sync version)
        code_confidences, review_reasons = self.confidence_combiner.combine_all(
            path_a_codes=path_a_result.codes,
            path_b_probabilities=path_b_result.confidences,
        )

        final_codes = [
            cc.code for cc in code_confidences
            if cc.confidence >= self.confidence_threshold
        ]

        final_confidences = {
            cc.code: cc.confidence for cc in code_confidences
            if cc.confidence >= self.confidence_threshold
        }

        explanations = {cc.code: cc.explanation for cc in code_confidences}
        needs_review = any(cc.needs_review for cc in code_confidences)

        total_time = (time.time() - start_time) * 1000

        return ParallelPathwayResult(
            final_codes=final_codes,
            final_confidences=final_confidences,
            path_a_result=path_a_result,
            path_b_result=path_b_result,
            code_confidences=code_confidences,
            needs_review=needs_review,
            review_reasons=review_reasons,
            explanations=explanations,
            total_time_ms=total_time,
        )
