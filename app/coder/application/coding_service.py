"""Coding Service - orchestrates the extraction-first coding pipeline.

This service coordinates registry extraction, deterministic CPT derivation,
and audit metadata to produce CodeSuggestion objects for review.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from config.settings import CoderSettings
from app.domain.knowledge_base.repository import KnowledgeBaseRepository
from app.domain.coding_rules.rule_engine import RuleEngine
from app.coder.adapters.nlp.keyword_mapping_loader import KeywordMappingRepository
from app.coder.adapters.nlp.simple_negation_detector import SimpleNegationDetector
from app.coder.adapters.llm.gemini_advisor import LLMAdvisorPort
from app.coder.adapters.ml_ranker import MLRankerPort
from app.coder.application.procedure_type_detector import detect_procedure_type
from app.domain.coding_rules import apply_mer_rules, apply_ncci_edits
from app.phi.ports import PHIScrubberPort
from proc_schemas.coding import CodeSuggestion, CodingResult
from proc_schemas.reasoning import ReasoningFields
from observability.timing import timed
from observability.logging_config import get_logger

if TYPE_CHECKING:
    from app.registry.application.registry_service import RegistryService

logger = get_logger("coding_service")


class CodingService:
    """Orchestrates the extraction-first coding pipeline.

    Pipeline Steps:
    1. Registry extraction → RegistryRecord
    2. Deterministic Registry → CPT derivation
    3. RAW-ML audit metadata (if enabled by RegistryService)
    4. Build CodeSuggestion[] → return for review
    """

    VERSION = "coding_service_v1"
    POLICY_VERSION = "extraction_first_v1"

    def __init__(
        self,
        kb_repo: KnowledgeBaseRepository,
        keyword_repo: KeywordMappingRepository,
        negation_detector: SimpleNegationDetector,
        rule_engine: RuleEngine,
        llm_advisor: Optional[LLMAdvisorPort],
        config: CoderSettings,
        phi_scrubber: Optional[PHIScrubberPort] = None,
        ml_ranker: Optional[MLRankerPort] = None,
        registry_service: "RegistryService | None" = None,
    ):
        self.kb_repo = kb_repo
        self.keyword_repo = keyword_repo
        self.negation_detector = negation_detector
        self.rule_engine = rule_engine
        self.llm_advisor = llm_advisor
        self.config = config
        self.phi_scrubber = phi_scrubber
        self.ml_ranker = ml_ranker
        self.registry_service = registry_service

        # Note: PHI scrubbing is now handled at route level (app/api/phi_redaction.py).
        # The phi_scrubber parameter is deprecated and ignored.
        if phi_scrubber:
            logger.debug(
                "phi_scrubber parameter is deprecated; PHI redaction is now handled at route level"
            )

        # Hybrid pipeline dependencies are accepted for compatibility, but unused in extraction-first.

    def generate_suggestions(
        self,
        procedure_id: str,
        report_text: str,
        use_llm: bool = True,
    ) -> tuple[list[CodeSuggestion], float]:
        """Generate code suggestions for a procedure note.

        Args:
            procedure_id: Unique identifier for the procedure.
            report_text: The procedure note text.
            use_llm: Ignored (LLM advisor is not used in extraction-first).

        Returns:
            Tuple of (List of CodeSuggestion objects, LLM latency in ms).
        """
        return self._generate_suggestions_extraction_first(procedure_id, report_text)

    def generate_result(
        self,
        procedure_id: str,
        report_text: str,
        use_llm: bool = True,
        procedure_type: str | None = None,
    ) -> CodingResult:
        """Generate a complete coding result with metadata.

        Args:
            procedure_id: Unique identifier for the procedure.
            report_text: The procedure note text.
            use_llm: Ignored (LLM advisor is not used in extraction-first).
            procedure_type: Classification of the procedure (e.g., bronch_diagnostic,
                          bronch_ebus, pleural, blvr). Used for metrics segmentation.
                          If None or "unknown", auto-detection is attempted.

        Returns:
            CodingResult with suggestions and metadata.
        """
        with timed("coding_service.generate_result") as timing:
            suggestions, llm_latency_ms = self.generate_suggestions(
                procedure_id, report_text, use_llm
            )

        # Auto-detect procedure type if not provided
        if not procedure_type or procedure_type == "unknown":
            suggestion_codes = [s.code for s in suggestions]
            detected_type = detect_procedure_type(
                report_text=report_text,
                codes=suggestion_codes,
            )
            procedure_type = detected_type
            logger.debug(
                "Auto-detected procedure type",
                extra={
                    "procedure_id": procedure_id,
                    "detected_type": detected_type,
                    "codes_used": suggestion_codes[:5],  # Log first 5 codes
                },
            )

        return CodingResult(
            procedure_id=procedure_id,
            suggestions=suggestions,
            final_codes=[],  # Populated after review
            procedure_type=procedure_type,
            warnings=[],
            ncci_notes=[],
            mer_notes=[],
            kb_version=self.kb_repo.version,
            policy_version=self.POLICY_VERSION,
            model_version="",
            processing_time_ms=timing.elapsed_ms,
            llm_latency_ms=llm_latency_ms,
        )

    @staticmethod
    def _base_confidence_from_difficulty(difficulty: str) -> float:
        normalized = (difficulty or "").strip().upper()
        if normalized == "HIGH_CONF":
            return 0.95
        if normalized in ("MEDIUM", "GRAY_ZONE"):
            return 0.80
        if normalized in ("LOW_CONF", "LOW"):
            return 0.70
        return 0.70

    def _generate_suggestions_extraction_first(
        self,
        procedure_id: str,
        report_text: str,
    ) -> tuple[list[CodeSuggestion], float]:
        """Extraction-first pipeline: Registry → Deterministic CPT → ML Audit.

        This pipeline:
        1. Extracts a RegistryRecord from the note text
        2. Derives CPT codes deterministically from the registry fields
        3. Optionally audits the derived codes against raw ML predictions

        Returns:
            Tuple of (List of CodeSuggestion objects, processing latency in ms).
        """
        from app.registry.application.registry_service import RegistryService

        start_time = time.time()

        logger.info(
            "Starting coding pipeline (extraction-first mode)",
            extra={
                "procedure_id": procedure_id,
                "text_length_chars": len(report_text),
            },
        )

        # Step 1: Extract registry fields + deterministic CPT codes
        registry_service = self.registry_service or RegistryService()
        extraction_result = registry_service.extract_fields_extraction_first(report_text)

        codes = list(extraction_result.cpt_codes or [])
        rationales = dict(extraction_result.code_rationales or {})
        derivation_warnings = list(extraction_result.derivation_warnings or [])

        if not codes:
            rule_result = self.rule_engine.generate_candidates(report_text)
            suggestions: list[CodeSuggestion] = []
            candidates = getattr(rule_result, "candidates", None)
            if candidates is None:
                legacy_codes = list(getattr(rule_result, "codes", []) or [])
                legacy_conf = dict(getattr(rule_result, "confidence", {}) or {})
                candidates = [
                    {
                        "code": code,
                        "confidence": legacy_conf.get(code, 0.9),
                        "rule_path": "RULE_ENGINE_FALLBACK",
                        "rationale": "rule_engine.fallback",
                    }
                    for code in legacy_codes
                ]

            for candidate in candidates:
                if isinstance(candidate, dict):
                    code = str(candidate.get("code", "")).strip()
                    confidence = float(candidate.get("confidence", 0.9))
                    rule_path = str(candidate.get("rule_path", "RULE_ENGINE"))
                    rationale = str(candidate.get("rationale", "derived"))
                else:
                    code = str(getattr(candidate, "code", "")).strip()
                    confidence = float(getattr(candidate, "confidence", 0.9))
                    rule_path = str(getattr(candidate, "rule_path", "RULE_ENGINE"))
                    rationale = str(getattr(candidate, "rationale", "derived"))

                if not code:
                    continue

                proc_info = self.kb_repo.get_procedure_info(code)
                description = proc_info.description if proc_info else ""

                reasoning = ReasoningFields(
                    trigger_phrases=[],
                    evidence_spans=[],
                    rule_paths=[rule_path, rationale],
                    ncci_notes="",
                    mer_notes="",
                    confidence=confidence,
                    kb_version=self.kb_repo.version,
                    policy_version=self.POLICY_VERSION,
                )

                suggestions.append(
                    CodeSuggestion(
                        code=code,
                        description=description,
                        source="rules",
                        hybrid_decision="kept_rule_priority",
                        rule_confidence=confidence,
                        llm_confidence=None,
                        final_confidence=confidence,
                        reasoning=reasoning,
                        review_flag="optional",
                        trigger_phrases=[],
                        evidence_verified=True,
                        suggestion_id=str(uuid.uuid4()),
                        procedure_id=procedure_id,
                    )
                )

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "Extraction-first derivation produced no CPT codes; using rules fallback",
                extra={
                    "procedure_id": procedure_id,
                    "fallback_code_count": len(suggestions),
                    "processing_time_ms": int(latency_ms),
                },
            )
            return suggestions, 0.0

        # Step 2: Build audit warnings
        audit_warnings: list[str] = list(extraction_result.audit_warnings or [])
        audit_warnings.extend(derivation_warnings)

        # Determine difficulty level
        base_confidence = self._base_confidence_from_difficulty(
            extraction_result.coder_difficulty
        )

        # Step 3: Build CodeSuggestion objects
        suggestions: list[CodeSuggestion] = []
        for code in codes:
            rationale = rationales.get(code, "derived")

            # Format audit warnings for mer_notes
            mer_notes = ""
            if audit_warnings:
                mer_notes = "AUDIT FLAGS:\n" + "\n".join(f"• {w}" for w in audit_warnings)

            reasoning = ReasoningFields(
                trigger_phrases=[],
                evidence_spans=[],
                rule_paths=[f"DETERMINISTIC: {rationale}"],
                ncci_notes="",
                mer_notes=mer_notes,
                confidence=base_confidence,
                kb_version=self.kb_repo.version,
                policy_version=self.POLICY_VERSION,
            )

            # Determine review flag
            if extraction_result.needs_manual_review:
                review_flag = "required"
            elif audit_warnings:
                review_flag = "recommended"
            else:
                review_flag = "optional"

            # Get procedure description
            proc_info = self.kb_repo.get_procedure_info(code)
            description = proc_info.description if proc_info else ""

            suggestion = CodeSuggestion(
                code=code,
                description=description,
                source="hybrid",  # Extraction-first is a form of hybrid
                hybrid_decision="EXTRACTION_FIRST",
                rule_confidence=base_confidence,
                llm_confidence=None,
                final_confidence=base_confidence,
                reasoning=reasoning,
                review_flag=review_flag,
                trigger_phrases=[],
                evidence_verified=True,
                suggestion_id=str(uuid.uuid4()),
                procedure_id=procedure_id,
            )
            suggestions.append(suggestion)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "Coding complete (extraction-first mode)",
            extra={
                "procedure_id": procedure_id,
                "num_suggestions": len(suggestions),
                "processing_time_ms": latency_ms,
                "codes": codes,
            },
        )

        return suggestions, latency_ms
