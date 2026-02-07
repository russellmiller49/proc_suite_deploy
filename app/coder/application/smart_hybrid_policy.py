"""Smart Hybrid Policy for merging rule-based and LLM advisor codes.

This module implements the core logic for combining deterministic rule-based
coding with LLM suggestions in a safe, explainable way.

Includes:
- HybridPolicy: Legacy merger for rule-based + LLM advisor codes
- SmartHybridOrchestrator: ML-first ternary classification (HIGH_CONF/GRAY_ZONE/LOW_CONF)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from config.settings import CoderSettings
from app.common.logger import get_logger
from app.domain.knowledge_base.repository import KnowledgeBaseRepository
from app.coder.adapters.nlp.keyword_mapping_loader import KeywordMappingRepository
from app.coder.adapters.nlp.simple_negation_detector import SimpleNegationDetector
from ml.lib.ml_coder.thresholds import CaseDifficulty

logger = get_logger("smart_hybrid_policy")


class HybridDecision(str, Enum):
    """Decision outcomes from the hybrid merge process."""

    ACCEPTED_AGREEMENT = "accepted_agreement"  # rule ∩ advisor
    ACCEPTED_HYBRID = "accepted_hybrid"  # advisor-only, verified
    KEPT_RULE_PRIORITY = "kept_rule_priority"  # rule-only, high confidence
    REJECTED_HYBRID = "rejected_hybrid"  # advisor-only, failed verification
    DROPPED_LOW_CONFIDENCE = "dropped_low_conf"  # rule-only, low conf + advisor omit
    HUMAN_REVIEW_REQUIRED = "human_review"  # high conf but verification failed


@dataclass
class RuleResult:
    """Result from the rule-based coding engine."""

    codes: List[str]
    confidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdvisorResult:
    """Result from the LLM advisor."""

    codes: List[str]
    confidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class HybridCandidate:
    """A candidate code with its merge decision and metadata."""

    code: str
    decision: HybridDecision
    rule_confidence: Optional[float]
    llm_confidence: Optional[float]
    flags: List[str] = field(default_factory=list)
    evidence_verified: bool = False
    trigger_phrases: List[str] = field(default_factory=list)


class HybridPolicy:
    """Smart hybrid policy that merges rule-based and LLM advisor codes.

    This implements the merge_autocode_and_advisor() logic using:
      - KnowledgeBaseRepository for VALID_CPT_LIST
      - KeywordMappingRepository for positive/negative phrases + context window
      - NegationDetectionPort for negation detection
      - Thresholds from CoderSettings
    """

    POLICY_VERSION = "smart_hybrid_v2"

    def __init__(
        self,
        kb_repo: KnowledgeBaseRepository,
        keyword_repo: KeywordMappingRepository,
        negation_detector: SimpleNegationDetector,
        config: CoderSettings,
    ) -> None:
        self.kb_repo = kb_repo
        self.keyword_repo = keyword_repo
        self.negation_detector = negation_detector
        self.config = config

        # Defensive filter to block hallucinated codes
        self.valid_cpt_set: Set[str] = set(kb_repo.get_all_codes())

    @property
    def version(self) -> str:
        return self.POLICY_VERSION

    def merge(
        self,
        rule_result: RuleResult,
        advisor_result: AdvisorResult,
        report_text: str,
        policy: str = "smart_hybrid",
    ) -> List[HybridCandidate]:
        """Merge rule-based and LLM advisor results.

        Args:
            rule_result: Codes and confidences from rule-based engine.
            advisor_result: Codes and confidences from LLM advisor.
            report_text: The original procedure note text.
            policy: Merge policy to use ("rules_only" or "smart_hybrid").

        Returns:
            List of HybridCandidate objects with merge decisions.
        """
        if policy == "rules_only":
            return [
                HybridCandidate(
                    code=c,
                    decision=HybridDecision.KEPT_RULE_PRIORITY,
                    rule_confidence=rule_result.confidence.get(c, 1.0),
                    llm_confidence=None,
                    flags=["rules_only_mode"],
                )
                for c in sorted(set(rule_result.codes))
            ]

        if policy != "smart_hybrid":
            raise ValueError(
                f"Unknown policy: {policy}. Use 'rules_only' or 'smart_hybrid'."
            )

        rule_codes = list(rule_result.codes)
        rule_conf = dict(rule_result.confidence)
        advisor_codes = list(advisor_result.codes)
        advisor_conf = dict(advisor_result.confidence)

        candidates: Dict[str, HybridCandidate] = {}
        rule_set = set(rule_codes)
        advisor_set = set(advisor_codes)

        # 1. Agreement (rule ∩ advisor)
        agreement = rule_set.intersection(advisor_set)
        for code in agreement:
            candidates[code] = HybridCandidate(
                code=code,
                decision=HybridDecision.ACCEPTED_AGREEMENT,
                rule_confidence=rule_conf.get(code),
                llm_confidence=advisor_conf.get(code),
                flags=["rules_and_advisor_agree"],
                evidence_verified=True,
            )

        # 2. LLM additions (advisor-only codes)
        additions = advisor_set - rule_set
        for code in additions:
            llm_conf = advisor_conf.get(code, 0.0)

            # 2A. Validate CPT code exists in KB
            if code not in self.valid_cpt_set:
                candidates[code] = HybridCandidate(
                    code=code,
                    decision=HybridDecision.REJECTED_HYBRID,
                    rule_confidence=None,
                    llm_confidence=llm_conf,
                    flags=[
                        f"DISCARDED_INVALID_CODE: Advisor suggested invalid CPT {code}"
                    ],
                )
                continue

            # 2B. Confidence + verification
            if llm_conf >= self.config.advisor_confidence_auto_accept:
                verified, trigger_phrases = self._verify_code_in_text(code, report_text)
                if verified:
                    candidates[code] = HybridCandidate(
                        code=code,
                        decision=HybridDecision.ACCEPTED_HYBRID,
                        rule_confidence=None,
                        llm_confidence=llm_conf,
                        flags=[
                            f"ACCEPTED_HYBRID: advisor_conf={llm_conf:.2f}, "
                            "verified_by_keywords"
                        ],
                        evidence_verified=True,
                        trigger_phrases=trigger_phrases,
                    )
                else:
                    candidates[code] = HybridCandidate(
                        code=code,
                        decision=HybridDecision.HUMAN_REVIEW_REQUIRED,
                        rule_confidence=None,
                        llm_confidence=llm_conf,
                        flags=[
                            "HUMAN_REVIEW_REQUIRED: high_conf_advisor_but_verification_failed"
                        ],
                    )
            else:
                candidates[code] = HybridCandidate(
                    code=code,
                    decision=HybridDecision.REJECTED_HYBRID,
                    rule_confidence=None,
                    llm_confidence=llm_conf,
                    flags=[
                        f"REJECTED_LOW_CONF_ADVISOR: llm_conf={llm_conf:.2f} "
                        "below_auto_accept"
                    ],
                )

        # 3. Rule-only codes (omissions from LLM)
        omissions = rule_set - advisor_set
        for code in omissions:
            rule_c = rule_conf.get(code, 0.0)

            if rule_c >= self.config.rule_confidence_low_threshold:
                # Trust rules when confidence is acceptable
                candidates[code] = HybridCandidate(
                    code=code,
                    decision=HybridDecision.KEPT_RULE_PRIORITY,
                    rule_confidence=rule_c,
                    llm_confidence=None,
                    flags=[
                        f"KEPT_RULE_PRIORITY: rule_conf={rule_c:.2f}, "
                        "advisor_omitted_code"
                    ],
                    evidence_verified=True,
                )
            else:
                # Low confidence rule that advisor also omitted → drop
                candidates[code] = HybridCandidate(
                    code=code,
                    decision=HybridDecision.DROPPED_LOW_CONFIDENCE,
                    rule_confidence=rule_c,
                    llm_confidence=None,
                    flags=[
                        f"DROPPED_LOW_CONF: rule_conf={rule_c:.2f}, "
                        "advisor_omitted_code"
                    ],
                )

        # Return in sorted code order for determinism
        return [candidates[c] for c in sorted(candidates.keys())]

    def _verify_code_in_text(
        self, code: str, report_text: str
    ) -> tuple[bool, list[str]]:
        """Verify that a CPT code is actually supported by the text.

        Uses YAML-backed positive/negative phrases and a context window.

        Args:
            code: The CPT code to verify.
            report_text: The procedure note text.

        Returns:
            Tuple of (verified, trigger_phrases_found).
        """
        mapping = self.keyword_repo.get_mapping(code)
        if not mapping:
            # Fail-safe: if we don't know how to verify, return False
            return False, []

        positive_phrases = mapping.positive_phrases
        negative_phrases = mapping.negative_phrases
        context_window = (
            mapping.context_window_chars or self.config.context_window_chars
        )

        text_lower = report_text.lower()
        found_triggers: list[str] = []

        for pos in positive_phrases:
            # Find all occurrences of the positive phrase
            pattern = r"\b" + re.escape(pos.lower()) + r"\b"
            for match in re.finditer(pattern, text_lower):
                start_idx = max(0, match.start() - context_window)
                end_idx = min(len(text_lower), match.end() + context_window)
                context = text_lower[start_idx:end_idx]

                # Use negation detector to check if this span is negated
                if self.negation_detector.is_negated_simple(context, negative_phrases):
                    continue

                # Found at least one non-negated positive phrase
                found_triggers.append(pos)
                return True, found_triggers

        return False, found_triggers

    def get_accepted_codes(self, candidates: List[HybridCandidate]) -> List[str]:
        """Get codes that should be accepted for billing.

        Args:
            candidates: List of hybrid candidates from merge().

        Returns:
            List of accepted CPT codes.
        """
        accepted_decisions = {
            HybridDecision.ACCEPTED_AGREEMENT,
            HybridDecision.ACCEPTED_HYBRID,
            HybridDecision.KEPT_RULE_PRIORITY,
        }

        return [c.code for c in candidates if c.decision in accepted_decisions]

    def get_review_required_codes(
        self, candidates: List[HybridCandidate]
    ) -> List[str]:
        """Get codes that require human review.

        Args:
            candidates: List of hybrid candidates from merge().

        Returns:
            List of CPT codes requiring review.
        """
        return [
            c.code
            for c in candidates
            if c.decision == HybridDecision.HUMAN_REVIEW_REQUIRED
        ]


# -----------------------------------------------------------------------------
# ML-First Smart Hybrid Orchestrator
# -----------------------------------------------------------------------------


@dataclass
class HybridCoderResult:
    """Result from the SmartHybridOrchestrator.

    Attributes:
        codes: Final list of validated CPT codes
        source: Where the final codes came from (ml_rules_fastpath, hybrid_llm_fallback)
        difficulty: ML case difficulty classification (HIGH_CONF/GRAY_ZONE/LOW_CONF)
        metadata: Additional context about the decision process
    """

    codes: List[str]
    source: str
    difficulty: CaseDifficulty | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Backwards compatibility alias
OrchestratorResult = HybridCoderResult


class SmartHybridOrchestrator:
    """ML-first hybrid orchestrator with ternary case difficulty classification.

    This implements the ML → Rules → LLM flow:

    1. ML predicts codes and classifies case difficulty (HIGH_CONF/GRAY_ZONE/LOW_CONF)
    2. Rules engine validates ML output, may veto impossible combinations
    3. Decision gate:
       - HIGH_CONF + rules OK → return codes via fast path (no LLM)
       - GRAY_ZONE or rules conflict → LLM as final judge (ML provides hints)
       - LOW_CONF → LLM as primary coder (ML opinion is weak context)

    Key difference from union approach: Once LLM is invoked, its output is treated
    as the candidate truth. We do NOT union ML and LLM codes.
    """

    def __init__(
        self,
        ml_predictor: Any,  # MLCoderPredictor
        rules_engine: Any,  # CodingRulesEngine
        llm_advisor: Any,  # LLMAdvisorPort
    ):
        """
        Initialize the orchestrator.

        Args:
            ml_predictor: MLCoderPredictor instance for ML predictions
            rules_engine: CodingRulesEngine instance for validation/veto
            llm_advisor: LLMAdvisorPort instance for LLM fallback
        """
        self._ml = ml_predictor
        self._rules = rules_engine
        self._llm = llm_advisor

    def get_codes(self, note_text: str) -> HybridCoderResult:
        """
        Get CPT codes using ML-first hybrid approach.

        Args:
            note_text: The procedure note text to code

        Returns:
            HybridCoderResult with final codes, difficulty classification, and metadata
        """
        # 1. ML Prediction + difficulty classification
        ml_result = self._ml.classify_case(note_text)
        difficulty = ml_result.difficulty.value

        # Get ML candidates (codes above lower threshold)
        ml_candidates = [
            p.cpt for p in ml_result.high_conf
        ] + [
            p.cpt for p in ml_result.gray_zone
        ]

        # Build predictions dict for metadata
        preds = [{"cpt": p.cpt, "prob": p.prob} for p in ml_result.predictions]

        logger.info(
            "ML classification: difficulty=%s, high_conf=%d, gray_zone=%d, candidates=%s",
            difficulty,
            len(ml_result.high_conf),
            len(ml_result.gray_zone),
            ml_candidates,
        )

        # 2. Try rules validation on ML candidates
        rules_cleaned_ml: List[str] = []
        rules_error: Optional[str] = None
        rules_error_type: Optional[str] = None

        candidates_for_rules = ml_candidates
        if difficulty == CaseDifficulty.HIGH_CONF.value:
            from app.coder.application.candidate_expansion import expand_candidates

            candidates_for_rules = expand_candidates(note_text, ml_candidates)

        if candidates_for_rules:
            try:
                rules_cleaned_ml = self._rules.validate(
                    candidates_for_rules, note_text, strict=True
                )
            except Exception as e:
                # RuleViolationError or other validation error
                rules_error = str(e)
                rules_error_type = type(e).__name__
                rules_cleaned_ml = []
                logger.warning(
                    "Rules validation failed: error_type=%s, message=%s",
                    rules_error_type,
                    rules_error,
                )

        # 3. Decision gate
        if difficulty == CaseDifficulty.HIGH_CONF.value and rules_cleaned_ml:
            # Fast path: ML + Rules agree, no LLM needed
            self._emit_telemetry(
                difficulty=difficulty,
                source="ml_rules_fastpath",
                llm_used=False,
                rules_error=rules_error,
                rules_error_type=rules_error_type,
                ml_candidates_count=len(ml_candidates),
                final_codes_count=len(rules_cleaned_ml),
            )
            return HybridCoderResult(
                codes=rules_cleaned_ml,
                source="ml_rules_fastpath",
                difficulty=ml_result.difficulty,
                metadata={
                    "ml_difficulty": difficulty,
                    "ml_candidates": ml_candidates,
                    "ml_predictions": preds,
                    "rules_error": rules_error,
                    "rules_error_type": rules_error_type,
                    "llm_called": False,
                },
            )

        # 4. LLM fallback — LLM is the final judge
        reason_for_fallback = "low_confidence"
        if difficulty == CaseDifficulty.GRAY_ZONE.value:
            reason_for_fallback = "gray_zone"
        if rules_error:
            reason_for_fallback = f"rule_conflict: {rules_error}"

        logger.info(
            "LLM fallback triggered: reason=%s",
            reason_for_fallback,
        )

        # Build context for LLM
        llm_context = {
            "ml_suggestion": ml_candidates,
            "difficulty": difficulty,
            "reason_for_fallback": reason_for_fallback,
            "ml_predictions": preds[:10],  # Top 10 predictions as context
        }

        llm_codes = self._call_llm_with_context(note_text, llm_context)

        if not llm_codes:
            fallback_codes = rules_cleaned_ml or candidates_for_rules or ml_candidates
            self._emit_telemetry(
                difficulty=difficulty,
                source="hybrid_llm_fallback",
                llm_used=True,
                rules_error=rules_error,
                rules_error_type=rules_error_type,
                ml_candidates_count=len(ml_candidates),
                final_codes_count=len(fallback_codes),
                llm_raw_count=0,
                rules_modified_llm=False,
                fallback_reason=f"{reason_for_fallback} (llm_empty)",
            )
            return HybridCoderResult(
                codes=fallback_codes,
                source="hybrid_llm_fallback",
                difficulty=ml_result.difficulty,
                metadata={
                    "ml_difficulty": difficulty,
                    "ml_candidates": ml_candidates,
                    "ml_predictions": preds,
                    "rules_error": rules_error,
                    "rules_error_type": rules_error_type,
                    "llm_called": True,
                    "llm_failed": True,
                    "llm_raw_codes": [],
                    "reason_for_fallback": reason_for_fallback,
                    "rules_modified_llm": False,
                },
            )

        # 5. Final safety check – rules veto LLM output if needed
        # Use non-strict validation to clean rather than reject
        final_codes = self._rules.validate(llm_codes, note_text, strict=False)

        # Track if rules modified LLM output
        rules_modified_llm = set(llm_codes) != set(final_codes)

        self._emit_telemetry(
            difficulty=difficulty,
            source="hybrid_llm_fallback",
            llm_used=True,
            rules_error=rules_error,
            rules_error_type=rules_error_type,
            ml_candidates_count=len(ml_candidates),
            final_codes_count=len(final_codes),
            llm_raw_count=len(llm_codes),
            rules_modified_llm=rules_modified_llm,
            fallback_reason=reason_for_fallback,
        )

        return HybridCoderResult(
            codes=final_codes,
            source="hybrid_llm_fallback",
            difficulty=ml_result.difficulty,
            metadata={
                "ml_difficulty": difficulty,
                "ml_candidates": ml_candidates,
                "ml_predictions": preds,
                "rules_error": rules_error,
                "rules_error_type": rules_error_type,
                "llm_called": True,
                "llm_raw_codes": llm_codes,
                "reason_for_fallback": reason_for_fallback,
                "rules_modified_llm": rules_modified_llm,
            },
        )

    def _call_llm_with_context(
        self, note_text: str, context: Dict[str, Any]
    ) -> List[str]:
        """
        Call the LLM advisor with ML context.

        The LLM is given ML predictions as hints but is the final judge.

        Args:
            note_text: The procedure note text
            context: Dict with ML predictions, difficulty, and fallback reason

        Returns:
            List of CPT codes from LLM
        """
        # Check if advisor supports context-aware suggestions
        if hasattr(self._llm, "suggest_with_context"):
            suggestions = self._llm.suggest_with_context(note_text, context)
        else:
            # Fall back to standard suggest_codes
            suggestions = self._llm.suggest_codes(note_text)

        # Extract codes from suggestions
        codes = []
        for s in suggestions:
            if hasattr(s, "code"):
                codes.append(s.code)
            elif isinstance(s, dict) and "code" in s:
                codes.append(s["code"])
            elif isinstance(s, str):
                codes.append(s)

        return codes

    def _emit_telemetry(
        self,
        difficulty: str,
        source: str,
        llm_used: bool,
        rules_error: Optional[str],
        rules_error_type: Optional[str],
        ml_candidates_count: int,
        final_codes_count: int,
        llm_raw_count: int = 0,
        rules_modified_llm: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """
        Emit structured telemetry for monitoring and debugging.

        This logs a single structured record with all key metrics from the
        orchestration decision. Can be easily parsed by log aggregators.

        Args:
            difficulty: Case difficulty classification (high_confidence/gray_zone/low_confidence)
            source: Decision source (ml_rules_fastpath or hybrid_llm_fallback)
            llm_used: Whether LLM was called
            rules_error: Error message if rules validation failed
            rules_error_type: Exception type name if rules failed
            ml_candidates_count: Number of ML candidate codes
            final_codes_count: Number of final output codes
            llm_raw_count: Number of codes returned by LLM (before rules)
            rules_modified_llm: Whether rules modified LLM output
            fallback_reason: Why LLM fallback was triggered
        """
        telemetry = {
            "event": "hybrid_orchestrator_decision",
            "difficulty": difficulty,
            "source": source,
            "llm_used": llm_used,
            "ml_candidates_count": ml_candidates_count,
            "final_codes_count": final_codes_count,
        }

        if llm_used:
            telemetry["llm_raw_count"] = llm_raw_count
            telemetry["rules_modified_llm"] = rules_modified_llm
            telemetry["fallback_reason"] = fallback_reason

        if rules_error:
            telemetry["rules_error"] = rules_error
            telemetry["rules_error_type"] = rules_error_type

        logger.info(
            "Orchestrator decision: source=%s, difficulty=%s, llm_used=%s, "
            "ml_candidates=%d, final_codes=%d%s%s",
            source,
            difficulty,
            llm_used,
            ml_candidates_count,
            final_codes_count,
            f", rules_error_type={rules_error_type}" if rules_error_type else "",
            f", fallback_reason={fallback_reason}" if fallback_reason else "",
            extra={"telemetry": telemetry},
        )


def build_hybrid_orchestrator(
    ml_predictor: Any = None,
    rules_engine: Any = None,
    llm_advisor: Any = None,
) -> SmartHybridOrchestrator:
    """
    Factory function to build a SmartHybridOrchestrator with default components.

    Args:
        ml_predictor: Optional MLCoderPredictor (creates default if None)
        rules_engine: Optional CodingRulesEngine (creates default if None)
        llm_advisor: Optional LLMAdvisorPort (creates default if None)

    Returns:
        Configured SmartHybridOrchestrator instance
    """
    if ml_predictor is None:
        from ml.lib.ml_coder.predictor import MLCoderPredictor
        ml_predictor = MLCoderPredictor()

    if rules_engine is None:
        from app.coder.rules_engine import CodingRulesEngine
        rules_engine = CodingRulesEngine()

    if llm_advisor is None:
        from ml.lib.ml_coder.data_prep import VALID_IP_CODES
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider == "openai_compat":
            from app.coder.adapters.llm.openai_compat_advisor import OpenAICompatAdvisorAdapter

            llm_advisor = OpenAICompatAdvisorAdapter(allowed_codes=list(VALID_IP_CODES))
        else:
            from app.coder.adapters.llm.gemini_advisor import GeminiAdvisorAdapter

            llm_advisor = GeminiAdvisorAdapter(allowed_codes=list(VALID_IP_CODES))

    return SmartHybridOrchestrator(
        ml_predictor=ml_predictor,
        rules_engine=rules_engine,
        llm_advisor=llm_advisor,
    )
