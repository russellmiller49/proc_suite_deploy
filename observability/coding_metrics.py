"""Coding-specific metrics for QA and observability.

This module provides high-level metrics helpers for tracking:
- Suggestion acceptance rates
- Manual code additions
- Pipeline latency
- Registry export success rates
"""

from __future__ import annotations

from .metrics import get_metrics_client


class CodingMetrics:
    """High-level metrics for the coding pipeline."""

    # Counter metric names
    SUGGESTIONS_TOTAL = "coder.suggestions_total"
    FINAL_CODES_TOTAL = "coder.final_codes_total"
    REVIEWS_TOTAL = "coder.reviews_total"
    REVIEW_ACCEPTED = "coder.reviews.accepted"
    REVIEW_REJECTED = "coder.reviews.rejected"
    REVIEW_MODIFIED = "coder.reviews.modified"
    MANUAL_CODES_TOTAL = "coder.manual_codes_total"
    REGISTRY_EXPORT_TOTAL = "coder.registry.exports_total"
    REGISTRY_EXPORT_SUCCESS = "coder.registry.exports_success"
    REGISTRY_EXPORT_PARTIAL = "coder.registry.exports_partial"

    # Timing/histogram metric names
    PIPELINE_LATENCY = "coder.pipeline_latency_ms"
    LLM_LATENCY = "coder.llm_latency_ms"
    RULE_ENGINE_LATENCY = "coder.rule_engine_latency_ms"
    REGISTRY_EXPORT_LATENCY = "coder.registry.export_latency_ms"

    # Gauge metric names
    ACCEPTANCE_RATE = "coder.acceptance_rate"
    COMPLETENESS_SCORE = "coder.registry.completeness_score"

    # LLM drift monitoring metrics
    LLM_SUGGESTIONS_REVIEWED = "coder.llm.suggestions_reviewed"
    LLM_SUGGESTIONS_ACCEPTED = "coder.llm.suggestions_accepted"

    @staticmethod
    def record_suggestions_generated(
        num_suggestions: int,
        procedure_type: str = "unknown",
        used_llm: bool = False,
    ) -> None:
        """Record that code suggestions were generated.

        Args:
            num_suggestions: Number of suggestions produced
            procedure_type: Type of procedure (for segmentation)
            used_llm: Whether LLM advisor was used
        """
        client = get_metrics_client()
        tags = {
            "procedure_type": procedure_type,
            "used_llm": str(used_llm).lower(),
        }
        client.incr(CodingMetrics.SUGGESTIONS_TOTAL, tags, num_suggestions)

    @staticmethod
    def record_review_action(
        action: str,
        procedure_type: str = "unknown",
        source: str = "hybrid",
    ) -> None:
        """Record a review action (accept, reject, modify).

        Args:
            action: The review action taken
            procedure_type: Type of procedure
            source: Source of the suggestion (rule, llm, hybrid)
        """
        client = get_metrics_client()
        tags = {
            "procedure_type": procedure_type,
            "source": source,
        }

        client.incr(CodingMetrics.REVIEWS_TOTAL, tags)

        if action == "accept":
            client.incr(CodingMetrics.REVIEW_ACCEPTED, tags)
        elif action == "reject":
            client.incr(CodingMetrics.REVIEW_REJECTED, tags)
        elif action == "modify":
            client.incr(CodingMetrics.REVIEW_MODIFIED, tags)

    @staticmethod
    def record_final_code_added(
        source: str = "hybrid",
        procedure_type: str = "unknown",
    ) -> None:
        """Record that a final code was added.

        Args:
            source: Source (hybrid, rule, llm, manual)
            procedure_type: Type of procedure
        """
        client = get_metrics_client()
        tags = {
            "source": source,
            "procedure_type": procedure_type,
        }
        client.incr(CodingMetrics.FINAL_CODES_TOTAL, tags)

    @staticmethod
    def record_manual_code_added(
        procedure_type: str = "unknown",
    ) -> None:
        """Record that a manual code was added.

        Args:
            procedure_type: Type of procedure
        """
        client = get_metrics_client()
        tags = {"procedure_type": procedure_type}
        client.incr(CodingMetrics.MANUAL_CODES_TOTAL, tags)

    @staticmethod
    def record_pipeline_latency(
        latency_ms: float,
        procedure_type: str = "unknown",
        used_llm: bool = False,
    ) -> None:
        """Record pipeline processing latency.

        Args:
            latency_ms: Latency in milliseconds
            procedure_type: Type of procedure
            used_llm: Whether LLM was used
        """
        client = get_metrics_client()
        tags = {
            "procedure_type": procedure_type,
            "used_llm": str(used_llm).lower(),
        }
        client.timing(CodingMetrics.PIPELINE_LATENCY, latency_ms, tags)

    @staticmethod
    def record_llm_latency(
        latency_ms: float,
        procedure_type: str = "unknown",
    ) -> None:
        """Record LLM advisor latency (separate from total pipeline latency).

        Args:
            latency_ms: LLM advisor latency in milliseconds
            procedure_type: Type of procedure (for segmentation)
        """
        if latency_ms <= 0:
            return  # No-op if LLM wasn't used or latency is 0

        client = get_metrics_client()
        tags = {"procedure_type": procedure_type}
        client.timing(CodingMetrics.LLM_LATENCY, latency_ms, tags)

    @staticmethod
    def record_acceptance_rate(
        rate: float,
        procedure_type: str = "unknown",
    ) -> None:
        """Record the current acceptance rate.

        Args:
            rate: Acceptance rate (0.0 to 1.0)
            procedure_type: Type of procedure
        """
        client = get_metrics_client()
        tags = {"procedure_type": procedure_type}
        client.observe(CodingMetrics.ACCEPTANCE_RATE, rate, tags)

    @staticmethod
    def record_registry_export(
        status: str,
        version: str = "v2",
        latency_ms: float = 0.0,
    ) -> None:
        """Record a registry export attempt.

        Args:
            status: Export status (success, partial, failed)
            version: Registry schema version
            latency_ms: Export latency
        """
        client = get_metrics_client()
        tags = {"version": version, "status": status}

        client.incr(CodingMetrics.REGISTRY_EXPORT_TOTAL, tags)

        if status == "success":
            client.incr(CodingMetrics.REGISTRY_EXPORT_SUCCESS, tags)
        elif status == "partial":
            client.incr(CodingMetrics.REGISTRY_EXPORT_PARTIAL, tags)

        if latency_ms > 0:
            client.timing(CodingMetrics.REGISTRY_EXPORT_LATENCY, latency_ms, {"version": version})

    @staticmethod
    def record_registry_completeness(
        score: float,
        version: str = "v2",
    ) -> None:
        """Record registry entry completeness score.

        Args:
            score: Completeness score (0.0 to 1.0)
            version: Registry schema version
        """
        client = get_metrics_client()
        tags = {"version": version}
        client.observe(CodingMetrics.COMPLETENESS_SCORE, score, tags)

    @staticmethod
    def record_llm_acceptance(
        accepted_count: int,
        reviewed_count: int,
        procedure_type: str = "unknown",
        source: str = "llm",
    ) -> None:
        """Record LLM suggestion acceptance metrics for drift monitoring.

        This tracks how many AI suggestions were reviewed and accepted,
        enabling drift detection over time. Only counts suggestions from
        AI sources (llm, hybrid, rule), not manual additions.

        Args:
            accepted_count: Number of suggestions accepted (or modified and accepted)
            reviewed_count: Total suggestions reviewed (accepted + rejected + modified)
            procedure_type: Type of procedure for segmentation
            source: Source filter (llm, hybrid, rule, or 'ai' for all non-manual)
        """
        if reviewed_count <= 0:
            return  # No-op if nothing was reviewed

        client = get_metrics_client()
        tags = {
            "procedure_type": procedure_type,
            "source": source,
        }

        # Record counts as counters for accurate rate calculation
        client.incr(CodingMetrics.LLM_SUGGESTIONS_REVIEWED, tags, reviewed_count)
        client.incr(CodingMetrics.LLM_SUGGESTIONS_ACCEPTED, tags, accepted_count)

        # Also record the instantaneous acceptance rate as a gauge
        acceptance_rate = accepted_count / reviewed_count
        CodingMetrics.record_acceptance_rate(acceptance_rate, procedure_type)
