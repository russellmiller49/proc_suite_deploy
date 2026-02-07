"""Reconciliation logic for parallel CPT coding pathways."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReconciledCode:
    """Result of reconciling NER evidence with ML probability."""

    code: str
    status: Literal["FINALIZED", "FLAGGED"]
    review_required: bool
    message: str | None = None


class CodeReconciler:
    """Apply evidence-gated reconciliation between NER and ML."""

    HIGH_CONF_THRESHOLD = 0.90
    LOW_CONF_THRESHOLD = 0.10

    def reconcile(self, ner_code: str | None, ml_prob: float) -> ReconciledCode:
        """Reconcile NER evidence and ML probability for a single code."""
        if ner_code and ml_prob > self.HIGH_CONF_THRESHOLD:
            return ReconciledCode(
                code=ner_code,
                status="FINALIZED",
                review_required=False,
            )

        if not ner_code and ml_prob > self.HIGH_CONF_THRESHOLD:
            return ReconciledCode(
                code="",
                status="FLAGGED",
                review_required=True,
                message=(
                    f"ML detected high probability ({ml_prob:.2f}) "
                    "but no supporting text evidence found."
                ),
            )

        if ner_code and ml_prob < self.LOW_CONF_THRESHOLD:
            return ReconciledCode(
                code=ner_code,
                status="FLAGGED",
                review_required=True,
                message=f"NER found keyword '{ner_code}', but ML strongly suggests negation/history.",
            )

        return ReconciledCode(
            code=ner_code or "",
            status="FLAGGED",
            review_required=True,
            message="Disagreement between NER evidence and ML probability; review required.",
        )
