"""Simple rule-based negation detector.

Implements the NegationDetectionPort interface using keyword matching.
Can be replaced with a more sophisticated NegEx or ML-based detector.
"""

from __future__ import annotations

from typing import Sequence

from app.domain.text.negation import NegationDetectionPort


class SimpleNegationDetector(NegationDetectionPort):
    """Simple keyword-based negation detector.

    Looks for negation phrases in a context window around the target span.
    """

    VERSION = "simple_v1"

    # Negation phrases that indicate something was NOT done
    NEGATION_PHRASES = [
        # Direct negation
        "no evidence of",
        "no",
        "not",
        "without",
        "none",
        "never",
        "neither",
        "negative for",
        "absent",
        # Patient refusal/cancellation
        "refused",
        "declined",
        "cancelled",
        "canceled",
        "deferred",
        # Future/planned (not yet done)
        "planned",
        "scheduled",
        "will consider",
        "will perform",
        "to be done",
        "consider",
        "may need",
        "might need",
        "recommend",
        # Attempted but not completed
        "attempted",
        "tried",
        "unable to",
        "could not",
        "failed to",
        "unsuccessful",
        "aborted",
        # Conditional
        "if needed",
        "if required",
        "as needed",
        "prn",
    ]

    # Additional context-specific phrases
    PROCEDURE_NEGATION_PHRASES = [
        "not performed",
        "was not done",
        "was deferred",
        "held",
        "postponed",
        "contraindicated",
    ]

    @property
    def version(self) -> str:
        return self.VERSION

    def is_negated(
        self,
        text: str,
        target_span_start: int,
        target_span_end: int,
        scope_chars: int = 60,
    ) -> bool:
        """Check if the target span appears to be negated.

        Args:
            text: The full text to analyze.
            target_span_start: Start index of the target phrase.
            target_span_end: End index of the target phrase.
            scope_chars: Number of characters around target to check.

        Returns:
            True if negation phrases are found near the target.
        """
        clues = self.get_negation_clues(
            text, target_span_start, target_span_end, scope_chars
        )
        return len(clues) > 0

    def is_negated_simple(
        self,
        context: str,
        negative_phrases: list[str],
    ) -> bool:
        """Simplified negation check for use with keyword mappings.

        Args:
            context: The text context around the target phrase.
            negative_phrases: List of phrases that indicate negation.

        Returns:
            True if any negative phrase is found in the context.
        """
        context_lower = context.lower()

        # Check custom negative phrases first
        for phrase in negative_phrases:
            if phrase.lower() in context_lower:
                return True

        # Check standard negation phrases
        for phrase in self.NEGATION_PHRASES:
            if phrase in context_lower:
                return True

        return False

    def get_negation_clues(
        self,
        text: str,
        target_span_start: int,
        target_span_end: int,
        scope_chars: int = 60,
    ) -> Sequence[str]:
        """Get negation phrases found near the target.

        Args:
            text: The full text to analyze.
            target_span_start: Start index of the target phrase.
            target_span_end: End index of the target phrase.
            scope_chars: Number of characters around target to check.

        Returns:
            List of negation phrases found in the context.
        """
        # Extract context window
        start = max(0, target_span_start - scope_chars)
        end = min(len(text), target_span_end + scope_chars)
        context = text[start:end].lower()

        found_clues: list[str] = []

        # Check all negation phrases
        all_phrases = self.NEGATION_PHRASES + self.PROCEDURE_NEGATION_PHRASES

        for phrase in all_phrases:
            if phrase in context:
                # Verify the phrase is before or close to the target
                phrase_pos = context.find(phrase)
                target_in_context = target_span_start - start

                # Negation typically precedes the target
                # or is very close after
                if phrase_pos < target_in_context + 30:
                    found_clues.append(phrase)

        return found_clues


# Singleton instance for convenience
_default_detector: SimpleNegationDetector | None = None


def get_negation_detector() -> SimpleNegationDetector:
    """Get the default negation detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = SimpleNegationDetector()
    return _default_detector
