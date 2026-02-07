"""Negation detection port (interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class NegationDetectionPort(ABC):
    """Port for detecting negation in clinical text.

    This is the domain interface that infrastructure adapters implement.
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version identifier for this detector."""
        ...

    @abstractmethod
    def is_negated(
        self,
        text: str,
        target_span_start: int,
        target_span_end: int,
        scope_chars: int = 60,
    ) -> bool:
        """Returns True if target span is negated or not performed.

        Args:
            text: The full text to analyze.
            target_span_start: Start character index of the target phrase.
            target_span_end: End character index of the target phrase.
            scope_chars: Number of characters around the target to consider.

        Returns:
            True if the target span appears to be negated.
        """
        ...

    @abstractmethod
    def get_negation_clues(
        self,
        text: str,
        target_span_start: int,
        target_span_end: int,
        scope_chars: int = 60,
    ) -> Sequence[str]:
        """Return the phrases used to decide negation.

        Args:
            text: The full text to analyze.
            target_span_start: Start character index of the target phrase.
            target_span_end: End character index of the target phrase.
            scope_chars: Number of characters around the target to consider.

        Returns:
            List of negation clue phrases found near the target.
        """
        ...
