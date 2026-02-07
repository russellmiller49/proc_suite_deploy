"""Text preprocessing transformers for ML coder pipeline."""

import re
from typing import Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin

from app.common.text_cleaning import strip_empty_table_rows
# Common boilerplate patterns to remove from clinical notes
BOILERPLATE_PATTERNS = [
    r"electronically signed by.*$",
    r"dictated but not read.*$",
    r"this note was generated using .*speech recognition.*$",
    r"please see dictated report.*$",
    r"signed electronically by.*$",
    r"authenticated by.*$",
    r"\*\*\* final \*\*\*",
    r"this is a confidential.*$",
]

_bp_regexes = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in BOILERPLATE_PATTERNS]


class NoteTextCleaner(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that cleans clinical note text.

    Removes common boilerplate patterns (signatures, disclaimers, etc.)
    that don't contribute to procedure code prediction.
    """

    def fit(self, X: Iterable[str], y=None):
        """No fitting required - returns self."""
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        """
        Clean each text in X by removing boilerplate patterns.

        Args:
            X: Iterable of note text strings

        Returns:
            List of cleaned text strings
        """
        cleaned = []
        for text in X:
            t = str(text) if text is not None else ""
            t, _removed = strip_empty_table_rows(t)
            for rx in _bp_regexes:
                t = rx.sub("", t)
            # Normalize whitespace
            t = re.sub(r"\s+", " ", t)
            cleaned.append(t.strip())
        return cleaned


__all__ = ["NoteTextCleaner", "BOILERPLATE_PATTERNS"]
