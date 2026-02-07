"""Shared text cleaning helpers."""

from __future__ import annotations

import re
from typing import Sequence

DEFAULT_TABLE_TOOL_KEYWORDS: tuple[str, ...] = (
    "apc",
    "argon plasma",
    "laser",
    "corecath",
    "core cath",
    "cryoprobe",
    "cryotherapy",
    "electrocautery",
    "electro cautery",
)

EMPTY_TABLE_TOKENS: set[str] = {
    "",
    "na",
    "n/a",
    "none",
    "nil",
    "-",
    "--",
}

_TABLE_SPLIT_RE = re.compile(r"\t+| {2,}")


def _split_table_columns(line: str) -> list[str]:
    return [col.strip() for col in _TABLE_SPLIT_RE.split(line)]


def _is_empty_cell(cell: str) -> bool:
    return (cell or "").strip().lower() in EMPTY_TABLE_TOKENS


def is_empty_table_row(
    line: str,
    *,
    keywords: Sequence[str] = DEFAULT_TABLE_TOOL_KEYWORDS,
) -> bool:
    """Return True if a table-like row lists a tool but no result values."""
    clean = (line or "").rstrip("\r\n")
    if not clean.strip():
        return False

    columns = _split_table_columns(clean)
    if len(columns) < 2:
        return False

    first_col = columns[0].lower()
    if not any(keyword in first_col for keyword in keywords):
        return False

    return all(_is_empty_cell(col) for col in columns[1:])


def find_empty_table_row_spans(
    text: str,
    *,
    keywords: Sequence[str] = DEFAULT_TABLE_TOOL_KEYWORDS,
) -> list[tuple[int, int]]:
    """Return (start, end) spans for empty template rows to be masked."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in (text or "").splitlines(keepends=True):
        line_len = len(line)
        if is_empty_table_row(line, keywords=keywords):
            spans.append((offset, offset + line_len))
        offset += line_len
    return spans


def strip_empty_table_rows(
    text: str,
    *,
    keywords: Sequence[str] = DEFAULT_TABLE_TOOL_KEYWORDS,
) -> tuple[str, int]:
    """Remove empty template rows and return (cleaned_text, removed_count)."""
    if not text:
        return text or "", 0

    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if is_empty_table_row(line, keywords=keywords):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


__all__ = [
    "DEFAULT_TABLE_TOOL_KEYWORDS",
    "EMPTY_TABLE_TOKENS",
    "find_empty_table_row_spans",
    "is_empty_table_row",
    "strip_empty_table_rows",
]
