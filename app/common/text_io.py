"""Utility helpers for loading notes and supporting knowledge assets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .knowledge import DEFAULT_KNOWLEDGE_FILE, get_knowledge

__all__ = [
    "DEFAULT_KNOWLEDGE_FILE",
    "load_note",
    "normalize_whitespace",
    "strip_headers",
    "load_knowledge_base",
]

_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*(dictated|transcribed)\s+by", re.IGNORECASE),
    re.compile(r"^\s*(signed|electronically signed)\b", re.IGNORECASE),
)

_FOOTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(cc:|copy to)", re.IGNORECASE),
    re.compile(r"^\s*end\s+of\s+note", re.IGNORECASE),
)


def load_note(source: str | Path) -> str:
    """Load and normalize a procedure note from a string or filesystem path."""

    text = _read_source(source)
    text = normalize_whitespace(text)
    text = strip_headers(text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and paragraph spacing for easier downstream parsing."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" ?\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_headers(text: str) -> str:
    """Remove simple headers/footers that commonly wrap exported procedure notes."""

    lines = text.splitlines()
    lines = _strip_matching(lines, _HEADER_PATTERNS, from_start=True)
    lines = _strip_matching(lines, _FOOTER_PATTERNS, from_start=False)
    return "\n".join(lines).strip()


def load_knowledge_base(path: str | Path | None = None) -> dict:
    """Load (and hot-reload) the shared coding knowledge base document."""

    return get_knowledge(path)


def _read_source(source: str | Path) -> str:
    if isinstance(source, Path):
        if not source.exists():  # pragma: no cover - simple guard
            raise FileNotFoundError(source)
        return source.read_text(encoding="utf-8")

    possible_path = Path(source)
    if possible_path.exists():
        return possible_path.read_text(encoding="utf-8")

    return str(source)


def _strip_matching(
    lines: Iterable[str],
    patterns: tuple[re.Pattern[str], ...],
    *,
    from_start: bool,
) -> list[str]:
    working = list(lines)
    matcher = (lambda line: any(regex.search(line) for regex in patterns))

    while working:
        index = 0 if from_start else -1
        if matcher(working[index]):
            working.pop(index)
            continue
        break
    return working
