"""Section-aware text truncation utilities for LLM prompts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Pattern, Tuple


@dataclass
class Section:
    name: str
    text: str
    priority: int


HEADER_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"^INDICATIONS?:", re.IGNORECASE | re.MULTILINE), "indications"),
    (re.compile(r"^PROCEDURE(?: DETAILS)?:", re.IGNORECASE | re.MULTILINE), "procedure"),
    (re.compile(r"^TECHNIQUE:", re.IGNORECASE | re.MULTILINE), "technique"),
    (re.compile(r"^FINDINGS?:", re.IGNORECASE | re.MULTILINE), "findings"),
    (re.compile(r"^HISTORY:", re.IGNORECASE | re.MULTILINE), "history"),
    (re.compile(r"^MEDICATIONS?:", re.IGNORECASE | re.MULTILINE), "medications"),
    (re.compile(r"^ALLERGIES?:", re.IGNORECASE | re.MULTILINE), "allergies"),
    (re.compile(r"^CONSENT:", re.IGNORECASE | re.MULTILINE), "consent"),
]

PRIORITY = {
    "procedure": 3,
    "technique": 3,
    "findings": 3,
    "indications": 2,
    "history": 1,
    "medications": 1,
    "allergies": 1,
    "consent": 1,
    "preamble": 1,
    "full": 1,
}


def sectionizer_enabled() -> bool:
    return os.getenv("CODING_SECTIONIZER_ENABLED", "false").lower() == "true"


def max_llm_input_tokens() -> int:
    try:
        return int(os.getenv("CODING_MAX_LLM_INPUT_TOKENS", "3000"))
    except ValueError:
        return 3000


def split_into_sections(text: str) -> List[Section]:
    if not text:
        return []

    matches: List[Tuple[int, int, str]] = []
    for pattern, name in HEADER_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), name))

    if not matches:
        return [Section(name="full", text=text.strip(), priority=PRIORITY["full"])]

    matches.sort(key=lambda item: item[0])
    sections: List[Section] = []

    first_start = matches[0][0]
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append(
                Section(name="preamble", text=preamble, priority=PRIORITY["preamble"])
            )

    for idx, (start, _, name) in enumerate(matches):
        next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        section_text = text[start:next_start].strip()
        if not section_text:
            continue
        priority = PRIORITY.get(name, 1)
        sections.append(Section(name=name, text=section_text, priority=priority))

    return sections


def approximate_token_count(text: str) -> int:
    return len(text.split())


def accordion_truncate(text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return text

    sections = split_into_sections(text)
    if not sections:
        return text

    ordered_sections = sorted(
        enumerate(sections),
        key=lambda pair: (-pair[1].priority, pair[0]),
    )

    selected_parts: List[str] = []
    used_tokens = 0

    for _, section in ordered_sections:
        section_tokens = approximate_token_count(section.text)
        if used_tokens >= max_tokens:
            break

        if used_tokens + section_tokens > max_tokens and section.priority < 3:
            continue

        if used_tokens + section_tokens > max_tokens:
            break

        selected_parts.append(section.text.strip())
        used_tokens += section_tokens

    if not selected_parts:
        return text

    return "\n\n".join(selected_parts)


__all__ = [
    "Section",
    "split_into_sections",
    "accordion_truncate",
    "approximate_token_count",
    "sectionizer_enabled",
    "max_llm_input_tokens",
]
