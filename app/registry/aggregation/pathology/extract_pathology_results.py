"""Deterministic extraction for final pathology summary fields.

This is intentionally conservative and designed for append-event aggregation:
- Prefer explicit headings (FINAL DIAGNOSIS / DIAGNOSIS / MICROBIOLOGY).
- Return null/None when unclear.
- Keep extracted strings short and scrub date-like substrings for ZK safety.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.registry.aggregation.sanitize import compact_text


_HEADING_DIAGNOSIS_RE = re.compile(r"(?im)^\s*(?:final\s+)?diagnos(?:is|es)\b\s*[:\-]?\s*(.*)\s*$")
_HEADING_MICRO_RE = re.compile(
    r"(?im)^\s*(?:microbiology|culture|gram\s+stain|afb|acid[-\s]?fast|fungal)\b\s*[:\-]?\s*(.*)\s*$"
)

# Generic "new section" heuristic: an all-caps heading (not specimen bullet like "A.")
_NEXT_SECTION_RE = re.compile(r"(?m)^\s*[A-Z][A-Z0-9 /()_-]{3,}\s*[:\-]?\s*$")

_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_US_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b")

_STAGE_RE = re.compile(
    r"\b(?:pathologic\s+)?stage\s*[:=]?\s*(?:p)?(0|[1-4]|I{1,4})\s*([ABC])?\b",
    re.IGNORECASE,
)
_STAGE_INLINE_RE = re.compile(r"\bStage\s*(0|[1-4]|I{1,4})\s*([ABC])?\b", re.IGNORECASE)
_TNM_RE = re.compile(
    r"\b(?:p|c)?T\d[a-d]?\s*N\d[a-c]?\s*M\d[a-c]?\b|\b(?:p|c)?T\d[a-d]?N\d[a-c]?M\d[a-c]?\b",
    re.IGNORECASE,
)


def _collapse_lines(lines: list[str], *, max_lines: int = 10) -> str:
    return "\n".join([ln.rstrip() for ln in lines[:max_lines] if str(ln or "").strip()]).strip()


def _extract_heading_block(text: str, heading_re: re.Pattern[str]) -> str | None:
    if not (text or "").strip():
        return None

    lines = (text or "").splitlines()
    for idx, line in enumerate(lines):
        match = heading_re.match(line)
        if not match:
            continue

        block_lines: list[str] = []
        first = str(match.group(1) or "").strip()
        if first:
            block_lines.append(first)

        for next_line in lines[idx + 1 : idx + 1 + 12]:
            if not next_line.strip() and block_lines:
                break
            if _NEXT_SECTION_RE.match(next_line) and block_lines:
                break
            block_lines.append(next_line.strip())

        raw_block = _collapse_lines(block_lines, max_lines=10)
        if not raw_block:
            continue
        return compact_text(raw_block, max_chars=300) or None

    return None


def _normalize_iso_date(year: str, month: str, day: str) -> str | None:
    try:
        parsed = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return parsed.isoformat()


def _extract_pathology_result_date(text: str) -> str | None:
    value = text or ""

    # Prefer dates close to "report/result/final/signed".
    context = re.search(r"(?is)\b(?:report|result|final|signed)\b.{0,120}", value)
    context_text = context.group(0) if context else value

    m = _ISO_DATE_RE.search(context_text)
    if m:
        return _normalize_iso_date(m.group(1), m.group(2), m.group(3))

    m = _US_DATE_RE.search(context_text)
    if m:
        return _normalize_iso_date(m.group(3), m.group(1).zfill(2), m.group(2).zfill(2))

    # Fallback to first ISO date in the doc.
    m = _ISO_DATE_RE.search(value)
    if m:
        return _normalize_iso_date(m.group(1), m.group(2), m.group(3))

    return None


def _extract_final_staging(text: str) -> str | None:
    value = text or ""
    match = _STAGE_RE.search(value) or _STAGE_INLINE_RE.search(value)
    if match:
        stage = str(match.group(1) or "").strip().upper()
        suffix = str(match.group(2) or "").strip().upper()
        return f"Stage {stage}{suffix}".strip()

    tnm = _TNM_RE.search(value)
    if tnm:
        raw = re.sub(r"\s+", "", tnm.group(0))
        return raw.strip() or None

    return None


def extract_pathology_results(text: str) -> dict[str, Any]:
    """Return a conservative pathology_results update dict from scrubbed pathology text."""

    qa_flags: list[str] = []

    final_diagnosis = _extract_heading_block(text, _HEADING_DIAGNOSIS_RE)
    if not final_diagnosis:
        # Fallback: try to capture a short malignant/non-malignant diagnosis line.
        diag_match = re.search(
            r"(?im)^\s*(?:positive|negative|suspicious|atypical|non[-\s]?diagnostic)[^\n]{0,180}$",
            text or "",
        )
        if diag_match:
            final_diagnosis = compact_text(diag_match.group(0), max_chars=220) or None
        else:
            qa_flags.append("no_final_diagnosis_found")

    final_staging = _extract_final_staging(text)
    microbiology_results = _extract_heading_block(text, _HEADING_MICRO_RE)
    pathology_result_date = _extract_pathology_result_date(text)

    out: dict[str, Any] = {
        "pathology_results_update": {
            "final_diagnosis": final_diagnosis,
            "final_staging": compact_text(final_staging or "", max_chars=80) or None,
            "microbiology_results": microbiology_results,
            "pathology_result_date": pathology_result_date,
        },
        "qa_flags": sorted(set(qa_flags)),
    }
    return out


__all__ = ["extract_pathology_results"]

