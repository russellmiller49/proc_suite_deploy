from __future__ import annotations

import re
from typing import Iterable

from app.common.text_cleaning import (
    DEFAULT_TABLE_TOOL_KEYWORDS,
    find_empty_table_row_spans,
)


PATTERNS: list[str] = [
    r"(?ims)^CPT\s+CODES?:.*?(?=\n\n|\Z)",
    r"(?ims)^BILLING:.*?(?=\n\n|\Z)",
    r"(?ims)^CODING\s+SUMMARY.*?(?=\n\n|\Z)",
]

NON_PROCEDURAL_HEADINGS: tuple[str, ...] = (
    "INDICATION",
    "INDICATIONS",
    "HISTORY",
    "CONSENT",
    "PLAN",
    "IMPRESSION/PLAN",
    "IMPRESSION / PLAN",
    "ASSESSMENT/PLAN",
    "ASSESSMENT / PLAN",
    "RECOMMENDATION",
    "RECOMMENDATIONS",
    "ASSESSMENT",
)

_HEADING_INLINE_RE = re.compile(
    r"^\s*(?P<header>[A-Za-z][A-Za-z /_-]{0,80})\s*:\s*(?P<rest>.*)$",
    re.MULTILINE,
)
_HEADING_STANDALONE_RE = re.compile(
    r"^\s*(?P<header>[A-Z][A-Z0-9 /()_-]{1,80})\s*$",
    re.MULTILINE,
)

_EXTERNAL_REPORT_HEADER_RE = re.compile(
    r"(?im)^\s*(?:🩺\s*)?(?:extraction\s+quality\s+report|external(?:\s+extraction)?\s+report)\b"
)

_PROCEDURE_HEADER_LINE_RE = re.compile(
    r"(?im)^(?P<header>\s*(?:PROCEDURES?\s+PERFORMED|PROCEDURE\s+PERFORMED|PROCEDURES|PROCEDURE))\s*:\s*(?P<rest>.*)$"
)
_CPT_CODE_RE = re.compile(r"\b\d{5}\b")
_CPT_LINE_RE = re.compile(r"(?im)^\s*(?:CPT:?)?\s*\d{5}\b.*$")

_IP_CODE_MOD_DETAILS_HEADER_RE = re.compile(
    r"(?im)^\s*IP\b[^\n]{0,80}CODE\s+MOD\s+DETAILS\b[^\n]*$"
)
_IP_BLOCK_END_RE = re.compile(
    r"(?im)^\s*(?:ANESTHESIA|MONITORING|INSTRUMENT|ESTIMATED\s+BLOOD\s+LOSS|COMPLICATIONS|PROCEDURE\s+IN\s+DETAIL|DESCRIPTION\s+OF\s+PROCEDURE)\b"
)
_CPT_CONTEXT_PRESERVE_RE = re.compile(
    r"\b(?:zephyr|spiration|endobronchial\s+valve|bronchial\s+valve|blvr)\b",
    re.IGNORECASE,
)

_CPT_EXAMPLE_TOKENS_RE = re.compile(
    r"(?i)\b(?:laser|cryotherap(?:y|ies)|cryo(?:probe|spray)?|thermal|apc|argon|radiofrequency|microwave)\b"
)

_CHECKBOX_TEMPLATE_LINE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?im)^\s*0\s*[—\-]\s+.*$"),
    re.compile(r"(?im)^\s*\[\s*\]\s+.*$"),
    re.compile(r"(?im)^\s*[☐□]\s+.*$"),
)


def _find_template_checkbox_spans(text: str) -> tuple[list[tuple[int, int]], int]:
    spans: list[tuple[int, int]] = []
    raw = text or ""
    if not raw:
        return spans, 0
    for pat in _CHECKBOX_TEMPLATE_LINE_RES:
        spans.extend(match.span() for match in pat.finditer(raw))
    return spans, len(spans)


def mask_offset_preserving(text: str, patterns: Iterable[str] = PATTERNS) -> str:
    """Mask matched spans with spaces while preserving length and newlines."""
    raw = text or ""
    masked = text or ""
    for pat in patterns:
        for match in re.finditer(pat, masked):
            start, end = match.span()
            chunk = masked[start:end]
            chunk_mask = re.sub(r"[^\n]", " ", chunk)
            masked = masked[:start] + chunk_mask + masked[end:]

    masked = _mask_ip_code_mod_details_cpt_codes(masked)
    masked = _mask_cpt_definition_lines(masked)
    # Prevent deterministic extractors from "reading" blank modality template rows
    # (e.g., APC/Cryoprobe listed but empty columns).
    masked = _mask_spans(masked, find_empty_table_row_spans(masked, keywords=DEFAULT_TABLE_TOOL_KEYWORDS))
    report_spans = [(start, end) for start, end, _ in _find_external_report_spans(raw)]
    masked = _mask_spans(masked, report_spans)
    checkbox_spans, _count = _find_template_checkbox_spans(raw)
    masked = _mask_spans(masked, checkbox_spans)
    return masked


def mask_extraction_noise(text: str) -> tuple[str, dict[str, object]]:
    """Mask template noise and non-procedural sections for extraction."""
    base = mask_offset_preserving(text or "")
    checkbox_spans, checkbox_line_count = _find_template_checkbox_spans(text or "")
    external_reports = _find_external_report_spans(text or "")
    external_report_spans = [(start, end) for start, end, _ in external_reports]
    sections = _find_non_procedural_section_spans(text or "")
    section_spans = [(start, end) for start, end, _ in sections]
    table_spans = find_empty_table_row_spans(text or "", keywords=DEFAULT_TABLE_TOOL_KEYWORDS)
    procedure_header_spans, procedure_header_line_count = _find_procedure_header_cpt_spans(text or "")
    spans = external_report_spans + section_spans + table_spans + procedure_header_spans

    masked = _mask_spans(base, spans)
    meta = {
        "masked_external_report_count": len(external_reports),
        "masked_external_report_markers": [title for _, _, title in external_reports],
        "masked_non_procedural_sections": sorted({title for _, _, title in sections}),
        "masked_non_procedural_section_count": len(sections),
        "masked_empty_table_rows": len(table_spans),
        "masked_procedure_header_cpt_lines": procedure_header_line_count,
        "masked_checkbox_template_lines": checkbox_line_count,
    }
    return masked, meta


def _mask_ip_code_mod_details_cpt_codes(text: str) -> str:
    """Mask CPT digits inside 'IP ... CODE MOD DETAILS' blocks while preserving details.

    These blocks are often selection-driven (not pure CPT menus). We mask only the
    5-digit CPT codes to reduce "menu reading" while keeping clinically relevant
    details (e.g., BLVR valve sizing, laterality, bilateral flags).
    """
    if not text:
        return text

    def _mask_full_line(raw: str) -> str:
        return re.sub(r"[^\n]", " ", raw)

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_block = False

    for line in lines:
        if _IP_CODE_MOD_DETAILS_HEADER_RE.search(line):
            in_block = True
            out.append(line)
            continue

        if in_block and _IP_BLOCK_END_RE.search(line):
            in_block = False
            out.append(line)
            continue

        if not in_block:
            out.append(line)
            continue

        # For code-definition list lines inside the modifier block, mask the entire line
        # unless it contains high-signal BLVR context we explicitly preserve.
        if _CPT_LINE_RE.match(line) and not _CPT_CONTEXT_PRESERVE_RE.search(line):
            out.append(_mask_full_line(line))
            continue

        masked_line = _CPT_CODE_RE.sub(lambda m: " " * len(m.group(0)), line)

        # Some blocks include "Apply to: <CPT definition>" fragments that add CPT-menu
        # noise (e.g., "(eg. laser therapy, cryotherapy)") inside otherwise clinical
        # narrative. Mask the apply-to clause when it contains example-only tokens.
        if _CPT_EXAMPLE_TOKENS_RE.search(masked_line) and not _CPT_CONTEXT_PRESERVE_RE.search(masked_line):
            m_apply = re.search(r"(?i)\bapply\s+to\s*:", masked_line)
            if m_apply:
                idx = m_apply.start()
                prefix = masked_line[:idx]
                suffix = _mask_full_line(masked_line[idx:])
                masked_line = prefix + suffix

        out.append(masked_line)

    return "".join(out)


def _mask_cpt_definition_lines(text: str) -> str:
    """Mask CPT-definition lines, preserving high-signal BLVR valve context.

    Some templates embed clinically meaningful BLVR valve details on lines that
    start with a CPT code (e.g., "31647 Zephyr size 4.0 ..."). For these lines we
    mask the code digits only and keep the remainder so extraction can still
    recover valve sizing/manufacturer when granular data is missing.
    """
    if not text:
        return text

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    after_cpt_definition_line = False
    continuation_paren_balance = 0
    continuation_force_next = False
    continuation_example_mode = False

    example_token_re = re.compile(
        r"(?i)\b(?:laser|cryotherapy|cryo(?:probe|spray)?|thermal|apc|argon|radiofrequency|microwave)\b"
    )

    def _mask_full_line(raw: str) -> str:
        return re.sub(r"[^\n]", " ", raw)

    def _stop_continuation(raw: str) -> bool:
        return not raw.strip() or _HEADING_INLINE_RE.match(raw) or _CPT_LINE_RE.match(raw)

    for line in lines:
        if after_cpt_definition_line or continuation_paren_balance or continuation_force_next or continuation_example_mode:
            if _stop_continuation(line):
                after_cpt_definition_line = False
                continuation_paren_balance = 0
                continuation_force_next = False
                continuation_example_mode = False
            else:
                is_indented = bool(re.match(r"^\s+\S", line))
                has_example_tokens = bool(example_token_re.search(line))
                should_mask = False

                if continuation_paren_balance > 0:
                    should_mask = True
                elif continuation_force_next:
                    should_mask = True
                elif continuation_example_mode and is_indented and has_example_tokens:
                    should_mask = True
                elif is_indented and has_example_tokens:
                    should_mask = True
                    continuation_example_mode = True

                after_cpt_definition_line = False
                if should_mask:
                    out.append(_mask_full_line(line))
                    continuation_paren_balance += line.count("(") - line.count(")")
                    if continuation_paren_balance <= 0:
                        continuation_paren_balance = 0
                    continuation_force_next = False
                    continue

                continuation_paren_balance = 0
                continuation_force_next = False
                continuation_example_mode = False

        if not _CPT_LINE_RE.match(line):
            out.append(line)
            continue

        if _CPT_CONTEXT_PRESERVE_RE.search(line):
            out.append(_CPT_CODE_RE.sub(lambda m: " " * len(m.group(0)), line))
            after_cpt_definition_line = False
            continuation_paren_balance = 0
            continuation_force_next = False
            continuation_example_mode = False
            continue

        # Default behavior: mask entire CPT-bearing line.
        out.append(_mask_full_line(line))
        after_cpt_definition_line = True

        continuation_paren_balance = line.count("(") - line.count(")")
        if continuation_paren_balance < 0:
            continuation_paren_balance = 0

        tail = line.rstrip().lower()
        continuation_force_next = tail.endswith("eg.") or tail.endswith("e.g.") or tail.endswith("(")
        continuation_example_mode = False

    return "".join(out)


def _normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).upper()


def _is_non_procedural_heading(header: str) -> bool:
    if header in NON_PROCEDURAL_HEADINGS:
        return True
    return any(
        header.startswith(prefix)
        for token in NON_PROCEDURAL_HEADINGS
        for prefix in (f"{token} ", f"{token}/", f"{token} -")
    )


def _find_non_procedural_section_spans(text: str) -> list[tuple[int, int, str]]:
    matches = list(_HEADING_INLINE_RE.finditer(text or ""))
    standalone_matches = list(_HEADING_STANDALONE_RE.finditer(text or ""))
    spans: list[tuple[int, int, str]] = []
    if not matches and not standalone_matches:
        return spans

    boundaries = sorted({m.start() for m in matches} | {m.start() for m in standalone_matches})

    def _next_boundary(current_start: int) -> int:
        for boundary in boundaries:
            if boundary > current_start:
                return boundary
        return len(text or "")

    for match in matches:
        header = _normalize_heading(match.group("header"))
        if not _is_non_procedural_heading(header):
            continue
        body_start = match.start("rest")
        body_end = _next_boundary(match.start())
        if body_end <= body_start:
            continue
        spans.append((body_start, body_end, header))

    for match in standalone_matches:
        header = _normalize_heading(match.group("header"))
        if not _is_non_procedural_heading(header):
            continue
        line_end = text.find("\n", match.start())
        body_start = len(text) if line_end == -1 else line_end + 1
        body_end = _next_boundary(match.start())
        if body_end <= body_start:
            continue
        spans.append((body_start, body_end, header))

    return spans


def _find_external_report_spans(text: str) -> list[tuple[int, int, str]]:
    """Mask appended reviewer/QA reports that are not part of the clinical note."""
    raw = text or ""
    if not raw:
        return []

    match = _EXTERNAL_REPORT_HEADER_RE.search(raw)
    if match is None:
        return []
    return [(match.start(), len(raw), (match.group(0) or "").strip())]


def _find_procedure_header_cpt_spans(text: str) -> tuple[list[tuple[int, int]], int]:
    """Mask CPT/definition lines inside PROCEDURE/PROCEDURES PERFORMED blocks.

    Many notes include CPT definitions under procedure headings (e.g., "31641... eg laser"),
    which can cause downstream "menu reading" false positives. This masks only the
    CPT-bearing lines while preserving offsets.
    """
    if not text:
        return ([], 0)

    lines = text.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    masked_lines = 0

    offset = 0
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = _PROCEDURE_HEADER_LINE_RE.match(line)
        if not match:
            offset += len(line)
            idx += 1
            continue

        # Scan forward until the next heading-like line to bound this section.
        section_end_idx = idx + 1
        while section_end_idx < len(lines):
            next_line = lines[section_end_idx]
            if _HEADING_INLINE_RE.match(next_line):
                break
            section_end_idx += 1

        # Mask CPT-bearing content on the header line (after the colon).
        rest = match.group("rest") or ""
        if _CPT_CODE_RE.search(rest):
            start = offset + match.start("rest")
            end = offset + len(line)
            spans.append((start, end))
            masked_lines += 1

        # Mask CPT-bearing lines within the section body.
        inner_offset = offset + len(line)
        for inner_line in lines[idx + 1 : section_end_idx]:
            if _CPT_CODE_RE.search(inner_line):
                spans.append((inner_offset, inner_offset + len(inner_line)))
                masked_lines += 1
            inner_offset += len(inner_line)

        # Advance.
        for consumed in lines[idx:section_end_idx]:
            offset += len(consumed)
        idx = section_end_idx

    return spans, masked_lines


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    masked = list(text)
    text_len = len(masked)
    for start, end in spans:
        if start >= text_len or end <= 0:
            continue
        s = max(0, start)
        e = min(text_len, end)
        for idx in range(s, e):
            if masked[idx] != "\n":
                masked[idx] = " "
    return "".join(masked)


__all__ = [
    "PATTERNS",
    "NON_PROCEDURAL_HEADINGS",
    "mask_offset_preserving",
    "mask_extraction_noise",
]
