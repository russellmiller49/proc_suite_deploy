from __future__ import annotations

import re

from app.common.sectionizer import SectionizerService


_PRIMARY_NARRATIVE_KEYWORDS: tuple[str, ...] = (
    "PROCEDURE IN DETAIL",
    "DESCRIPTION OF PROCEDURE",
    "PROCEDURE DESCRIPTION",
    "FINDINGS",
    "AIRWAY INSPECTION",
    "EBUS FINDINGS",
    "LYMPH NODES EVALUATED",
)

_SUPPORTING_SECTION_KEYWORDS: tuple[str, ...] = (
    "PROCEDURE",
    "TECHNIQUE",
    "OPERATIVE REPORT",
    "ANESTHESIA",
    "MONITORING",
    "COMPLICATIONS",
    "DISPOSITION",
    "INSTRUMENT",
)

_SUPPORTING_DATA_KEYWORDS: tuple[str, ...] = (
    "SPECIMEN",
    "IMPRESSION",
)

_EXCLUDED_FOCUS_KEYWORDS: tuple[str, ...] = (
    # Exclude history/indication/plan content from procedure extraction context.
    "HISTORY",
    "INDICATION",
    "PLAN",
)

_POST_PROCEDURE_TAIL_RE = re.compile(
    r"(?i)\b(?:prior\s+to\s+extubation|extubat(?:ion|ed)|transported\s+to\s+(?:the\s+)?recovery|recovery\s+room|pacu)\b"
)


def _canonical_heading(value: str) -> str:
    """Return a canonicalized heading token for matching.

    Examples:
    - "EBUS-Findings" -> "EBUS FINDINGS"
    - "IMPRESSION/PLAN" -> "IMPRESSION PLAN"
    - "SPECIMEN(S)" -> "SPECIMENS"
    """
    raw = (value or "").strip().upper()
    if not raw:
        return ""
    raw = re.sub(r"[/_\\-]+", " ", raw)
    raw = re.sub(r"[^A-Z0-9 ]+", "", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _heading_matches_any(heading: str, keywords: tuple[str, ...]) -> bool:
    if not heading:
        return False
    for keyword in keywords:
        if keyword and keyword in heading:
            return True
    return False


def _heading_excluded(heading: str) -> bool:
    if not heading:
        return False
    return any(token in heading for token in _EXCLUDED_FOCUS_KEYWORDS)


def _trim_post_procedure_tail(text: str) -> str:
    if not text:
        return ""
    match = _POST_PROCEDURE_TAIL_RE.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


def _extract_target_sections_by_regex(note_text: str) -> dict[str, list[str]]:
    pattern = re.compile(r"^(?P<header>[A-Za-z][A-Za-z0-9 /()_-]{0,80})\s*:\s*(?P<rest>.*)$", re.MULTILINE)
    matches = list(pattern.finditer(note_text or ""))
    if not matches:
        return {}

    extracted: dict[str, list[str]] = {}
    for idx, match in enumerate(matches):
        header_raw = match.group("header").strip()
        header = _canonical_heading(header_raw)
        if not header:
            continue
        if _heading_excluded(header):
            continue

        wanted = (
            _heading_matches_any(header, _PRIMARY_NARRATIVE_KEYWORDS)
            or _heading_matches_any(header, _SUPPORTING_SECTION_KEYWORDS)
            or _heading_matches_any(header, _SUPPORTING_DATA_KEYWORDS)
        )
        if not wanted:
            continue

        body_start = match.start("rest")
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(note_text)
        body = (note_text[body_start:body_end] or "").strip()
        if body:
            extracted.setdefault(header_raw, []).append(body)

    return extracted


def _dedupe_bodies(bodies: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for body in bodies:
        clean = (body or "").strip()
        if not clean:
            continue
        key = re.sub(r"\s+", " ", clean)
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def get_procedure_focus(note_text: str) -> str:
    """Return a focused view of the note for procedure extraction.

    If no relevant section headings are found, returns the full original note text.
    """
    original = note_text or ""
    if not original.strip():
        return note_text

    # Prefer an explicit sectionizer for common headings, but always apply
    # narrative-first ordering and tagging so downstream extractors prefer
    # rich procedural details over specimen lists/plan text.
    extracted: dict[str, list[str]] = {}

    # 1) ParserAgent handles inline headings like "PROCEDURE: text..." well, but
    # its segment types are not consistently granular across note styles. Treat
    # it as a helper for the canonical "PROCEDURE/FINDINGS/IMPRESSION/TECHNIQUE" block.
    try:
        from app.agents.contracts import ParserIn
        from app.agents.parser.parser_agent import ParserAgent

        parser_out = ParserAgent().run(ParserIn(note_id="", raw_text=original))
        for seg in getattr(parser_out, "segments", []) or []:
            seg_type_raw = str(getattr(seg, "type", "") or "")
            seg_type = _canonical_heading(seg_type_raw)
            if not seg_type:
                continue
            if _heading_excluded(seg_type):
                continue
            if not (
                _heading_matches_any(seg_type, _PRIMARY_NARRATIVE_KEYWORDS)
                or _heading_matches_any(seg_type, _SUPPORTING_SECTION_KEYWORDS)
                or _heading_matches_any(seg_type, _SUPPORTING_DATA_KEYWORDS)
            ):
                continue
            seg_text = (getattr(seg, "text", "") or "").strip()
            if seg_text:
                extracted.setdefault(seg_type_raw.strip(), []).append(seg_text)
    except Exception:
        pass

    # 2) Sectionizer handles isolated headings and formatting quirks for a small
    # curated set. Include both narrative and supporting headings.
    try:
        sectionizer = SectionizerService(
            headings=tuple(
                sorted(
                    {
                        "PROCEDURE",
                        "PROCEDURE IN DETAIL",
                        "DESCRIPTION OF PROCEDURE",
                        "FINDINGS",
                        "EBUS-FINDINGS",
                        "EBUS FINDINGS",
                        "LYMPH NODES EVALUATED",
                        "AIRWAY INSPECTION",
                        "TECHNIQUE",
                        "OPERATIVE REPORT",
                        "SPECIMEN(S)",
                        "SPECIMENS",
                        "IMPRESSION",
                    }
                )
            )
        )
        sections = sectionizer.sectionize(original)
        for section in sections:
            title_raw = (section.title or "").strip()
            title = _canonical_heading(title_raw)
            if not title:
                continue
            if _heading_excluded(title):
                continue
            if not (
                _heading_matches_any(title, _PRIMARY_NARRATIVE_KEYWORDS)
                or _heading_matches_any(title, _SUPPORTING_SECTION_KEYWORDS)
                or _heading_matches_any(title, _SUPPORTING_DATA_KEYWORDS)
            ):
                continue
            clean = (section.text or "").strip()
            if clean:
                extracted.setdefault(title_raw, []).append(clean)
    except Exception:
        pass

    # 3) Regex fallback for any remaining colon-delimited headings.
    regex_extracted = _extract_target_sections_by_regex(original)
    for title_raw, bodies in regex_extracted.items():
        for body in bodies:
            extracted.setdefault(title_raw, [])
            if body not in extracted[title_raw]:
                extracted[title_raw].append(body)

    if not extracted:
        return note_text

    # De-dupe while preserving insertion ordering from the extractors above.
    ordered_titles = list(extracted.keys())
    for title in ordered_titles:
        extracted[title] = _dedupe_bodies(extracted[title])

    primary_parts: list[str] = []
    supporting_parts: list[str] = []

    for title_raw in ordered_titles:
        title = _canonical_heading(title_raw)
        bucket = "ignore"
        if _heading_matches_any(title, _PRIMARY_NARRATIVE_KEYWORDS):
            bucket = "primary"
        elif _heading_matches_any(title, _SUPPORTING_DATA_KEYWORDS) or _heading_matches_any(
            title, _SUPPORTING_SECTION_KEYWORDS
        ):
            bucket = "support"

        if bucket == "ignore":
            continue

        for text in extracted.get(title_raw, []) or []:
            clean = (text or "").strip()
            if not clean:
                continue
            if bucket == "primary":
                clean = _trim_post_procedure_tail(clean)
                if not clean:
                    continue
            rendered = f"{title_raw.strip().upper()}:\n{clean}"
            if bucket == "primary":
                primary_parts.append(rendered)
            else:
                supporting_parts.append(rendered)

    # If we failed to capture any narrative, fall back to supporting headings.
    if not primary_parts:
        primary_parts = supporting_parts
        supporting_parts = []

    primary_text = "\n\n".join(primary_parts).strip()
    supporting_text = "\n\n".join(supporting_parts).strip()

    if not primary_text and not supporting_text:
        return note_text

    if supporting_text:
        return (
            "<primary_narrative>\n"
            f"{primary_text}\n"
            "</primary_narrative>\n\n"
            "<supporting_data>\n"
            f"{supporting_text}\n"
            "</supporting_data>"
        ).strip()

    return f"<primary_narrative>\n{primary_text}\n</primary_narrative>".strip()


__all__ = ["get_procedure_focus"]
