from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_ICD10_RE = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b")
_CPT_RE = re.compile(r"\b\d{5}\b")


@dataclass(slots=True)
class CanonicalNote:
    vendor: str
    template_family: str
    page_num: int
    page_type: str
    demographics: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    codes: dict[str, list[str]] = field(default_factory=dict)
    raw_text_by_section: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEMO_PATTERNS: dict[str, re.Pattern[str]] = {
    "patient_name": re.compile(r"(?im)^\s*Patient\s+Name\s*:\s*(?P<v>.+?)\s*$"),
    "dob": re.compile(r"(?im)^\s*(?:Date\s+of\s+Birth|DOB)\s*:\s*(?P<v>.+?)\s*$"),
    "record_number": re.compile(r"(?im)^\s*(?:Record\s+Number|MRN)\s*:\s*(?P<v>.+?)\s*$"),
    "datetime": re.compile(
        r"(?im)^\s*(?:Date\s*/\s*Time\s+of\s+Procedure|Date\s+of\s+Procedure)\s*:\s*(?P<v>.+?)\s*$"
    ),
    "referring_physician": re.compile(r"(?im)^\s*Referring\s+Physician\s*:\s*(?P<v>.+?)\s*$"),
    "pulmonologist": re.compile(r"(?im)^\s*Pulmonologist\s*:\s*(?P<v>.+?)\s*$"),
}


_HEADING_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("procedure_performed", ("PROCEDURE PERFORMED", "PROCEDURES PERFORMED")),
    ("indications", ("INDICATIONS FOR EXAMINATION", "INDICATIONS FOR EXAM", "INDICATIONS")),
    ("technique", ("PROCEDURE TECHNIQUE", "TECHNIQUE")),
    ("findings", ("FINDINGS",)),
    ("impression_or_postop_dx", ("IMPRESSION", "POSTOPERATIVE DIAGNOSIS", "POST OP DIAGNOSIS")),
    ("recommendations", ("RECOMMENDATIONS", "RECOMMENDATION")),
    ("codes", ("ICD 10 CODES", "ICD-10 CODES", "ICD10 CODES", "CPT CODE", "CPT CODES")),
)


def _build_heading_patterns() -> list[tuple[str, re.Pattern[str]]]:
    out: list[tuple[str, re.Pattern[str]]] = []
    for key, variants in _HEADING_SPECS:
        # Inline heading: HEADING: rest-of-line
        inline = re.compile(
            r"(?im)^\s*(?:{})\s*:\s*(?P<rest>.*)$".format("|".join(re.escape(v) for v in variants))
        )
        standalone = re.compile(
            r"(?im)^\s*(?:{})\s*:?\s*$".format("|".join(re.escape(v) for v in variants))
        )
        out.append((key, inline))
        out.append((key, standalone))
    return out


_HEADING_PATTERNS = _build_heading_patterns()


@dataclass(slots=True)
class _HeadingMatch:
    key: str
    start: int
    end: int
    content_start: int


def _find_heading_matches(text: str) -> list[_HeadingMatch]:
    raw = text or ""
    matches: list[_HeadingMatch] = []
    for key, pat in _HEADING_PATTERNS:
        for m in pat.finditer(raw):
            content_start = m.start("rest") if "rest" in m.groupdict() and (m.group("rest") or "").strip() else m.end()
            matches.append(_HeadingMatch(key=key, start=m.start(), end=m.end(), content_start=content_start))
    # De-dupe overlaps: keep the earliest (inline usually wins because content_start < end).
    matches.sort(key=lambda x: (x.start, x.content_start, -(x.end - x.start)))
    deduped: list[_HeadingMatch] = []
    last_span: tuple[int, int] | None = None
    for item in matches:
        span = (item.start, item.end)
        if last_span and span[0] == last_span[0] and span[1] == last_span[1]:
            continue
        deduped.append(item)
        last_span = span
    deduped.sort(key=lambda x: x.start)
    return deduped


def _extract_demographics(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, pat in _DEMO_PATTERNS.items():
        m = pat.search(text or "")
        if not m:
            continue
        value = (m.group("v") or "").strip()
        if value:
            out[key] = value
    return out


def parse_endosoft_procedure_pages(
    *,
    clean_pages: list[str],
    page_types: list[str],
    raw_pages: list[str] | None = None,
    template_family: str = "endosoft",
) -> list[CanonicalNote]:
    """Parse cleaned EndoSoft pages into canonical procedure-report notes."""
    if len(clean_pages) != len(page_types):
        raise ValueError("clean_pages length must match page_types")
    if raw_pages is not None and len(raw_pages) != len(clean_pages):
        raise ValueError("raw_pages length must match clean_pages")

    notes: list[CanonicalNote] = []
    for idx, (clean_text, page_type) in enumerate(zip(clean_pages, page_types, strict=True)):
        if page_type != "procedure_report":
            continue
        raw_text = raw_pages[idx] if raw_pages is not None else clean_text
        if raw_pages is not None and len(raw_text) != len(clean_text):
            raw_text = clean_text

        demo = _extract_demographics(clean_text)
        heading_matches = _find_heading_matches(clean_text)

        sections: dict[str, str] = {}
        raw_by_section: dict[str, str] = {}

        for h_idx, h in enumerate(heading_matches):
            start = h.content_start
            end = heading_matches[h_idx + 1].start if h_idx + 1 < len(heading_matches) else len(clean_text)
            chunk_clean = (clean_text[start:end] or "").strip()
            chunk_raw = (raw_text[start:end] or "").strip()
            if not chunk_clean:
                continue
            if h.key in sections:
                sections[h.key] = (sections[h.key] + "\n" + chunk_clean).strip()
                raw_by_section[h.key] = (raw_by_section.get(h.key, "") + "\n" + chunk_raw).strip()
            else:
                sections[h.key] = chunk_clean
                raw_by_section[h.key] = chunk_raw

        code_source = sections.get("codes") or clean_text
        icd10 = sorted({m.group(0) for m in _ICD10_RE.finditer(code_source or "")})
        cpt = sorted({m.group(0) for m in _CPT_RE.finditer(code_source or "")})

        notes.append(
            CanonicalNote(
                vendor="endosoft",
                template_family=template_family,
                page_num=idx + 1,
                page_type=page_type,
                demographics=demo,
                sections=sections,
                codes={"icd10": icd10, "cpt": cpt},
                raw_text_by_section=raw_by_section,
            )
        )

    return notes


__all__ = ["CanonicalNote", "parse_endosoft_procedure_pages"]

