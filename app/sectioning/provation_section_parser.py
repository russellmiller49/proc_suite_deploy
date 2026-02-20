from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_ICD_RE = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b")
_CPT_RE = re.compile(r"\b\d{5}\b")

_LABEL_LINE_RE = re.compile(r"^\s*(?P<label>[^:]{1,40})\s*:\s*(?P<value>.+?)\s*$")

_SIGNED_BY_RE = re.compile(
    r"(?im)\b(e-?\s*signed|electronically\s+signed)\s+by\s*:\s*(?P<v>.+?)\s*$"
)
_SIGNED_DT_RE = re.compile(
    r"(?im)\b(?:signed|finalized)\s*(?:date|date/time|datetime)\s*:\s*(?P<v>.+?)\s*$"
)


@dataclass(slots=True)
class CanonicalNote:
    vendor: str
    template_family: str
    page_num: int
    page_type: str
    demographics: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    codes: dict[str, list[str]] = field(default_factory=dict)
    signatures: dict[str, Any] = field(default_factory=dict)
    raw_text_by_section: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("indications", ("INDICATIONS", "INDICATION")),
    ("medications", ("MEDICATIONS",)),
    ("procedure", ("PROCEDURE",)),
    ("findings", ("FINDINGS",)),
    ("impression", ("IMPRESSION",)),
    ("recommendation", ("RECOMMENDATIONS", "RECOMMENDATION")),
    ("codes", ("PROCEDURE CODE(S)", "DIAGNOSIS CODE(S)", "CODES")),
)


def _build_heading_patterns() -> list[tuple[str, re.Pattern[str]]]:
    out: list[tuple[str, re.Pattern[str]]] = []
    for key, variants in _SECTION_SPECS:
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


def _parse_va_like_header(text: str, *, max_lines: int = 60) -> dict[str, str]:
    out: dict[str, str] = {}
    lines = (text or "").splitlines()[:max_lines]
    for line in lines:
        m = _LABEL_LINE_RE.match(line)
        if not m:
            continue
        label = (m.group("label") or "").strip().lower()
        value = (m.group("value") or "").strip()
        if not value:
            continue
        if label in {"patient name", "name"}:
            out.setdefault("patient_name", value)
        elif label in {"mrn", "medical record number", "record number"}:
            out.setdefault("mrn", value)
        elif label in {"dob", "date of birth"}:
            out.setdefault("dob", value)
        elif label == "age":
            out.setdefault("age", value)
        elif label in {"sex", "gender"}:
            out.setdefault("sex", value)
        elif label in {"procedure date", "date/time", "date / time", "date/time of procedure"}:
            out.setdefault("procedure_datetime", value)
        elif label in {"location", "facility"}:
            out.setdefault("location", value)
        elif label in {"system", "service"}:
            out.setdefault("system", value)
        elif label in {"note status"}:
            out.setdefault("note_status", value)
    return out


def _parse_signatures(text: str) -> dict[str, Any]:
    signed_by: list[str] = []
    for m in _SIGNED_BY_RE.finditer(text or ""):
        value = (m.group("v") or "").strip()
        if value and value not in signed_by:
            signed_by.append(value)
    signed_dt = None
    m_dt = _SIGNED_DT_RE.search(text or "")
    if m_dt:
        signed_dt = (m_dt.group("v") or "").strip() or None
    status = None
    for line in (text or "").splitlines()[:80]:
        m = _LABEL_LINE_RE.match(line)
        if m and (m.group("label") or "").strip().lower() == "note status":
            status = (m.group("value") or "").strip() or None
            break
    return {"signed_by": signed_by, "signed_datetime": signed_dt, "note_status": status}


def parse_provation_procedure_pages(
    *,
    clean_pages: list[str],
    page_types: list[str],
    raw_pages: list[str] | None = None,
    template_family: str = "provation",
) -> list[CanonicalNote]:
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

        header_demo = _parse_va_like_header(clean_text)
        demographics: dict[str, str] = {k: v for k, v in header_demo.items() if k in {"patient_name", "mrn", "dob", "age", "sex", "procedure_datetime", "location", "system"}}

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

        signatures = _parse_signatures(clean_text)
        if header_demo.get("note_status") and not signatures.get("note_status"):
            signatures["note_status"] = header_demo.get("note_status")

        code_source = sections.get("codes") or clean_text
        cpt = sorted({m.group(0) for m in _CPT_RE.finditer(code_source or "")})
        icd = sorted({m.group(0) for m in _ICD_RE.finditer(code_source or "")})

        notes.append(
            CanonicalNote(
                vendor="provation",
                template_family=template_family,
                page_num=idx + 1,
                page_type=page_type,
                demographics=demographics,
                sections=sections,
                codes={"cpt": cpt, "icd": icd},
                signatures=signatures,
                raw_text_by_section=raw_by_section,
            )
        )

    return notes


__all__ = ["CanonicalNote", "parse_provation_procedure_pages"]
