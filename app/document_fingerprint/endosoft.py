from __future__ import annotations

import re
from typing import Any

from app.document_fingerprint.registry import FingerprintResult


_BRAND_ENDOSOFT_RE = re.compile(r"(?i)\bendosoft\b")
_BRAND_PHOTOREPORT_RE = re.compile(r"(?i)\bphotoreport\b")
_PROCEDURE_REPORT_RE = re.compile(r"(?i)\bprocedure\s+report\b")

_DEMO_LABEL_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("patient_name", re.compile(r"(?i)\bpatient\s+name\b")),
    ("dob", re.compile(r"(?i)\bdate\s+of\s+birth\b|\bDOB\b")),
    ("record_number", re.compile(r"(?i)\brecord\s+number\b|\bMRN\b")),
    ("procedure_datetime", re.compile(r"(?i)\bdate\s*/\s*time\s+of\s+procedure\b|\bdate\s+of\s+procedure\b")),
)

_SECTION_HEADER_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("procedure_performed", re.compile(r"(?i)^\s*PROCEDURE\s+PERFORMED\b", re.M)),
    ("indications", re.compile(r"(?i)^\s*INDICATIONS?\s+FOR\s+EXAM", re.M)),
    ("technique", re.compile(r"(?i)^\s*PROCEDURE\s+TECHNIQUE\b", re.M)),
    ("findings", re.compile(r"(?i)^\s*FINDINGS\b", re.M)),
    ("recommendations", re.compile(r"(?i)^\s*RECOMMENDATIONS?\b", re.M)),
    ("icd10", re.compile(r"(?i)^\s*ICD\s*10\s+Codes?\b", re.M)),
    ("cpt", re.compile(r"(?i)^\s*CPT\s+Codes?\b|\bCPT\s+Code\b", re.M)),
    ("impression", re.compile(r"(?i)^\s*(?:IMPRESSION|POST\s*OP(?:ERATIVE)?\s*DIAGNOSIS)\b", re.M)),
)

_DISCHARGE_RE = re.compile(r"(?i)\bdischarge\s+instructions\b")
_PATH_REQUISITION_RE = re.compile(r"(?i)\bpathology\s+requisition\b")
_SIGNED_RE = re.compile(r"(?i)\belectronically\s+signed\b|\bsigned\s+off\b")

_CAPTION_NUMBER_ONLY_RE = re.compile(r"^\s*\d+\s*$")
_CAPTION_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s+[A-Za-z][A-Za-z0-9/()_-]{0,20}\b.*$")
_SHORT_ANATOMY_LABEL_RE = re.compile(
    r"(?i)^(left|right|upper|lower|middle|mainstem|segment|bronchus|airway|carina|trachea|lingula)(\s+\w+){0,6}$"
)


def _count_section_headers(text: str) -> int:
    return sum(1 for _name, pat in _SECTION_HEADER_RES if pat.search(text or ""))


def classify_page_type(page_text: str) -> str:
    text = page_text or ""
    if _DISCHARGE_RE.search(text):
        return "discharge_instructions"
    if _PATH_REQUISITION_RE.search(text):
        return "pathology_requisition"

    header_hits = _count_section_headers(text)
    if _PROCEDURE_REPORT_RE.search(text) and header_hits >= 2:
        return "procedure_report"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        caption_hits = 0
        for line in lines:
            if _CAPTION_NUMBER_ONLY_RE.match(line):
                caption_hits += 1
                continue
            if _CAPTION_NUMBER_PREFIX_RE.match(line) and len(line) <= 60:
                caption_hits += 1
                continue
            if _SHORT_ANATOMY_LABEL_RE.match(line) and len(line) <= 60:
                caption_hits += 1
                continue

        # EndoSoft PhotoReport image pages can be short; use a moderate threshold.
        if caption_hits >= 6 and (caption_hits / max(1, len(lines))) >= 0.45 and header_hits == 0:
            return "images_page"

    if _SIGNED_RE.search(text) and len(re.sub(r"\s+", "", text)) < 800:
        return "signature_only"

    return "unknown"


def fingerprint(text: str, page_texts: list[str]) -> FingerprintResult:
    raw = text or ""
    pages = list(page_texts or [])

    signals: dict[str, Any] = {}
    score = 0.0

    has_endosoft = bool(_BRAND_ENDOSOFT_RE.search(raw))
    has_photoreport = bool(_BRAND_PHOTOREPORT_RE.search(raw))
    has_proc_report = bool(_PROCEDURE_REPORT_RE.search(raw))

    if has_endosoft:
        score += 0.45
    if has_photoreport:
        score += 0.45
    if has_proc_report:
        score += 0.20

    demo_hits: list[str] = []
    for key, pat in _DEMO_LABEL_RES:
        if pat.search(raw):
            demo_hits.append(key)
            score += 0.05
    score += 0.0  # explicit for readability

    header_hits: list[str] = []
    for key, pat in _SECTION_HEADER_RES:
        if pat.search(raw):
            header_hits.append(key)
            score += 0.04

    score = min(1.0, score)
    signals.update(
        {
            "brand_endosoft": has_endosoft,
            "brand_photoreport": has_photoreport,
            "has_procedure_report": has_proc_report,
            "demographics_labels": demo_hits,
            "section_headers": header_hits,
            "score_raw": score,
        }
    )

    page_types = [classify_page_type(p) for p in pages] if pages else []
    template_family = "photoreport" if (has_photoreport or "images_page" in page_types) else "endosoft_basic_report"

    strong_brand = has_endosoft or has_photoreport
    if not strong_brand:
        # Avoid false positives on generic "Procedure Report" notes with common headings.
        return FingerprintResult(
            vendor="unknown",
            template_family="unknown",
            confidence=min(0.54, score),
            page_types=["unknown" for _ in page_types] if page_types else [],
            signals=signals,
        )

    return FingerprintResult(
        vendor="endosoft",
        template_family=template_family,
        confidence=score,
        page_types=page_types,
        signals=signals,
    )


__all__ = ["classify_page_type", "fingerprint"]
