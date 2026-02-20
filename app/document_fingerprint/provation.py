from __future__ import annotations

import re
from typing import Any

from app.document_fingerprint.registry import FingerprintResult


_BRAND_POWERED_RE = re.compile(r"(?i)\bpowered\s+by\s+provation\b")
_BRAND_PROVATION_MD_RE = re.compile(r"(?i)\bprovation(?:\u00ae)?\s*md\b")

_PROC_CODES_RE = re.compile(r"(?i)\bprocedure\s+code\(s\)\s*:?\s*")
_DX_CODES_RE = re.compile(r"(?i)\bdiagnosis\s+code\(s\)\s*:?\s*")
_NOTE_STATUS_FINAL_RE = re.compile(r"(?i)\bnote\s+status\s*:\s*finalized\b")
_ADDENDA_RE = re.compile(r"(?i)\bnumber\s+of\s+addenda\b")
_SIGNED_RE = re.compile(r"(?i)\b(e-?\s*signed|electronically\s+signed)\s+by\b")

_VA_SD_RE = re.compile(r"(?i)\bVA\s+San\s+Diego\b|\bSan\s+Diego\s+VA\b")

_ADDL_IMAGES_RE = re.compile(r"(?i)\badd(?:'|i)?l\s+images\b|\badditional\s+images\b")

_CPT_RE = re.compile(r"\b\d{5}\b")

_SECTION_KEYWORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("indications", re.compile(r"(?i)^\s*indications?\b", re.M)),
    ("medications", re.compile(r"(?i)^\s*medications?\b", re.M)),
    ("procedure", re.compile(r"(?i)^\s*procedure\s*:", re.M)),
    ("findings", re.compile(r"(?i)^\s*findings\b", re.M)),
    ("impression", re.compile(r"(?i)^\s*impression\b", re.M)),
    ("recommendation", re.compile(r"(?i)^\s*recommendations?\b", re.M)),
)


def _count_section_hits(text: str) -> int:
    return sum(1 for _name, pat in _SECTION_KEYWORDS if pat.search(text or ""))


def classify_page_type(page_text: str) -> str:
    text = page_text or ""
    if _ADDL_IMAGES_RE.search(text):
        return "images_page"

    section_hits = _count_section_hits(text)
    code_count = len(_CPT_RE.findall(text))

    if section_hits >= 2:
        return "procedure_report"

    if (_PROC_CODES_RE.search(text) or _DX_CODES_RE.search(text)) and code_count >= 4 and section_hits == 0:
        return "codes_page"

    signature_hits = sum(
        1
        for pat in (_NOTE_STATUS_FINAL_RE, _ADDENDA_RE, _SIGNED_RE)
        if pat.search(text)
    )
    if signature_hits >= 2 and len(re.sub(r"\s+", "", text)) < 900 and section_hits == 0:
        return "signature_only"

    return "unknown"


def fingerprint(text: str, page_texts: list[str]) -> FingerprintResult:
    raw = text or ""
    pages = list(page_texts or [])

    signals: dict[str, Any] = {}
    score = 0.0

    has_powered = bool(_BRAND_POWERED_RE.search(raw))
    has_provation_md = bool(_BRAND_PROVATION_MD_RE.search(raw))

    if has_powered:
        score += 0.50
    if has_provation_md:
        score += 0.35

    proc_codes = bool(_PROC_CODES_RE.search(raw))
    dx_codes = bool(_DX_CODES_RE.search(raw))
    note_final = bool(_NOTE_STATUS_FINAL_RE.search(raw))
    addenda = bool(_ADDENDA_RE.search(raw))
    signed = bool(_SIGNED_RE.search(raw))

    if proc_codes:
        score += 0.20
    if dx_codes:
        score += 0.20
    if note_final:
        score += 0.15
    if signed:
        score += 0.15
    if addenda:
        score += 0.08

    score = min(1.0, score)

    page_types = [classify_page_type(p) for p in pages] if pages else []

    template_family = "provation_md"
    if _VA_SD_RE.search(raw):
        template_family = "provation_va_sd"
    else:
        lower = raw.lower()
        if "procedure note sample" in lower and any(token in lower for token in ("bronchoscopy", "ebus", "tbna")):
            template_family = "provation_sample_bronchoscopy"
        elif any(token in lower for token in ("colonoscopy", "egd", "esophagogastroduodenoscopy")):
            template_family = "provation_gi_sample"

    signals.update(
        {
            "brand_powered": has_powered,
            "brand_provation_md": has_provation_md,
            "has_proc_codes": proc_codes,
            "has_dx_codes": dx_codes,
            "note_status_finalized": note_final,
            "has_addenda_label": addenda,
            "has_signature": signed,
            "score_raw": score,
        }
    )

    strong_brand = has_powered or has_provation_md
    if not strong_brand:
        return FingerprintResult(
            vendor="unknown",
            template_family="unknown",
            confidence=min(0.54, score),
            page_types=["unknown" for _ in page_types] if page_types else [],
            signals=signals,
        )

    return FingerprintResult(
        vendor="provation",
        template_family=template_family,
        confidence=score,
        page_types=page_types,
        signals=signals,
    )


__all__ = ["classify_page_type", "fingerprint"]
