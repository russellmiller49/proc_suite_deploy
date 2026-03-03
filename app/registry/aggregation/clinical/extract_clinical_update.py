"""Deterministic clinical-update extractor (MVP)."""

from __future__ import annotations

import re
from typing import Any

from app.registry.aggregation.sanitize import compact_text


_PERFORMANCE_RE = re.compile(
    r"\b(?:ECOG\s*[:=]?\s*\d(?:\s*[-/]\s*\d)?|performance\s+status\s*[:=]?\s*[^\n\.;]{1,50})\b",
    re.IGNORECASE,
)

_TREATMENT_RE = re.compile(
    r"\b(?:started|initiated|completed|discontinued|held|resumed)\b[^\n\.;]{0,120}\b(?:therapy|immunotherapy|chemotherapy|radiation|treatment|steroid|antibiotic)s?\b",
    re.IGNORECASE,
)

_COMPLICATION_KEYWORDS = [
    "pneumothorax",
    "hemoptysis",
    "bleeding",
    "hypoxia",
    "fever",
    "infection",
    "respiratory failure",
]

_HOSPITAL_ADMIT_NEG_RE = re.compile(r"\b(?:no|not)\s+(?:admitted|hospitalized|inpatient)\b", re.IGNORECASE)
_HOSPITAL_ADMIT_POS_RE = re.compile(
    r"\b(?:hospital\s+admission|admitted\s+to\s+the\s+hospital|hospitalized|inpatient)\b",
    re.IGNORECASE,
)
_ICU_NEG_RE = re.compile(r"\b(?:no|not)\s+(?:icu|intensive\s+care)\b", re.IGNORECASE)
_ICU_POS_RE = re.compile(r"\b(?:icu|micu|sicu|ccu|intensive\s+care)\b", re.IGNORECASE)
_DECEASED_NEG_RE = re.compile(r"\b(?:not|no)\s+deceased\b", re.IGNORECASE)
_DECEASED_POS_RE = re.compile(r"\b(?:deceased|died|passed\s+away|expired)\b", re.IGNORECASE)

_DISEASE_PROGRESSION_RE = re.compile(
    r"\b(?:progression|progressive)\b[^\n\.;]{0,30}\b(?:disease|cancer|tumou?r)\b|\bprogression\s+of\s+disease\b",
    re.IGNORECASE,
)
_DISEASE_STABLE_RE = re.compile(
    r"\bstable\b[^\n\.;]{0,30}\b(?:disease|cancer|tumou?r)\b|\bstable\s+disease\b",
    re.IGNORECASE,
)
_DISEASE_RESPONSE_RE = re.compile(
    r"\b(?:partial|complete)\s+response\b|\bresponding\b[^\n\.;]{0,30}\b(?:disease|cancer|tumou?r)\b",
    re.IGNORECASE,
)
_DISEASE_MIXED_RE = re.compile(r"\bmixed\s+response\b", re.IGNORECASE)
_DISEASE_INDET_RE = re.compile(r"\bindeterminate\b[^\n\.;]{0,30}\b(?:response|disease)\b", re.IGNORECASE)


def _symptom_change(text: str) -> str | None:
    value = text.lower()
    if re.search(r"\b(?:improved|better|resolving)\b", value):
        return "Better"
    if re.search(r"\b(?:worse|worsened|progressive|declined)\b", value):
        return "Worse"
    if re.search(r"\b(?:stable|unchanged)\b", value):
        return "Stable"
    return None


def _extract_complication_text(text: str) -> str | None:
    value = text or ""
    for keyword in _COMPLICATION_KEYWORDS:
        match = re.search(rf"\b{re.escape(keyword)}\b[^\n\.;]{{0,100}}", value, re.IGNORECASE)
        if match:
            return compact_text(match.group(0), max_chars=160)
    return None


def _extract_tri_state(text: str, *, neg_re: re.Pattern[str], pos_re: re.Pattern[str]) -> bool | None:
    value = text or ""
    if neg_re.search(value):
        return False
    if pos_re.search(value):
        return True
    return None


def _extract_disease_status(text: str) -> str | None:
    value = text or ""
    if _DISEASE_MIXED_RE.search(value):
        return "Mixed"
    if _DISEASE_PROGRESSION_RE.search(value):
        return "Progression"
    if _DISEASE_RESPONSE_RE.search(value):
        return "Response"
    if _DISEASE_STABLE_RE.search(value):
        return "Stable"
    if _DISEASE_INDET_RE.search(value):
        return "Indeterminate"
    return None


def extract_clinical_update_event(
    text: str,
    *,
    update_type: str,
    relative_day_offset: int | None,
) -> dict[str, Any]:
    """Extract minimal structured clinical update fields."""

    clean = text or ""
    qa_flags: list[str] = []

    hospital_admission = _extract_tri_state(clean, neg_re=_HOSPITAL_ADMIT_NEG_RE, pos_re=_HOSPITAL_ADMIT_POS_RE)
    icu_admission = _extract_tri_state(clean, neg_re=_ICU_NEG_RE, pos_re=_ICU_POS_RE)
    deceased = _extract_tri_state(clean, neg_re=_DECEASED_NEG_RE, pos_re=_DECEASED_POS_RE)
    disease_status = _extract_disease_status(clean)

    performance_status_text = None
    performance_match = _PERFORMANCE_RE.search(clean)
    if performance_match:
        performance_status_text = compact_text(performance_match.group(0), max_chars=80)

    treatment_change_text = None
    treatment_match = _TREATMENT_RE.search(clean)
    if treatment_match:
        treatment_change_text = compact_text(treatment_match.group(0), max_chars=140)

    complication_text = _extract_complication_text(clean)
    symptom_change = _symptom_change(clean)

    summary_text = compact_text(clean, max_chars=220)
    if not any(
        [
            performance_status_text,
            treatment_change_text,
            complication_text,
            symptom_change,
            hospital_admission is not None,
            icu_admission is not None,
            deceased is not None,
            disease_status is not None,
        ]
    ):
        qa_flags.append("minimal_structure_extracted")

    return {
        "clinical_update": {
            "relative_day_offset": int(relative_day_offset or 0),
            "update_type": update_type,
            "performance_status_text": performance_status_text,
            "symptom_change": symptom_change,
            "treatment_change_text": treatment_change_text,
            "complication_text": complication_text,
            "summary_text": summary_text or None,
            "hospital_admission": hospital_admission,
            "icu_admission": icu_admission,
            "deceased": deceased,
            "disease_status": disease_status,
            "qa_flags": sorted(set(qa_flags)),
        },
        "qa_flags": sorted(set(qa_flags)),
    }


__all__ = ["extract_clinical_update_event"]
