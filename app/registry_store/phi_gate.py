"""Lightweight PHI risk scan before persistence.

This is a *safety gate*, not a full PHI scrubber. It should:
- be fast
- avoid returning matched PHI snippets
- catch obvious high-risk leftovers (DOB/MRN/SSN/email/phone, etc.)
"""

from __future__ import annotations

import re


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
_MRN_RE = re.compile(r"\b(?:mrn|medical record number|patient id)\s*[:#]?\s*\d{4,}\b", re.IGNORECASE)
_DOB_RE = re.compile(
    r"\b(?:dob|date of birth|birth date)\s*[:#]?\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},\s*\d{4})\b",
    re.IGNORECASE,
)


def scan_text_for_phi_risk(text: str) -> list[str]:
    """Return a list of non-sensitive PHI risk reasons."""

    reasons: list[str] = []

    if _EMAIL_RE.search(text):
        reasons.append("Email address pattern detected")
    if _SSN_RE.search(text):
        reasons.append("SSN pattern detected")
    if _PHONE_RE.search(text):
        reasons.append("Phone number pattern detected")
    if _MRN_RE.search(text):
        reasons.append("MRN/patient ID pattern detected")
    if _DOB_RE.search(text):
        reasons.append("DOB/date-of-birth pattern detected")

    return reasons


__all__ = ["scan_text_for_phi_risk"]

