"""Vendor-aware text cleanup utilities.

These cleaners are designed to be safe for evidence anchoring: by default they
mask (space-out) removed spans while preserving offsets and newlines.
"""

from __future__ import annotations

from app.text_cleaning.camera_ocr_cleaner import (
    CameraOcrCleanerUnavailable,
    CameraOcrSanitizeResult,
    sanitize_camera_ocr_text,
)
from app.text_cleaning.camera_ocr_fuzzy import (
    CameraOcrFuzzyReplacement,
    CameraOcrFuzzyResult,
    clear_camera_ocr_fuzzy_phrase_cache,
    normalize_camera_ocr_for_extraction,
)
from app.text_cleaning.endosoft_cleaner import clean_endosoft, clean_endosoft_page
from app.text_cleaning.provation_cleaner import clean_provation, clean_provation_page

__all__ = [
    "CameraOcrCleanerUnavailable",
    "CameraOcrSanitizeResult",
    "CameraOcrFuzzyReplacement",
    "CameraOcrFuzzyResult",
    "clear_camera_ocr_fuzzy_phrase_cache",
    "clean_endosoft",
    "clean_endosoft_page",
    "clean_provation",
    "clean_provation_page",
    "sanitize_camera_ocr_text",
    "normalize_camera_ocr_for_extraction",
]
