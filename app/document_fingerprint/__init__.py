"""Vendor/template fingerprinting for extracted clinical documents."""

from __future__ import annotations

from app.document_fingerprint.registry import (
    FingerprintResult,
    PdfPageChunk,
    PdfTextDocument,
    fingerprint_document,
    split_pdf_fulltext,
)

__all__ = [
    "FingerprintResult",
    "PdfPageChunk",
    "PdfTextDocument",
    "fingerprint_document",
    "split_pdf_fulltext",
]

