from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FingerprintResult:
    vendor: str
    template_family: str
    confidence: float
    page_types: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PdfPageChunk:
    """A single page chunk from the client PDF pipeline fullText.

    `header` is the "===== PAGE N (...) =====" marker line (including its newline).
    `body` is the page text content up to (but not including) the next header line.
    """

    page_num: int
    header: str
    body: str


@dataclass(slots=True)
class PdfTextDocument:
    """Parsed representation of client-side PDF extraction `fullText`."""

    prefix: str
    pages: list[PdfPageChunk]

    @property
    def page_texts(self) -> list[str]:
        return [p.body for p in self.pages]

    def reassemble(self, *, cleaned_page_bodies: list[str] | None = None) -> str:
        if cleaned_page_bodies is None:
            cleaned_page_bodies = self.page_texts
        if len(cleaned_page_bodies) != len(self.pages):
            raise ValueError("cleaned_page_bodies length must match pages")
        out = [self.prefix]
        for page, body in zip(self.pages, cleaned_page_bodies, strict=True):
            out.append(page.header)
            out.append(body)
        return "".join(out)


_PAGE_HEADER_RE = re.compile(r"(?m)^===== PAGE (?P<num>\d+) \((?P<label>[^\n]*)\) =====\s*$")


def split_pdf_fulltext(text: str) -> PdfTextDocument:
    """Split the client-side PDF `fullText` into per-page chunks.

    If no page markers are present, returns a single-page document with empty header.
    """
    raw = text or ""
    matches = list(_PAGE_HEADER_RE.finditer(raw))
    if not matches:
        return PdfTextDocument(prefix="", pages=[PdfPageChunk(page_num=1, header="", body=raw)])

    prefix = raw[: matches[0].start()]
    pages: list[PdfPageChunk] = []

    for idx, match in enumerate(matches):
        header_start = match.start()
        header_end = match.end()
        # Include the header line terminator so we can reassemble byte-for-byte (length-preserving).
        if raw[header_end : header_end + 2] == "\r\n":
            header_end += 2
        elif raw[header_end : header_end + 1] == "\n":
            header_end += 1

        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        try:
            page_num = int(match.group("num"))
        except ValueError:
            page_num = idx + 1
        pages.append(
            PdfPageChunk(
                page_num=page_num,
                header=raw[header_start:header_end],
                body=raw[header_end:body_end],
            )
        )

    return PdfTextDocument(prefix=prefix, pages=pages)


def fingerprint_document(text: str, page_texts: list[str] | None = None) -> FingerprintResult:
    """Best-effort vendor/template fingerprint for already-extracted text."""
    doc = split_pdf_fulltext(text)
    pages = list(page_texts) if page_texts is not None else doc.page_texts

    # Lazy imports to avoid import cycles.
    from app.document_fingerprint import endosoft, provation

    candidates: list[FingerprintResult] = [
        endosoft.fingerprint(text, pages),
        provation.fingerprint(text, pages),
    ]
    best = max(candidates, key=lambda r: float(r.confidence or 0.0))
    if float(best.confidence or 0.0) < 0.55:
        return FingerprintResult(
            vendor="unknown",
            template_family="unknown",
            confidence=float(best.confidence or 0.0),
            page_types=["unknown" for _ in pages],
            signals={"candidates": [c.to_dict() for c in candidates]},
        )

    merged_signals: dict[str, Any] = dict(best.signals or {})
    merged_signals["candidates"] = [c.to_dict() for c in candidates]
    return FingerprintResult(
        vendor=str(best.vendor),
        template_family=str(best.template_family),
        confidence=float(best.confidence),
        page_types=list(best.page_types or ["unknown" for _ in pages]),
        signals=merged_signals,
    )


__all__ = [
    "FingerprintResult",
    "PdfPageChunk",
    "PdfTextDocument",
    "fingerprint_document",
    "split_pdf_fulltext",
]

