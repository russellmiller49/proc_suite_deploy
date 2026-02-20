#!/usr/bin/env python3
"""EndoSoft fingerprint + cleaning + canonical section debug tool (local only).

This tool does not upload anything. It is intended to run on already-extracted text
(ideally PHI-scrubbed). If `--pdf` is provided, it performs a best-effort text-layer
extraction (no OCR) using an optional dependency (`pypdf`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.document_fingerprint.registry import fingerprint_document, split_pdf_fulltext  # noqa: E402
from app.sectioning.endosoft_section_parser import parse_endosoft_procedure_pages  # noqa: E402
from app.text_cleaning.endosoft_cleaner import clean_endosoft_page_with_meta  # noqa: E402


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Missing dependency: pypdf") from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        parts.append(f"===== PAGE {idx + 1} (PDF) =====\n{text}\n")
    return "\n".join(parts)


def _mask_diff_lines(raw: str, clean: str, *, limit: int = 80) -> list[str]:
    """Return a small sample of lines that were fully masked."""
    out: list[str] = []
    raw_lines = (raw or "").splitlines()
    clean_lines = (clean or "").splitlines()
    for r, c in zip(raw_lines, clean_lines, strict=False):
        if not r.strip():
            continue
        if r.strip() and not c.strip():
            out.append(r.strip()[:200])
        if len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="Path to a local PDF (text-layer only; no OCR).")
    parser.add_argument("--text", type=Path, help="Path to an already-extracted text file.")
    parser.add_argument("--out", type=Path, required=True, help="Write debug JSON output here.")
    parser.add_argument("--dump-clean-text", type=Path, help="Optional: write cleaned full text here.")
    args = parser.parse_args(argv)

    if bool(args.pdf) == bool(args.text):
        parser.error("Provide exactly one of --pdf or --text")

    if args.text:
        note_text = args.text.read_text(encoding="utf-8")
        source = str(args.text)
    else:
        pdf_path: Path = args.pdf
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        note_text = _read_pdf_text(pdf_path)
        source = str(pdf_path)

    doc = split_pdf_fulltext(note_text)
    fp = fingerprint_document(note_text, doc.page_texts)

    cleaned_pages: list[str] = []
    page_debug: list[dict[str, Any]] = []
    for body, page_type, chunk in zip(doc.page_texts, fp.page_types, doc.pages, strict=False):
        cleaned, meta = clean_endosoft_page_with_meta(body, page_type)
        cleaned_pages.append(cleaned)
        page_debug.append(
            {
                "page_num": chunk.page_num,
                "page_type": page_type,
                "clean_meta": {
                    "masked_footer_lines": meta.masked_footer_lines,
                    "masked_caption_lines": meta.masked_caption_lines,
                    "masked_dedup_blocks": meta.masked_dedup_blocks,
                },
                "masked_line_samples": _mask_diff_lines(body, cleaned, limit=40),
            }
        )

    cleaned_full = doc.reassemble(cleaned_page_bodies=cleaned_pages) if cleaned_pages else note_text
    if args.dump_clean_text:
        args.dump_clean_text.write_text(cleaned_full, encoding="utf-8")

    canonical = parse_endosoft_procedure_pages(
        raw_pages=doc.page_texts,
        clean_pages=cleaned_pages or doc.page_texts,
        page_types=fp.page_types,
        template_family=fp.template_family,
    )

    payload: dict[str, Any] = {
        "source": source,
        "fingerprint": fp.to_dict(),
        "pages": page_debug,
        "canonical_notes": [note.to_dict() for note in canonical],
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # Console output is metrics-only (avoid printing clinical text).
    counts: dict[str, int] = {}
    for t in fp.page_types:
        counts[t] = counts.get(t, 0) + 1
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(
        f"vendor={fp.vendor} family={fp.template_family} conf={fp.confidence:.2f} pages={len(doc.pages)} {counts_str}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
