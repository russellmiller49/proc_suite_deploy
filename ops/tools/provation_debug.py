#!/usr/bin/env python3
"""ProVation fingerprint + cleaning + canonical section debug tool (local only).

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
from app.sectioning.provation_section_parser import parse_provation_procedure_pages  # noqa: E402
from app.text_cleaning.provation_cleaner import clean_provation  # noqa: E402


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


def _parse_page_spec(spec: str | None, max_pages: int) -> list[int]:
    if not spec:
        return list(range(max_pages))
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = max(1, int(start_s))
            end = min(max_pages, int(end_s))
            for v in range(start, end + 1):
                out.add(v - 1)
        else:
            v = int(token)
            if 1 <= v <= max_pages:
                out.add(v - 1)
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="Path to a local PDF (text-layer only; no OCR).")
    parser.add_argument("--text", type=Path, help="Path to an already-extracted text file.")
    parser.add_argument("--out", type=Path, required=True, help="Write debug JSON output here.")
    parser.add_argument("--dump-clean", type=Path, help="Optional: write cleaned full text here.")
    parser.add_argument("--pages", type=str, help="Optional page selector (e.g., '1-5' or '1,3').")
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
    keep = set(_parse_page_spec(args.pages, max_pages=len(doc.pages)))
    if keep and len(keep) != len(doc.pages):
        selected_pages = [p for i, p in enumerate(doc.pages) if i in keep]
        doc = type(doc)(prefix=doc.prefix, pages=selected_pages)
        note_text = doc.reassemble()

    fp = fingerprint_document(note_text, doc.page_texts)
    page_meta = clean_provation(doc.page_texts, fp.page_types)
    cleaned_pages = [p.clean_text for p in page_meta]
    cleaned_full = doc.reassemble(cleaned_page_bodies=cleaned_pages) if cleaned_pages else note_text
    if args.dump_clean:
        args.dump_clean.write_text(cleaned_full, encoding="utf-8")

    canonical = parse_provation_procedure_pages(
        raw_pages=doc.page_texts,
        clean_pages=cleaned_pages or doc.page_texts,
        page_types=fp.page_types,
        template_family=fp.template_family,
    )

    payload: dict[str, Any] = {
        "source": source,
        "fingerprint": fp.to_dict(),
        "pages": [p.to_dict() for p in page_meta],
        "canonical_notes": [note.to_dict() for note in canonical],
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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
