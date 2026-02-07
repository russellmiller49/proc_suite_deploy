#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


LETTER_WIDTH = 612
LETTER_HEIGHT = 792


def _pdf_escape(text: str) -> str:
    # PDF string literals use parentheses. Escape what could break them.
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@dataclass(frozen=True)
class _TextLine:
    font: str
    size: float
    x: float
    y: float
    text: str


def _wrap(text: str, max_chars: int) -> list[str]:
    return textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)


def _build_content_stream(lines: list[_TextLine], draw_ops: list[str]) -> bytes:
    ops: list[str] = []
    ops.append("q\n")

    # Vector ops (lines, rectangles) live outside BT/ET.
    for op in draw_ops:
        ops.append(op.rstrip("\n") + "\n")

    ops.append("BT\n")
    for line in lines:
        # Set font, position, and draw text.
        ops.append(f"/{line.font} {line.size:.2f} Tf\n")
        ops.append(f"1 0 0 1 {line.x:.2f} {line.y:.2f} Tm\n")
        ops.append(f"({_pdf_escape(line.text)}) Tj\n")
    ops.append("ET\n")
    ops.append("Q\n")

    return "".join(ops).encode("ascii", errors="strict")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "output" / "pdf"
    tmp_dir = repo_root / "tmp" / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_dir / "procedure_suite_one_pager.pdf"

    margin = 54
    x0 = margin
    x1 = LETTER_WIDTH - margin
    y = LETTER_HEIGHT - margin

    title_font = "F2"
    subtitle_font = "F3"
    heading_font = "F2"
    body_font = "F1"

    title_size = 16
    subtitle_size = 10
    heading_size = 11
    body_size = 9

    leading = 11
    section_gap = 6

    lines: list[_TextLine] = []
    draw_ops: list[str] = []

    def add_text(font: str, size: float, text: str, x: float | None = None) -> None:
        nonlocal y
        lines.append(_TextLine(font=font, size=size, x=float(x0 if x is None else x), y=float(y), text=text))
        y -= leading

    def add_heading(text: str) -> None:
        nonlocal y
        y -= section_gap
        lines.append(_TextLine(font=heading_font, size=heading_size, x=float(x0), y=float(y), text=text))
        y -= leading + 1

    def add_paragraph(text: str, max_chars: int) -> None:
        for part in _wrap(text, max_chars=max_chars):
            add_text(body_font, body_size, part)

    def add_bullets(items: list[str], max_chars: int) -> None:
        nonlocal y
        bullet_x = x0
        cont_x = x0 + 12
        for item in items:
            wrapped = _wrap(item, max_chars=max_chars)
            if not wrapped:
                continue
            lines.append(_TextLine(font=body_font, size=body_size, x=float(bullet_x), y=float(y), text=f"- {wrapped[0]}"))
            y -= leading
            for cont in wrapped[1:]:
                lines.append(_TextLine(font=body_font, size=body_size, x=float(cont_x), y=float(y), text=cont))
                y -= leading

    # Title block
    lines.append(_TextLine(font=title_font, size=title_size, x=float(x0), y=float(y), text="Procedure Suite"))
    y -= 20

    subtitle = (
        "Automated CPT coding, registry extraction, and synoptic reporting "
        "for interventional pulmonology."
    )
    for part in _wrap(subtitle, max_chars=92):
        lines.append(_TextLine(font=subtitle_font, size=subtitle_size, x=float(x0), y=float(y), text=part))
        y -= 12

    # Divider
    divider_y = y - 4
    draw_ops.append("0.75 0.75 0.75 RG 1 w")
    draw_ops.append(f"{x0:.2f} {divider_y:.2f} m {x1:.2f} {divider_y:.2f} l S")
    y -= 18

    # What it is
    add_heading("What it is")
    add_paragraph(
        "A FastAPI web UI and API that turns (scrubbed) procedure note text into "
        "validated registry data and derived CPT billing codes, with evidence and QA flags.",
        max_chars=96,
    )
    add_paragraph(
        "Current production mode is extraction-first and stateless: text in -> registry + CPT out via POST /api/v1/process.",
        max_chars=96,
    )

    # Who it's for
    add_heading("Who it's for")
    add_paragraph(
        "Primary users: interventional pulmonology coding/billing specialists and registry coordinators; "
        "also clinicians and QA reviewers validating documentation and extracted fields.",
        max_chars=96,
    )

    # What it does
    add_heading("What it does")
    add_bullets(
        [
            "Web UI (/ui/) for note paste, PHI detection/redaction workflow, and results review.",
            "Unified endpoint: POST /api/v1/process (scrubbed text in -> registry + CPT out).",
            "Extraction-first pipeline: registry extraction (engine via REGISTRY_EXTRACTION_ENGINE; recommended parallel_ner) then deterministic Registry->CPT rules.",
            "Returns UI-ready JSON with evidence spans and review/status flags.",
            "Omission scan + RAW-ML auditor; optional guarded self-correction judge (REGISTRY_SELF_CORRECT_ENABLED=1).",
            "Exports: raw JSON and flattened editable tables (Excel-readable .xls); table edits emit Edited JSON (Training).",
            "CLI + tests for validation and batch runs (make test, make validate-registry, ops/tools/registry_pipeline_smoke*.py).",
        ],
        max_chars=94,
    )

    # How it works
    add_heading("How it works (repo evidence)")
    add_bullets(
        [
            "Client UI: static PHI redactor/dashboard at ui/static/phi_redactor/ served on /ui/.",
            "Data flow: UI redacts PHI in-browser, then submits scrubbed note text (or sets already_scrubbed=true).",
            "API: FastAPI app in app/api/fastapi_app.py exposes POST /api/v1/process (app/api/routes/unified_process.py).",
            "Pipeline: run_unified_pipeline_logic -> RegistryService.extract_fields -> deterministic RegistryRecord -> CPT derivation (CodingService) -> audit/self-correct -> response adapter.",
            "Key knowledge + schemas: data/knowledge/ip_coding_billing_v3_0.json; proc_schemas/registry/; schemas/.",
        ],
        max_chars=94,
    )

    # How to run
    add_heading("How to run (minimal)")
    add_bullets(
        [
            "Install deps: make install (Python 3.11+).",
            "Set required env: PROCSUITE_PIPELINE_MODE=extraction_first (service will not start otherwise).",
            "Configure LLM (optional for some features): GEMINI_API_KEY=... (or use offline flags like GEMINI_OFFLINE=1 / OPENAI_OFFLINE=1).",
            "Start: ./ops/devserver.sh",
            "Open: http://localhost:8000/ui/ and http://localhost:8000/docs",
        ],
        max_chars=94,
    )

    # Footer
    y_footer = margin - 18
    footer_lines = [
        "Sources: README.md; docs/USER_GUIDE.md; docs/ARCHITECTURE.md; app/api/routes/unified_process.py",
    ]
    for i, ft in enumerate(footer_lines):
        lines.append(_TextLine(font=subtitle_font, size=7.5, x=float(x0), y=float(y_footer - (i * 9)), text=ft))

    if y < margin:
        raise SystemExit(
            f"Content overflow: y={y:.2f} < margin={margin}. Tighten copy or adjust layout."
        )

    writer = PdfWriter()
    page = writer.add_blank_page(width=LETTER_WIDTH, height=LETTER_HEIGHT)

    # Fonts: standard 14 PDF fonts (no embedding required).
    font_regular = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_bold = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
        }
    )
    font_oblique = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica-Oblique"),
        }
    )

    font_regular_ref = writer._add_object(font_regular)
    font_bold_ref = writer._add_object(font_bold)
    font_oblique_ref = writer._add_object(font_oblique)

    resources = DictionaryObject()
    resources[NameObject("/Font")] = DictionaryObject(
        {
            NameObject("/F1"): font_regular_ref,
            NameObject("/F2"): font_bold_ref,
            NameObject("/F3"): font_oblique_ref,
        }
    )
    page[NameObject("/Resources")] = resources

    stream = DecodedStreamObject()
    stream.set_data(_build_content_stream(lines=lines, draw_ops=draw_ops))
    page[NameObject("/Contents")] = writer._add_object(stream)

    with out_pdf.open("wb") as f:
        writer.write(f)

    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
