#!/usr/bin/env python3
"""Quick OCR debug runner for one PDF (Provation-style hardening).

This tool is intentionally standalone and uses optional local dependencies:
- PyMuPDF (`fitz`) for rendering PDF pages
- Pillow (`PIL`) for image operations
- pytesseract + local tesseract binary for OCR

Example:
  python ops/tools/ocr_debug_one_pdf.py \
    --pdf provation_examples.pdf \
    --out out.txt \
    --dump-json out.json \
    --pages 1-3 \
    --save-debug-images
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

HEADER_BAND_FRAC = 0.22
HEADER_RETRY_PSMS = [6, 4, 11]
FIGURE_OVERLAP_THRESHOLD = 0.35
SHORT_LOW_CONF_THRESHOLD = 30.0

BOILERPLATE_PATTERNS = [
    re.compile(r"Powered\s+by\s+Provation", re.I),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.I),
    re.compile(r"AMA.*copyright", re.I),
]

CAPTION_PATTERN = re.compile(
    r"^(left|right|upper|lower|middle|lobe|mainstem|entrance|segment|bronchus|airway|carina|trachea|lingula)(\s+\w+){0,6}$",
    re.I,
)


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class OcrLine:
    text: str
    conf: float
    bbox: BBox
    page_index: int


@dataclass
class PageMetrics:
    char_count: int
    alpha_ratio: float
    mean_line_conf: float | None
    low_conf_line_frac: float | None
    num_lines: int
    median_token_len: float
    footer_boilerplate_hits: int



def parse_page_spec(page_spec: str | None, max_pages: int) -> list[int]:
    if not page_spec:
        return list(range(max_pages))
    out: set[int] = set()
    for part in page_spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = max(1, int(start_s))
            end = min(max_pages, int(end_s))
            for value in range(start, end + 1):
                out.add(value - 1)
        else:
            value = int(token)
            if 1 <= value <= max_pages:
                out.add(value - 1)
    return sorted(out)



def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()



def _line_key(text: str) -> str:
    return _normalize_text(text).lower()



def _overlap_ratio(a: BBox, b: BBox) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    area = w * h
    denom = max(1.0, a.w * a.h)
    return area / denom



def _has_valid_dob(text: str) -> bool:
    return re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text or "") is not None



def _age_is_valid(text: str) -> bool:
    source = text or ""
    age_match = re.search(r"\bAge\b[:\s]*([0-9]{1,3})\b", source, re.I)
    if not age_match:
        return re.search(r"\bAge\b", source, re.I) is None
    age = int(age_match.group(1))
    return 0 <= age <= 120



def _compute_metrics(lines: Iterable[OcrLine]) -> PageMetrics:
    line_list = list(lines)
    text = "\n".join(line.text for line in line_list)
    chars = len(text)
    alpha = sum(1 for ch in text if ch.isalpha())
    confs = [line.conf for line in line_list if line.conf >= 0]
    low_conf = [value for value in confs if value < SHORT_LOW_CONF_THRESHOLD]
    tokens = [
        re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token)
        for token in re.split(r"\s+", text)
        if token
    ]
    token_lens = [len(token) for token in tokens if token]

    footer_hits = 0
    for line in line_list:
        if any(pattern.search(line.text) for pattern in BOILERPLATE_PATTERNS):
            footer_hits += 1

    return PageMetrics(
        char_count=chars,
        alpha_ratio=(alpha / chars) if chars else 0.0,
        mean_line_conf=(sum(confs) / len(confs)) if confs else None,
        low_conf_line_frac=(len(low_conf) / len(confs)) if confs else None,
        num_lines=len(line_list),
        median_token_len=float(median(token_lens)) if token_lens else 0.0,
        footer_boilerplate_hits=footer_hits,
    )



def _detect_figure_regions(pil_img: Any) -> list[BBox]:
    import numpy as np

    arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
    if arr.size == 0:
        return []

    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    grad = gx + gy

    h, w = arr.shape
    cell = 16
    gh = math.ceil(h / cell)
    gw = math.ceil(w / cell)
    active = [[False for _ in range(gw)] for _ in range(gh)]

    for cy in range(gh):
        for cx in range(gw):
            y0 = cy * cell
            y1 = min(h, (cy + 1) * cell)
            x0 = cx * cell
            x1 = min(w, (cx + 1) * cell)
            block = grad[y0:y1, x0:x1]
            gray_block = arr[y0:y1, x0:x1]
            if block.size == 0:
                continue
            edge_density = float((block > 45).mean())
            mid_ratio = float(((gray_block >= 35) & (gray_block <= 225)).mean())
            active[cy][cx] = edge_density >= 0.16 and mid_ratio >= 0.30

    visited = [[False for _ in range(gw)] for _ in range(gh)]
    out: list[BBox] = []

    for cy in range(gh):
        for cx in range(gw):
            if visited[cy][cx] or not active[cy][cx]:
                continue
            stack = [(cx, cy)]
            visited[cy][cx] = True
            min_x = min_y = 10**9
            max_x = max_y = -1
            count = 0

            while stack:
                sx, sy = stack.pop()
                count += 1
                min_x = min(min_x, sx)
                min_y = min(min_y, sy)
                max_x = max(max_x, sx)
                max_y = max(max_y, sy)
                for nx, ny in ((sx - 1, sy), (sx + 1, sy), (sx, sy - 1), (sx, sy + 1)):
                    if nx < 0 or ny < 0 or nx >= gw or ny >= gh:
                        continue
                    if visited[ny][nx] or not active[ny][nx]:
                        continue
                    visited[ny][nx] = True
                    stack.append((nx, ny))

            area_ratio = count / float(max(1, gw * gh))
            if area_ratio < 0.05:
                continue

            rx = min_x * cell
            ry = min_y * cell
            rw = (max_x - min_x + 1) * cell
            rh = (max_y - min_y + 1) * cell
            if rw / max(1, w) < 0.2 and rh / max(1, h) < 0.2:
                continue
            out.append(BBox(x=float(rx), y=float(ry), w=float(min(rw, w - rx)), h=float(min(rh, h - ry))))

    return out



def _run_tesseract_lines(pil_img: Any, page_index: int, psm: int, y_offset: int = 0) -> list[OcrLine]:
    import pytesseract

    data = pytesseract.image_to_data(
        pil_img,
        output_type=pytesseract.Output.DICT,
        config=f"--oem 1 --psm {psm}",
    )

    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, raw_text in enumerate(data.get("text", [])):
        text = _normalize_text(raw_text)
        if not text:
            continue
        key = (
            int(data.get("block_num", [0])[i]),
            int(data.get("par_num", [0])[i]),
            int(data.get("line_num", [0])[i]),
        )
        grouped[key].append(i)

    out: list[OcrLine] = []
    for _, idxs in grouped.items():
        parts: list[str] = []
        confs: list[float] = []
        lefts: list[int] = []
        tops: list[int] = []
        rights: list[int] = []
        bottoms: list[int] = []

        for idx in idxs:
            token = _normalize_text(data["text"][idx])
            if not token:
                continue
            conf = float(data["conf"][idx]) if str(data["conf"][idx]).strip() not in {"", "-1"} else -1.0
            if conf >= 0:
                confs.append(conf)
            l = int(data["left"][idx])
            t = int(data["top"][idx]) + y_offset
            w = int(data["width"][idx])
            h = int(data["height"][idx])
            lefts.append(l)
            tops.append(t)
            rights.append(l + w)
            bottoms.append(t + h)
            parts.append(token)

        text = _normalize_text(" ".join(parts))
        if not text:
            continue

        out.append(
            OcrLine(
                text=text,
                conf=(sum(confs) / len(confs)) if confs else -1.0,
                bbox=BBox(
                    x=float(min(lefts) if lefts else 0),
                    y=float(min(tops) if tops else 0),
                    w=float((max(rights) - min(lefts)) if rights else 0),
                    h=float((max(bottoms) - min(tops)) if bottoms else 0),
                ),
                page_index=page_index,
            )
        )

    out.sort(key=lambda line: (line.bbox.y, line.bbox.x))
    deduped: list[OcrLine] = []
    prev_key = ""
    for line in out:
        key = _line_key(line.text)
        if key == prev_key:
            continue
        prev_key = key
        deduped.append(line)
    return deduped



def _filter_lines(lines: list[OcrLine], figure_regions: list[BBox]) -> tuple[list[OcrLine], list[dict[str, Any]]]:
    kept: list[OcrLine] = []
    dropped: list[dict[str, Any]] = []

    for line in lines:
        text = _normalize_text(line.text)
        if not text:
            continue

        if any(p.search(text) for p in BOILERPLATE_PATTERNS):
            dropped.append({"reason": "boilerplate", "text": text, "conf": line.conf})
            continue

        if CAPTION_PATTERN.match(text) and len(text) <= 58:
            dropped.append({"reason": "caption", "text": text, "conf": line.conf})
            continue

        overlap = 0.0
        for region in figure_regions:
            overlap = max(overlap, _overlap_ratio(line.bbox, region))
        if overlap > FIGURE_OVERLAP_THRESHOLD:
            dropped.append({"reason": "figure_overlap", "text": text, "conf": line.conf, "overlap": overlap})
            continue

        if line.conf >= 0 and line.conf < SHORT_LOW_CONF_THRESHOLD and len(text) < 6:
            dropped.append({"reason": "low_conf_short", "text": text, "conf": line.conf})
            continue

        kept.append(line)

    return kept, dropped



def _ensure_optional_deps() -> tuple[Any, Any, Any, Any]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("Missing dependency: PyMuPDF (`fitz`). Install with `pip install pymupdf`.") from exc

    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("Missing dependency: Pillow. Install with `pip install pillow`.") from exc

    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("Missing dependency: pytesseract. Install with `pip install pytesseract`.") from exc

    return fitz, Image, ImageDraw, pytesseract



def _draw_debug_image(pil_img: Any, figure_regions: list[BBox], out_path: Path, ImageDraw: Any) -> None:
    canvas = pil_img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for region in figure_regions:
        draw.rectangle(
            [region.x, region.y, region.x + region.w, region.y + region.h],
            outline=(255, 0, 0),
            width=3,
        )
    canvas.save(out_path)



def run(args: argparse.Namespace) -> int:
    fitz, Image, ImageDraw, _pytesseract = _ensure_optional_deps()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else None
    json_path = Path(args.dump_json).expanduser().resolve() if args.dump_json else None
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else pdf_path.parent / "ocr_debug_images"

    doc = fitz.open(pdf_path)
    page_indices = parse_page_spec(args.pages, doc.page_count)

    report: dict[str, Any] = {
        "pdf": str(pdf_path),
        "pages": [],
    }
    text_pages: list[str] = []

    for page_index in page_indices:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(args.zoom, args.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        figure_regions = _detect_figure_regions(pil_img)

        header_h = max(1, int(pil_img.height * HEADER_BAND_FRAC))
        header_img = pil_img.crop((0, 0, pil_img.width, header_h))
        body_img = pil_img.crop((0, header_h, pil_img.width, pil_img.height))

        header_lines: list[OcrLine] = []
        header_attempts = []
        for psm in HEADER_RETRY_PSMS:
            lines = _run_tesseract_lines(header_img, page_index, psm=psm, y_offset=0)
            text = "\n".join(line.text for line in lines)
            header_attempts.append({"psm": psm, "char_count": len(text), "dob": _has_valid_dob(text), "age": _age_is_valid(text)})
            header_lines = lines
            if _has_valid_dob(text) and _age_is_valid(text):
                break

        body_lines = _run_tesseract_lines(body_img, page_index, psm=args.body_psm, y_offset=header_h)
        raw_lines = header_lines + body_lines
        raw_metrics = _compute_metrics(raw_lines)

        filtered_lines, dropped = _filter_lines(raw_lines, figure_regions)
        filtered_metrics = _compute_metrics(filtered_lines)

        page_text = "\n".join(line.text for line in filtered_lines)
        text_pages.append(f"===== PAGE {page_index + 1} =====\n{page_text}\n")

        page_json = {
            "page_index": page_index,
            "header_band_frac": HEADER_BAND_FRAC,
            "header_attempts": header_attempts,
            "figure_region_count": len(figure_regions),
            "figure_regions": [asdict(region) for region in figure_regions],
            "metrics": {
                "pre_filter": asdict(raw_metrics),
                "post_filter": asdict(filtered_metrics),
            },
            "dropped_lines": dropped,
        }
        report["pages"].append(page_json)

        print(
            " | ".join(
                [
                    f"p{page_index + 1}",
                    f"chars={filtered_metrics.char_count}",
                    f"alpha={filtered_metrics.alpha_ratio:.2f}",
                    f"conf={(filtered_metrics.mean_line_conf if filtered_metrics.mean_line_conf is not None else float('nan')):.1f}",
                    f"lowConf={(filtered_metrics.low_conf_line_frac if filtered_metrics.low_conf_line_frac is not None else float('nan')):.2f}",
                    f"lines={filtered_metrics.num_lines}",
                    f"medTok={filtered_metrics.median_token_len:.1f}",
                    f"footerHits={filtered_metrics.footer_boilerplate_hits}",
                ]
            )
        )

        if args.save_debug_images:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"page_{page_index + 1:03d}_regions.png"
            _draw_debug_image(pil_img, figure_regions, debug_path, ImageDraw)

    doc.close()

    full_text = "\n".join(text_pages)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_text, encoding="utf-8")
    else:
        print(full_text)

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug OCR extraction for a single PDF.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", default="", help="Path to extracted text output")
    parser.add_argument("--dump-json", default="", dest="dump_json", help="Path to JSON debug output")
    parser.add_argument("--pages", default="", help="Page spec like '1-3,5' (1-based)")
    parser.add_argument("--save-debug-images", action="store_true", help="Save debug images with figure regions")
    parser.add_argument("--debug-dir", default="", help="Directory for debug images")
    parser.add_argument("--zoom", type=float, default=2.0, help="Render zoom (default: 2.0)")
    parser.add_argument("--body-psm", type=int, default=6, help="Tesseract PSM for body OCR")
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
