from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from typing import Any


_PAGE_X_OF_Y_RE = re.compile(r"(?i)^\s*page\s+\d+\s+of\s+\d+\s*$")
_POWERED_BY_RE = re.compile(r"(?i)\bpowered\s+by\s+provation\b")
_AMA_COPYRIGHT_RE = re.compile(r"(?i)\bAMA\b.*\bcopyright\b|\bcopyright\b.*\bAMA\b")
_SAMPLE_DISCLAIMER_RE = re.compile(r"(?i)\bprocedure\s+note\s+sample\b|\bsample\s+note\b")

_PHONE_RE = re.compile(r"\b(?:\(\d{3}\)\s*\d{3}-\d{4}|\d{3}[-.]\d{3}[-.]\d{4})\b")

_ADDL_IMAGES_RE = re.compile(r"(?i)\badd(?:'|i)?l\s+images\b|\badditional\s+images\b")

_CAPTION_NUMBER_ONLY_RE = re.compile(r"^\s*\d+\s*$")
_CAPTION_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s+.{1,60}$")
_SHORT_ANATOMY_LABEL_RE = re.compile(
    r"(?i)^(left|right|upper|lower|middle|mainstem|entrance|segment|bronchus|airway|carina|trachea|lingula)(\s+\w+){0,8}$"
)
_HEADER_DOB_BIRCH_RE = re.compile(r"(?i)\b(?:data|date)\s+(?:of|nf)\s+birch\b")
_HEADER_ACCOUNT_LABEL_RE = re.compile(r"(?i)\b(?:account|acct)\b")

_GUARD_VERBS: tuple[str, ...] = (
    "performed",
    "removed",
    "instilled",
    "returned",
    "needle",
    "tbna",
    "biopsy",
    "rose",
    "lavage",
    "dilated",
    "placed",
    "exchanged",
    "inserted",
    "advanced",
)

_MEASUREMENT_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:ml|mL|mm|cm|fr|french|gauge)\b")


def _mask_non_newline_chars(text: str) -> str:
    return re.sub(r"[^\n\r]", " ", text)


def _normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _normalize_letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").lower())


def _token_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(a=left, b=right).ratio()


def _looks_like_garbled_account_token(token: str) -> bool:
    normalized = _normalize_letters(token)
    if len(normalized) < 6:
        return False
    if normalized.startswith("aeecrnimt"):
        return True
    return _token_similarity(normalized, "aeecrnimt") >= 0.72


def _looks_like_account_token(token: str) -> bool:
    normalized = _normalize_letters(token)
    if len(normalized) < 5:
        return False
    return _token_similarity(normalized, "account") >= 0.66


def _is_corrupt_dob_value(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return False
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", clean):
        return False
    return bool(re.search(r"[0-9][A-Za-z]|[A-Za-z][0-9]", clean))


def _looks_like_header_noise(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    tokens = re.findall(r"[A-Za-z0-9#]+", clean)
    first = tokens[0] if tokens else ""

    if _looks_like_garbled_account_token(first):
        return True
    if _HEADER_ACCOUNT_LABEL_RE.search(clean):
        return True
    if _looks_like_account_token(first) and re.search(r"(?i)#|number|num|acct", clean):
        return True
    if _HEADER_DOB_BIRCH_RE.search(clean) and _is_corrupt_dob_value(clean):
        return True
    return False


def _is_guarded_clinical_line(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    lower = clean.lower()
    if any(token in lower for token in _GUARD_VERBS):
        return True
    if _MEASUREMENT_RE.search(clean):
        return True
    return False


def _looks_like_boilerplate(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if _PAGE_X_OF_Y_RE.match(clean):
        return True
    if _POWERED_BY_RE.search(clean):
        return True
    if _AMA_COPYRIGHT_RE.search(clean):
        return True
    if _SAMPLE_DISCLAIMER_RE.search(clean):
        return True
    if ("phone" in clean.lower() or "fax" in clean.lower() or _PHONE_RE.search(clean)) and len(clean) < 90:
        return True
    return False


def _looks_like_caption_noise(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if _is_guarded_clinical_line(clean):
        return False
    if _CAPTION_NUMBER_ONLY_RE.match(clean):
        return True
    if _CAPTION_NUMBER_PREFIX_RE.match(clean) and not re.search(r"[.?!:;]", clean):
        return True
    if _SHORT_ANATOMY_LABEL_RE.match(clean) and len(clean) <= 70 and not re.search(r"[.?!:;]", clean):
        return True
    return False


@dataclass(slots=True)
class PageTextWithMeta:
    page_num: int
    raw_text: str
    clean_text: str
    page_type: str
    removed_lines: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_provation(pages: list[str], page_types: list[str] | None = None) -> list[PageTextWithMeta]:
    types = list(page_types or [])
    if types and len(types) != len(pages):
        raise ValueError("page_types length must match pages")
    out: list[PageTextWithMeta] = []
    for idx, raw in enumerate(pages):
        page_type = types[idx] if idx < len(types) else "unknown"
        out.append(clean_provation_page(raw, page_type, page_num=idx + 1))
    return out


def clean_provation_page(text: str, page_type: str, *, page_num: int = 1) -> PageTextWithMeta:
    raw = text or ""
    if not raw:
        return PageTextWithMeta(page_num=page_num, raw_text=raw, clean_text=raw, page_type=page_type)

    lines = list(raw.splitlines(keepends=True))
    out_lines = lines[:]
    removed: list[dict[str, Any]] = []

    masked_boilerplate = 0
    masked_header_noise = 0
    masked_captions = 0
    masked_line_dupes = 0
    masked_block_dupes = 0

    # 1) Boilerplate + caption stripping (offset-preserving)
    for i, line in enumerate(lines):
        no_nl = line.rstrip("\r\n")
        if _looks_like_header_noise(no_nl):
            out_lines[i] = _mask_non_newline_chars(line)
            masked_header_noise += 1
            removed.append({"reason": "header_noise", "line": (no_nl.strip()[:200] if no_nl.strip() else "")})
            continue

        if _looks_like_boilerplate(no_nl):
            out_lines[i] = _mask_non_newline_chars(line)
            masked_boilerplate += 1
            removed.append({"reason": "boilerplate", "line": (no_nl.strip()[:200] if no_nl.strip() else "")})
            continue

        if page_type == "images_page":
            # Image pages are mostly captions; keep only clearly clinical lines.
            if _looks_like_caption_noise(no_nl) or (_ADDL_IMAGES_RE.search(no_nl) and len(no_nl.strip()) <= 40):
                out_lines[i] = _mask_non_newline_chars(line)
                masked_captions += 1
                removed.append({"reason": "caption", "line": (no_nl.strip()[:200] if no_nl.strip() else "")})
                continue
        else:
            if _looks_like_caption_noise(no_nl):
                out_lines[i] = _mask_non_newline_chars(line)
                masked_captions += 1
                removed.append({"reason": "caption", "line": (no_nl.strip()[:200] if no_nl.strip() else "")})
                continue

    # 2) Consecutive line de-dupe (hybrid/OCR artifacts)
    prev_key = None
    for i, line in enumerate(out_lines):
        key = _normalize_line(line)
        if not key:
            continue
        if prev_key is not None and key == prev_key:
            out_lines[i] = _mask_non_newline_chars(line)
            masked_line_dupes += 1
            continue
        prev_key = key

    # 3) Paragraph/block de-dupe with very high similarity threshold
    def _is_blank(ln: str) -> bool:
        return not (ln or "").strip()

    blocks: list[tuple[int, int, str]] = []
    idx = 0
    while idx < len(out_lines):
        if _is_blank(out_lines[idx]):
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < len(out_lines) and not _is_blank(out_lines[idx]):
            idx += 1
        end = idx
        block_text = "".join(out_lines[start:end])
        key = _normalize_block(block_text)
        if key:
            blocks.append((start, end, key))

    prev_block_key: str | None = None
    for start, end, key in blocks:
        if prev_block_key is None:
            prev_block_key = key
            continue
        if key == prev_block_key:
            for j in range(start, end):
                out_lines[j] = _mask_non_newline_chars(out_lines[j])
            masked_block_dupes += 1
            continue
        if len(key) < 160:
            prev_block_key = key
            continue
        ratio = difflib.SequenceMatcher(a=prev_block_key, b=key).ratio()
        if ratio >= 0.98:
            for j in range(start, end):
                out_lines[j] = _mask_non_newline_chars(out_lines[j])
            masked_block_dupes += 1
        else:
            prev_block_key = key

    clean_text = "".join(out_lines)

    return PageTextWithMeta(
        page_num=page_num,
        raw_text=raw,
        clean_text=clean_text,
        page_type=page_type,
        removed_lines=removed,
        metrics={
            "masked_boilerplate_lines": masked_boilerplate,
            "masked_header_noise_lines": masked_header_noise,
            "masked_caption_lines": masked_captions,
            "masked_consecutive_line_dupes": masked_line_dupes,
            "masked_block_dupes": masked_block_dupes,
        },
    )


__all__ = [
    "PageTextWithMeta",
    "clean_provation",
    "clean_provation_page",
]
