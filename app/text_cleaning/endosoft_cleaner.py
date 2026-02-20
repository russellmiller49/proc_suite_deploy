from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


_PAGE_X_OF_Y_RE = re.compile(r"(?i)^\s*page\s+\d+\s*(?:/|of)\s*\d+\s*$")
_COPYRIGHT_RE = re.compile(r"(?i)\bcopyright\b|\ball\s+rights\s+reserved\b")
_SIGNED_RE = re.compile(r"(?i)\belectronically\s+signed\b|\bsigned\s+off\b")
_PHOTOREPORT_RE = re.compile(r"(?i)\bphotoreport\b")

_PHONE_RE = re.compile(r"\b(?:\(\d{3}\)\s*\d{3}-\d{4}|\d{3}[-.]\d{3}[-.]\d{4})\b")

_CAPTION_NUMBER_ONLY_RE = re.compile(r"^\s*\d+\s*$")
_CAPTION_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s+[A-Za-z][A-Za-z0-9/()_-]{0,20}\b.*$")
_CAPTION_VERB_RE = re.compile(
    r"(?i)\b(?:is|are|was|were|be|been|being|shows?|showed|noted?|seen|performed|placed|inserted|advanced|removed|biops(?:y|ied)|lavage|aspirat(?:e|ed)|examined)\b"
)
_WORD_RE = re.compile(r"[A-Za-z]+")
_ANATOMY_LOCATION_TOKENS: set[str] = {
    "left",
    "right",
    "upper",
    "lower",
    "middle",
    "lobe",
    "lobar",
    "mainstem",
    "entrance",
    "segment",
    "bronchus",
    "airway",
    "carina",
    "trachea",
    "lingula",
    "lul",
    "lll",
    "rul",
    "rml",
    "rll",
}

_ALLOWLIST_TOKENS: tuple[str, ...] = (
    "biopsy",
    "tbna",
    "rose",
    "needle",
    "specimen",
    "jar",
)


def _mask_non_newline_chars(text: str) -> str:
    return re.sub(r"[^\n\r]", " ", text)

def _normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _looks_like_short_anatomy_label(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if re.search(r"[.?!:;,]", clean):
        return False
    if _CAPTION_VERB_RE.search(clean):
        return False
    if re.search(r"\d", clean):
        return False

    tokens = [token.lower() for token in _WORD_RE.findall(clean)]
    if len(tokens) < 2 or len(tokens) > 5:
        return False
    if any(token in _ALLOWLIST_TOKENS for token in tokens):
        return False

    anatomy_count = sum(1 for token in tokens if token in _ANATOMY_LOCATION_TOKENS)
    if anatomy_count < 2:
        return False
    return anatomy_count / max(1, len(tokens)) >= 0.5


def _looks_like_boilerplate_footer(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    if _PAGE_X_OF_Y_RE.match(clean):
        return True
    if _PHOTOREPORT_RE.search(clean):
        return True
    if _SIGNED_RE.search(clean):
        return True
    if _COPYRIGHT_RE.search(clean):
        return True
    # Conservative address/phone heuristic: only strip if the line is clearly contact info.
    if ("phone" in clean.lower() or "fax" in clean.lower() or _PHONE_RE.search(clean)) and len(clean) < 80:
        return True
    return False


def _looks_like_caption_noise(line: str) -> bool:
    clean = (line or "").strip()
    if not clean:
        return False
    lower = clean.lower()
    if any(token in lower for token in _ALLOWLIST_TOKENS):
        return False
    # Do not treat standalone CPT-like codes as caption noise.
    if re.fullmatch(r"\d{5}", clean):
        return False
    if _CAPTION_NUMBER_ONLY_RE.match(clean):
        return True
    if _CAPTION_NUMBER_PREFIX_RE.match(clean) and len(clean) <= 60:
        return True
    if _looks_like_short_anatomy_label(clean) and len(clean) <= 60:
        return True
    return False


def _mask_caption_chunks_in_line(line: str) -> tuple[str, bool]:
    """Mask short caption chunks in mixed lines split by wide spacing.

    Some native extraction paths collapse right-side photo labels into the same
    visual row as narrative text with 3+ spaces between chunks.
    """
    if not line:
        return line, False
    no_nl = line.rstrip("\r\n")
    if not re.search(r"\s{3,}", no_nl):
        return line, False

    line_ending = line[len(no_nl):]
    parts = re.split(r"(\s{3,})", no_nl)
    if len(parts) <= 1:
        return line, False

    changed = False
    out_parts: list[str] = []
    for part in parts:
        if re.fullmatch(r"\s{3,}", part or ""):
            out_parts.append(part)
            continue
        if _looks_like_caption_noise(part):
            out_parts.append(re.sub(r"[^\s]", " ", part))
            changed = True
        else:
            out_parts.append(part)

    return ("".join(out_parts) + line_ending, changed)


@dataclass(slots=True)
class EndoSoftCleanMeta:
    masked_footer_lines: int = 0
    masked_caption_lines: int = 0
    masked_dedup_blocks: int = 0


def clean_endosoft(pages: list[str], page_types: list[str] | None = None) -> list[str]:
    """Clean EndoSoft pages (offset-preserving masking)."""
    types = list(page_types or [])
    if types and len(types) != len(pages):
        raise ValueError("page_types length must match pages")
    out: list[str] = []
    for idx, page in enumerate(pages):
        page_type = types[idx] if idx < len(types) else "unknown"
        out.append(clean_endosoft_page(page, page_type))
    return out


def clean_endosoft_page(text: str, page_type: str) -> str:
    clean, _meta = clean_endosoft_page_with_meta(text, page_type)
    return clean


def clean_endosoft_page_with_meta(text: str, page_type: str) -> tuple[str, EndoSoftCleanMeta]:
    """Clean a single EndoSoft page (offset-preserving).

    Strategy:
    - Mask boilerplate footer/header lines (PHOTOREPORT, page x/y, signatures, copyright)
    - Mask caption-like noise lines (photo captions) with guardrails
    - Mask consecutive duplicate blocks (hybrid duplication artifacts)
    """
    raw = text or ""
    if not raw:
        return raw, EndoSoftCleanMeta()

    meta = EndoSoftCleanMeta()

    # 1) Line-level masking (boilerplate + captions)
    lines = list(raw.splitlines(keepends=True))
    masked = lines[:]

    for i, line in enumerate(lines):
        no_nl = line.rstrip("\r\n")
        if _looks_like_boilerplate_footer(no_nl):
            masked[i] = _mask_non_newline_chars(line)
            meta.masked_footer_lines += 1
            continue

        chunk_masked_line, chunk_changed = _mask_caption_chunks_in_line(line)
        if chunk_changed:
            masked[i] = chunk_masked_line
            meta.masked_caption_lines += 1
            continue

        # On image-heavy pages, be more aggressive (still preserve allowlist tokens).
        if page_type in {"images_page", "image_caption_page"}:
            if _looks_like_caption_noise(no_nl) or len(no_nl.strip()) <= 2:
                masked[i] = _mask_non_newline_chars(line)
                meta.masked_caption_lines += 1
                continue
        else:
            if _looks_like_caption_noise(no_nl):
                masked[i] = _mask_non_newline_chars(line)
                meta.masked_caption_lines += 1
                continue

    # 2) Block-level dedupe (consecutive exact/near-exact after normalization)
    # Split into blocks by blank lines, but mask duplicates rather than deleting.
    out_lines = masked[:]

    def _is_blank_line(ln: str) -> bool:
        return not (ln or "").strip()

    blocks: list[tuple[int, int, str]] = []
    start = 0
    idx = 0
    while idx < len(out_lines):
        if _is_blank_line(out_lines[idx]):
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < len(out_lines) and not _is_blank_line(out_lines[idx]):
            idx += 1
        end = idx  # exclusive
        block_text = "".join(out_lines[start:end])
        key = _normalize_block(block_text)
        if key:
            blocks.append((start, end, key))

    prev_key = None
    prev_text = None
    for start, end, key in blocks:
        block_text = "".join(out_lines[start:end])
        if prev_key is not None:
            if key == prev_key:
                for j in range(start, end):
                    out_lines[j] = _mask_non_newline_chars(out_lines[j])
                meta.masked_dedup_blocks += 1
                continue
            # High-threshold near-duplicate masking (avoid over-masking short blocks).
            if prev_text and len(key) >= 120:
                ratio = difflib.SequenceMatcher(a=prev_text, b=key).ratio()
                if ratio >= 0.96:
                    for j in range(start, end):
                        out_lines[j] = _mask_non_newline_chars(out_lines[j])
                    meta.masked_dedup_blocks += 1
                    continue
        prev_key = key
        prev_text = key

    return "".join(out_lines), meta


__all__ = [
    "EndoSoftCleanMeta",
    "clean_endosoft",
    "clean_endosoft_page",
    "clean_endosoft_page_with_meta",
]
