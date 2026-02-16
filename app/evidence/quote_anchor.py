from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from rapidfuzz import fuzz

from app.common.spans import Span


class AnchorMethod(str, Enum):
    exact = "exact"
    case_insensitive = "case_insensitive"
    whitespace = "whitespace"
    whitespace_case_insensitive = "whitespace_case_insensitive"
    fuzzy = "fuzzy"
    not_found = "not_found"


@dataclass(frozen=True)
class QuoteAnchorResult:
    span: Span | None
    method: AnchorMethod
    score: float | None = None


_WS_RE = re.compile(r"\s+")


def _normalize_for_alignment(text: str) -> str:
    """Return a length-preserving normalization for fuzzy alignment."""

    if not text:
        return ""
    out_chars: list[str] = []
    for ch in text:
        if ch.isalnum():
            out_chars.append(ch.lower())
        else:
            out_chars.append(" ")
    return "".join(out_chars)


def _whitespace_flexible_pattern(quote: str) -> str | None:
    tokens = [t for t in _WS_RE.split((quote or "").strip()) if t]
    if not tokens:
        return None
    return r"\s+".join(re.escape(token) for token in tokens)


def anchor_quote(
    document: str,
    quote: str,
    *,
    prefix: str | None = None,
    suffix: str | None = None,
    fuzzy_threshold: float = 90.0,
    context_window_chars: int = 2500,
) -> QuoteAnchorResult:
    """Anchor *quote* into *document* and return a Span with offsets.

    The returned span's .text is always sourced from *document*.

    Args:
        document: Full (scrubbed) note text.
        quote: Proposed evidence quote (may have minor formatting drift).
        prefix/suffix: Optional small context strings around the quote.
        fuzzy_threshold: Minimum RapidFuzz alignment score for fuzzy matches.
        context_window_chars: When prefix/suffix are provided, use a bounded search
            window around them before falling back to the full document.
    """

    doc = document or ""
    q = (quote or "").strip()
    if not doc or not q:
        return QuoteAnchorResult(span=None, method=AnchorMethod.not_found)

    def _search_region(region: str, region_offset: int) -> QuoteAnchorResult:
        idx = region.find(q)
        if idx >= 0:
            start = region_offset + idx
            end = start + len(q)
            return QuoteAnchorResult(
                span=Span(text=doc[start:end], start=start, end=end, confidence=1.0),
                method=AnchorMethod.exact,
                score=100.0,
            )

        lowered_region = region.lower()
        lowered_q = q.lower()
        idx = lowered_region.find(lowered_q)
        if idx >= 0:
            start = region_offset + idx
            end = start + len(q)
            return QuoteAnchorResult(
                span=Span(text=doc[start:end], start=start, end=end, confidence=0.98),
                method=AnchorMethod.case_insensitive,
                score=98.0,
            )

        pattern = _whitespace_flexible_pattern(q)
        if pattern:
            match = re.search(pattern, region, flags=re.DOTALL)
            if match:
                start = region_offset + match.start()
                end = region_offset + match.end()
                return QuoteAnchorResult(
                    span=Span(text=doc[start:end], start=start, end=end, confidence=0.95),
                    method=AnchorMethod.whitespace,
                    score=95.0,
                )

            match = re.search(pattern, region, flags=re.DOTALL | re.IGNORECASE)
            if match:
                start = region_offset + match.start()
                end = region_offset + match.end()
                return QuoteAnchorResult(
                    span=Span(text=doc[start:end], start=start, end=end, confidence=0.93),
                    method=AnchorMethod.whitespace_case_insensitive,
                    score=93.0,
                )

        normalized_region = _normalize_for_alignment(region)
        normalized_quote = _normalize_for_alignment(q)
        if normalized_region and normalized_quote and len(normalized_quote.strip()) >= 12:
            alignment = fuzz.partial_ratio_alignment(normalized_quote, normalized_region)
            if alignment and float(alignment.score) >= float(fuzzy_threshold):
                start = region_offset + int(alignment.dest_start)
                end = region_offset + int(alignment.dest_end)
                return QuoteAnchorResult(
                    span=Span(
                        text=doc[start:end],
                        start=start,
                        end=end,
                        confidence=float(alignment.score) / 100.0,
                    ),
                    method=AnchorMethod.fuzzy,
                    score=float(alignment.score),
                )

        return QuoteAnchorResult(span=None, method=AnchorMethod.not_found)

    # 1) If prefix/suffix are present, try a bounded region first.
    doc_lower = doc.lower()
    start_hint = 0
    end_hint = len(doc)

    if prefix:
        prefix_clean = str(prefix).strip()
        if prefix_clean:
            idx = doc_lower.find(prefix_clean.lower())
            if idx >= 0:
                start_hint = max(0, idx - int(context_window_chars))

    if suffix:
        suffix_clean = str(suffix).strip()
        if suffix_clean:
            idx = doc_lower.find(suffix_clean.lower(), start_hint)
            if idx >= 0:
                end_hint = min(len(doc), idx + len(suffix_clean) + int(context_window_chars))

    if start_hint != 0 or end_hint != len(doc):
        region = doc[start_hint:end_hint]
        result = _search_region(region, start_hint)
        if result.span is not None:
            return result

    # 2) Fallback to full-document search.
    return _search_region(doc, 0)


__all__ = ["AnchorMethod", "QuoteAnchorResult", "anchor_quote"]

