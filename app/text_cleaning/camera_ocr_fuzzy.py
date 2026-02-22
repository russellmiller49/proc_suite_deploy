"""Narrow fuzzy normalization for camera OCR extraction."""

from __future__ import annotations

import difflib
import functools
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from rapidfuzz import fuzz as _rapid_fuzz
except Exception:  # pragma: no cover - fallback path only
    _rapid_fuzz = None

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_token(value: str) -> str:
    return _NON_ALNUM_RE.sub("", (value or "").lower())


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if _rapid_fuzz is not None:
        return float(_rapid_fuzz.ratio(left, right))
    return float(difflib.SequenceMatcher(a=left, b=right).ratio() * 100.0)


@dataclass(frozen=True)
class _PhraseSpec:
    phrase: str
    threshold_full: float
    threshold_token_avg: float
    threshold_token_min: float

    @property
    def tokens(self) -> list[str]:
        return self.phrase.split()

    @property
    def normalized(self) -> str:
        return _normalize_token(self.phrase)


_DEFAULT_PHRASE_SPECS: tuple[_PhraseSpec, ...] = (
    _PhraseSpec("therapeutic aspiration", threshold_full=88.0, threshold_token_avg=74.0, threshold_token_min=58.0),
    _PhraseSpec("bronchoalveolar lavage", threshold_full=86.0, threshold_token_avg=72.0, threshold_token_min=55.0),
    _PhraseSpec("endobronchial biopsy", threshold_full=88.0, threshold_token_avg=75.0, threshold_token_min=58.0),
    _PhraseSpec("transbronchial needle aspiration", threshold_full=88.0, threshold_token_avg=74.0, threshold_token_min=58.0),
    _PhraseSpec("transbronchial biopsy", threshold_full=88.0, threshold_token_avg=75.0, threshold_token_min=58.0),
    _PhraseSpec("navigational bronchoscopy", threshold_full=88.0, threshold_token_avg=74.0, threshold_token_min=58.0),
    _PhraseSpec("radial ebus", threshold_full=84.0, threshold_token_avg=72.0, threshold_token_min=52.0),
    _PhraseSpec("fiducial marker", threshold_full=88.0, threshold_token_avg=74.0, threshold_token_min=58.0),
    _PhraseSpec("ultrasound elastography", threshold_full=86.0, threshold_token_avg=72.0, threshold_token_min=55.0),
)
_DEFAULT_THRESHOLDS = (88.0, 74.0, 58.0)


def _clamp_threshold(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(max(0.0, min(100.0, parsed)))


def _clean_phrase(value: Any) -> str:
    phrase = " ".join(str(value or "").strip().split())
    return phrase


def _threshold_defaults_for_phrase(phrase: str) -> tuple[float, float, float]:
    phrase_lower = phrase.lower()
    for spec in _DEFAULT_PHRASE_SPECS:
        if spec.phrase.lower() == phrase_lower:
            return (spec.threshold_full, spec.threshold_token_avg, spec.threshold_token_min)
    return _DEFAULT_THRESHOLDS


def _parse_phrase_spec_entry(entry: Any) -> _PhraseSpec | None:
    if isinstance(entry, str):
        phrase = _clean_phrase(entry)
        if not phrase:
            return None
        full_d, avg_d, min_d = _threshold_defaults_for_phrase(phrase)
        return _PhraseSpec(
            phrase=phrase,
            threshold_full=full_d,
            threshold_token_avg=avg_d,
            threshold_token_min=min_d,
        )

    if not isinstance(entry, dict):
        return None

    phrase = _clean_phrase(entry.get("phrase"))
    if not phrase:
        return None

    full_d, avg_d, min_d = _threshold_defaults_for_phrase(phrase)
    return _PhraseSpec(
        phrase=phrase,
        threshold_full=_clamp_threshold(entry.get("threshold_full"), full_d),
        threshold_token_avg=_clamp_threshold(entry.get("threshold_token_avg"), avg_d),
        threshold_token_min=_clamp_threshold(entry.get("threshold_token_min"), min_d),
    )


def _parse_phrase_specs(raw_payload: Any) -> tuple[_PhraseSpec, ...]:
    entries = raw_payload
    if isinstance(raw_payload, dict):
        entries = raw_payload.get("phrases")
    if not isinstance(entries, list):
        return tuple()

    specs: list[_PhraseSpec] = []
    seen: set[str] = set()
    for entry in entries:
        spec = _parse_phrase_spec_entry(entry)
        if spec is None:
            continue
        key = spec.phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
    return tuple(specs)


def _load_phrase_specs_from_env() -> tuple[_PhraseSpec, ...]:
    raw_json = os.getenv("CAMERA_OCR_FUZZY_PHRASES_JSON", "").strip()
    raw_path = os.getenv("CAMERA_OCR_FUZZY_PHRASES_PATH", "").strip()

    if raw_json:
        try:
            parsed = json.loads(raw_json)
            parsed_specs = _parse_phrase_specs(parsed)
            if parsed_specs:
                return parsed_specs
        except Exception:
            pass

    if raw_path:
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            parsed_specs = _parse_phrase_specs(parsed)
            if parsed_specs:
                return parsed_specs
        except Exception:
            pass

    return _DEFAULT_PHRASE_SPECS


@functools.lru_cache(maxsize=1)
def _get_phrase_specs() -> tuple[_PhraseSpec, ...]:
    return _load_phrase_specs_from_env()


def clear_camera_ocr_fuzzy_phrase_cache() -> None:
    _get_phrase_specs.cache_clear()


@dataclass(frozen=True)
class CameraOcrFuzzyReplacement:
    start: int
    end: int
    original: str
    replacement: str
    score: float


@dataclass
class CameraOcrFuzzyResult:
    text: str
    replacements: list[CameraOcrFuzzyReplacement] = field(default_factory=list)

    @property
    def replacement_count(self) -> int:
        return len(self.replacements)


def _preserve_case(replacement: str, original: str) -> str:
    if not original:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


def normalize_camera_ocr_for_extraction(
    text: str,
    *,
    max_replacements: int = 16,
) -> CameraOcrFuzzyResult:
    """Apply narrow fuzzy phrase normalization for camera OCR text."""
    source = str(text or "")
    if not source.strip():
        return CameraOcrFuzzyResult(text=source)

    token_matches = list(_WORD_RE.finditer(source))
    if not token_matches:
        return CameraOcrFuzzyResult(text=source)

    candidates: list[CameraOcrFuzzyReplacement] = []
    for spec in _get_phrase_specs():
        phrase_tokens = spec.tokens
        n = len(phrase_tokens)
        if n <= 0 or len(token_matches) < n:
            continue

        normalized_phrase_tokens = [_normalize_token(token) for token in phrase_tokens]
        normalized_phrase = spec.normalized
        for idx in range(0, len(token_matches) - n + 1):
            window = token_matches[idx : idx + n]
            start = int(window[0].start())
            end = int(window[-1].end())
            candidate_span = source[start:end]

            # Never rewrite bracketed tokens like [REDACTED], [DATE: ...], [SYSTEM: ...].
            if "[" in candidate_span or "]" in candidate_span:
                continue

            candidate_tokens = [match.group(0) for match in window]
            normalized_candidate_tokens = [_normalize_token(tok) for tok in candidate_tokens]
            if any(not tok for tok in normalized_candidate_tokens):
                continue

            normalized_candidate = _normalize_token(" ".join(candidate_tokens))
            if normalized_candidate == normalized_phrase:
                continue

            full_score = _ratio(normalized_candidate, normalized_phrase)
            if full_score < spec.threshold_full:
                continue

            token_scores = [
                _ratio(left, right)
                for left, right in zip(normalized_candidate_tokens, normalized_phrase_tokens, strict=True)
            ]
            if not token_scores:
                continue

            token_avg = sum(token_scores) / float(len(token_scores))
            token_min = min(token_scores)
            if token_avg < spec.threshold_token_avg or token_min < spec.threshold_token_min:
                continue

            replacement = _preserve_case(spec.phrase, candidate_span)
            candidates.append(
                CameraOcrFuzzyReplacement(
                    start=start,
                    end=end,
                    original=candidate_span,
                    replacement=replacement,
                    score=float(round(full_score, 1)),
                )
            )

    if not candidates:
        return CameraOcrFuzzyResult(text=source, replacements=[])

    # Keep highest-confidence non-overlapping replacements.
    candidates.sort(key=lambda item: (item.score, item.end - item.start), reverse=True)
    selected: list[CameraOcrFuzzyReplacement] = []
    for candidate in candidates:
        if len(selected) >= int(max_replacements):
            break
        if any(not (candidate.end <= taken.start or candidate.start >= taken.end) for taken in selected):
            continue
        selected.append(candidate)

    if not selected:
        return CameraOcrFuzzyResult(text=source, replacements=[])

    out = source
    for candidate in sorted(selected, key=lambda item: item.start, reverse=True):
        out = f"{out[:candidate.start]}{candidate.replacement}{out[candidate.end:]}"

    selected_sorted = sorted(selected, key=lambda item: item.start)
    return CameraOcrFuzzyResult(text=out, replacements=selected_sorted)


__all__ = [
    "CameraOcrFuzzyReplacement",
    "CameraOcrFuzzyResult",
    "clear_camera_ocr_fuzzy_phrase_cache",
    "normalize_camera_ocr_for_extraction",
]
