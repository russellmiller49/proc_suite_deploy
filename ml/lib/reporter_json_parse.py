"""Strict JSON parsing utilities for reporter prompt model outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from proc_schemas.clinical.common import ProcedureBundle

from ml.lib.reporter_bundle_codec import decode_bundle_keys_v1


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

# T5 sentencepiece vocab in `google/flan-t5-*` cannot represent `{` / `}` (and some
# other punctuation) and will emit `<unk>` unless we pre-encode braces to tokens
# that exist in-vocab. Our training/eval scripts use `<extra_id_0>` / `<extra_id_1>`
# as reversible placeholders; normalize them back before `json.loads()`.
_T5_JSON_OBJ_OPEN = "<extra_id_0>"
_T5_JSON_OBJ_CLOSE = "<extra_id_1>"

# When decoding with `skip_special_tokens=False`, tokenizers may emit these
# literal strings. Strip them before attempting JSON parsing.
_DECODED_SPECIAL_TOKEN_STRINGS = ("<pad>", "</s>", "<s>")


class ReporterJSONParseError(ValueError):
    """Raised when JSON parsing fails after bounded repair attempts."""


def normalize_decoded_model_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    for token in _DECODED_SPECIAL_TOKEN_STRINGS:
        cleaned = cleaned.replace(token, "")

    cleaned = cleaned.replace(_T5_JSON_OBJ_OPEN, "{").replace(_T5_JSON_OBJ_CLOSE, "}")
    return cleaned.strip()


def strip_markdown_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def extract_balanced_json_object(text: str) -> str | None:
    """Extract first balanced JSON object from free-form text.

    Returns None when no balanced top-level object is found.
    """
    if not text:
        return None

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def remove_trailing_commas(text: str) -> str:
    prev = text
    while True:
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", prev)
        if cleaned == prev:
            return cleaned
        prev = cleaned


def parse_json_object_strict(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse JSON object with bounded, explicit cleanup steps.

    Steps:
    1. Strip markdown fences.
    2. Normalize decoded-model artifacts (special tokens / brace placeholders).
    3. Direct json.loads.
    3. Balanced object extraction.
    4. Trailing-comma cleanup.
    """
    notes: list[str] = []
    cleaned = normalize_decoded_model_text(strip_markdown_fences(raw_text))

    def _parse(text: str) -> dict[str, Any]:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ReporterJSONParseError("JSON payload is not an object")
        return payload

    try:
        return _parse(cleaned), notes
    except Exception as exc_direct:
        notes.append(f"direct_parse_failed:{exc_direct.__class__.__name__}")

    balanced = extract_balanced_json_object(cleaned)
    if balanced:
        try:
            notes.append("used_balanced_object_extraction")
            return _parse(balanced), notes
        except Exception as exc_balanced:
            notes.append(f"balanced_parse_failed:{exc_balanced.__class__.__name__}")

    candidate = balanced or cleaned
    with_commas_fixed = remove_trailing_commas(candidate)
    if with_commas_fixed != candidate:
        notes.append("removed_trailing_commas")
    try:
        return _parse(with_commas_fixed), notes
    except Exception as exc_final:
        notes.append(f"final_parse_failed:{exc_final.__class__.__name__}")
        raise ReporterJSONParseError(
            "Failed to parse JSON object after bounded repairs"
        ) from exc_final


def parse_and_validate_bundle(
    raw_text: str,
    *,
    decode_codec: bool = True,
) -> tuple[ProcedureBundle, dict[str, Any], list[str]]:
    """Parse model output into a validated ProcedureBundle."""
    payload, parse_notes = parse_json_object_strict(raw_text)
    if decode_codec:
        payload = decode_bundle_keys_v1(payload)
    bundle = ProcedureBundle.model_validate(payload)
    return bundle, payload, parse_notes


__all__ = [
    "ReporterJSONParseError",
    "extract_balanced_json_object",
    "normalize_decoded_model_text",
    "parse_and_validate_bundle",
    "parse_json_object_strict",
    "remove_trailing_commas",
    "strip_markdown_fences",
]
