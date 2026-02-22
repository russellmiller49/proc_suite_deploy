"""LLM-assisted cleanup for scrubbed camera OCR notes."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from app.common.llm import OpenAILLM

_REDACTED_TOKEN = "[REDACTED]"
_DATE_TOKEN_RE = re.compile(r"\[DATE:[^\]\n]{0,120}\]")
_SYSTEM_TOKEN_RE = re.compile(r"\[SYSTEM:[^\]\n]{0,220}\]")
_BRACKET_TOKEN_RE = re.compile(r"\[[^\]\n]{1,220}\]")

_OCR_CLEANER_PROMPT = """
You repair OCR artifacts in a de-identified clinical procedure note captured by camera OCR.

Return ONLY cleaned plain text.

Hard rules:
1. Correct obvious OCR errors (character swaps, split/merged words, punctuation, spacing).
2. Do NOT add, infer, or remove clinical facts, measurements, diagnoses, procedures, medications, or codes.
3. Preserve section order and line breaks as much as possible.
4. Preserve bracketed tokens exactly as-is (examples: [REDACTED], [DATE: T+5 DAYS], [SYSTEM: ...], [unreadable]).
5. If uncertain, keep the original token/text unchanged.
6. Never expand missing values; keep unclear text unchanged.
"""


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y"}


def _resolve_max_chars() -> int:
    raw = os.getenv("CAMERA_OCR_CLEANER_MAX_CHARS", "").strip()
    if not raw:
        return 60_000
    try:
        parsed = int(raw)
    except ValueError:
        return 60_000
    return max(5_000, parsed)


class CameraOcrCleanerUnavailable(RuntimeError):
    """Raised when OCR cleaner cannot run (provider/key/offline constraints)."""


@dataclass
class CameraOcrSanitizeResult:
    cleaned_text: str
    changed: bool
    correction_applied: bool
    model: str | None = None
    warnings: list[str] = field(default_factory=list)


def _strip_markdown_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`").strip()
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:].strip()
        elif cleaned.lower().startswith("markdown"):
            cleaned = cleaned[8:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned.strip()


def _guard_failed(before: str, after: str) -> bool:
    if after.count(_REDACTED_TOKEN) < before.count(_REDACTED_TOKEN):
        return True
    if len(_DATE_TOKEN_RE.findall(after)) < len(_DATE_TOKEN_RE.findall(before)):
        return True
    if len(_SYSTEM_TOKEN_RE.findall(after)) < len(_SYSTEM_TOKEN_RE.findall(before)):
        return True

    before_tokens = _BRACKET_TOKEN_RE.findall(before)
    after_tokens = set(_BRACKET_TOKEN_RE.findall(after))
    for token in before_tokens:
        if token.startswith("[REDACTED") or token.startswith("[DATE:") or token.startswith("[SYSTEM:"):
            if token not in after_tokens:
                return True

    if before and len(after) < int(len(before) * 0.55):
        return True
    return False


def _resolve_openai_llm() -> tuple[OpenAILLM, str]:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider != "openai_compat":
        raise CameraOcrCleanerUnavailable(
            f"LLM_PROVIDER must be openai_compat for camera OCR correction (got {provider!r})"
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if _truthy_env("OPENAI_OFFLINE") or not api_key:
        raise CameraOcrCleanerUnavailable(
            "Camera OCR correction unavailable (OPENAI_OFFLINE=1 or OPENAI_API_KEY not set)"
        )

    model = (
        os.getenv("OPENAI_MODEL_OCR_CLEANER")
        or os.getenv("OPENAI_MODEL_STRUCTURER")
        or "gpt-5-mini"
    ).strip()
    if not model:
        model = "gpt-5-mini"

    return OpenAILLM(api_key=api_key, model=model, task="structurer"), model


def sanitize_camera_ocr_text(raw_text: str) -> CameraOcrSanitizeResult:
    """Run a safe OCR correction pass for scrubbed camera OCR text."""
    text = str(raw_text or "")
    if not text.strip():
        return CameraOcrSanitizeResult(
            cleaned_text=text,
            changed=False,
            correction_applied=False,
            warnings=["OCR_SANITIZE_SKIPPED: empty_text"],
        )

    max_chars = _resolve_max_chars()
    if len(text) > max_chars:
        return CameraOcrSanitizeResult(
            cleaned_text=text,
            changed=False,
            correction_applied=False,
            warnings=[f"OCR_SANITIZE_SKIPPED: input_too_long>{max_chars}"],
        )

    llm, model = _resolve_openai_llm()
    prompt = (
        f"{_OCR_CLEANER_PROMPT.strip()}\n\n"
        "Input note:\n"
        "[BEGIN_NOTE]\n"
        f"{text}\n"
        "[END_NOTE]\n"
    )

    raw = llm.generate(prompt, task="structurer", temperature=0.1)
    cleaned = _strip_markdown_code_fences(raw)
    if not cleaned:
        return CameraOcrSanitizeResult(
            cleaned_text=text,
            changed=False,
            correction_applied=False,
            model=model,
            warnings=["OCR_SANITIZE_SKIPPED: empty_model_output"],
        )

    if _guard_failed(text, cleaned):
        return CameraOcrSanitizeResult(
            cleaned_text=text,
            changed=False,
            correction_applied=False,
            model=model,
            warnings=["OCR_SANITIZE_SKIPPED: redaction_guard_triggered"],
        )

    return CameraOcrSanitizeResult(
        cleaned_text=cleaned,
        changed=cleaned != text,
        correction_applied=True,
        model=model,
    )


__all__ = [
    "CameraOcrCleanerUnavailable",
    "CameraOcrSanitizeResult",
    "sanitize_camera_ocr_text",
]
