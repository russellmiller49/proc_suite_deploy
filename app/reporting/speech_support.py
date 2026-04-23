"""Reporter speech transcription and scrubbed-text cleanup helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Final

import httpx

from app.common.llm import OpenAILLM, _normalize_openai_base_url, _resolve_openai_timeout

_logger = logging.getLogger(__name__)

_REDACTED_TOKEN: Final[str] = "[REDACTED]"
_DATE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\[DATE:[^\]\n]{0,120}\]")
_SYSTEM_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\[SYSTEM:[^\]\n]{0,220}\]")
_BRACKET_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\[[^\]\n]{1,220}\]")

_REPORTER_SPEECH_CLEANER_PROMPT: Final[str] = """
You repair transcription errors in a de-identified interventional pulmonology procedure note.

Return ONLY cleaned plain text.

Hard rules:
1. Correct only obvious transcription mistakes, punctuation, spacing, and broken medical terms.
2. Do NOT add, infer, remove, or reorder clinical facts.
3. Preserve negation, laterality, station identifiers, measurements, medication doses,
   counts, and procedure names exactly unless there is an obvious transcription artifact.
4. Preserve bracketed tokens exactly as-is
   (examples: [REDACTED], [DATE: T+5 DAYS], [SYSTEM: ...], [unreadable]).
5. If uncertain, keep the original token or phrase unchanged.
6. Never fill in missing details or make ambiguous wording more specific than the input.
7. Preserve section order and line breaks as much as possible.
"""

_REPORTER_SPEECH_TRANSCRIBE_PROMPT: Final[str] = """
This is a short de-identified interventional pulmonology procedure dictation.

Prefer these domain terms when the audio supports them:
Ion, Monarch, robotic bronchoscopy, bronchoscopy, EBUS, EBUS-guided staging, EBUS-TBNA,
TBNA, TBBx, EBBx, ROSE, cone beam CT, fluoroscopy, cryobiopsy, cryobiopsies, forceps biopsy,
mediastinal lymphadenopathy, granulomas, sarcoidosis, atypical cells,
LUL, LLL, RUL, RML, RLL, station 4R, station 4L, station 7, station 11R, station 11L,
ground glass opacity, lesion, mass, catheter, guide sheath, radial probe,
22-gauge, 25-gauge, 1.1 mm, 1.8 cm, no complications.

Return the most likely transcript without adding facts that are not supported by the audio.
"""

_POTENTIAL_PHI_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"\b(?:patient|pt|mrn|medical\s+record|dob|date\s+of\s+birth|phone|address)\b\s*[:#-]",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])\b"),
    re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b"),
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s+(?:19|20)\d{2}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"),
)

_ALLOWED_AUDIO_EXTENSIONS: Final[tuple[str, ...]] = (
    ".wav",
    ".webm",
    ".ogg",
    ".mp3",
    ".m4a",
)
_RETRYABLE_TRANSCRIBE_STATUS_CODES: Final[frozenset[int]] = frozenset({400, 404, 415, 422})


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


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
        if (
            token.startswith("[REDACTED")
            or token.startswith("[DATE:")
            or token.startswith("[SYSTEM:")
        ):
            if token not in after_tokens:
                return True

    if before and len(after) < int(len(before) * 0.55):
        return True
    return False


def contains_potential_phi(text: str) -> bool:
    source = str(text or "")
    return any(pattern.search(source) for pattern in _POTENTIAL_PHI_PATTERNS)


def _resolve_reporter_speech_enabled() -> bool:
    return _truthy_env("REPORTER_SPEECH_ENABLED", default=True)


def _resolve_cloud_fallback_enabled() -> bool:
    return _truthy_env("REPORTER_SPEECH_ALLOW_CLOUD_FALLBACK", default=False)


def _resolve_audio_max_bytes() -> int:
    raw = os.getenv("REPORTER_SPEECH_MAX_AUDIO_BYTES", "").strip()
    if not raw:
        return 10 * 1024 * 1024
    try:
        parsed = int(raw)
    except ValueError:
        return 10 * 1024 * 1024
    return max(256_000, parsed)


def _resolve_cleanup_max_chars() -> int:
    raw = os.getenv("REPORTER_SPEECH_CLEANER_MAX_CHARS", "").strip()
    if not raw:
        return 60_000
    try:
        parsed = int(raw)
    except ValueError:
        return 60_000
    return max(5_000, parsed)


def _resolve_transcribe_model() -> str:
    return (os.getenv("REPORTER_SPEECH_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()


def _resolve_cleanup_model() -> str:
    return (os.getenv("REPORTER_SPEECH_CLEANUP_MODEL") or "gpt-5.4-mini").strip()


def _resolve_transcribe_prompt() -> str:
    override = (os.getenv("REPORTER_SPEECH_TRANSCRIBE_PROMPT") or "").strip()
    return override or _REPORTER_SPEECH_TRANSCRIBE_PROMPT.strip()


def _resolve_openai_api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def _resolve_provider_is_openai() -> bool:
    return (os.getenv("LLM_PROVIDER") or "gemini").strip().lower() == "openai_compat"


def _safe_error_preview(text: str, *, limit: int = 500) -> str:
    preview = " ".join(str(text or "").split())
    if len(preview) <= limit:
        return preview
    return f"{preview[: limit - 1]}…"


def _validate_audio_input(
    filename: str | None,
    content_type: str | None,
    audio_bytes: bytes,
) -> None:
    if not audio_bytes:
        raise ReporterSpeechUnavailable("Audio upload was empty")

    if len(audio_bytes) > _resolve_audio_max_bytes():
        raise ReporterSpeechUnavailable("Audio upload exceeded the configured size limit")

    name = (filename or "").strip().lower()
    content = (content_type or "").strip().lower()
    valid_ext = any(name.endswith(ext) for ext in _ALLOWED_AUDIO_EXTENSIONS)
    valid_type = content.startswith("audio/") or content in {
        "application/octet-stream",
        "application/ogg",
    }
    if not valid_ext and not valid_type:
        raise ReporterSpeechUnavailable("Unsupported audio upload type")


@dataclass
class ReporterSpeechTranscriptionResult:
    transcript: str
    provider: str
    model: str | None = None
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReporterSpeechCleanupResult:
    cleaned_text: str
    changed: bool
    correction_applied: bool
    model: str | None = None
    warnings: list[str] = field(default_factory=list)


class ReporterSpeechUnavailable(RuntimeError):
    """Raised when reporter speech support is unavailable."""


class ReporterSpeechUnsafeInput(RuntimeError):
    """Raised when speech cleanup receives text that appears unsanitized."""


@dataclass(frozen=True)
class _ReporterSpeechTranscribeAttempt:
    model: str
    response_format: str
    include_prompt: bool
    include_language: bool


def _resolve_transcribe_models() -> list[str]:
    primary = _resolve_transcribe_model()
    raw_fallbacks = (os.getenv("REPORTER_SPEECH_TRANSCRIBE_FALLBACK_MODELS") or "").strip()
    fallback_models = [part.strip() for part in raw_fallbacks.split(",") if part.strip()]
    if not fallback_models and primary != "whisper-1":
        fallback_models = ["whisper-1"]

    ordered: list[str] = []
    for candidate in [primary, *fallback_models]:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _build_transcribe_attempts_for_model(model: str) -> list[_ReporterSpeechTranscribeAttempt]:
    normalized = str(model or "").strip().lower()
    if normalized in {"gpt-4o-mini-transcribe", "gpt-4o-transcribe"}:
        return [
            _ReporterSpeechTranscribeAttempt(
                model=model,
                response_format="text",
                include_prompt=True,
                include_language=True,
            ),
            _ReporterSpeechTranscribeAttempt(
                model=model,
                response_format="json",
                include_prompt=True,
                include_language=True,
            ),
            _ReporterSpeechTranscribeAttempt(
                model=model,
                response_format="text",
                include_prompt=False,
                include_language=True,
            ),
            _ReporterSpeechTranscribeAttempt(
                model=model,
                response_format="text",
                include_prompt=False,
                include_language=False,
            ),
        ]

    return [
        _ReporterSpeechTranscribeAttempt(
            model=model,
            response_format="text",
            include_prompt=False,
            include_language=True,
        ),
        _ReporterSpeechTranscribeAttempt(
            model=model,
            response_format="json",
            include_prompt=False,
            include_language=True,
        ),
        _ReporterSpeechTranscribeAttempt(
            model=model,
            response_format="text",
            include_prompt=False,
            include_language=False,
        ),
    ]


def _build_transcribe_form_data(attempt: _ReporterSpeechTranscribeAttempt) -> dict[str, str]:
    data: dict[str, str] = {
        "model": attempt.model,
        "response_format": attempt.response_format,
    }
    if attempt.include_language:
        data["language"] = "en"
    if attempt.include_prompt:
        data["prompt"] = _resolve_transcribe_prompt()
    return data


def _extract_provider_error_message(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    try:
        payload = json.loads(source)
    except Exception:
        return _safe_error_preview(source, limit=240)

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return _safe_error_preview(message, limit=240)
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return _safe_error_preview(message, limit=240)
    return _safe_error_preview(source, limit=240)


def _is_model_rejection_error(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False
    return (
        "model" in normalized
        and (
            "does not exist" in normalized
            or "not found" in normalized
            or "not available" in normalized
            or "do not have access" in normalized
            or "don't have access" in normalized
            or "unsupported model" in normalized
            or "invalid model" in normalized
        )
    )


def _should_retry_transcribe_status(status_code: int) -> bool:
    return int(status_code) in _RETRYABLE_TRANSCRIBE_STATUS_CODES


def _extract_transcript_text(response: httpx.Response) -> str:
    raw_text = str(response.text or "").strip()
    if raw_text:
        try:
            payload = json.loads(raw_text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            transcript = str(payload.get("text") or "").strip()
            if transcript:
                return transcript
        if not raw_text.startswith("{"):
            return raw_text
    return ""


def _build_transcribe_success_warnings(
    *,
    primary_model: str,
    attempt: _ReporterSpeechTranscribeAttempt,
    model_index: int,
    attempt_index: int,
) -> list[str]:
    warnings: list[str] = []
    if model_index > 0 and attempt.model != primary_model:
        warnings.append(
            f"Cloud transcription used {attempt.model} after the primary model was rejected."
        )
    elif attempt_index > 0:
        warnings.append("Cloud transcription retried with a compatibility request shape.")
    return warnings


def _build_transcribe_failure_message(provider_message: str) -> str:
    if provider_message:
        return (
            "Reporter cloud transcription was rejected by the transcription provider: "
            f"{provider_message}"
        )
    return "Reporter cloud transcription was rejected by the transcription provider"


async def transcribe_reporter_audio(
    *,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    source: str | None,
    cloud_fallback_confirmed: bool,
) -> ReporterSpeechTranscriptionResult:
    if not _resolve_reporter_speech_enabled():
        raise ReporterSpeechUnavailable("Reporter speech support is disabled")

    if not _resolve_cloud_fallback_enabled():
        raise ReporterSpeechUnavailable("Cloud transcription is disabled")

    if not cloud_fallback_confirmed:
        raise ReporterSpeechUnavailable("Cloud transcription requires explicit confirmation")

    if not _resolve_provider_is_openai():
        raise ReporterSpeechUnavailable(
            "Reporter cloud transcription requires LLM_PROVIDER=openai_compat"
        )

    api_key = _resolve_openai_api_key()
    if _truthy_env("OPENAI_OFFLINE") or not api_key:
        raise ReporterSpeechUnavailable("Reporter cloud transcription unavailable in offline mode")

    _validate_audio_input(filename=filename, content_type=content_type, audio_bytes=audio_bytes)

    models = _resolve_transcribe_models()
    primary_model = models[0]
    url = f"{_normalize_openai_base_url(os.getenv('OPENAI_BASE_URL'))}/v1/audio/transcriptions"
    timeout = _resolve_openai_timeout("structurer")
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {
        "file": (
            filename or "reporter_dictation.webm",
            audio_bytes,
            content_type or "audio/webm",
        )
    }
    last_provider_message = ""
    last_status_code = 0

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for model_index, model in enumerate(models):
                attempts = _build_transcribe_attempts_for_model(model)
                for attempt_index, attempt in enumerate(attempts):
                    data = _build_transcribe_form_data(attempt)
                    response = await client.post(url, headers=headers, data=data, files=files)

                    if response.status_code >= 400:
                        request_id = (
                            response.headers.get("x-request-id")
                            or response.headers.get("request-id")
                        )
                        provider_message = _extract_provider_error_message(response.text)
                        last_provider_message = provider_message
                        last_status_code = int(response.status_code)
                        can_retry = _should_retry_transcribe_status(response.status_code)
                        is_model_rejection = _is_model_rejection_error(provider_message)

                        log_method = _logger.warning if can_retry else _logger.error
                        log_method(
                            "Reporter speech transcription API error",
                            extra={
                                "model": attempt.model,
                                "source": source or "unknown",
                                "status_code": response.status_code,
                                "request_id": request_id or "",
                                "response_body": _safe_error_preview(response.text),
                                "attempt_index": attempt_index,
                                "model_index": model_index,
                                "response_format": attempt.response_format,
                                "include_prompt": attempt.include_prompt,
                                "include_language": attempt.include_language,
                            },
                        )

                        if not can_retry:
                            raise ReporterSpeechUnavailable(
                                _build_transcribe_failure_message(provider_message)
                            )
                        if is_model_rejection:
                            break
                        if attempt_index < len(attempts) - 1:
                            continue
                        break

                    transcript = _extract_transcript_text(response)
                    if not transcript:
                        last_provider_message = (
                            "The transcription provider returned an empty transcript."
                        )
                        if attempt_index < len(attempts) - 1:
                            continue
                        break

                    warnings = _build_transcribe_success_warnings(
                        primary_model=primary_model,
                        attempt=attempt,
                        model_index=model_index,
                        attempt_index=attempt_index,
                    )
                    return ReporterSpeechTranscriptionResult(
                        transcript=transcript,
                        provider="openai",
                        model=attempt.model,
                        fallback_used=bool(model_index or attempt_index),
                        warnings=warnings,
                    )
    except ReporterSpeechUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "Reporter speech transcription request failed",
            extra={
                "model": primary_model,
                "source": source or "unknown",
                "error_type": type(exc).__name__,
            },
        )
        raise ReporterSpeechUnavailable("Reporter cloud transcription request failed") from exc

    if last_status_code:
        raise ReporterSpeechUnavailable(_build_transcribe_failure_message(last_provider_message))
    raise ReporterSpeechUnavailable("Reporter cloud transcription returned an empty transcript")


def _resolve_cleanup_llm() -> tuple[OpenAILLM, str]:
    if not _resolve_reporter_speech_enabled():
        raise ReporterSpeechUnavailable("Reporter speech support is disabled")

    if not _resolve_provider_is_openai():
        raise ReporterSpeechUnavailable(
            "Reporter speech cleanup requires LLM_PROVIDER=openai_compat"
        )

    api_key = _resolve_openai_api_key()
    if _truthy_env("OPENAI_OFFLINE") or not api_key:
        raise ReporterSpeechUnavailable("Reporter speech cleanup unavailable in offline mode")

    model = _resolve_cleanup_model()
    return OpenAILLM(api_key=api_key, model=model, task="structurer"), model


def clean_scrubbed_reporter_transcript(
    text: str,
    *,
    already_scrubbed: bool,
    strict: bool,
) -> ReporterSpeechCleanupResult:
    source = str(text or "")
    if not source.strip():
        return ReporterSpeechCleanupResult(
            cleaned_text=source,
            changed=False,
            correction_applied=False,
            warnings=["REPORTER_SPEECH_CLEANUP_SKIPPED: empty_text"],
        )

    if not already_scrubbed:
        raise ReporterSpeechUnsafeInput("Reporter speech cleanup requires already_scrubbed=true")

    if strict and contains_potential_phi(source):
        raise ReporterSpeechUnsafeInput(
            "Reporter speech cleanup rejected text that appears to contain PHI"
        )

    max_chars = _resolve_cleanup_max_chars()
    if len(source) > max_chars:
        return ReporterSpeechCleanupResult(
            cleaned_text=source,
            changed=False,
            correction_applied=False,
            warnings=[f"REPORTER_SPEECH_CLEANUP_SKIPPED: input_too_long>{max_chars}"],
        )

    llm, model = _resolve_cleanup_llm()
    prompt = (
        f"{_REPORTER_SPEECH_CLEANER_PROMPT.strip()}\n\n"
        "Input note:\n"
        "[BEGIN_NOTE]\n"
        f"{source}\n"
        "[END_NOTE]\n"
    )

    raw = llm.generate(prompt, task="structurer", temperature=0.1)
    cleaned = _strip_markdown_code_fences(raw)
    if not cleaned:
        return ReporterSpeechCleanupResult(
            cleaned_text=source,
            changed=False,
            correction_applied=False,
            model=model,
            warnings=["REPORTER_SPEECH_CLEANUP_SKIPPED: empty_model_output"],
        )

    if _guard_failed(source, cleaned):
        return ReporterSpeechCleanupResult(
            cleaned_text=source,
            changed=False,
            correction_applied=False,
            model=model,
            warnings=["REPORTER_SPEECH_CLEANUP_SKIPPED: redaction_guard_triggered"],
        )

    return ReporterSpeechCleanupResult(
        cleaned_text=cleaned,
        changed=cleaned != source,
        correction_applied=True,
        model=model,
    )


__all__ = [
    "ReporterSpeechCleanupResult",
    "ReporterSpeechTranscriptionResult",
    "ReporterSpeechUnavailable",
    "ReporterSpeechUnsafeInput",
    "clean_scrubbed_reporter_transcript",
    "contains_potential_phi",
    "transcribe_reporter_audio",
]
