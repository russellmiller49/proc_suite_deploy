"LLM client infrastructure shared across app."

from __future__ import annotations

import atexit
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
try:
    from google.auth import default as google_auth_default
    from google.auth.transport.requests import Request as GoogleAuthRequest
except ImportError:  # google-auth is optional unless OAuth is used
    google_auth_default = None  # type: ignore[assignment]
    GoogleAuthRequest = None  # type: ignore[assignment]

from app.common.logger import get_logger
from app.common.exceptions import LLMError
from app.common.model_capabilities import filter_payload_for_model, is_gpt5
from app.common.openai_responses import (
    get_primary_api,
    is_fallback_enabled,
    build_responses_payload,
    parse_responses_text,
    post_responses,
    ResponsesEndpointNotFound,
)
from app.infra.cache import get_llm_memory_cache
from app.infra.llm_control import (
    backoff_seconds,
    llm_slot,
    make_llm_cache_key,
    parse_retry_after_seconds,
)
from app.infra.settings import get_infra_settings

logger = get_logger("common.llm")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


@dataclass
class _UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


_USAGE_TOTALS_BY_MODEL: dict[str, _UsageTotals] = {}
_USAGE_TOTALS_ALL = _UsageTotals()
_PRICING_CACHE: dict[str, dict[str, float]] | None = None
_USAGE_ATEXIT_REGISTERED = False


def _load_pricing() -> dict[str, dict[str, float]]:
    """
    Optional pricing config for cost estimation.

    Supported env vars:
      - OPENAI_PRICING_JSON: JSON like {"gpt-4o-mini":{"input_per_1k":0.00015,"output_per_1k":0.0006}, ...}
      - OPENAI_COST_INPUT_PER_1K, OPENAI_COST_OUTPUT_PER_1K: global fallback floats
    """
    global _PRICING_CACHE
    if _PRICING_CACHE is not None:
        return _PRICING_CACHE

    pricing: dict[str, dict[str, float]] = {}

    raw = os.getenv("OPENAI_PRICING_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for model, entry in parsed.items():
                    if not isinstance(model, str) or not isinstance(entry, dict):
                        continue
                    inp = entry.get("input_per_1k")
                    out = entry.get("output_per_1k")
                    if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
                        pricing[model] = {"input_per_1k": float(inp), "output_per_1k": float(out)}
        except Exception:
            # Ignore malformed JSON; cost estimation will be disabled.
            pricing = {}

    inp = os.getenv("OPENAI_COST_INPUT_PER_1K", "").strip()
    out = os.getenv("OPENAI_COST_OUTPUT_PER_1K", "").strip()
    if inp and out:
        try:
            pricing["*"] = {"input_per_1k": float(inp), "output_per_1k": float(out)}
        except Exception:
            pass

    _PRICING_CACHE = pricing
    return pricing


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = _load_pricing()
    entry = pricing.get(model) or pricing.get("*")
    if not entry:
        return None
    in_per_1k = entry.get("input_per_1k")
    out_per_1k = entry.get("output_per_1k")
    if not isinstance(in_per_1k, float) or not isinstance(out_per_1k, float):
        return None
    return (input_tokens / 1000.0) * in_per_1k + (output_tokens / 1000.0) * out_per_1k


def _record_usage(
    *,
    model: str,
    api_style: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    latency_s: float,
) -> None:
    totals = _USAGE_TOTALS_BY_MODEL.setdefault(model, _UsageTotals())
    totals.calls += 1
    totals.input_tokens += int(input_tokens)
    totals.output_tokens += int(output_tokens)
    totals.total_tokens += int(total_tokens)

    _USAGE_TOTALS_ALL.calls += 1
    _USAGE_TOTALS_ALL.input_tokens += int(input_tokens)
    _USAGE_TOTALS_ALL.output_tokens += int(output_tokens)
    _USAGE_TOTALS_ALL.total_tokens += int(total_tokens)

    cost = _estimate_cost_usd(model, int(input_tokens), int(output_tokens))
    if cost is not None:
        totals.cost_usd += float(cost)
        _USAGE_TOTALS_ALL.cost_usd += float(cost)

    if _truthy_env("OPENAI_LOG_USAGE_PER_CALL"):
        cost_str = f"${cost:.6f}" if cost is not None else "(pricing not configured)"
        msg = (
            f"[llm_usage] api={api_style} model={model} "
            f"in={input_tokens} out={output_tokens} total={total_tokens} "
            f"latency_s={latency_s:.2f} cost={cost_str}"
        )
        # Ensure visibility even when logs aren't shown.
        try:
            print(msg, file=os.sys.stderr)
        except Exception:
            pass
        logger.info(msg)


def _print_usage_summary() -> None:
    if not _truthy_env("OPENAI_LOG_USAGE_SUMMARY"):
        return
    if _USAGE_TOTALS_ALL.calls <= 0:
        return

    header = (
        f"[llm_usage_summary] calls={_USAGE_TOTALS_ALL.calls} "
        f"in={_USAGE_TOTALS_ALL.input_tokens} out={_USAGE_TOTALS_ALL.output_tokens} total={_USAGE_TOTALS_ALL.total_tokens} "
        + (
            f"cost=${_USAGE_TOTALS_ALL.cost_usd:.6f}"
            if _USAGE_TOTALS_ALL.cost_usd > 0
            else "cost=(pricing not configured)"
        )
    )
    lines: list[str] = [header]

    top = sorted(_USAGE_TOTALS_BY_MODEL.items(), key=lambda kv: kv[1].total_tokens, reverse=True)[:10]
    for model, t in top:
        lines.append(
            f"  - {model}: calls={t.calls} in={t.input_tokens} out={t.output_tokens} total={t.total_tokens} "
            + (f"cost=${t.cost_usd:.6f}" if t.cost_usd > 0 else "cost=(pricing not configured)")
        )

    msg = "\n".join(lines)
    try:
        print(msg, file=os.sys.stderr)
    except Exception:
        pass
    logger.info(msg)


# Load environment variables from a .env file if present so GEMINI_* keys are available locally.
# Important: do NOT override explicitly-exported environment variables.
# Tests can opt out (and avoid accidental real API keys) by setting `PROCSUITE_SKIP_DOTENV=1`.
if not _truthy_env("PROCSUITE_SKIP_DOTENV"):
    try:
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
    except Exception as e:
        logger.warning(
            "Failed to load .env via python-dotenv (%s); proceeding with OS env only",
            type(e).__name__,
        )


def _normalize_openai_base_url(base_url: str | None) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized or "https://api.openai.com"


def _resolve_openai_model(task: str | None) -> str:
    task_key = (task or "").strip().lower()
    if task_key == "summarizer":
        return (os.getenv("OPENAI_MODEL_SUMMARIZER") or os.getenv("OPENAI_MODEL") or "").strip()
    if task_key == "structurer":
        return (os.getenv("OPENAI_MODEL_STRUCTURER") or os.getenv("OPENAI_MODEL") or "").strip()
    if task_key == "judge":
        return (os.getenv("OPENAI_MODEL_JUDGE") or os.getenv("OPENAI_MODEL") or "").strip()
    return (os.getenv("OPENAI_MODEL") or "").strip()


def _resolve_openai_timeout_seconds(task: str | None) -> float:
    # Back-compat shim for older callers/tests. Prefer `_resolve_openai_timeout()`.
    return float(_resolve_openai_timeout(task).read)


def _get_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _resolve_openai_timeout(task: str | None) -> httpx.Timeout:
    """Resolve an OpenAI httpx.Timeout based on workload/task.

    Env overrides:
    - `OPENAI_TIMEOUT_READ_REGISTRY_SECONDS` (default 180)
    - `OPENAI_TIMEOUT_READ_DEFAULT_SECONDS` (default 60)

    Legacy env (read-timeout shims):
    - `OPENAI_TIMEOUT_SECONDS`, `OPENAI_TIMEOUT_SECONDS_STRUCTURER`,
      `OPENAI_TIMEOUT_SECONDS_JUDGE`, `OPENAI_TIMEOUT_SECONDS_SUMMARIZER`
    """

    task_key = (task or "").strip().lower()

    is_registry_task = False
    if task_key:
        if task_key in {"registry", "registry_extraction", "registry-extraction", "structurer"}:
            is_registry_task = True
        elif "registry" in task_key and "extract" in task_key:
            is_registry_task = True

    if is_registry_task:
        read_seconds = (
            _get_float_env("OPENAI_TIMEOUT_READ_REGISTRY_SECONDS")
            or _get_float_env("OPENAI_TIMEOUT_SECONDS_STRUCTURER")
            or _get_float_env("OPENAI_TIMEOUT_SECONDS")
            or 180.0
        )
    else:
        legacy_task_read: float | None = None
        if task_key == "judge":
            legacy_task_read = _get_float_env("OPENAI_TIMEOUT_SECONDS_JUDGE")
        elif task_key == "summarizer":
            legacy_task_read = _get_float_env("OPENAI_TIMEOUT_SECONDS_SUMMARIZER")

        read_seconds = (
            _get_float_env("OPENAI_TIMEOUT_READ_DEFAULT_SECONDS")
            or legacy_task_read
            or _get_float_env("OPENAI_TIMEOUT_SECONDS")
            or 60.0
        )

    return httpx.Timeout(connect=10.0, read=read_seconds, write=30.0, pool=30.0)


def _openai_request_id(response: httpx.Response) -> str | None:
    for header_name in ("x-request-id", "request-id", "x-openai-request-id", "openai-request-id"):
        value = response.headers.get(header_name)
        if value:
            return value
    return None


def _openai_error_details(response: httpx.Response) -> tuple[str, str | None, str | None]:
    message = ""
    error_type: str | None = None
    error_param: str | None = None

    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        data = None

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or "")
            error_type = str(err.get("type") or "") or None
            error_param = str(err.get("param") or "") or None

    if not message:
        message = str((response.text or "")).strip()

    message = " ".join(message.split())
    if len(message) > 500:
        message = message[:500] + "…"

    if not message:
        message = f"HTTP {response.status_code} from OpenAI"

    return message, error_type, error_param


def _looks_like_unsupported_parameter_error(
    *,
    message: str,
    error_type: str | None,
    error_param: str | None,
) -> bool:
    if error_param and error_param in {
        "response_format",
        "temperature",
        "top_p",
        "seed",
        "logprobs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }:
        return True

    haystack = f"{error_type or ''} {message}".lower()
    return any(
        token in haystack
        for token in (
            "unsupported",
            "unknown parameter",
            "unrecognized request argument",
            "unrecognized",
            "unexpected",
            "additional properties",
            "extra fields",
            "invalid_request_error",
        )
    )


def _error_suggests_tools_unsupported(*, message: str, error_param: str | None) -> bool:
    if error_param and error_param in {"tools", "tool_choice", "parallel_tool_calls"}:
        return True
    haystack = message.lower()
    return any(token in haystack for token in (" tools", "tool_choice", "parallel_tool_calls", "function calling"))


def _build_unsupported_param_retry_payload(
    payload: dict[str, Any],
    *,
    message: str,
    error_param: str | None,
) -> tuple[dict[str, Any], list[str]]:
    retry_payload: dict[str, Any] = dict(payload)
    removed: list[str] = []

    def _pop(key: str) -> None:
        if key in retry_payload:
            retry_payload.pop(key, None)
            removed.append(key)

    _pop("response_format")
    for key in ("temperature", "top_p", "seed", "logprobs", "top_logprobs"):
        _pop(key)

    if _error_suggests_tools_unsupported(message=message, error_param=error_param):
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            _pop(key)

    return retry_payload, removed


def _prepend_json_object_instruction(prompt: str) -> str:
    instruction = "Return exactly one JSON object. No markdown. No code fences."
    if instruction.lower() in (prompt or "").lower():
        return prompt
    return f"{instruction}\n\n{prompt}"


class LLMInterface(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class OpenAILLM:
    """OpenAI client supporting both Responses API and Chat Completions.

    By default, uses Responses API (POST /v1/responses) for first-party OpenAI.
    Falls back to Chat Completions on 404 or when configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        task: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.1")
        self.base_url = _normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))
        self.task = task
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else None

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set; OpenAILLM calls will fail.")

    def _is_openai_endpoint(self) -> bool:
        """Check if base_url points to OpenAI's API."""
        base = self.base_url.lower()
        return "api.openai.com" in base or not base or base == "https://api.openai.com"

    def _get_timeout(self, task_key: str | None) -> httpx.Timeout:
        """Get timeout configuration for the task."""
        resolved_timeout = _resolve_openai_timeout(task_key)
        read_override = self.timeout_seconds
        if read_override is not None:
            return httpx.Timeout(
                connect=resolved_timeout.connect,
                read=read_override,
                write=resolved_timeout.write,
                pool=resolved_timeout.pool,
            )
        return resolved_timeout

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with auth."""
        return {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        response_schema: dict | None = None,
        *,
        task: str | None = None,
        **kwargs,
    ) -> str:
        """Generate a response from OpenAI.

        Uses Responses API by default for first-party OpenAI endpoints.
        Falls back to Chat Completions on 404 or when configured.

        Args:
            prompt: The prompt text
            response_schema: Currently ignored (Gemini-only schema shape)
            task: Task identifier for timeout/capability selection
            **kwargs: Optional parameters (best-effort, capability-filtered)
        """
        if _truthy_env("OPENAI_OFFLINE") or not self.api_key:
            return "{}"

        settings = get_infra_settings()
        prompt_version = str(
            kwargs.pop("prompt_version", "") or os.getenv("LLM_PROMPT_VERSION") or "default"
        ).strip() or "default"

        cache_key: str | None = None
        if settings.enable_llm_cache:
            temperature = kwargs.get("temperature")
            cacheable = temperature is None or float(temperature) == 0.0
            if cacheable:
                cache_key = make_llm_cache_key(
                    model=self.model,
                    prompt=prompt,
                    prompt_version=prompt_version,
                )
                cached = get_llm_memory_cache().get(cache_key)
                if isinstance(cached, str) and cached:
                    return cached

        task_key = task if task is not None else self.task
        primary_api = get_primary_api()
        started = time.monotonic()
        usage: dict[str, Any] | None = None

        # Use Responses API for first-party OpenAI when configured
        if primary_api == "responses" and self._is_openai_endpoint():
            try:
                response_text, usage = self._generate_via_responses(prompt, task=task_key, **kwargs)
            except ResponsesEndpointNotFound:
                if is_fallback_enabled():
                    logger.info(
                        "Responses API not available; falling back to Chat Completions model=%s",
                        self.model,
                    )
                    response_text, usage = self._generate_via_chat(prompt, task=task_key, **kwargs)
                else:
                    raise

        else:
            # Use Chat Completions for compat endpoints or when configured
            response_text, usage = self._generate_via_chat(prompt, task=task_key, **kwargs)

        if cache_key is not None and response_text:
            get_llm_memory_cache().set(cache_key, response_text, ttl_s=3600)

        # Best-effort usage reporting (tokens are present only when upstream includes them)
        if isinstance(usage, dict):
            try:
                input_tokens = int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
                if input_tokens or output_tokens or total_tokens:
                    global _USAGE_ATEXIT_REGISTERED
                    if not _USAGE_ATEXIT_REGISTERED:
                        atexit.register(_print_usage_summary)
                        _USAGE_ATEXIT_REGISTERED = True
                    _record_usage(
                        model=self.model,
                        api_style=str(usage.get("api") or "unknown"),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_s=max(0.0, time.monotonic() - started),
                    )
            except Exception:
                pass

        return response_text

    def _generate_via_responses(
        self,
        prompt: str,
        *,
        task: str | None = None,
        **kwargs,
    ) -> tuple[str, dict[str, Any]]:
        """Generate using Responses API (POST /v1/responses)."""
        url = f"{self.base_url}/v1/responses"
        headers = self._get_headers()
        timeout = self._get_timeout(task)

        # Build extra params from kwargs (capability-filtered later)
        extra: dict[str, Any] = {}
        for float_key in ("temperature", "top_p"):
            if float_key in kwargs and kwargs[float_key] is not None:
                extra[float_key] = float(kwargs[float_key])
        for int_key in ("max_tokens", "max_completion_tokens", "seed"):
            if int_key in kwargs and kwargs[int_key] is not None:
                extra[int_key] = int(kwargs[int_key])

        payload = build_responses_payload(
            self.model,
            prompt,
            wants_json=True,
            task=task,
            extra=extra,
        )

        # Apply capability filtering for responses API
        payload = filter_payload_for_model(payload, self.model, api_style="responses")

        try:
            resp_json = post_responses(
                url=url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                model=self.model,
            )
            text = parse_responses_text(resp_json)
            usage: dict[str, Any] = {"api": "responses"}
            usage_raw = resp_json.get("usage") if isinstance(resp_json, dict) else None
            if isinstance(usage_raw, dict):
                usage["input_tokens"] = int(usage_raw.get("input_tokens") or 0)
                usage["output_tokens"] = int(usage_raw.get("output_tokens") or 0)
                usage["total_tokens"] = int(
                    usage_raw.get("total_tokens") or (usage["input_tokens"] + usage["output_tokens"])
                )
            return text, usage
        except ResponsesEndpointNotFound:
            raise
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Unexpected error in Responses API (model={self.model}): {exc}") from exc

    def _generate_via_chat(
        self,
        prompt: str,
        *,
        task: str | None = None,
        **kwargs,
    ) -> tuple[str, dict[str, Any]]:
        """Generate using Chat Completions API (POST /v1/chat/completions)."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._get_headers()
        timeout = self._get_timeout(task)

        wants_json = True
        outgoing_prompt = _prepend_json_object_instruction(prompt) if is_gpt5(self.model) and wants_json else prompt

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": outgoing_prompt}],
        }

        # Prefer native structured outputs where supported; GPT-5 rejects response_format.
        if wants_json and not is_gpt5(self.model):
            payload["response_format"] = {"type": "json_object"}

        # Best-effort optional parameters (ignored when unsupported by the model).
        for float_key in ("temperature", "top_p"):
            if float_key in kwargs and kwargs[float_key] is not None:
                payload[float_key] = float(kwargs[float_key])
        for int_key in ("max_tokens", "max_completion_tokens", "seed", "top_logprobs"):
            if int_key in kwargs and kwargs[int_key] is not None:
                payload[int_key] = int(kwargs[int_key])
        if "logprobs" in kwargs and kwargs["logprobs"] is not None:
            payload["logprobs"] = kwargs["logprobs"]
        for raw_key in ("tools", "tool_choice", "parallel_tool_calls"):
            if raw_key in kwargs and kwargs[raw_key] is not None:
                payload[raw_key] = kwargs[raw_key]

        payload = filter_payload_for_model(payload, self.model, api_style="chat")

        removed_on_retry: list[str] = []

        try:
            deadline = time.monotonic() + float(get_infra_settings().llm_timeout_s)
            with httpx.Client(timeout=timeout) as client:
                attempt_payload = payload
                did_retry_timeout = False
                did_retry_unsupported = False

                for attempt in range(3):
                    try:
                        with llm_slot():
                            response = client.post(url, headers=headers, json=attempt_payload)
                    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.TransportError) as exc:
                        if not did_retry_timeout:
                            did_retry_timeout = True
                            backoff = random.uniform(0.8, 1.5)
                            logger.warning(
                                "Chat Completions API transient transport error; retrying once endpoint=%s model=%s",
                                url,
                                self.model,
                            )
                            time.sleep(backoff)
                            continue
                        raise LLMError(
                            f"Chat Completions transport error after retry (model={self.model}): {type(exc).__name__}"
                        ) from exc

                    if response.status_code < 400:
                        data = response.json()
                        choices = data.get("choices", [])
                        if not choices:
                            raise LLMError("No choices returned from Chat Completions API")
                        content = choices[0].get("message", {}).get("content", "")
                        usage: dict[str, Any] = {"api": "chat"}
                        usage_raw = data.get("usage") if isinstance(data, dict) else None
                        if isinstance(usage_raw, dict):
                            usage["input_tokens"] = int(usage_raw.get("prompt_tokens") or 0)
                            usage["output_tokens"] = int(usage_raw.get("completion_tokens") or 0)
                            usage["total_tokens"] = int(
                                usage_raw.get("total_tokens") or (usage["input_tokens"] + usage["output_tokens"])
                            )
                        return content, usage

                    message, error_type, error_param = _openai_error_details(response)
                    request_id = _openai_request_id(response)
                    request_id_suffix = f" request_id={request_id}" if request_id else ""
                    logger.warning(
                        "Chat Completions API error status=%s endpoint=%s model=%s%s",
                        response.status_code,
                        url,
                        self.model,
                        request_id_suffix,
                    )

                    if response.status_code == 429 or 500 <= response.status_code <= 599:
                        if attempt < 2 and time.monotonic() < deadline:
                            retry_after = parse_retry_after_seconds(response.headers)
                            sleep_s = retry_after if retry_after is not None else backoff_seconds(attempt)
                            remaining = max(0.0, deadline - time.monotonic())
                            if remaining > 0:
                                time.sleep(min(sleep_s, remaining))
                                continue

                    should_retry = (
                        not did_retry_unsupported
                        and response.status_code == 400
                        and _looks_like_unsupported_parameter_error(
                            message=message, error_type=error_type, error_param=error_param
                        )
                    )
                    if should_retry:
                        retry_payload, removed_on_retry = _build_unsupported_param_retry_payload(
                            attempt_payload,
                            message=message,
                            error_param=error_param,
                        )
                        if removed_on_retry:
                            attempt_payload = retry_payload
                            did_retry_unsupported = True
                            continue

                    removed_summary = ", ".join(removed_on_retry) if removed_on_retry else "none"
                    raise LLMError(
                        f"Chat Completions request failed (status={response.status_code}, model={self.model}, "
                        f"removed_on_retry={removed_summary}): {message}"
                    )

        except httpx.RequestError as exc:
            raise LLMError(f"Network error contacting Chat Completions API (model={self.model}): {exc}") from exc
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            raise LLMError(f"Unexpected error in Chat Completions (model={self.model}): {exc}") from exc

        # Should not reach here
        raise LLMError(f"Chat Completions request failed after all attempts (model={self.model})")


class GeminiLLM:
    """Implementation of LLMInterface using Google's Gemini API via HTTP.
    
    Supports both API key and OAuth2/service account authentication.
    - API key: Pass api_key parameter
    - OAuth2: Set GEMINI_USE_OAUTH=true and configure Google Cloud credentials
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        use_oauth: bool | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # Determine authentication method
        if use_oauth is None:
            use_oauth = os.getenv("GEMINI_USE_OAUTH", "").lower() in ("true", "1", "yes")
        
        self.use_oauth = use_oauth
        
        if self.use_oauth:
            if api_key:
                logger.warning("Both API key and OAuth enabled. Using OAuth authentication.")
            logger.info("Using OAuth2/service account authentication for Gemini API")
            self._credentials = None
            self._refresh_credentials()
        elif not api_key:
            # Attempt to find key in env if not passed explicitly and not using oauth
            if not api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")
            
            if not self.api_key:
                 # As a fallback for local dev without keys, we might want to allow initialization 
                 # but fail on generate, or let the caller handle it.
                 # For now, we'll log a warning and expect the user to fix it.
                 logger.warning("No GEMINI_API_KEY found and OAuth not enabled.")

    def _refresh_credentials(self) -> None:
        """Refresh OAuth2 credentials using Application Default Credentials."""
        if not google_auth_default or not GoogleAuthRequest:
            raise RuntimeError(
                "google-auth is required for OAuth2 Gemini access. Install google-auth and retry."
            )
        try:
            credentials, _ = google_auth_default()
            # Ensure we have a valid token
            if not credentials.valid:
                credentials.refresh(GoogleAuthRequest())
            self._credentials = credentials
            logger.debug("OAuth2 credentials refreshed successfully")
        except Exception as e:
            logger.error(f"Failed to obtain OAuth2 credentials: {e}")
            raise RuntimeError(
                "OAuth2 authentication failed. Ensure you have:\n"
                "1. Set up Application Default Credentials (gcloud auth application-default login), or\n"
                "2. Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file, or\n"
                "3. Running on GCP with appropriate IAM roles"
            ) from e

    def _get_access_token(self) -> str:
        """Get a valid OAuth2 access token."""
        if not self._credentials or not self._credentials.valid:
            self._refresh_credentials()
        return self._credentials.token  # type: ignore[return-value]

    def generate(
        self,
        prompt: str,
        response_schema: dict | None = None,
        max_retries: int = 3,
        *,
        temperature: float | None = None,
        task: str | None = None,
        prompt_version: str | None = None,
    ) -> str:
        settings = get_infra_settings()
        deadline = time.monotonic() + float(settings.llm_timeout_s)

        prompt_version_value = str(prompt_version or os.getenv("LLM_PROMPT_VERSION") or "default").strip() or "default"
        cache_key: str | None = None
        if settings.enable_llm_cache:
            cacheable = temperature is None or float(temperature) == 0.0
            if cacheable:
                cache_key = make_llm_cache_key(
                    model=self.model,
                    prompt=prompt,
                    prompt_version=prompt_version_value,
                )
                cached = get_llm_memory_cache().get(cache_key)
                if isinstance(cached, str) and cached:
                    return cached

        if self.use_oauth:
            url = f"{self.base_url}/{self.model}:generateContent"
            access_token = self._get_access_token()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            }
        else:
            if not self.api_key:
                 logger.error("Attempted to generate without API key or OAuth.")
                 return "{}"
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}

        generation_config: dict[str, Any] = {"response_mime_type": "application/json"}
        if response_schema:
            generation_config["response_schema"] = response_schema
        if temperature is not None:
            generation_config["temperature"] = float(temperature)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config
        }

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=float(settings.llm_timeout_s),
                    write=30.0,
                    pool=10.0,
                )
                with httpx.Client(timeout=timeout) as client:
                    with llm_slot():
                        response = client.post(url, headers=headers, json=payload)

                    if response.status_code >= 400:
                        if response.status_code == 429 or response.status_code >= 500:
                            raise httpx.HTTPStatusError(
                                f"Transient HTTP {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        logger.error("Gemini API HTTP error status=%s", response.status_code)
                        return "{}"

                    data = response.json()

                    # Extract text from response structure
                    # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
                    candidates = data.get("candidates", [])
                    if not candidates:
                        logger.error("No candidates returned from Gemini API")
                        return "{}"

                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if cache_key is not None and text:
                        get_llm_memory_cache().set(cache_key, text, ttl_s=3600)
                    return text
            except httpx.RequestError as e:
                last_error = e
                logger.warning("Gemini API transport error attempt=%s error=%s", attempt + 1, type(e).__name__)
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = getattr(e.response, "status_code", None)
                logger.warning("Gemini API transient HTTP error attempt=%s status=%s", attempt + 1, status_code)
            except Exception as e:
                logger.error(f"Unexpected error in GeminiLLM: {e}")
                return "{}"

            if attempt >= max_retries - 1 or time.monotonic() >= deadline:
                break

            retry_after = None
            if isinstance(last_error, httpx.HTTPStatusError) and last_error.response is not None:
                retry_after = parse_retry_after_seconds(last_error.response.headers)

            sleep_s = retry_after if retry_after is not None else backoff_seconds(attempt)
            remaining = max(0.0, deadline - time.monotonic())
            if remaining > 0:
                time.sleep(min(sleep_s, remaining))

        # All retries exhausted
        logger.error(f"All {max_retries} retries exhausted. Last error: {last_error}")
        return "{}"


class DeterministicStubLLM:
    """Simple deterministic LLM stub used for tests and local runs."""

    def __init__(self, payload: dict | None = None, *, reason: str | None = None) -> None:
        self.payload = payload or {
            "indication": "Peripheral nodule",
            "anesthesia": "Moderate Sedation",
            "survey": ["Airways inspected"],
            "localization": "Navigated to RUL",
            "sampling": ["EBUS 4R"],
            "therapeutics": ["Stent RMB"],
            "complications": [],
            "disposition": "Home",
        }
        self.reason = reason
        self._warned = False

    def generate(self, prompt: str, **_kwargs: Any) -> str:
        if not self._warned:
            if self.reason:
                logger.warning("Using DeterministicStubLLM (%s)", self.reason)
            else:
                logger.warning("Using DeterministicStubLLM")
            self._warned = True
        return json.dumps(self.payload)

TModel = TypeVar("TModel", bound=BaseModel)


class LLMService:
    """Small helper for structured (JSON) generations.

    This wraps the repo's LLM clients and provides a convenience method that:
    - requests JSON output
    - parses the JSON payload
    - validates it against a Pydantic model
    """

    def __init__(self, llm: LLMInterface | None = None, *, task: str | None = None) -> None:
        if llm is not None:
            self._llm = llm
            return

        stub_reason: str | None = None
        use_stub = os.getenv("REGISTRY_USE_STUB_LLM", "").lower() in ("1", "true", "yes")
        use_stub = use_stub or os.getenv("GEMINI_OFFLINE", "").lower() in ("1", "true", "yes")

        if use_stub:
            if os.getenv("REGISTRY_USE_STUB_LLM", "").lower() in ("1", "true", "yes"):
                stub_reason = "REGISTRY_USE_STUB_LLM enabled"
            elif os.getenv("GEMINI_OFFLINE", "").lower() in ("1", "true", "yes"):
                stub_reason = "GEMINI_OFFLINE enabled"
            self._llm = DeterministicStubLLM(reason=stub_reason)
            return

        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider == "openai_compat":
            openai_offline = _truthy_env("OPENAI_OFFLINE") or not bool(os.getenv("OPENAI_API_KEY"))
            if openai_offline:
                reason = "OPENAI_OFFLINE enabled" if _truthy_env("OPENAI_OFFLINE") else "OPENAI_API_KEY not set"
                self._llm = DeterministicStubLLM(reason=reason)
                return

            model = _resolve_openai_model(task)
            if not model:
                self._llm = DeterministicStubLLM(reason="OPENAI_MODEL not set")
                return

            self._llm = OpenAILLM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model,
                task=task,
            )
            return

        if provider != "gemini":
            logger.warning("Unknown LLM_PROVIDER='%s'; defaulting to gemini", provider)

        if not os.getenv("GEMINI_API_KEY"):
            self._llm = DeterministicStubLLM(reason="GEMINI_API_KEY not set (LLM_PROVIDER=gemini)")
            return

        self._llm = GeminiLLM()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
        temperature: float = 0.0,
    ) -> TModel:
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n"

        # Prefer prompt-only enforcement for now; Gemini response_schema requires a
        # provider-specific schema shape (see LLMDetailedExtractor for conversion).
        raw = self._generate(prompt, temperature=temperature)
        cleaned = _strip_markdown_code_fences(raw)

        if cleaned.strip() in {"null", "None", ""}:
            raise ValueError("LLM returned null/empty response")

        data = json.loads(cleaned)
        return response_model.model_validate(data)

    def _generate(self, prompt: str, *, temperature: float) -> str:
        llm = self._llm
        if isinstance(llm, GeminiLLM):
            return llm.generate(prompt, temperature=temperature)
        return llm.generate(prompt)


def _strip_markdown_code_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3].strip()
    return cleaned.strip()


__all__ = ["LLMInterface", "GeminiLLM", "OpenAILLM", "DeterministicStubLLM", "LLMService"]
