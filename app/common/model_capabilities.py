"""Model capability detection + request payload filtering.

This module exists to prevent hard-to-debug OpenAI 400 "unsupported parameter"
errors when switching between model families (notably GPT-5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Capabilities:
    supports_response_format_json_object: bool
    supports_response_format_json_schema: bool
    supports_tools: bool
    supports_parallel_tool_calls_with_strict: bool
    supports_temperature: bool
    supports_top_p: bool
    supports_seed: bool
    supports_logprobs: bool


def is_gpt5(model: str) -> bool:
    """Return True when the model looks like a GPT-5 family identifier."""
    return (model or "").strip().lower().startswith("gpt-5")


def capabilities_for(model: str) -> Capabilities:
    """Best-effort capability matrix by model family.

    We keep this intentionally conservative: if a parameter is known to trigger
    400s for a family, we mark it unsupported here so it can be filtered out.
    """
    if is_gpt5(model):
        return Capabilities(
            supports_response_format_json_object=False,
            supports_response_format_json_schema=False,
            supports_tools=True,
            supports_parallel_tool_calls_with_strict=False,
            supports_temperature=False,
            supports_top_p=False,
            supports_seed=False,
            supports_logprobs=False,
        )

    # Default OpenAI Chat Completions feature set for non-GPT-5 models.
    return Capabilities(
        supports_response_format_json_object=True,
        supports_response_format_json_schema=True,
        supports_tools=True,
        supports_parallel_tool_calls_with_strict=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_seed=True,
        supports_logprobs=True,
    )


def filter_payload_for_model(
    payload: dict[str, Any],
    model: str,
    *,
    api_style: str = "chat",
) -> dict[str, Any]:
    """Return a filtered copy of `payload` with unsupported keys removed.

    Args:
        payload: The request payload
        model: Model identifier
        api_style: "chat" for Chat Completions, "responses" for Responses API

    Returns:
        Filtered payload safe for the target API
    """
    filtered: dict[str, Any] = dict(payload)
    caps = capabilities_for(model)

    if api_style == "responses":
        # Responses API uses different structure; remove chat-specific keys
        for chat_key in ("messages", "response_format", "stream"):
            filtered.pop(chat_key, None)

        # Responses API may not support all sampling params for all models
        if not caps.supports_temperature:
            filtered.pop("temperature", None)
        if not caps.supports_top_p:
            filtered.pop("top_p", None)
        if not caps.supports_seed:
            filtered.pop("seed", None)
        if not caps.supports_logprobs:
            filtered.pop("logprobs", None)

    else:  # api_style == "chat"
        if not (caps.supports_response_format_json_object or caps.supports_response_format_json_schema):
            filtered.pop("response_format", None)

        if not caps.supports_temperature:
            filtered.pop("temperature", None)

        if not caps.supports_top_p:
            filtered.pop("top_p", None)

        if not caps.supports_seed:
            filtered.pop("seed", None)

        if not caps.supports_logprobs:
            filtered.pop("logprobs", None)

        if not caps.supports_tools:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                filtered.pop(key, None)

    return filtered


__all__ = ["Capabilities", "capabilities_for", "filter_payload_for_model", "is_gpt5"]

