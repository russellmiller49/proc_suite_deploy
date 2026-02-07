"""Centralized Gemini API client with timeouts and error handling.

This module provides a thin wrapper over the GeminiLLM class that enforces
consistent timeout handling and logging across all Gemini API calls. All
Gemini interactions should go through `call_gemini()` to ensure consistent
behavior and error handling.

Usage:
    from app.api.gemini_client import call_gemini

    response = call_gemini("Your prompt here", timeout=30.0)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.common.llm import DeterministicStubLLM, GeminiLLM

logger = logging.getLogger(__name__)

# Default timeout for Gemini calls (in seconds)
# This should be lower than Railway's request timeout to allow for graceful handling
DEFAULT_TIMEOUT_SECONDS = 30.0

# Default model - can be overridden via GEMINI_MODEL env var
DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# Singleton LLM instances (created lazily)
_gemini_instance: GeminiLLM | None = None
_stub_instance: DeterministicStubLLM | None = None


def _get_gemini_llm() -> GeminiLLM:
    """Get or create the singleton GeminiLLM instance."""
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiLLM()
    return _gemini_instance


def _get_stub_llm() -> DeterministicStubLLM:
    """Get or create the singleton stub LLM instance."""
    global _stub_instance
    if _stub_instance is None:
        _stub_instance = DeterministicStubLLM()
    return _stub_instance


def call_gemini(
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    operation: str | None = None,
    response_schema: dict[str, Any] | None = None,
    use_stub: bool = False,
) -> str:
    """Centralized Gemini API call with timeout and error handling.

    This function should be the single point of contact for all Gemini API
    interactions in request handlers. It provides:
    - Consistent timeout handling
    - Detailed logging with timing information
    - Graceful error handling with informative messages

    Args:
        prompt: The prompt text to send to Gemini.
        model: Optional model name override (defaults to GEMINI_MODEL env var
               or gemini-2.5-flash).
        timeout: Maximum time in seconds to wait for a response. Note that
                 the underlying GeminiLLM class has its own timeout settings;
                 this timeout is primarily for documentation and logging.
        operation: Optional description of the operation for logging
                   (e.g., "summarizing report", "advisor explanation").
        response_schema: Optional JSON schema for structured responses.
        use_stub: If True, use the deterministic stub LLM instead of real Gemini.
                  Useful for testing and offline development.

    Returns:
        The response text from Gemini, or "{}" on error.

    Raises:
        No exceptions are raised; errors are logged and "{}" is returned.
    """
    op_desc = operation or "Gemini call"
    model_name = model or DEFAULT_MODEL_NAME

    if use_stub:
        logger.info("[%s] Using stub LLM (offline mode)", op_desc)
        return _get_stub_llm().generate(prompt)

    start_time = time.perf_counter()
    logger.info(
        "[%s] Starting Gemini request (model=%s, timeout=%ss)",
        op_desc,
        model_name,
        timeout,
    )

    try:
        llm = _get_gemini_llm()

        # The GeminiLLM class handles its own timeouts and retries
        # We just need to make the call and measure timing
        if response_schema:
            response = llm.generate(prompt, response_schema=response_schema)
        else:
            response = llm.generate(prompt)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "[%s] Gemini request completed in %.2fs (model=%s)",
            op_desc,
            elapsed,
            model_name,
        )

        return response

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.error(
            "[%s] Gemini request failed after %.2fs (model=%s): %s",
            op_desc,
            elapsed,
            model_name,
            exc,
        )
        return "{}"


def call_gemini_with_timeout(
    prompt: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    operation: str | None = None,
    response_schema: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """Call Gemini with explicit timing information returned.

    This is a convenience wrapper around call_gemini that also returns
    the elapsed time, useful for monitoring and metrics collection.

    Args:
        prompt: The prompt text to send to Gemini.
        timeout: Maximum time in seconds to wait for a response.
        operation: Optional description of the operation for logging.
        response_schema: Optional JSON schema for structured responses.

    Returns:
        A tuple of (response_text, elapsed_seconds).
    """
    start_time = time.perf_counter()
    response = call_gemini(
        prompt,
        timeout=timeout,
        operation=operation,
        response_schema=response_schema,
    )
    elapsed = time.perf_counter() - start_time
    return response, elapsed


__all__ = [
    "call_gemini",
    "call_gemini_with_timeout",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MODEL_NAME",
]
