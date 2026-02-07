"""LLM Advisor adapter using Gemini API.

Provides CPT code suggestions from the LLM based on procedure note text.
"""

from __future__ import annotations

import json
import os
import re
import time

# Ensure .env is loaded before reading API keys
from pathlib import Path
from dotenv import load_dotenv


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


# Find and load the .env file from project root
_env_path = Path(__file__).resolve().parents[4] / ".env"
# Tests can opt out (and avoid accidental real network calls) by setting `PROCSUITE_SKIP_DOTENV=1`.
if not _truthy_env("PROCSUITE_SKIP_DOTENV"):
    # Prefer explicitly-exported environment variables over values in `.env`.
    load_dotenv(_env_path, override=False)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from observability.logging_config import get_logger
from app.infra.cache import get_llm_memory_cache
from app.infra.llm_control import backoff_seconds, llm_slot, make_llm_cache_key
from app.infra.settings import get_infra_settings

logger = get_logger("llm_advisor")


@dataclass
class LLMCodeSuggestion:
    """A code suggestion from the LLM advisor."""

    code: str
    confidence: float
    rationale: str


class LLMAdvisorPort(ABC):
    """Abstract port for LLM-based code advisors."""

    @abstractmethod
    def suggest_codes(self, report_text: str) -> list[LLMCodeSuggestion]:
        """Get code suggestions from the LLM.

        Args:
            report_text: The procedure note text to analyze.

        Returns:
            List of code suggestions with confidences.
        """
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the model version identifier."""
        ...


class GeminiAdvisorAdapter(LLMAdvisorPort):
    """Advisor adapter using Google's Gemini API."""

    PROMPT_TEMPLATE = '''You are a medical coding expert specializing in Interventional Pulmonology procedures.
Analyze the following procedure note and suggest appropriate CPT codes.

IMPORTANT CONSTRAINTS:
- Suggest 31640 (tumor excision) ONLY when explicit resection/debulking language is present. Pure ablation terminology should map to 31641 instead.
- Photodynamic therapy without independent stent/cryotherapy/debulking work → 31641 only; suppress 31635/31649/31651/31654 unless separate interventions are clearly described.
- 31634 (hemorrhage control) should be added when massive/brisk hemoptysis is treated with iced saline, epinephrine/TXA instillation, balloon tamponade, etc.; drop navigation/foreign-body codes unless those tasks are explicitly performed.
- 31654 (Radial EBUS) should only be suggested when targeting a peripheral lung lesion with radial probe localization.
- 31645 vs 31646: routine stent cleaning/surveillance without reposition/exchange supports 31645. Only suggest 31646 when the stent is repositioned, exchanged, upsized/downsized, or removed/replaced.
- Do NOT suggest 31622 (Diagnostic bronchoscopy) if any therapeutic/surgical bronchoscopy code (31625-31661) is applicable.
- 32550 (tunneled pleural catheter) requires documentation of tunnel creation, cuff placement, or brand names (PleurX/Aspira). Planning phrases such as "consider PleurX" do NOT qualify.
- When talc/doxycycline pleurodesis is performed through an existing tube without new tunneled catheter placement, prefer 32560 and suppress 32550.

For each code you suggest, provide:
1. The CPT code
2. Your confidence (0.0-1.0) that this code applies
3. A brief rationale

Only suggest codes from this allowed list: {allowed_codes}

Return your response as a JSON array of objects with keys: code, confidence, rationale

Example format:
[
  {{"code": "31628", "confidence": 0.95, "rationale": "Transbronchial biopsy clearly documented"}},
  {{"code": "31652", "confidence": 0.85, "rationale": "EBUS-TBNA of 2 stations mentioned"}}
]

Procedure Note:
{report_text}

Return ONLY the JSON array, no other text.
'''

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        allowed_codes: list[str] | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.allowed_codes = set(allowed_codes) if allowed_codes else set()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        self._client: Optional[object] = None

    @property
    def version(self) -> str:
        return self.model_name

    def _get_client(self) -> object:
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except ImportError:
                logger.warning(
                    "google-generativeai not installed. LLM advisor will return empty suggestions."
                )
                return None  # type: ignore
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                return None  # type: ignore

        return self._client

    # Maximum text size to send to LLM (in characters)
    # Gemini Pro 1.5 supports ~128K tokens, but we limit to avoid excessive costs
    # and ensure reasonable response times. 32K chars ~= 8K tokens is a safe limit.
    MAX_TEXT_SIZE = 32000

    # Context-aware prompt template for ML-first hybrid policy
    CONTEXT_PROMPT_TEMPLATE = '''You are the final judge for CPT code assignment in an ML-assisted coding pipeline.

The ML model predicted the following CPT codes with these confidence scores:
{ml_predictions}

ML Classification: {difficulty}
Reason for LLM Review: {reason_for_fallback}

Given the full procedure note below, evaluate whether you agree with the ML suggestions.
If not, explain briefly and provide the corrected list of CPT codes.

IMPORTANT CONSTRAINTS:
- Suggest 31640 (tumor excision) ONLY when explicit resection/debulking language is present.
- 31654 (Radial EBUS) should only be suggested when targeting a peripheral lung lesion.
- Do NOT suggest 31622 (Diagnostic bronchoscopy) if any therapeutic code applies.
- 32550 (tunneled pleural catheter) requires documentation of tunnel creation or brand names.

Only suggest codes from this allowed list: {allowed_codes}

Return your response as a JSON array of objects with keys: code, confidence, rationale

Example format:
[
  {{"code": "31628", "confidence": 0.95, "rationale": "Transbronchial biopsy clearly documented"}},
  {{"code": "31652", "confidence": 0.85, "rationale": "EBUS-TBNA of 2 stations mentioned"}}
]

Procedure Note:
{report_text}

Return ONLY the JSON array, no other text.
'''

    def suggest_codes(self, report_text: str) -> list[LLMCodeSuggestion]:
        """Get code suggestions from Gemini.

        Args:
            report_text: The procedure note text to analyze.

        Returns:
            List of code suggestions with confidences.
        """
        client = self._get_client()
        if client is None:
            return []

        # Build prompt with smart text handling
        allowed_codes_str = ", ".join(sorted(self.allowed_codes)[:50])  # Limit for prompt size
        processed_text = self._prepare_text_for_llm(report_text)
        prompt = self.PROMPT_TEMPLATE.format(
            allowed_codes=allowed_codes_str,
            report_text=processed_text,
        )

        settings = get_infra_settings()
        deadline = time.monotonic() + float(settings.llm_timeout_s)

        cache_key: str | None = None
        if settings.enable_llm_cache:
            prompt_version = (os.getenv("LLM_PROMPT_VERSION") or "default").strip() or "default"
            cache_key = make_llm_cache_key(
                model=self.model_name,
                prompt=prompt,
                prompt_version=prompt_version,
            )
            cached = get_llm_memory_cache().get(cache_key)
            if isinstance(cached, str) and cached:
                return self._parse_response(cached)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with llm_slot():
                    response = client.generate_content(prompt)  # type: ignore
                response_text = response.text
                if cache_key is not None and response_text:
                    get_llm_memory_cache().set(cache_key, response_text, ttl_s=3600)
                return self._parse_response(response_text)
            except Exception as e:  # noqa: BLE001
                if attempt >= max_retries - 1 or time.monotonic() >= deadline:
                    logger.error("Gemini API call failed: %s", type(e).__name__)
                    return []

                sleep_s = backoff_seconds(attempt)
                remaining = max(0.0, deadline - time.monotonic())
                if remaining > 0:
                    time.sleep(min(sleep_s, remaining))
        return []

    def suggest_with_context(
        self,
        report_text: str,
        context: dict,
    ) -> list[LLMCodeSuggestion]:
        """Get code suggestions with ML context for hybrid pipeline.

        This method is used when the LLM acts as the final judge after
        ML has provided initial predictions.

        Args:
            report_text: The procedure note text to analyze.
            context: Dict containing:
                - ml_suggestion: List of ML-suggested codes
                - difficulty: ML case difficulty classification
                - reason_for_fallback: Why LLM was invoked
                - ml_predictions: List of {cpt, prob} dicts

        Returns:
            List of code suggestions with confidences.
        """
        client = self._get_client()
        if client is None:
            return []

        # Format ML predictions for the prompt
        ml_preds = context.get("ml_predictions", [])
        ml_pred_str = "\n".join(
            f"  - {p['cpt']}: {p['prob']:.2f}" for p in ml_preds[:10]
        ) or "  (none)"

        difficulty = context.get("difficulty", "unknown")
        reason = context.get("reason_for_fallback", "unknown")

        allowed_codes_str = ", ".join(sorted(self.allowed_codes)[:50])
        processed_text = self._prepare_text_for_llm(report_text)

        prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
            ml_predictions=ml_pred_str,
            difficulty=difficulty,
            reason_for_fallback=reason,
            allowed_codes=allowed_codes_str,
            report_text=processed_text,
        )

        settings = get_infra_settings()
        deadline = time.monotonic() + float(settings.llm_timeout_s)

        cache_key: str | None = None
        if settings.enable_llm_cache:
            prompt_version = (os.getenv("LLM_PROMPT_VERSION") or "default").strip() or "default"
            cache_key = make_llm_cache_key(
                model=self.model_name,
                prompt=prompt,
                prompt_version=prompt_version,
            )
            cached = get_llm_memory_cache().get(cache_key)
            if isinstance(cached, str) and cached:
                return self._parse_response(cached)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with llm_slot():
                    response = client.generate_content(prompt)  # type: ignore
                response_text = response.text
                if cache_key is not None and response_text:
                    get_llm_memory_cache().set(cache_key, response_text, ttl_s=3600)
                return self._parse_response(response_text)
            except Exception as e:  # noqa: BLE001
                if attempt >= max_retries - 1 or time.monotonic() >= deadline:
                    logger.error("Gemini API call with context failed: %s", type(e).__name__)
                    return []

                sleep_s = backoff_seconds(attempt)
                remaining = max(0.0, deadline - time.monotonic())
                if remaining > 0:
                    time.sleep(min(sleep_s, remaining))
        return []

    def _prepare_text_for_llm(self, text: str) -> str:
        """Prepare text for LLM processing with smart truncation.

        If text exceeds MAX_TEXT_SIZE, this method preserves the most important
        parts of the procedure note (beginning and end) while indicating that
        content was truncated from the middle.

        This prevents the common issue of losing middle sections in long notes
        while still staying within token limits.

        Args:
            text: Raw procedure note text.

        Returns:
            Processed text that fits within size limits.
        """
        if len(text) <= self.MAX_TEXT_SIZE:
            return text

        # For long texts, preserve beginning and end
        # Procedure notes typically have:
        # - Beginning: Indication, patient info, procedure start
        # - Middle: Detailed procedure steps (may be lengthy)
        # - End: Findings summary, specimens, complications, disposition

        # Allocate 40% to beginning, 40% to end, leaving room for truncation marker
        begin_size = int(self.MAX_TEXT_SIZE * 0.4)
        end_size = int(self.MAX_TEXT_SIZE * 0.4)

        begin_text = text[:begin_size]
        end_text = text[-end_size:]

        # Find natural break points (sentence boundaries)
        begin_break = begin_text.rfind('. ')
        if begin_break > begin_size * 0.8:  # Only use if we keep >80% of allocated space
            begin_text = begin_text[:begin_break + 1]

        end_break = end_text.find('. ')
        if end_break > 0 and end_break < end_size * 0.2:  # Only use if near start
            end_text = end_text[end_break + 2:]

        truncated_chars = len(text) - len(begin_text) - len(end_text)
        truncation_marker = (
            f"\n\n[... {truncated_chars} characters of detailed procedure content omitted "
            f"due to length. Key procedures may be in this section. ...]\n\n"
        )

        logger.warning(
            f"Text truncated for LLM: {len(text)} chars -> {len(begin_text) + len(end_text)} chars "
            f"({truncated_chars} chars removed from middle)"
        )

        return begin_text + truncation_marker + end_text

    def _parse_response(self, response_text: str) -> list[LLMCodeSuggestion]:
        """Parse the JSON response from the LLM.

        Args:
            response_text: Raw response text from the LLM.

        Returns:
            List of parsed code suggestions.
        """
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)

            if not isinstance(data, list):
                logger.warning("LLM response is not a list")
                return []

            suggestions = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                code = str(item.get("code", "")).strip()
                if not code:
                    continue

                # Validate against allowed codes
                if self.allowed_codes and code not in self.allowed_codes:
                    logger.debug(f"Skipping invalid code from LLM: {code}")
                    continue

                confidence = float(item.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

                rationale = str(item.get("rationale", ""))

                suggestions.append(
                    LLMCodeSuggestion(
                        code=code,
                        confidence=confidence,
                        rationale=rationale,
                    )
                )

            return suggestions

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return []


class MockLLMAdvisor(LLMAdvisorPort):
    """Mock advisor for testing without making API calls."""

    def __init__(self, suggestions: list[LLMCodeSuggestion] | None = None):
        self._suggestions = suggestions or []
        self._context_suggestions: list[LLMCodeSuggestion] | None = None
        self._last_context: dict | None = None

    @property
    def version(self) -> str:
        return "mock-advisor-v1"

    def suggest_codes(self, report_text: str) -> list[LLMCodeSuggestion]:
        return self._suggestions

    def suggest_with_context(
        self, report_text: str, context: dict
    ) -> list[LLMCodeSuggestion]:
        """Mock context-aware suggestion for hybrid pipeline testing."""
        self._last_context = context
        if self._context_suggestions is not None:
            return self._context_suggestions
        return self._suggestions

    def set_suggestions(self, suggestions: list[LLMCodeSuggestion]) -> None:
        """Set the suggestions to return."""
        self._suggestions = suggestions

    def set_context_suggestions(
        self, suggestions: list[LLMCodeSuggestion] | None
    ) -> None:
        """Set suggestions specifically for context-aware calls."""
        self._context_suggestions = suggestions

    @property
    def last_context(self) -> dict | None:
        """Get the last context passed to suggest_with_context."""
        return self._last_context
