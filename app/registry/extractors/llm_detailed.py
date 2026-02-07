"""LLM-based detailed extractor for registry data.

Implements a ReAct/self-correction loop around a Pydantic schema to
extract structured registry data from procedure notes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type
import os

from pydantic import BaseModel, ValidationError

from app.common.llm import (
    DeterministicStubLLM,
    GeminiLLM,
    OpenAILLM,
    _resolve_openai_model,
)
from app.common.logger import get_logger
from app.common.sectionizer import Section
from app.registry.prompts import build_registry_prompt
from app.registry.slots.base import SlotResult
from config.settings import LLMExtractionConfig
from observability.timing import timed
from observability.logging_config import get_logger as get_obs_logger

logger = get_logger("registry.extractors.llm")
obs_logger = get_obs_logger("llm_extractor")


@dataclass
class ExtractionAttempt:
    """Record of a single extraction attempt."""

    attempt_number: int
    response_text: str
    parsed_data: Optional[dict] = None
    validation_error: Optional[str] = None
    elapsed_ms: float = 0.0
    success: bool = False


@dataclass
class ExtractionResult:
    """Full result of the extraction process with all attempts."""

    value: Optional[dict] = None
    confidence: float = 0.0
    attempts: list[ExtractionAttempt] = field(default_factory=list)
    cache_hit: bool = False
    note_hash: str = ""
    schema_name: str = ""


class NoteHashCache:
    """Simple in-memory cache keyed by note hash + schema name."""

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, SlotResult] = {}
        self._max_size = max_size

    def _make_key(self, note_hash: str, schema_name: str) -> str:
        return f"{note_hash}:{schema_name}"

    def get(self, note_hash: str, schema_name: str) -> Optional[SlotResult]:
        key = self._make_key(note_hash, schema_name)
        return self._cache.get(key)

    def set(self, note_hash: str, schema_name: str, result: SlotResult) -> None:
        key = self._make_key(note_hash, schema_name)

        # Simple LRU: if at max size, remove oldest entry
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result

    def clear(self) -> None:
        self._cache.clear()


def hash_text(text: str) -> str:
    """Compute a hash of the text for caching."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class LLMDetailedExtractor:
    """LLM-based extractor with self-correction loop.

    Features:
    - ReAct-style correction loop for validation errors
    - Caching by note hash to avoid redundant LLM calls
    - Fast path for high-confidence first attempts
    - Configurable timeouts and retry limits
    """

    slot_name = "llm_detailed"
    VERSION = "llm_extractor_v2"

    def __init__(
        self,
        llm: GeminiLLM | OpenAILLM | None = None,
        config: LLMExtractionConfig | None = None,
    ) -> None:
        if llm is not None:
            self.llm = llm
        else:
            use_stub = os.getenv("REGISTRY_USE_STUB_LLM", "").lower() in ("1", "true", "yes")
            use_stub = use_stub or os.getenv("GEMINI_OFFLINE", "").lower() in ("1", "true", "yes")

            if use_stub:
                self.llm = DeterministicStubLLM()
            else:
                # Check LLM_PROVIDER to determine which LLM to use
                provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
                if provider == "openai_compat":
                    openai_offline = os.getenv("OPENAI_OFFLINE", "").strip().lower() in ("1", "true", "yes")
                    api_key = os.getenv("OPENAI_API_KEY")
                    model = _resolve_openai_model("structurer") or "gpt-5.2"
                    if openai_offline or not api_key:
                        self.llm = DeterministicStubLLM()
                    elif model:
                        self.llm = OpenAILLM(
                            api_key=api_key,
                            model=model,
                            task="registry_extraction",
                        )
                        logger.info(f"Using OpenAI LLM with model: {model}")
                    else:
                        logger.warning("OPENAI_MODEL not set, falling back to stub")
                        self.llm = DeterministicStubLLM()
                else:
                    # Default to Gemini
                    if not os.getenv("GEMINI_API_KEY"):
                        logger.warning("GEMINI_API_KEY not set, falling back to stub")
                        self.llm = DeterministicStubLLM()
                    else:
                        self.llm = GeminiLLM()
                        logger.info("Using Gemini LLM")

        self.config = config or LLMExtractionConfig()
        self.cache = NoteHashCache()

    @property
    def version(self) -> str:
        return self.VERSION

    def extract(
        self,
        text: str,
        sections: list[Section],
        context: dict[str, Any] | None = None,
    ) -> SlotResult:
        """Extract registry data from text.

        Args:
            text: The procedure note text.
            sections: Pre-parsed sections.
            context: Optional extraction context with hints from hybrid coder:
                - verified_cpt_codes: List of CPT codes from hybrid coder
                - coder_difficulty: Case difficulty classification
                - hybrid_source: Source of codes (ml_rules_fastpath, hybrid_llm_fallback)

        Returns:
            SlotResult with extracted data.
        """
        context = context or {}

        # Filter relevant sections to reduce context window and noise
        relevant_text = self._filter_relevant_text(text, sections)

        # Pass context to prompt builder for CPT code guidance
        prompt = build_registry_prompt(relevant_text, context=context)

        try:
            response = self.llm.generate(prompt, task="registry_extraction")
            # Basic cleanup of markdown code blocks if present
            response = self._clean_response(response)
            data = json.loads(response)
            return SlotResult(value=data, evidence=[], confidence=0.9)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return SlotResult(value=None, evidence=[], confidence=0.0)

    def extract_with_schema(
        self,
        text: str,
        schema: Type[BaseModel],
        sections: list[Section] | None = None,
    ) -> SlotResult:
        """Extract registry data with Pydantic schema validation and self-correction.

        Args:
            text: The procedure note text.
            schema: Pydantic model class for validation.
            sections: Optional pre-parsed sections (to filter text).

        Returns:
            SlotResult with extracted data and confidence.
        """
        # Check cache first
        note_hash = hash_text(text)
        schema_name = schema.__name__

        cached = self.cache.get(note_hash, schema_name)
        if cached is not None:
            obs_logger.debug("Cache hit", extra={"note_hash": note_hash, "schema": schema_name})
            return cached

        # Filter text if sections provided
        if sections:
            relevant_text = self._filter_relevant_text(text, sections)
        else:
            relevant_text = text

        # Run extraction with self-correction loop
        with timed("llm_extractor.extract_with_schema") as timing:
            result = self._extract_with_correction(relevant_text, schema)

        # Build SlotResult
        slot_result = SlotResult(
            value=result.value,
            evidence=[],
            confidence=result.confidence,
        )

        # Cache successful results
        if result.value is not None:
            self.cache.set(note_hash, schema_name, slot_result)

        obs_logger.info(
            "Extraction complete",
            extra={
                "note_hash": note_hash,
                "schema": schema_name,
                "attempts": len(result.attempts),
                "success": result.value is not None,
                "confidence": result.confidence,
                "elapsed_ms": timing.elapsed_ms,
            },
        )

        return slot_result

    def _extract_with_correction(
        self,
        text: str,
        schema: Type[BaseModel],
    ) -> ExtractionResult:
        """Run the extraction with self-correction loop.

        Args:
            text: The text to extract from.
            schema: Pydantic model for validation.

        Returns:
            ExtractionResult with attempts and final value.
        """
        result = ExtractionResult(
            note_hash=hash_text(text),
            schema_name=schema.__name__,
        )

        prompt = self._build_extraction_prompt(text, schema)

        # Build Gemini response_schema from Pydantic model for structured output
        # This helps the LLM adhere to enum constraints at the API level
        response_schema = self._build_gemini_schema(schema)

        for attempt_num in range(self.config.max_retries + 1):
            start_time = time.perf_counter()

            try:
                response = self.llm.generate(prompt, response_schema=response_schema, task="registry_extraction")
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Clean and parse response
                cleaned = self._clean_response(response)
                data = json.loads(cleaned)

                # Validate against schema
                validated = schema.model_validate(data)

                # Success!
                attempt = ExtractionAttempt(
                    attempt_number=attempt_num + 1,
                    response_text=response,
                    parsed_data=validated.model_dump(),
                    elapsed_ms=elapsed_ms,
                    success=True,
                )
                result.attempts.append(attempt)

                # Fast path: skip correction loop if first attempt is high confidence
                if attempt_num == 0:
                    result.confidence = self.config.fast_path_confidence_threshold
                else:
                    # Degrade confidence with each retry
                    result.confidence = max(0.5, 0.95 - (attempt_num * 0.1))

                result.value = validated.model_dump()
                return result

            except json.JSONDecodeError as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                attempt = ExtractionAttempt(
                    attempt_number=attempt_num + 1,
                    response_text=response if 'response' in locals() else "",
                    validation_error=f"JSON parse error: {e}",
                    elapsed_ms=elapsed_ms,
                )
                result.attempts.append(attempt)

                # Build correction prompt for next attempt
                if attempt_num < self.config.max_retries:
                    prompt = self._build_correction_prompt(prompt, response if 'response' in locals() else "", str(e))

            except ValidationError as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                attempt = ExtractionAttempt(
                    attempt_number=attempt_num + 1,
                    response_text=response if 'response' in locals() else "",
                    parsed_data=data if 'data' in locals() else None,
                    validation_error=str(e),
                    elapsed_ms=elapsed_ms,
                )
                result.attempts.append(attempt)

                # Build correction prompt for next attempt
                if attempt_num < self.config.max_retries:
                    prompt = self._build_correction_prompt(prompt, response if 'response' in locals() else "", str(e))

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                attempt = ExtractionAttempt(
                    attempt_number=attempt_num + 1,
                    response_text=response if 'response' in locals() else "",
                    validation_error=f"Unexpected error: {e}",
                    elapsed_ms=elapsed_ms,
                )
                result.attempts.append(attempt)
                logger.error(f"LLM extraction attempt {attempt_num + 1} failed: {e}")

        # All attempts failed
        result.confidence = 0.0
        return result

    def _filter_relevant_text(self, text: str, sections: list[Section]) -> str:
        """Filter text to relevant sections."""
        relevant_text = ""
        target_headers = ["DESCRIPTION", "PROCEDURE", "TECHNIQUE", "FINDINGS", "IMPRESSION"]

        for section in sections:
            if any(h in section.title.upper() for h in target_headers):
                relevant_text += f"\n\n{section.title}:\n{section.text}"

        # If no relevant sections found, fall back to full text
        if not relevant_text.strip():
            relevant_text = text

        return relevant_text

    def _clean_response(self, response: str) -> str:
        """Clean LLM response by removing markdown code blocks."""
        # Remove markdown code block wrappers
        response = re.sub(r"^```json\s*", "", response, flags=re.MULTILINE)
        response = re.sub(r"^```\s*$", "", response, flags=re.MULTILINE)
        response = re.sub(r"```$", "", response)
        return response.strip()

    def _build_gemini_schema(self, schema: Type[BaseModel]) -> dict | None:
        """Build Gemini-compatible response_schema from Pydantic model.

        Gemini's structured output feature enforces schema constraints at the API level,
        improving consistency for enum fields, required fields, and type constraints.

        Returns:
            A dict suitable for Gemini's response_schema parameter, or None if unavailable.
        """
        try:
            json_schema = schema.model_json_schema()

            # Gemini expects a specific format - convert from JSON Schema
            # The schema should have 'type', 'properties', etc.
            gemini_schema = self._convert_json_schema_to_gemini(json_schema)
            return gemini_schema
        except Exception as e:
            logger.warning(f"Failed to build Gemini schema, falling back to prompt-only: {e}")
            return None

    def _convert_json_schema_to_gemini(self, json_schema: dict) -> dict:
        """Convert JSON Schema to Gemini's response_schema format.

        Gemini uses a subset of JSON Schema. This method handles:
        - Type mappings
        - Enum constraints
        - Nested objects and arrays
        - $defs/definitions resolution
        """
        defs = json_schema.get("$defs", json_schema.get("definitions", {}))

        def resolve_ref(schema_part: dict) -> dict:
            """Resolve $ref references."""
            if "$ref" in schema_part:
                ref_path = schema_part["$ref"]
                # Handle "#/$defs/ModelName" or "#/definitions/ModelName"
                if ref_path.startswith("#/$defs/"):
                    ref_name = ref_path.split("/")[-1]
                    return resolve_ref(defs.get(ref_name, {}))
                elif ref_path.startswith("#/definitions/"):
                    ref_name = ref_path.split("/")[-1]
                    return resolve_ref(defs.get(ref_name, {}))
            return schema_part

        def convert_property(prop: dict) -> dict:
            """Convert a single property to Gemini format."""
            prop = resolve_ref(prop)

            # Handle anyOf/oneOf (used for Optional types)
            if "anyOf" in prop:
                # Find the non-null type
                for option in prop["anyOf"]:
                    if option.get("type") != "null":
                        return convert_property(option)
                return {"type": "STRING", "nullable": True}

            prop_type = prop.get("type", "string")
            result: dict = {}

            if prop_type == "string":
                result["type"] = "STRING"
                if "enum" in prop:
                    result["enum"] = prop["enum"]
            elif prop_type == "integer":
                result["type"] = "INTEGER"
            elif prop_type == "number":
                result["type"] = "NUMBER"
            elif prop_type == "boolean":
                result["type"] = "BOOLEAN"
            elif prop_type == "array":
                result["type"] = "ARRAY"
                if "items" in prop:
                    result["items"] = convert_property(prop["items"])
            elif prop_type == "object":
                result["type"] = "OBJECT"
                if "properties" in prop:
                    result["properties"] = {
                        k: convert_property(v)
                        for k, v in prop["properties"].items()
                    }
            else:
                result["type"] = "STRING"

            if prop.get("description"):
                result["description"] = prop["description"]

            return result

        # Convert the root schema
        gemini_schema = {
            "type": "OBJECT",
            "properties": {}
        }

        for prop_name, prop_def in json_schema.get("properties", {}).items():
            gemini_schema["properties"][prop_name] = convert_property(prop_def)

        if "required" in json_schema:
            gemini_schema["required"] = json_schema["required"]

        return gemini_schema

    def _build_extraction_prompt(self, text: str, schema: Type[BaseModel]) -> str:
        """Build the extraction prompt with schema information."""
        schema_json = schema.model_json_schema()

        return f"""Extract structured data from this procedure note according to the schema below.

Return ONLY valid JSON that conforms to the schema. No other text.

Schema:
{json.dumps(schema_json, indent=2)}

Procedure Note:
{text[:8000]}

JSON Output:"""

    def _build_correction_prompt(
        self,
        original_prompt: str,
        previous_response: str,
        error_message: str,
    ) -> str:
        """Build a correction prompt after a failed attempt with specific guidance."""
        # Parse the error to provide targeted guidance
        guidance = self._get_error_guidance(error_message)

        return f"""{original_prompt}

VALIDATION ERROR - Your previous response failed validation:
{error_message}

{guidance}

Previous response (truncated):
{previous_response[:800]}

IMPORTANT: Return ONLY the corrected JSON with the exact enum values from the schema. Do not add explanatory text."""

    def _get_error_guidance(self, error_message: str) -> str:
        """Generate specific guidance based on the error type."""
        guidance_parts = []

        # Check for common enum/literal errors
        if "literal_error" in error_message.lower() or "input should be" in error_message.lower():
            guidance_parts.append(
                "ENUM ERROR: You used an invalid value. Use ONLY the exact values listed in the schema. "
                "For example, 'rebus_view' must be exactly one of: \"Concentric\", \"Eccentric\", \"Adjacent\", \"Not visualized\" - "
                "NOT descriptive phrases like 'Concentric radial EBUS view of lesion'."
            )

        # Check for null/required field errors
        if "string_type" in error_message.lower() and "none" in error_message.lower():
            guidance_parts.append(
                "REQUIRED FIELD ERROR: A required string field was null. "
                "Fields like 'source_location' must have a value - use the procedure type or 'Unknown' if not documented."
            )

        # Check for type errors
        if "type_error" in error_message.lower():
            guidance_parts.append(
                "TYPE ERROR: A field has the wrong type. Check that numbers are not quoted as strings, "
                "booleans are true/false not \"true\"/\"false\", and arrays use [] not strings."
            )

        if not guidance_parts:
            guidance_parts.append(
                "Review the schema carefully and ensure all values match the expected types and enums exactly."
            )

        return "\n".join(guidance_parts)
