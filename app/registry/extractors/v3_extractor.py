from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING, TypeVar

import httpx
from pydantic import BaseModel

from app.common.exceptions import LLMError
from app.common.model_capabilities import filter_payload_for_model, is_gpt5
from app.common.logger import get_logger
from app.infra.llm_control import backoff_seconds, llm_slot, parse_retry_after_seconds
from app.registry.schema.ip_v3_extraction import IPRegistryV3

logger = get_logger("registry.v3_extractor")

TModel = TypeVar("TModel", bound=BaseModel)

if TYPE_CHECKING:  # pragma: no cover
    from app.common.llm import DeterministicStubLLM, GeminiLLM, OpenAILLM


SYSTEM_PROMPT = (
    "You are an expert clinical information extraction system.\n"
    "Extract interventional pulmonology events.\n"
    "You MUST provide a verbatim evidence_quote for every event.\n"
    "The focused note text may include <primary_narrative> and <supporting_data> blocks.\n"
    "You MUST prefer evidence quotes from <primary_narrative>.\n"
    "Only use <supporting_data> when the procedure is not described in the narrative.\n"
    "Output JSON only."
)


def extract_v3_draft(focused_text: str, *, prompt_context: dict[str, Any] | None = None) -> IPRegistryV3:
    schema = IPRegistryV3.model_json_schema()

    context_block = ""
    if prompt_context:
        context_block = (
            "Deterministic anchors (use these to populate target/measurements when mentioned):\n"
            f"{json.dumps(prompt_context, indent=2)}\n\n"
        )

    user_prompt = (
        "Extract interventional pulmonology events from the focused note text.\n\n"
        "Return ONLY valid JSON that conforms to the provided schema.\n"
        "You MUST provide a verbatim evidence_quote for every event.\n\n"
        "Granularity requirements:\n"
        "- If a procedure is documented at a specific station/lobe/segment, you MUST populate ProcedureEvent.target.\n"
        "- If BAL includes instilled/returned volumes, you MUST populate ProcedureEvent.measurements.\n"
        "- If linear EBUS uses elastography, capture the specific pattern (e.g., Type 1/2/3, blue pattern) in findings when present.\n"
        "- When the same procedure occurs at multiple stations/lobes/segments, emit separate events per distinct target.\n\n"
        f"{context_block}"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Focused note text:\n{focused_text}\n"
    )

    # Use a timeout profile appropriate for registry extraction (see `_resolve_openai_timeout`).
    llm = _resolve_llm(task="registry_extraction")
    raw = _generate_structured_json(
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=IPRegistryV3,
        response_json_schema=schema,
    )
    return IPRegistryV3.model_validate(raw)


def _resolve_llm(*, task: str | None) -> "GeminiLLM | OpenAILLM | DeterministicStubLLM":
    # Import lazily so `REGISTRY_USE_STUB_LLM=1` can be honored even when the repo's
    # dotenv loader would otherwise override env vars at import time.
    from app.common.llm import DeterministicStubLLM, GeminiLLM, OpenAILLM

    use_stub = os.getenv("REGISTRY_USE_STUB_LLM", "").strip().lower() in ("1", "true", "yes")
    use_stub = use_stub or os.getenv("GEMINI_OFFLINE", "").strip().lower() in ("1", "true", "yes")

    if use_stub:
        return DeterministicStubLLM(payload={"note_id": "unknown", "source_filename": "unknown", "schema_version": "v3", "procedures": []})

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "openai_compat":
        openai_offline = os.getenv("OPENAI_OFFLINE", "").strip().lower() in ("1", "true", "yes") or not bool(os.getenv("OPENAI_API_KEY"))
        if openai_offline:
            return DeterministicStubLLM(payload={"note_id": "unknown", "source_filename": "unknown", "schema_version": "v3", "procedures": []})
        return OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL"), task=task)

    if not os.getenv("GEMINI_API_KEY"):
        return DeterministicStubLLM(payload={"note_id": "unknown", "source_filename": "unknown", "schema_version": "v3", "procedures": []})
    return GeminiLLM()


def _generate_structured_json(
    *,
    llm: "GeminiLLM | OpenAILLM | DeterministicStubLLM",
    system_prompt: str,
    user_prompt: str,
    response_model: type[TModel],
    response_json_schema: dict[str, Any],
) -> dict[str, Any]:
    from app.common.llm import DeterministicStubLLM, GeminiLLM, OpenAILLM

    if isinstance(llm, DeterministicStubLLM):
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n"
        return json.loads(llm.generate(prompt))

    if isinstance(llm, GeminiLLM):
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n"
        gemini_schema = _convert_json_schema_to_gemini(response_json_schema)
        text = llm.generate(prompt, response_schema=gemini_schema, temperature=0.0)
        return _parse_json_model(text, response_model=response_model)

    # OpenAI-compatible Chat Completions JSON schema when supported; prompt-only fallback otherwise.
    if isinstance(llm, OpenAILLM) and is_gpt5(llm.model):
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n"
        text = llm.generate(prompt, temperature=0.0)
        return _parse_json_model(text, response_model=response_model)

    return _openai_chat_json_schema(
        base_url=llm.base_url,
        api_key=llm.api_key,
        model=llm.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=response_model,
        json_schema=response_json_schema,
    )


def _parse_json_model(text: str, *, response_model: type[TModel]) -> dict[str, Any]:
    cleaned = _strip_markdown_code_fences(text)
    if cleaned.strip() in {"", "null", "None"}:
        raise ValueError("LLM returned empty response")
    data = json.loads(cleaned)
    # Validate shape early for clearer error messages.
    response_model.model_validate(data)
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object")
    return data


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


def _openai_chat_json_schema(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[TModel],
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    if not api_key:
        raise LLMError("OPENAI_API_KEY not configured")

    url = f"{(base_url or 'https://api.openai.com').rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": response_model.__name__, "schema": json_schema, "strict": True},
        },
        "temperature": 0.0,
    }
    payload = filter_payload_for_model(payload, model, api_style="chat")

    removed_response_format = "response_format" not in payload
    if removed_response_format:
        # Model doesn't support JSON schema response formatting; fall back to prompt-only JSON.
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        payload = filter_payload_for_model(payload, model, api_style="chat")

    deadline_s = float(os.getenv("REGISTRY_LLM_TIMEOUT_S", "60").strip() or "60")
    deadline = httpx.Timeout(connect=10.0, read=deadline_s, write=30.0, pool=10.0)

    last_error: Exception | None = None
    with httpx.Client(timeout=deadline) as client:
        for attempt in range(3):
            try:
                with llm_slot():
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code < 400:
                    data = resp.json()
                    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                    return _parse_json_model(content, response_model=response_model)

                if resp.status_code in {429} or 500 <= resp.status_code <= 599:
                    retry_after = parse_retry_after_seconds(resp.headers)
                    sleep_s = retry_after if retry_after is not None else backoff_seconds(attempt)
                    # Best-effort backoff (no logging of PHI-bearing prompt).
                    import time

                    time.sleep(sleep_s)
                    continue

                msg = " ".join((resp.text or "").split())
                raise LLMError(f"OpenAI chat error HTTP {resp.status_code}: {msg[:300]}")
            except (httpx.TransportError, httpx.ReadTimeout) as exc:
                last_error = exc
                logger.warning("OpenAI chat transient error attempt=%s error=%s", attempt + 1, type(exc).__name__)
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break

    raise LLMError(f"OpenAI chat failed after retries: {type(last_error).__name__ if last_error else 'unknown'}")


def _convert_json_schema_to_gemini(json_schema: dict[str, Any]) -> dict[str, Any]:
    defs = json_schema.get("$defs", json_schema.get("definitions", {}))

    def resolve_ref(schema_part: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in schema_part:
            ref_path = str(schema_part["$ref"])
            if ref_path.startswith("#/$defs/") or ref_path.startswith("#/definitions/"):
                ref_name = ref_path.split("/")[-1]
                return resolve_ref(defs.get(ref_name, {}))
        return schema_part

    def convert_property(prop: dict[str, Any]) -> dict[str, Any]:
        prop = resolve_ref(prop)

        if "anyOf" in prop:
            for option in prop["anyOf"]:
                if option.get("type") != "null":
                    converted = convert_property(option)
                    converted["nullable"] = True
                    return converted
            return {"type": "STRING", "nullable": True}

        if "oneOf" in prop:
            for option in prop["oneOf"]:
                if option.get("type") != "null":
                    converted = convert_property(option)
                    converted["nullable"] = True
                    return converted
            return {"type": "STRING", "nullable": True}

        prop_type = prop.get("type", "string")
        result: dict[str, Any] = {}

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
        elif prop_type == "object" or "properties" in prop:
            result["type"] = "OBJECT"
            props = prop.get("properties") or {}
            result["properties"] = {k: convert_property(v) for k, v in props.items()}
        else:
            result["type"] = "STRING"

        if prop.get("description"):
            result["description"] = prop["description"]
        return result

    gemini_schema: dict[str, Any] = {"type": "OBJECT", "properties": {}}
    for prop_name, prop_def in (json_schema.get("properties") or {}).items():
        gemini_schema["properties"][prop_name] = convert_property(prop_def)

    if "required" in json_schema:
        gemini_schema["required"] = json_schema["required"]

    return gemini_schema


__all__ = ["extract_v3_draft"]
