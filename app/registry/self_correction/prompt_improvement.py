"""Self-correction helper for registry extraction prompts and post-processing."""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List

from config.settings import KnowledgeSettings
from app.common.llm import GeminiLLM, OpenAILLM
from app.registry.prompts import FIELD_INSTRUCTIONS

_SCHEMA_PATH = KnowledgeSettings().registry_schema_path


@dataclass
class RegistryErrorExample:
    field_name: str
    gold_value: str
    predicted_value: str
    note_text: str


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def get_allowed_values(field_name: str) -> list[str]:
    schema = _load_schema()
    prop = schema.get("properties", {}).get(field_name, {})
    enum = prop.get("enum") or []
    return enum


def load_errors(path: str, target_field: str, max_examples: int = 20) -> List[RegistryErrorExample]:
    errors: List[RegistryErrorExample] = []
    error_path = Path(path)
    if not error_path.exists():
        return errors

    with error_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("field_name") != target_field:
                continue
            errors.append(
                RegistryErrorExample(
                    field_name=entry.get("field_name", ""),
                    gold_value=str(entry.get("gold_value", "")),
                    predicted_value=str(entry.get("predicted_value", "")),
                    note_text=str(entry.get("note_text", ""))[:600],
                )
            )
            if len(errors) >= max_examples:
                break
    return errors


def build_self_correction_prompt(
    field_name: str,
    instruction_text: str,
    errors: list[RegistryErrorExample],
    allowed_values: list[str],
) -> str:
    allowed_text = ", ".join(allowed_values) if allowed_values else "Any text value"
    lines = [
        f"We are extracting the field '{field_name}' from interventional pulmonology procedure notes.",
        f"Allowed values: {allowed_text}.",
        "Here is the current instruction text being used:",
        instruction_text,
        "",
        "Recent extraction errors:",
    ]

    for idx, ex in enumerate(errors, start=1):
        lines.append(
            f"Example {idx}:\n  Note excerpt: {ex.note_text}\n  Gold: {ex.gold_value}\n  Predicted: {ex.predicted_value}"
        )

    lines.append(
        (
            "Suggest improvements. Return JSON with:\n"
            '{\n  "updated_instruction": "new prompt text for this field",\n'
            '  "python_postprocessing_rules": "Python code snippet that maps raw LLM text to allowed values",\n'
            '  "comments": "brief explanation of what changed and why"\n}'
        )
    )
    return "\n".join(lines)


def suggest_improvements_for_field(
    field_name: str, allowed_values: list[str], max_examples: int = 20
) -> dict:
    errors = load_errors("data/registry_errors.jsonl", field_name, max_examples=max_examples)
    if not errors:
        return {"error": f"No errors found for field '{field_name}'. Run validation first."}

    instruction_text = FIELD_INSTRUCTIONS.get(field_name, "No instruction available.")
    prompt = build_self_correction_prompt(field_name, instruction_text, errors, allowed_values)

    model_name = os.getenv("REGISTRY_SELF_CORRECTION_MODEL", "gpt-5.1")
    llm = OpenAILLM(model=model_name) if model_name.startswith("gpt") else GeminiLLM(model=model_name)

    try:
        raw = llm.generate(prompt)
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        suggestion = json.loads(cleaned)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"LLM suggestion failed: {exc}"}

    return suggestion


__all__ = [
    "RegistryErrorExample",
    "load_errors",
    "build_self_correction_prompt",
    "suggest_improvements_for_field",
    "get_allowed_values",
]
