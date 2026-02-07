"""LLM-backed extraction of EBUS lymph node evidence."""

from __future__ import annotations

import json
import logging
import os
from typing import List

from app.common.llm import DeterministicStubLLM, GeminiLLM
from app.coder.types import EBUSNodeEvidence
from app.coder.sectionizer import (
    accordion_truncate,
    max_llm_input_tokens,
    sectionizer_enabled,
)

logger = logging.getLogger(__name__)


class EBUSEvidenceExtractor:
    """Uses an LLM to extract structured EBUS node tuples."""

    def __init__(self, llm: GeminiLLM | DeterministicStubLLM | None = None):
        self._llm = llm or self._create_llm()

    def _create_llm(self):
        use_stub = os.getenv("GEMINI_OFFLINE", "").lower() in ("1", "true", "yes")
        if use_stub:
            return DeterministicStubLLM(payload=[])
        return GeminiLLM()

    def extract(self, scrubbed_text: str) -> List[EBUSNodeEvidence]:
        if not scrubbed_text:
            return []

        source_text = scrubbed_text
        if sectionizer_enabled():
            source_text = accordion_truncate(source_text, max_llm_input_tokens())

        prompt = (
            "You are assisting with coding for an EBUS bronchoscopy.\n"
            "Read the following procedure note and output ONLY a JSON array with no explanation.\n"
            "Each object must include:\n"
            "  - \"Station\": lymph node station identifier (e.g., \"7\", \"4R\", \"11L\")\n"
            "  - \"Action\": exactly \"Inspection\" or \"Sampling\"\n"
            "  - \"Method\": \"EBUS-linear\", \"EBUS-radial\", or null if not specified\n"
            "List every station mentioned, even if only inspected.\n\n"
            "Procedure Note:\n"
            f"{source_text}\n\n"
            "JSON:"
        )

        try:
            response = self._llm.generate(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("EBUS evidence extraction failed: %s", exc)
            return []

        try:
            raw = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("EBUS extractor returned non-JSON payload")
            return []

        evidence: list[EBUSNodeEvidence] = []
        if not isinstance(raw, list):
            return evidence

        for item in raw:
            if not isinstance(item, dict):
                continue

            station = str(item.get("Station", "")).strip()
            action = str(item.get("Action", "")).strip()
            method_raw = item.get("Method")
            method = str(method_raw).strip() if method_raw is not None else None
            if method == "":
                method = None

            if not station or action not in ("Inspection", "Sampling"):
                continue

            evidence.append(
                EBUSNodeEvidence(
                    station=station,
                    action="Sampling" if action == "Sampling" else "Inspection",
                    method=method,
                )
            )

        return evidence


__all__ = ["EBUSEvidenceExtractor"]
