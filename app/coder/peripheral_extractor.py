"""LLM-backed extraction of peripheral lesion evidence."""

from __future__ import annotations

import json
import os
from typing import Any, List

from app.common.llm import DeterministicStubLLM, GeminiLLM
from app.coder.sectionizer import accordion_truncate, max_llm_input_tokens, sectionizer_enabled
from app.coder.types import PeripheralLesionEvidence


class PeripheralLesionExtractor:
    """Extract structured peripheral lesion data from scrubbed notes."""

    def __init__(self, llm_client: Any | None = None):
        self._llm = llm_client
        self._offline = os.getenv("GEMINI_OFFLINE", "").lower() in ("1", "true", "yes")

    def _ensure_llm(self):
        if self._llm is None:
            if self._offline:
                self._llm = DeterministicStubLLM(payload={"lesions": []})
            else:
                self._llm = GeminiLLM()
        return self._llm

    def _build_prompt(self, note_text: str) -> str:
        return (
            "You are extracting peripheral lung lesion interventions from a bronchoscopy note.\n"
            "Output ONLY JSON matching this structure:\n"
            "{\n  \"lesions\": [\n    {\n      \"Lobe\": \"RUL\"|\"RML\"|\"RLL\"|\"LUL\"|\"LLL\"|null,\n"
            "      \"Segment\": string or null,\n"
            "      \"Actions\": [\"Cryobiopsy\"|\"TBNA\"|\"Brush\"|\"BAL\"|\"Fiducial\"],\n"
            "      \"Navigation\": true|false,\n"
            "      \"RadialEBUS\": true|false\n    }\n  ]\n}\n\n"
            "Include a lesion entry only when sampling/intervention occurred.\n"
            "Set Navigation true when robotic/EMN/Ion guidance used for that lesion.\n"
            "Set RadialEBUS true when a radial probe confirmed lesion position.\n\n"
            "Note:\n"
            f"{note_text}\n\nJSON:"
        )

    def extract(self, scrubbed_text: str) -> List[PeripheralLesionEvidence]:
        if not scrubbed_text:
            return []

        text = scrubbed_text
        if sectionizer_enabled():
            text = accordion_truncate(text, max_llm_input_tokens())

        llm = self._ensure_llm()

        try:
            response = llm.generate(self._build_prompt(text))
        except Exception:
            return []

        try:
            raw = json.loads(response)
        except json.JSONDecodeError:
            return []

        lesions_data = raw.get("lesions") or []
        evidence: list[PeripheralLesionEvidence] = []
        for entry in lesions_data:
            if not isinstance(entry, dict):
                continue
            lobe = entry.get("Lobe")
            if lobe is not None:
                lobe = str(lobe).strip() or None
            segment = entry.get("Segment")
            if segment is not None:
                segment = str(segment).strip() or None
            actions_raw = entry.get("Actions") or []
            actions = [str(a) for a in actions_raw if a]
            navigation = bool(entry.get("Navigation", False))
            radial = bool(entry.get("RadialEBUS", False))

            evidence.append(
                PeripheralLesionEvidence(
                    lobe=lobe,
                    segment=segment,
                    actions=actions,
                    navigation=navigation,
                    radial_ebus=radial,
                )
            )

        return evidence


__all__ = ["PeripheralLesionExtractor"]
