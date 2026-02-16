"""Structurer-first extraction wiring (optional; behind feature flag).

This provides an experimental extraction engine (`REGISTRY_EXTRACTION_ENGINE=agents_structurer`)
that uses the LLM-backed V3 event-log extractor and projects it into the
dynamic `RegistryRecord` shape via the `v3_to_v2` adapter.

Guardrails:
- If an LLM provider is not configured, raise NotImplementedError so the
  registry service falls back to the deterministic engine.
- Evidence spans are attached (best-effort) to the performed flags that map
  cleanly from the V3 event types.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

from app.common.spans import Span
from app.registry.schema import RegistryRecord


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _llm_configured() -> bool:
    if _truthy_env("REGISTRY_USE_STUB_LLM") or _truthy_env("GEMINI_OFFLINE"):
        return False

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "openai_compat":
        if _truthy_env("OPENAI_OFFLINE"):
            return False
        if not os.getenv("OPENAI_API_KEY"):
            return False
        model = (os.getenv("OPENAI_MODEL_STRUCTURER") or os.getenv("OPENAI_MODEL") or "").strip()
        if not model:
            return False
        return True

    # Default: Gemini
    return bool((os.getenv("GEMINI_API_KEY") or "").strip())


_NON_ALNUM_RE = re.compile(r"[^a-z0-9_]+")


def _event_type_key(value: str | None) -> str:
    raw = (value or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    raw = _NON_ALNUM_RE.sub("", raw)
    return raw


_TYPE_TO_PERFORMED_FIELD: dict[str, str] = {
    "diagnostic": "procedures_performed.diagnostic_bronchoscopy.performed",
    "diagnostic_bronchoscopy": "procedures_performed.diagnostic_bronchoscopy.performed",
    "bal": "procedures_performed.bal.performed",
    "brushing": "procedures_performed.brushings.performed",
    "brushings": "procedures_performed.brushings.performed",
    "endobronchial_biopsy": "procedures_performed.endobronchial_biopsy.performed",
    "endobronchialbiopsy": "procedures_performed.endobronchial_biopsy.performed",
    "tbna": "procedures_performed.tbna_conventional.performed",
    "tbna_conventional": "procedures_performed.tbna_conventional.performed",
    "ebus_tbna": "procedures_performed.linear_ebus.performed",
    "linear_ebus": "procedures_performed.linear_ebus.performed",
    "ebus_inspection": "procedures_performed.linear_ebus.performed",
    "radial_ebus": "procedures_performed.radial_ebus.performed",
    "navigation": "procedures_performed.navigational_bronchoscopy.performed",
    "navigational_bronchoscopy": "procedures_performed.navigational_bronchoscopy.performed",
    "tbbx": "procedures_performed.transbronchial_biopsy.performed",
    "transbronchial_biopsy": "procedures_performed.transbronchial_biopsy.performed",
    "cryobiopsy": "procedures_performed.transbronchial_cryobiopsy.performed",
    "transbronchial_cryobiopsy": "procedures_performed.transbronchial_cryobiopsy.performed",
    "therapeutic_aspiration": "procedures_performed.therapeutic_aspiration.performed",
    "airway_dilation": "procedures_performed.airway_dilation.performed",
    "recanalizationdilation": "procedures_performed.airway_dilation.performed",
    "airway_stent": "procedures_performed.airway_stent.performed",
    "stent": "procedures_performed.airway_stent.performed",
    "blvr": "procedures_performed.blvr.performed",
    "thoracentesis": "pleural_procedures.thoracentesis.performed",
    "chest_tube": "pleural_procedures.chest_tube.performed",
    "ipc": "pleural_procedures.ipc.performed",
    "pleurx": "pleural_procedures.ipc.performed",
    "pleurodesis": "pleural_procedures.pleurodesis.performed",
    "fibrinolytic_therapy": "pleural_procedures.fibrinolytic_therapy.performed",
}


def structure_note_to_registry_record(
    note_text: str,
    *,
    note_id: str | None = None,
) -> tuple[RegistryRecord, dict[str, Any]]:
    if not _llm_configured():
        raise NotImplementedError(
            "REGISTRY_EXTRACTION_ENGINE=agents_structurer unavailable (LLM not configured/offline); "
            "falling back to deterministic RegistryEngine extraction"
        )

    from app.registry.pipelines.v3_pipeline import run_v3_extraction
    from app.registry.schema.adapters.v3_to_v2 import convert_v3_to_v2

    v3 = run_v3_extraction(note_text)
    record = convert_v3_to_v2(v3)

    evidence = getattr(record, "evidence", None)
    if not isinstance(evidence, dict):
        evidence = {}

    type_counts: Counter[str] = Counter()
    for event in v3.procedures:
        typ = _event_type_key(getattr(event, "type", None))
        if typ:
            type_counts[typ] += 1
        field_path = _TYPE_TO_PERFORMED_FIELD.get(typ)
        if not field_path:
            continue

        ev = getattr(event, "evidence", None)
        if ev is None:
            continue
        start = getattr(ev, "start", None)
        end = getattr(ev, "end", None)
        if start is None or end is None:
            continue
        try:
            start_val = int(start)
            end_val = int(end)
        except (TypeError, ValueError):
            continue
        if start_val < 0 or end_val <= start_val or end_val > len(note_text):
            continue
        text = (getattr(ev, "quote", None) or "").strip()
        if not text:
            text = note_text[start_val:end_val]
        evidence.setdefault(field_path, []).append(Span(text=text, start=start_val, end=end_val, confidence=0.9))

    record.evidence = evidence

    meta: dict[str, Any] = {
        "status": "ok",
        "note_id": note_id,
        "v3_event_count": len(v3.procedures),
        "v3_event_types": dict(type_counts),
    }
    if not note_id:
        meta.pop("note_id", None)

    return record, meta


__all__ = ["structure_note_to_registry_record"]
