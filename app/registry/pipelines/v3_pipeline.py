from __future__ import annotations

import os

from app.registry.schema.ip_v3_extraction import IPRegistryV3


def run_v3_extraction(note_text: str) -> IPRegistryV3:
    from app.registry.processing.focus import get_procedure_focus
    from app.registry.extractors.v3_extractor import extract_v3_draft, repair_v3_evidence_quotes
    from app.registry.deterministic.anatomy import (
        extract_deterministic_anatomy,
        extract_volume_anchors,
        to_prompt_payload,
    )
    from app.registry.evidence.verifier import verify_registry

    focused = get_procedure_focus(note_text)
    prompt_context = to_prompt_payload(
        anatomy=extract_deterministic_anatomy(focused),
        volumes=extract_volume_anchors(focused),
    )
    draft = extract_v3_draft(focused, prompt_context=prompt_context)
    final = verify_registry(draft, note_text)

    repair_enabled = os.getenv("REGISTRY_V3_QUOTE_REPAIR_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    if repair_enabled and len(final.procedures) < len(draft.procedures):
        try:
            repaired = repair_v3_evidence_quotes(draft, note_text=note_text)
            repaired_final = verify_registry(repaired, note_text)
            if len(repaired_final.procedures) >= len(final.procedures):
                return repaired_final
        except Exception:
            # Best-effort: if quote repair fails, keep the original verified output.
            pass

    return final


__all__ = ["run_v3_extraction"]
