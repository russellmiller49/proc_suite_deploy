"""Coverage check helpers for extraction-first fallback logic."""

from __future__ import annotations

import re
from typing import Any, Callable

from app.registry.schema import RegistryRecord


def coverage_failures(note_text: str, record_in: RegistryRecord) -> list[str]:
    failures: list[str] = []
    text = note_text or ""

    ebus_hit = re.search(r"(?i)EBUS[- ]Findings|EBUS Lymph Nodes Sampled|\blinear\s+ebus\b", text)
    ebus_performed = False
    try:
        ebus_obj = (
            record_in.procedures_performed.linear_ebus if record_in.procedures_performed else None
        )
        ebus_performed = bool(getattr(ebus_obj, "performed", False))
    except Exception:  # noqa: BLE001
        ebus_performed = False
    if ebus_hit and not ebus_performed:
        failures.append("linear_ebus missing")

    eus_hit = re.search(
        r"(?i)\bEUS-?B\b|\bleft adrenal\b|\btransgastric\b|\btransesophageal\b",
        text,
    )
    procedures = record_in.procedures_performed if record_in.procedures_performed else None
    if eus_hit and procedures is not None and hasattr(procedures, "eus_b"):
        eus_b_performed = False
        try:
            eus_b_obj = getattr(procedures, "eus_b", None)
            eus_b_performed = bool(getattr(eus_b_obj, "performed", False)) if eus_b_obj else False
        except Exception:  # noqa: BLE001
            eus_b_performed = False
        if not eus_b_performed:
            failures.append("eus_b missing")

    nav_hit = re.search(
        r"(?i)\b(navigational bronchoscopy|robotic bronchoscopy|electromagnetic navigation|\benb\b|\bion\b|monarch|galaxy)\b",
        text,
    )
    if nav_hit:
        target_mentions = len(re.findall(r"(?i)target lesion", text))
        try:
            nav_targets = (
                record_in.granular_data.navigation_targets if record_in.granular_data is not None else None
            )
            nav_count = len(nav_targets or [])
        except Exception:  # noqa: BLE001
            nav_count = 0
        if target_mentions and nav_count < target_mentions:
            failures.append(f"navigation_targets {nav_count} < {target_mentions}")

    return failures


def run_structurer_fallback(
    note_text: str,
    *,
    registry_engine: Any,
    granular_propagator: Callable[[RegistryRecord], tuple[RegistryRecord, list[str]]],
) -> tuple[RegistryRecord | None, list[str]]:
    warnings: list[str] = []
    context: dict[str, Any] = {"schema_version": "v3"}
    try:
        run_with_warnings = getattr(registry_engine, "run_with_warnings", None)
        if callable(run_with_warnings):
            record_out, engine_warnings = run_with_warnings(note_text, context=context)
            warnings.extend(engine_warnings or [])
        else:
            record_out = registry_engine.run(note_text, context=context)
            if isinstance(record_out, tuple):
                record_out = record_out[0]

        record_out, granular_warnings = granular_propagator(record_out)
        warnings.extend(granular_warnings)

        from app.registry.evidence.verifier import verify_evidence_integrity
        from app.registry.postprocess import (
            cull_hollow_ebus_claims,
            populate_ebus_node_events_fallback,
            sanitize_ebus_events,
        )

        record_out, verifier_warnings = verify_evidence_integrity(record_out, note_text)
        warnings.extend(verifier_warnings)
        warnings.extend(sanitize_ebus_events(record_out, note_text))
        warnings.extend(populate_ebus_node_events_fallback(record_out, note_text))
        warnings.extend(cull_hollow_ebus_claims(record_out, note_text))
        return record_out, warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"STRUCTURER_FALLBACK_FAILED: {exc}")
        return None, warnings


__all__ = ["coverage_failures", "run_structurer_fallback"]
