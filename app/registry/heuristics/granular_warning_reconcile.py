"""Postprocess helpers for granular validation warning cleanup."""

from __future__ import annotations

from app.registry.constants import (
    BRONCHIAL_WASH_PERFORMED,
    BRUSHINGS_PERFORMED,
    LINEAR_EBUS_PERFORMED,
    PERIPHERAL_TBNA_PERFORMED,
    TBNA_CONVENTIONAL_PERFORMED,
    TRANSBRONCHIAL_BIOPSY_PERFORMED,
)
from app.registry.schema import RegistryRecord


def reconcile_granular_validation_warnings(
    record_in: RegistryRecord,
) -> tuple[RegistryRecord, set[str]]:
    """Drop stale granular warnings after postprocess flips performed flags."""
    warnings_in = getattr(record_in, "granular_validation_warnings", None)
    if not isinstance(warnings_in, list) or not warnings_in:
        return record_in, set()

    procs = getattr(record_in, "procedures_performed", None)

    def _performed(proc_name: str) -> bool:
        if procs is None:
            return False
        proc = getattr(procs, proc_name, None)
        if proc is None:
            return False
        return bool(getattr(proc, "performed", False))

    linear_performed = _performed("linear_ebus")
    tbna_performed = _performed("tbna_conventional")
    peripheral_tbna_performed = _performed("peripheral_tbna")
    brushings_performed = _performed("brushings")
    tbbx_performed = _performed("transbronchial_biopsy")
    bronchial_wash_performed = _performed("bronchial_wash")

    removed: set[str] = set()
    cleaned: list[str] = []
    seen: set[str] = set()
    for warning in warnings_in:
        if not isinstance(warning, str) or not warning.strip():
            continue
        if LINEAR_EBUS_PERFORMED in warning and not linear_performed:
            removed.add(warning)
            continue
        if TBNA_CONVENTIONAL_PERFORMED in warning and not tbna_performed:
            removed.add(warning)
            continue
        if PERIPHERAL_TBNA_PERFORMED in warning and not peripheral_tbna_performed:
            removed.add(warning)
            continue
        if BRUSHINGS_PERFORMED in warning and not brushings_performed:
            removed.add(warning)
            continue
        if TRANSBRONCHIAL_BIOPSY_PERFORMED in warning and not tbbx_performed:
            removed.add(warning)
            continue
        if BRONCHIAL_WASH_PERFORMED in warning and not bronchial_wash_performed:
            removed.add(warning)
            continue
        if warning in seen:
            continue
        seen.add(warning)
        cleaned.append(warning)

    if not removed and len(cleaned) == len(warnings_in):
        return record_in, set()

    record_data = record_in.model_dump()
    record_data["granular_validation_warnings"] = cleaned
    return RegistryRecord(**record_data), removed


__all__ = ["reconcile_granular_validation_warnings"]
