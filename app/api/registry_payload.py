"""Shared registry response payload shaping utilities."""

from __future__ import annotations

from typing import Any

from app.api.adapters.response_adapter import build_v3_evidence_payload
from app.api.normalization import simplify_billing_cpt_codes
from app.api.routes_registry import _prune_none
from app.common.spans import Span
from app.registry.schema import RegistryRecord
from app.registry.summarize import add_procedure_summaries


def shape_registry_payload(
    record: RegistryRecord,
    evidence: dict[str, list[Span]] | None,
    *,
    codes: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a registry record + evidence into a JSON-safe, null-pruned payload."""
    payload = _prune_none(record.model_dump(exclude_none=True))
    simplify_billing_cpt_codes(payload)
    add_procedure_summaries(payload)
    payload["evidence"] = build_v3_evidence_payload(
        record=record,
        evidence=evidence,
        codes=codes,
    )
    return payload


__all__ = ["shape_registry_payload"]
