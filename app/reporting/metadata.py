from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Literal, TypedDict


class ProcedureAutocodeResult(TypedDict, total=False):
    cpt: list[str]
    modifiers: list[str]
    icd: list[str]
    notes: str


@dataclass
class ProcedureMetadata:
    proc_id: str
    proc_type: str
    label: str
    cpt_candidates: list[str]
    icd_candidates: list[str]
    modifiers: list[str]
    section: str
    templates_used: list[str]
    has_critical_missing: bool
    missing_critical_fields: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportMetadata:
    patient_id: str | None
    mrn: str | None
    encounter_id: str | None
    date_of_procedure: dt.date | None
    attending: str | None
    location: str | None
    procedures: list[ProcedureMetadata]
    autocode_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredReport:
    text: str
    metadata: ReportMetadata
    warnings: list[str] = field(default_factory=list)
    issues: list["MissingFieldIssue"] = field(default_factory=list)


@dataclass
class MissingFieldIssue:
    proc_id: str
    proc_type: str
    template_id: str
    field_path: str
    severity: Literal["warning", "recommended"]
    message: str


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclass objects to JSON-friendly primitives."""
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if is_dataclass(obj):
        return {key: _serialize(val) for key, val in asdict(obj).items()}
    if isinstance(obj, dict):
        return {key: _serialize(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize(val) for val in obj]
    return obj


def metadata_to_dict(metadata: ReportMetadata) -> dict[str, Any]:
    """Convert ReportMetadata to a JSON-serializable dict."""
    return _serialize(metadata)


__all__ = [
    "ProcedureAutocodeResult",
    "ProcedureMetadata",
    "ReportMetadata",
    "StructuredReport",
    "MissingFieldIssue",
    "metadata_to_dict",
]
