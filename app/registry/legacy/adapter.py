"""Map Procedure Suite outputs to registry-ready dictionaries."""

from __future__ import annotations

from typing import Any, Dict, List

from proc_schemas.billing import BillingResult
from proc_schemas.procedure_report import ProcedureReport

try:  # optional validation if Test_reg repo is installed
    from bronch_schema.models import NodeSampling, StentPlacement  # type: ignore
except ImportError:  # pragma: no cover
    NodeSampling = None  # type: ignore
    StentPlacement = None  # type: ignore


def report_to_registry(report: ProcedureReport, billing: BillingResult) -> Dict[str, Any]:
    lineage = _lineage(report)
    specimens = _specimens(report, lineage)
    devices = _devices(report, lineage)
    complications = _complications(report, lineage)
    billing_lines = [line.model_dump() for line in billing.codes]

    bundle = {
        "bronchoscopy_procedure": {
            "procedure_type": report.procedure_core.type,
            "laterality": report.procedure_core.laterality,
            "stations_sampled": report.procedure_core.stations_sampled,
            "meta": report.meta,
            "indication": report.indication,
            "lineage": lineage,
        },
        "specimens": specimens,
        "devices": devices,
        "complications": complications,
        "billing_lines": billing_lines,
    }
    return bundle


def _lineage(report: ProcedureReport) -> Dict[str, Any]:
    return {
        "paragraph_hashes": report.nlp.paragraph_hashes,
        "cuis": [concept.get("cui") for concept in report.nlp.umls],
    }


def _specimens(report: ProcedureReport, lineage: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for target in report.procedure_core.targets:
        station = target.segment or target.lobe or "unspecified"
        payload = {
            "station": station,
            "specimens": target.specimens,
            "guidance": target.guidance,
            "lineage": lineage,
        }
        if NodeSampling is not None:
            payload["registry_model"] = NodeSampling(
                station=station,
                passes=target.specimens.get("fna"),
            ).model_dump()
        entries.append(payload)
    return entries


def _devices(report: ProcedureReport, lineage: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for key, value in (report.procedure_core.devices or {}).items():
        device_payload = {"device_key": key, "value": value, "lineage": lineage}
        if StentPlacement is not None and key == "stent":
            device_payload["registry_model"] = StentPlacement(
                location=value.get("location", "unknown"),
                type=value.get("type", "metal"),
            ).model_dump()
        entries.append(device_payload)
    return entries


def _complications(report: ProcedureReport, lineage: Dict[str, Any]) -> List[Dict[str, Any]]:
    comp = report.postop.get("complications", {}) if isinstance(report.postop, dict) else {}
    if isinstance(comp, dict):
        if not comp:
            return []
        return [
            {
                "type": key,
                "detail": value,
                "lineage": lineage,
            }
            for key, value in comp.items()
        ]
    return []
