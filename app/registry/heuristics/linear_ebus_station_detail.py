"""Linear EBUS station detail heuristic extraction."""

from __future__ import annotations

import re
from typing import Any

from app.registry.constants.warnings import ebus_station_detail_parsed
from app.registry.schema import RegistryRecord


def apply_linear_ebus_station_detail_heuristics(
    note_text: str,
    record_in: RegistryRecord,
) -> tuple[RegistryRecord, list[str]]:
    if record_in is None:
        return RegistryRecord(), []

    text = note_text or ""
    if not re.search(r"(?i)\b(?:ebus|endobronchial\s+ultrasound|ebus-tbna)\b", text):
        return record_in, []

    from app.registry.processing.linear_ebus_stations_detail import (
        extract_linear_ebus_stations_detail,
    )

    parsed = extract_linear_ebus_stations_detail(text)
    if not parsed:
        return record_in, []

    evidence_by_station: dict[str, dict[str, dict[str, Any]]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        station = str(item.get("station") or "").strip()
        if not station:
            continue

        for field, meta_key in (
            ("lymphocytes_present", "_lymphocytes_present_evidence"),
            ("needle_gauge", "_needle_gauge_evidence"),
            ("number_of_passes", "_number_of_passes_evidence"),
            ("short_axis_mm", "_short_axis_mm_evidence"),
            ("long_axis_mm", "_long_axis_mm_evidence"),
        ):
            ev = item.get(meta_key)
            if not isinstance(ev, dict):
                continue
            start = ev.get("start")
            end = ev.get("end")
            snippet = ev.get("text")
            if start is None or end is None or not snippet:
                continue
            try:
                evidence_by_station.setdefault(station, {})[field] = {
                    "start": int(start),
                    "end": int(end),
                    "text": str(snippet),
                }
            except Exception:  # noqa: BLE001
                continue

    record_data = record_in.model_dump()
    granular = record_data.get("granular_data")
    if granular is None or not isinstance(granular, dict):
        granular = {}

    evidence = record_data.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        record_data["evidence"] = evidence

    existing_raw = granular.get("linear_ebus_stations_detail")
    existing: list[dict[str, Any]] = []
    if isinstance(existing_raw, list):
        existing = [dict(item) for item in existing_raw if isinstance(item, dict)]

    by_station: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in existing:
        station = str(item.get("station") or "").strip()
        if not station:
            continue
        if station not in by_station:
            order.append(station)
        by_station[station] = item

    for item in parsed:
        if not isinstance(item, dict):
            continue
        station = str(item.get("station") or "").strip()
        if not station:
            continue
        existing_item = by_station.get(station)
        if existing_item is None:
            by_station[station] = dict(item)
            order.append(station)
            continue
        updated = False
        for key, value in item.items():
            if key == "station":
                continue
            if value in (None, "", [], {}):
                continue
            if existing_item.get(key) in (None, "", [], {}):
                existing_item[key] = value
                updated = True
        if updated:
            by_station[station] = existing_item

    merged = [by_station[station] for station in order if station in by_station]

    try:
        from app.common.spans import Span

        for idx, item in enumerate(merged):
            if not isinstance(item, dict):
                continue
            station = str(item.get("station") or "").strip()
            if not station:
                continue

            station_evidence = evidence_by_station.get(station) or {}

            def _add_field_evidence(field: str) -> None:
                ev = station_evidence.get(field)
                if not ev:
                    return
                key = f"granular_data.linear_ebus_stations_detail.{idx}.{field}"
                if evidence.get(key):
                    return
                evidence.setdefault(key, []).append(
                    Span(
                        text=str(ev.get("text") or ""),
                        start=int(ev.get("start") or 0),
                        end=int(ev.get("end") or 0),
                        confidence=0.9,
                    )
                )

            lymph = item.get("lymphocytes_present")
            if lymph in (True, False):
                _add_field_evidence("lymphocytes_present")

            gauge = item.get("needle_gauge")
            if isinstance(gauge, int):
                _add_field_evidence("needle_gauge")

            passes = item.get("number_of_passes")
            if isinstance(passes, int):
                _add_field_evidence("number_of_passes")

            short_axis = item.get("short_axis_mm")
            if isinstance(short_axis, (int, float)):
                _add_field_evidence("short_axis_mm")

            long_axis = item.get("long_axis_mm")
            if isinstance(long_axis, (int, float)):
                _add_field_evidence("long_axis_mm")
    except Exception:  # noqa: BLE001
        pass

    granular["linear_ebus_stations_detail"] = merged

    record_data["granular_data"] = granular
    record_out = RegistryRecord(**record_data)
    return record_out, [ebus_station_detail_parsed(len(parsed))]


class LinearEbusStationDetailHeuristic:
    def apply(self, note_text: str, record: RegistryRecord) -> tuple[RegistryRecord, list[str]]:
        return apply_linear_ebus_station_detail_heuristics(note_text, record)


__all__ = [
    "LinearEbusStationDetailHeuristic",
    "apply_linear_ebus_station_detail_heuristics",
]
