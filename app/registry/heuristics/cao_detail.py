"""CAO intervention detail heuristic extraction."""

from __future__ import annotations

import re
from typing import Any

from app.registry.constants.warnings import cao_detail_parsed
from app.registry.schema import RegistryRecord


def apply_cao_detail_heuristics(
    note_text: str,
    record_in: RegistryRecord,
) -> tuple[RegistryRecord, list[str]]:
    if record_in is None:
        return RegistryRecord(), []

    text = note_text or ""
    if not re.search(
        r"(?i)\b(?:"
        r"central\s+airway|airway\s+obstruct\w*|"
        r"rigid\s+bronchos\w*|"
        r"debulk\w*|"
        r"tumou?r\s+(?:ablation|destruction)|"
        r"stent"
        r")\b",
        text,
    ):
        return record_in, []

    from app.registry.processing.cao_interventions_detail import (
        extract_cao_interventions_detail,
    )

    parsed = extract_cao_interventions_detail(text)
    if not parsed:
        return record_in, []

    record_data = record_in.model_dump()
    granular = record_data.get("granular_data")
    if granular is None or not isinstance(granular, dict):
        granular = {}

    existing_raw = granular.get("cao_interventions_detail")
    existing: list[dict[str, Any]] = []
    if isinstance(existing_raw, list):
        existing = [dict(item) for item in existing_raw if isinstance(item, dict)]

    def _key(item: dict[str, Any]) -> str:
        return str(item.get("location") or "").strip()

    by_loc: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in existing:
        loc = _key(item)
        if not loc:
            continue
        if loc not in by_loc:
            order.append(loc)
        by_loc[loc] = item

    for item in parsed:
        if not isinstance(item, dict):
            continue
        loc = _key(item)
        if not loc:
            continue
        existing_item = by_loc.get(loc)
        if existing_item is None:
            by_loc[loc] = dict(item)
            order.append(loc)
            continue

        updated = False
        for key, value in item.items():
            if key == "location":
                continue
            if value in (None, "", [], {}):
                continue
            if key == "modalities_applied":
                existing_apps = existing_item.get("modalities_applied")
                if not isinstance(existing_apps, list):
                    existing_apps = []
                existing_mods = {
                    str(app.get("modality"))
                    for app in existing_apps
                    if isinstance(app, dict) and app.get("modality")
                }
                new_apps = []
                if isinstance(value, list):
                    for app in value:
                        if not isinstance(app, dict):
                            continue
                        mod = app.get("modality")
                        if not mod or str(mod) in existing_mods:
                            continue
                        new_apps.append(app)
                if new_apps:
                    existing_apps.extend(new_apps)
                    existing_item["modalities_applied"] = existing_apps
                    updated = True
                continue
            if existing_item.get(key) in (None, "", [], {}):
                existing_item[key] = value
                updated = True
        if updated:
            by_loc[loc] = existing_item

    merged = [by_loc[loc] for loc in order if loc in by_loc]
    granular["cao_interventions_detail"] = merged

    record_data["granular_data"] = granular
    record_out = RegistryRecord(**record_data)
    return record_out, [cao_detail_parsed(len(parsed))]


class CaoDetailHeuristic:
    def apply(self, note_text: str, record: RegistryRecord) -> tuple[RegistryRecord, list[str]]:
        return apply_cao_detail_heuristics(note_text, record)


__all__ = ["CaoDetailHeuristic", "apply_cao_detail_heuristics"]
