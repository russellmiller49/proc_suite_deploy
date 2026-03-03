"""Apply imaging event extraction updates to canonical registry JSON."""

from __future__ import annotations

from typing import Any

from app.registry.aggregation.locks import (
    assign_if_unlocked,
    ensure_dict,
    ensure_list,
    is_pointer_locked,
    pointer_join,
)
from app.registry.aggregation.sanitize import compact_text


_PERIPHERAL_FIELDS = [
    "target_key",
    "laterality",
    "lobe",
    "segment",
    "size_mm_long",
    "size_mm_short",
    "size_mm_cc",
    "density",
    "pet_avid",
    "pet_suvmax",
    "pet_delayed_suvmax",
    "comparative_change",
]

_MEDIASTINAL_FIELDS = [
    "station",
    "location_text",
    "short_axis_mm",
    "pet_avid",
    "pet_suvmax",
    "pet_delayed_suvmax",
    "comparative_change",
]


def _find_peripheral_target_index(
    items: list[dict[str, Any]],
    *,
    target_key: str | None,
    laterality: str | None,
    lobe: str | None,
    segment: str | None,
) -> int | None:
    if target_key:
        lower_key = target_key.strip().lower()
        for idx, item in enumerate(items):
            if str(item.get("target_key") or "").strip().lower() == lower_key:
                return idx

    for idx, item in enumerate(items):
        if laterality and str(item.get("laterality") or "") != laterality:
            continue
        if lobe and str(item.get("lobe") or "").upper() != lobe.upper():
            continue
        if segment and str(item.get("segment") or "").strip().lower() != segment.strip().lower():
            continue
        if laterality or lobe or segment:
            return idx
    return None


def _find_mediastinal_target_index(
    items: list[dict[str, Any]],
    *,
    station: str | None,
    location_text: str | None,
) -> int | None:
    if station:
        normalized_station = station.strip().upper()
        for idx, item in enumerate(items):
            if str(item.get("station") or "").strip().upper() == normalized_station:
                return idx
    if location_text:
        location_key = location_text.strip().lower()
        for idx, item in enumerate(items):
            if str(item.get("location_text") or "").strip().lower() == location_key:
                return idx
    return None


def _snapshot_key(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        snapshot.get("relative_day_offset"),
        snapshot.get("modality"),
        snapshot.get("subtype"),
        snapshot.get("event_title"),
    )


def patch_imaging_update(
    registry_json: dict[str, Any],
    *,
    extracted: dict[str, Any],
    event_id: str,
    relative_day_offset: int | None,
    event_subtype: str | None,
    event_title: str | None,
    source_modality: str | None,
    manual_overrides: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Patch imaging sections and target upserts while honoring locks."""

    changed = False
    qa_flags = list(extracted.get("qa_flags") or [])

    imaging_summary = ensure_dict(registry_json, "imaging_summary")
    snapshot = dict(extracted.get("imaging_snapshot") or {})
    if snapshot:
        if relative_day_offset is not None:
            snapshot["relative_day_offset"] = int(relative_day_offset)
        if source_modality:
            snapshot["modality"] = str(source_modality).strip().lower()
        if event_subtype:
            snapshot["subtype"] = str(event_subtype).strip().lower()
        if event_title:
            snapshot["event_title"] = compact_text(event_title, max_chars=120)
        snapshot["qa_flags"] = sorted(set(list(snapshot.get("qa_flags") or []) + qa_flags))

        subtype = str(snapshot.get("subtype") or "").lower()
        if subtype == "preop":
            baseline_pointer = pointer_join("imaging_summary", "baseline")
            if not is_pointer_locked(manual_overrides, baseline_pointer):
                if imaging_summary.get("baseline") != snapshot:
                    imaging_summary["baseline"] = snapshot
                    changed = True
        else:
            followups = ensure_list(imaging_summary, "followups")
            followups_pointer = pointer_join("imaging_summary", "followups")
            dedupe_key = _snapshot_key(snapshot)
            has_match = any(_snapshot_key(item) == dedupe_key for item in followups if isinstance(item, dict))
            if not has_match and not is_pointer_locked(manual_overrides, followups_pointer):
                followups.append(snapshot)
                changed = True

    targets = ensure_dict(registry_json, "targets")

    peripheral_targets = ensure_list(targets, "peripheral_targets")
    peripheral_list_pointer = pointer_join("targets", "peripheral_targets")
    for raw in list((extracted.get("targets_update") or {}).get("peripheral_targets") or []):
        if not isinstance(raw, dict):
            continue

        target_idx = _find_peripheral_target_index(
            peripheral_targets,
            target_key=raw.get("target_key"),
            laterality=raw.get("laterality"),
            lobe=raw.get("lobe"),
            segment=raw.get("segment"),
        )

        if target_idx is None:
            if is_pointer_locked(manual_overrides, peripheral_list_pointer):
                qa_flags.append("peripheral_target_locked")
                continue
            new_target = {"target_key": raw.get("target_key") or f"target_{len(peripheral_targets) + 1}"}
            peripheral_targets.append(new_target)
            target_idx = len(peripheral_targets) - 1
            changed = True

        target = peripheral_targets[target_idx]
        base_pointer = pointer_join("targets", "peripheral_targets", target_idx)
        for field in _PERIPHERAL_FIELDS:
            changed |= assign_if_unlocked(
                target,
                key=field,
                value=raw.get(field),
                pointer=f"{base_pointer}/{field}",
                manual_overrides=manual_overrides,
            )
        changed |= assign_if_unlocked(
            target,
            key="source_event_id",
            value=event_id,
            pointer=f"{base_pointer}/source_event_id",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            target,
            key="source_relative_day_offset",
            value=relative_day_offset,
            pointer=f"{base_pointer}/source_relative_day_offset",
            manual_overrides=manual_overrides,
        )

    mediastinal_targets = ensure_list(targets, "mediastinal_targets")
    mediastinal_list_pointer = pointer_join("targets", "mediastinal_targets")
    for raw in list((extracted.get("targets_update") or {}).get("mediastinal_targets") or []):
        if not isinstance(raw, dict):
            continue

        target_idx = _find_mediastinal_target_index(
            mediastinal_targets,
            station=raw.get("station"),
            location_text=raw.get("location_text"),
        )

        if target_idx is None:
            if is_pointer_locked(manual_overrides, mediastinal_list_pointer):
                qa_flags.append("mediastinal_target_locked")
                continue
            mediastinal_targets.append({})
            target_idx = len(mediastinal_targets) - 1
            changed = True

        target = mediastinal_targets[target_idx]
        base_pointer = pointer_join("targets", "mediastinal_targets", target_idx)
        for field in _MEDIASTINAL_FIELDS:
            changed |= assign_if_unlocked(
                target,
                key=field,
                value=raw.get(field),
                pointer=f"{base_pointer}/{field}",
                manual_overrides=manual_overrides,
            )
        changed |= assign_if_unlocked(
            target,
            key="source_event_id",
            value=event_id,
            pointer=f"{base_pointer}/source_event_id",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            target,
            key="source_relative_day_offset",
            value=relative_day_offset,
            pointer=f"{base_pointer}/source_relative_day_offset",
            manual_overrides=manual_overrides,
        )

    return changed, sorted(set(qa_flags))


__all__ = ["patch_imaging_update"]
