"""Apply pathology extraction updates to canonical registry JSON."""

from __future__ import annotations

from typing import Any

from app.registry.aggregation.locks import (
    assign_if_unlocked,
    ensure_dict,
    ensure_list,
    is_pointer_locked,
    pointer_join,
)
from app.registry.aggregation.pathology.normalize import build_target_key, normalize_station


def _find_node_event_index(node_events: list[dict[str, Any]], station: str) -> int | None:
    target = normalize_station(station)
    if not target:
        return None
    for idx, node_event in enumerate(node_events):
        if normalize_station(str(node_event.get("station") or "")) == target:
            return idx
    return None


def _find_peripheral_target_index(
    peripheral_targets: list[dict[str, Any]],
    *,
    target_key: str | None,
    laterality: str | None,
    lobe: str | None,
    segment: str | None,
) -> int | None:
    normalized_key = (target_key or "").strip().lower()
    for idx, target in enumerate(peripheral_targets):
        existing_key = str(target.get("target_key") or "").strip().lower()
        if normalized_key and existing_key and normalized_key == existing_key:
            return idx

    for idx, target in enumerate(peripheral_targets):
        if laterality and str(target.get("laterality") or "") != laterality:
            continue
        if lobe and str(target.get("lobe") or "").upper() != lobe.upper():
            continue
        if segment and str(target.get("segment") or "").strip().lower() != segment.strip().lower():
            continue
        if laterality or lobe or segment:
            return idx
    return None


def patch_pathology_update(
    registry_json: dict[str, Any],
    *,
    extracted: dict[str, Any],
    event_id: str,
    relative_day_offset: int | None,
    manual_overrides: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Patch pathology-derived fields while honoring manual locks."""

    changed = False
    qa_flags: list[str] = list(extracted.get("qa_flags") or [])

    procedures = ensure_dict(registry_json, "procedures_performed")
    linear_ebus = ensure_dict(procedures, "linear_ebus")
    node_events = ensure_list(linear_ebus, "node_events")

    for raw_update in list(extracted.get("node_updates") or []):
        if not isinstance(raw_update, dict):
            continue
        station = normalize_station(str(raw_update.get("station") or ""))
        if not station:
            continue

        node_idx = _find_node_event_index(node_events, station)
        if node_idx is None:
            qa_flags.append("station_not_in_node_events")
            continue

        node_event = node_events[node_idx]
        base_pointer = pointer_join("procedures_performed", "linear_ebus", "node_events", node_idx)
        changed |= assign_if_unlocked(
            node_event,
            key="path_result",
            value=raw_update.get("path_result"),
            pointer=f"{base_pointer}/path_result",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            node_event,
            key="path_diagnosis_text",
            value=raw_update.get("path_diagnosis_text"),
            pointer=f"{base_pointer}/path_diagnosis_text",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            node_event,
            key="path_source_event_id",
            value=event_id,
            pointer=f"{base_pointer}/path_source_event_id",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            node_event,
            key="path_relative_day_offset",
            value=relative_day_offset,
            pointer=f"{base_pointer}/path_relative_day_offset",
            manual_overrides=manual_overrides,
        )

    targets = ensure_dict(registry_json, "targets")
    peripheral_targets = ensure_list(targets, "peripheral_targets")
    peripheral_list_pointer = pointer_join("targets", "peripheral_targets")

    for raw_update in list(extracted.get("peripheral_updates") or []):
        if not isinstance(raw_update, dict):
            continue

        laterality = raw_update.get("laterality")
        lobe = raw_update.get("lobe")
        segment = raw_update.get("segment")
        target_key = raw_update.get("target_key") or build_target_key(
            laterality=laterality,
            lobe=lobe,
            segment=segment,
        )

        target_idx = _find_peripheral_target_index(
            peripheral_targets,
            target_key=target_key,
            laterality=laterality,
            lobe=lobe,
            segment=segment,
        )

        if target_idx is None:
            if is_pointer_locked(manual_overrides, peripheral_list_pointer):
                qa_flags.append("peripheral_target_locked")
                continue
            new_target = {
                "target_key": target_key or f"target_{len(peripheral_targets) + 1}",
                "laterality": laterality,
                "lobe": lobe,
                "segment": segment,
            }
            peripheral_targets.append(new_target)
            target_idx = len(peripheral_targets) - 1
            changed = True

        target = peripheral_targets[target_idx]
        base_pointer = pointer_join("targets", "peripheral_targets", target_idx)
        changed |= assign_if_unlocked(
            target,
            key="path_result",
            value=raw_update.get("path_result"),
            pointer=f"{base_pointer}/path_result",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            target,
            key="path_diagnosis_text",
            value=raw_update.get("path_diagnosis_text"),
            pointer=f"{base_pointer}/path_diagnosis_text",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            target,
            key="path_source_event_id",
            value=event_id,
            pointer=f"{base_pointer}/path_source_event_id",
            manual_overrides=manual_overrides,
        )
        changed |= assign_if_unlocked(
            target,
            key="path_relative_day_offset",
            value=relative_day_offset,
            pointer=f"{base_pointer}/path_relative_day_offset",
            manual_overrides=manual_overrides,
        )

    return changed, sorted(set(qa_flags))


__all__ = ["patch_pathology_update"]
