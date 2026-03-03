"""Apply clinical update events to canonical registry JSON."""

from __future__ import annotations

from typing import Any

from app.registry.aggregation.locks import ensure_dict, ensure_list, is_pointer_locked, pointer_join
from app.registry.aggregation.sanitize import compact_text


def _dedupe_key(update: dict[str, Any], event_title: str | None) -> tuple[Any, ...]:
    title = str(update.get("event_title") or event_title or "").strip().lower()
    return (
        update.get("relative_day_offset"),
        update.get("update_type"),
        title,
    )


def patch_clinical_update(
    registry_json: dict[str, Any],
    *,
    extracted: dict[str, Any],
    event_id: str,
    event_title: str | None,
    manual_overrides: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    changed = False
    qa_flags = list(extracted.get("qa_flags") or [])

    clinical_course = ensure_dict(registry_json, "clinical_course")
    updates = ensure_list(clinical_course, "updates")

    update = extracted.get("clinical_update")
    if isinstance(update, dict):
        update["source_event_id"] = event_id
        if event_title:
            update["event_title"] = compact_text(event_title, max_chars=120)
        key = _dedupe_key(update, event_title)
        seen = {_dedupe_key(item, None) for item in updates if isinstance(item, dict)}
        updates_pointer = pointer_join("clinical_course", "updates")
        was_appended = False
        if key not in seen and not is_pointer_locked(manual_overrides, updates_pointer):
            updates.append(update)
            changed = True
            was_appended = True

        current_state_pointer = pointer_join("clinical_course", "current_state")
        if was_appended and not is_pointer_locked(manual_overrides, current_state_pointer):
            next_state = {
                "relative_day_offset": update.get("relative_day_offset"),
                "performance_status_text": update.get("performance_status_text"),
                "symptom_change": update.get("symptom_change"),
                "treatment_change_text": update.get("treatment_change_text"),
                "complication_text": update.get("complication_text"),
                "summary_text": update.get("summary_text"),
                "hospital_admission": update.get("hospital_admission"),
                "icu_admission": update.get("icu_admission"),
                "deceased": update.get("deceased"),
                "disease_status": update.get("disease_status"),
                "source_event_id": event_id,
            }
            if clinical_course.get("current_state") != next_state:
                clinical_course["current_state"] = next_state
                changed = True

    return changed, sorted(set(qa_flags))


__all__ = ["patch_clinical_update"]
