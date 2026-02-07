"""Schema filtering utilities for registry prompt construction."""

from __future__ import annotations

from typing import Any

from app.registry.tags import FIELD_APPLICABLE_TAGS


def filter_schema_properties(
    schema_properties: dict[str, Any],
    active_families: set[str] | None,
    *,
    force_include_all: bool = False,
) -> dict[str, Any]:
    """Filter schema properties based on active procedure families.

    - Fields not present in `FIELD_APPLICABLE_TAGS` are treated as universal.
    - When `active_families` is None, return all fields (caller is signaling "no gating").
    - When `active_families` is an empty set, return universal fields only.
    """
    if force_include_all or active_families is None:
        return dict(schema_properties)

    filtered: dict[str, Any] = {}
    for field_name, field_def in schema_properties.items():
        applicable = FIELD_APPLICABLE_TAGS.get(field_name)
        if not applicable:
            filtered[field_name] = field_def
            continue
        if applicable.intersection(active_families):
            filtered[field_name] = field_def

    return filtered


