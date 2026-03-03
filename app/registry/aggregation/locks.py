"""JSON-pointer lock helpers for case aggregation."""

from __future__ import annotations

from typing import Any


def escape_pointer_token(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def pointer_join(*parts: str | int) -> str:
    if not parts:
        return "/"
    return "/" + "/".join(escape_pointer_token(str(part)) for part in parts)


def _parent_pointers(pointer: str) -> list[str]:
    if not pointer or pointer == "/":
        return ["/"]
    parts = [part for part in pointer.split("/") if part]
    parents: list[str] = []
    for idx in range(len(parts), -1, -1):
        if idx == 0:
            parents.append("/")
        else:
            parents.append("/" + "/".join(parts[:idx]))
    return parents


def is_pointer_locked(manual_overrides: dict[str, Any] | None, pointer: str) -> bool:
    overrides = manual_overrides or {}
    for candidate in _parent_pointers(pointer):
        payload = overrides.get(candidate)
        if isinstance(payload, dict) and bool(payload.get("locked")):
            return True
    return False


def assign_if_unlocked(
    target: dict[str, Any],
    *,
    key: str,
    value: Any,
    pointer: str,
    manual_overrides: dict[str, Any] | None,
) -> bool:
    if value is None:
        return False
    if is_pointer_locked(manual_overrides, pointer):
        return False
    if target.get(key) == value:
        return False
    target[key] = value
    return True


def ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    current = parent.get(key)
    if isinstance(current, dict):
        return current
    parent[key] = {}
    return parent[key]


def ensure_list(parent: dict[str, Any], key: str) -> list[Any]:
    current = parent.get(key)
    if isinstance(current, list):
        return current
    parent[key] = []
    return parent[key]


__all__ = [
    "assign_if_unlocked",
    "ensure_dict",
    "ensure_list",
    "is_pointer_locked",
    "pointer_join",
]
