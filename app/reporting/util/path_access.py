from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def get_path(data: Any, path: str) -> Any:
    """Safely traverse a dict/BaseModel/list by a dot + [idx] path.

    Supported:
    - "a.b.c"
    - "a.b[0].c"
    - "a[0][1].b"

    Returns None when the path cannot be resolved.
    """
    if not path:
        return data

    current: Any = data
    buffer = ""
    idx = 0

    def _step_key(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        if isinstance(value, BaseModel):
            return getattr(value, key, None)
        return None

    while idx < len(path):
        ch = path[idx]
        if ch == ".":
            if buffer:
                current = _step_key(current, buffer)
                buffer = ""
            idx += 1
            continue

        if ch == "[":
            if buffer:
                current = _step_key(current, buffer)
                buffer = ""
            close = path.find("]", idx + 1)
            if close == -1:
                return None
            raw = path[idx + 1 : close].strip()
            if raw == "":
                # "[]" is intentionally unsupported by get_path (callers should expand first).
                return None
            try:
                list_idx = int(raw)
            except Exception:
                return None
            if not isinstance(current, list):
                return None
            try:
                current = current[list_idx]
            except Exception:
                return None
            idx = close + 1
            continue

        buffer += ch
        idx += 1

    if buffer:
        current = _step_key(current, buffer)
    return current
