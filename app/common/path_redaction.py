from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_relative_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    path = Path(raw)
    if path.is_absolute():
        try:
            return path.resolve(strict=False).relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return None

    return path.as_posix()


def sanitize_path_fields(payload: Any) -> Any:
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(key, str) and (key == "path" or key.endswith("_path")):
                if isinstance(value, (str, Path)) or value is None:
                    out[key] = repo_relative_path(value)
                else:
                    out[key] = sanitize_path_fields(value)
            else:
                out[key] = sanitize_path_fields(value)
        return out
    if isinstance(payload, list):
        return [sanitize_path_fields(item) for item in payload]
    return payload


__all__ = ["REPO_ROOT", "repo_relative_path", "sanitize_path_fields"]
