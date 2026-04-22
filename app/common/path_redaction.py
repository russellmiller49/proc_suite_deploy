from __future__ import annotations

from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_TOP_LEVEL_NAMES = {child.name for child in REPO_ROOT.iterdir() if child.name and not child.name.startswith(".")}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _repo_root_aliases() -> set[str]:
    normalized = _normalize_name(REPO_ROOT.name)
    aliases = {normalized}
    if normalized.startswith("proc"):
        aliases.add("procedure" + normalized[len("proc") :])
    if normalized.startswith("procedure"):
        aliases.add("proc" + normalized[len("procedure") :])
    return aliases


_REPO_ROOT_ALIASES = _repo_root_aliases()


def _looks_like_repo_root_segment(segment: str) -> bool:
    normalized = _normalize_name(segment)
    if not normalized:
        return False
    if normalized in _REPO_ROOT_ALIASES:
        return True
    return ("proc" in normalized or "procedure" in normalized) and "suite" in normalized


def _strip_machine_local_repo_prefix(raw: str) -> str | None:
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2:
        return None

    for index in range(1, len(parts)):
        if parts[index] not in _REPO_TOP_LEVEL_NAMES:
            continue
        if not _looks_like_repo_root_segment(parts[index - 1]):
            continue
        return Path(*parts[index:]).as_posix()

    return None


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
            return _strip_machine_local_repo_prefix(raw)

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
