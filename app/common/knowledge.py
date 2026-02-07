"""Knowledge base loader with hot-reload support."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, cast

from jsonschema import Draft7Validator

from config.settings import KnowledgeSettings

from .knowledge_schema import KNOWLEDGE_SCHEMA

KNOWLEDGE_ENV_VAR = "PSUITE_KNOWLEDGE_FILE"
KNOWLEDGE_WATCH_ENV_VAR = "PSUITE_KNOWLEDGE_WATCH"
KNOWLEDGE_ALLOW_VERSION_MISMATCH_ENV_VAR = "PSUITE_KNOWLEDGE_ALLOW_VERSION_MISMATCH"

_WATCH_INTERVAL_SECONDS = 1.0

logger = logging.getLogger(__name__)


class KnowledgeValidationError(RuntimeError):
    """Raised when the knowledge document fails schema validation."""


def _allow_version_mismatch() -> bool:
    value = os.environ.get(KNOWLEDGE_ALLOW_VERSION_MISMATCH_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes"}


def _extract_semver_from_filename(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_v(\d+)[._](\d+)\.json$", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_semver_from_kb_version(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().lstrip("v").split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _validate_filename_semver_matches_version(document: dict[str, Any], target: Path) -> None:
    if _allow_version_mismatch():
        return
    file_semver = _extract_semver_from_filename(target)
    kb_semver = _extract_semver_from_kb_version(document.get("version"))
    if not file_semver or not kb_semver:
        return
    if file_semver != kb_semver:
        raise KnowledgeValidationError(
            "KB filename semver "
            f"v{file_semver[0]}_{file_semver[1]} does not match internal version "
            f"{document.get('version')!r} ({target}). "
            f"Set {KNOWLEDGE_ALLOW_VERSION_MISMATCH_ENV_VAR}=1 to override."
        )


@dataclass(frozen=True)
class RVUEntry:
    code: str
    work: float
    pe: float
    mp: float
    total: float


@dataclass(frozen=True)
class KnowledgeSnapshot:
    version: str
    sha256: str
    rvus: list[RVUEntry]
    add_on_codes: list[str]
    bundling_rules: list[str]

_cache: dict[str, Any] | None = None
_knowledge_path: Path | None = None
_mtime: float | None = None
_checksum: str | None = None
_version: str | None = None
_lock = Lock()
_validator = Draft7Validator(KNOWLEDGE_SCHEMA)
_watch_thread: Thread | None = None


def get_knowledge(path: str | Path | None = None, *, force_reload: bool = False) -> dict[str, Any]:
    """Return the parsed knowledge document, reloading if the file changes."""

    target = _resolve_path(path)
    stat = target.stat()
    global _cache
    with _lock:
        if (
            force_reload
            or _cache is None
            or _knowledge_path != target
            or _mtime != stat.st_mtime
        ):
            _refresh_cache(target, stat.st_mtime)
    if _watch_enabled():
        _maybe_start_watcher()
    return _cache if isinstance(_cache, dict) else {}


def reset_cache() -> None:
    """Clear the cached knowledge data forcing a reload on next access."""

    global _cache, _knowledge_path, _mtime, _checksum, _version
    with _lock:
        _cache = None
        _knowledge_path = None
        _mtime = None
        _checksum = None
        _version = None


def knowledge_version() -> str | None:
    """Return the semantic version of the loaded knowledge file."""

    get_knowledge()
    return _version


def knowledge_hash() -> str | None:
    """Return the SHA256 hash of the current knowledge file contents."""

    get_knowledge()
    return _checksum


def get_rvu(cpt: str) -> dict[str, float] | None:
    data = get_knowledge()
    normalized = cpt.strip().lstrip("+")

    # Prefer the consolidated master index when present.
    master = data.get("master_code_index")
    if isinstance(master, dict):
        entry = master.get(normalized)
        if isinstance(entry, dict):
            simplified = entry.get("rvu_simplified")
            if isinstance(simplified, dict):
                return {
                    "work": float(simplified.get("work", 0.0)),
                    "pe": float(simplified.get("pe", 0.0)),
                    "mp": float(simplified.get("mp", 0.0)),
                }

            financials = entry.get("financials")
            if isinstance(financials, dict):
                cms_2026 = financials.get("cms_pfs_2026")
                if isinstance(cms_2026, dict):
                    work = cms_2026.get("work_rvu")
                    pe = cms_2026.get("facility_pe_rvu")
                    mp = cms_2026.get("mp_rvu")
                    if work is not None and pe is not None and mp is not None:
                        return {"work": float(work), "pe": float(pe), "mp": float(mp)}

    # Legacy fallback: rvus map (some add-ons are stored with '+' prefixes).
    rvus = data.get("rvus", {})
    if not isinstance(rvus, dict):
        return None

    entry = rvus.get(cpt)
    if not entry and not cpt.startswith("+"):
        entry = rvus.get(f"+{cpt}")
    if not entry and cpt.startswith("+"):
        entry = rvus.get(cpt.lstrip("+"))
    if not isinstance(entry, dict):
        return None

    return {
        "work": float(entry.get("work", 0.0)),
        "pe": float(entry.get("pe", 0.0)),
        "mp": float(entry.get("mp", 0.0)),
    }


def total_rvu(cpt: str) -> float:
    entry = get_rvu(cpt)
    if not entry:
        return 0.0
    return entry["work"] + entry["pe"] + entry["mp"]


def is_add_on_code(cpt: str) -> bool:
    data = get_knowledge()
    add_ons = set(data.get("add_on_codes", []))
    if cpt.startswith("+"):
        return True
    if cpt in add_ons:
        return True
    return f"+{cpt}" in add_ons


def bundling_rules() -> dict[str, Any]:
    return cast(dict[str, Any], get_knowledge().get("bundling_rules", {}))


def ncci_pairs() -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], get_knowledge().get("ncci_pairs", []))


def synonym_list(key: str) -> list[str]:
    synonyms = cast(dict[str, list[str]], get_knowledge().get("synonyms", {}))
    return list(synonyms.get(key, []))


def lobe_aliases() -> dict[str, list[str]]:
    data = get_knowledge()
    lobes = data.get("lobes")
    if isinstance(lobes, dict) and lobes:
        return cast(dict[str, list[str]], lobes)
    anatomy = data.get("anatomy")
    if isinstance(anatomy, dict):
        return cast(dict[str, list[str]], anatomy.get("lobes", {}) or {})
    return {}


def station_aliases() -> dict[str, list[str]]:
    data = get_knowledge()
    stations = data.get("stations")
    if isinstance(stations, dict) and stations:
        return cast(dict[str, list[str]], stations)
    anatomy = data.get("anatomy")
    if isinstance(anatomy, dict):
        return cast(dict[str, list[str]], anatomy.get("lymph_node_stations", {}) or {})
    return {}


def airway_map() -> dict[str, dict[str, Any]]:
    data = get_knowledge()
    airways = data.get("airways")
    if isinstance(airways, dict) and airways:
        return cast(dict[str, dict[str, Any]], airways)
    anatomy = data.get("anatomy")
    if isinstance(anatomy, dict):
        return cast(dict[str, dict[str, Any]], anatomy.get("airways", {}) or {})
    return {}


def blvr_config() -> dict[str, Any]:
    return cast(dict[str, Any], get_knowledge().get("blvr", {}))


def knowledge_snapshot(top_n: int = 20) -> KnowledgeSnapshot:
    """Return a structured view of the top RVU codes and rule metadata."""

    data = get_knowledge()
    rvus = data.get("rvus", {})
    entries: list[RVUEntry] = []
    for code, metrics in rvus.items():
        if not isinstance(metrics, dict):
            continue
        work = float(metrics.get("work", 0.0))
        pe = float(metrics.get("pe", 0.0))
        mp = float(metrics.get("mp", 0.0))
        entries.append(RVUEntry(code=code, work=work, pe=pe, mp=mp, total=work + pe + mp))
    entries.sort(key=lambda entry: entry.total, reverse=True)
    top_entries = entries[:top_n]

    add_on_codes = sorted({str(code) for code in data.get("add_on_codes", [])})
    bundling_rules = sorted((data.get("bundling_rules") or {}).keys())

    return KnowledgeSnapshot(
        version=str(data.get("version") or "unknown"),
        sha256=knowledge_hash() or "",
        rvus=top_entries,
        add_on_codes=add_on_codes,
        bundling_rules=bundling_rules,
    )


def _resolve_path(override: str | Path | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return KnowledgeSettings().kb_path


def _refresh_cache(target: Path, mtime: float | None = None) -> None:
    raw = target.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    _validate_document(data)
    _validate_filename_semver_matches_version(data, target)
    resolved_mtime = mtime if mtime is not None else target.stat().st_mtime
    global _cache, _knowledge_path, _mtime, _checksum, _version
    _cache = data
    _knowledge_path = target
    _mtime = resolved_mtime
    _checksum = hashlib.sha256(raw).hexdigest()
    _version = data.get("version")


def _validate_document(document: dict[str, Any]) -> None:
    errors = sorted(_validator.iter_errors(document), key=lambda err: list(err.absolute_path))
    if not errors:
        return
    error = errors[0]
    path = "/".join(str(part) for part in error.absolute_path) or "<root>"
    raise KnowledgeValidationError(f"{path}: {error.message}")


def _watch_enabled() -> bool:
    value = os.environ.get(KNOWLEDGE_WATCH_ENV_VAR, "").lower()
    return value in {"1", "true", "yes"}


def _maybe_start_watcher() -> None:
    if not _watch_enabled():
        return
    global _watch_thread
    if _watch_thread and _watch_thread.is_alive():
        return
    _watch_thread = Thread(target=_watch_loop, name="knowledge-watch", daemon=True)
    _watch_thread.start()


def _watch_loop() -> None:
    while True:
        time.sleep(_WATCH_INTERVAL_SECONDS)
        with _lock:
            target = _knowledge_path
            cached_mtime = _mtime
        if not target:
            continue
        try:
            stat = target.stat()
        except FileNotFoundError:
            continue
        if cached_mtime is not None and stat.st_mtime <= cached_mtime:
            continue
        try:
            with _lock:
                _refresh_cache(target, stat.st_mtime)
            logger.info("Reloaded knowledge base from %s", target)
        except Exception as exc:  # pragma: no cover - watcher logs and continues
            logger.warning("Knowledge hot-reload failed: %s", exc)


__all__ = [
    "get_knowledge",
    "reset_cache",
    "knowledge_version",
    "knowledge_hash",
    "get_rvu",
    "total_rvu",
    "is_add_on_code",
    "bundling_rules",
    "ncci_pairs",
    "synonym_list",
    "lobe_aliases",
    "station_aliases",
    "airway_map",
    "blvr_config",
    "knowledge_snapshot",
    "KnowledgeSnapshot",
    "RVUEntry",
    "KnowledgeValidationError",
    "KNOWLEDGE_ENV_VAR",
    "KNOWLEDGE_WATCH_ENV_VAR",
    "KNOWLEDGE_ALLOW_VERSION_MISMATCH_ENV_VAR",
]
