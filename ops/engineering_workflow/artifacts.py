"""Artifact helpers for the engineering workflow."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import SessionManifest


def resolve_artifact_root(explicit_root: str | Path | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()

    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return (Path(codex_home).expanduser() / "agent_runs" / "proc_suite").resolve()
    return (Path.home() / ".codex" / "agent_runs" / "proc_suite").resolve()


def ensure_session_dir(root: Path, session_id: str) -> Path:
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def append_event(session_dir: Path, event_type: str, payload: dict[str, Any]) -> Path:
    event_path = session_dir / "events.jsonl"
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event_type": event_type,
        "payload": payload,
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return event_path


def load_manifest(session_dir: Path) -> SessionManifest:
    return SessionManifest.model_validate_json((session_dir / "session_manifest.json").read_text(encoding="utf-8"))


def save_manifest(session_dir: Path, manifest: SessionManifest) -> Path:
    manifest.updated_at = datetime.utcnow()
    return write_json(session_dir / "session_manifest.json", manifest)

