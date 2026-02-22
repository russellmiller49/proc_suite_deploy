from __future__ import annotations

import logging
from typing import Any

from config.settings import UmlsSettings


_logger = logging.getLogger(__name__)


def _get_store():
    settings = UmlsSettings()
    if not settings.enable_linker:
        return None
    try:
        from app.umls.ip_umls_store import get_ip_umls_store

        return get_ip_umls_store()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("UMLS store unavailable for template filter: %s", exc)
        return None


def umls_pref(term: Any, category: str | None = None) -> str:
    raw = "" if term in (None, "") else str(term)
    store = _get_store()
    if store is None or not raw:
        return raw
    try:
        match = store.match(raw, category=category)
        return str(match["preferred_name"]) if match else raw
    except Exception:  # noqa: BLE001
        return raw


def umls_cui(term: Any, category: str | None = None) -> str | None:
    raw = "" if term in (None, "") else str(term)
    store = _get_store()
    if store is None or not raw:
        return None
    try:
        match = store.match(raw, category=category)
        return str(match["chosen_cui"]) if match else None
    except Exception:  # noqa: BLE001
        return None


__all__ = ["umls_cui", "umls_pref"]

