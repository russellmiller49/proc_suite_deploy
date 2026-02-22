from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from config.settings import UmlsSettings


router = APIRouter(tags=["umls"])


def _require_store():
    settings = UmlsSettings()
    if not settings.enable_linker:
        raise HTTPException(status_code=503, detail="UMLS disabled")
    try:
        from app.umls.ip_umls_store import get_ip_umls_store

        return get_ip_umls_store()
    except Exception:
        raise HTTPException(status_code=503, detail="UMLS map unavailable")


@router.get("/v1/umls/suggest")
def suggest(
    q: str = Query(..., min_length=1),
    category: str | None = None,
    limit: int = Query(20, ge=1, le=50),
) -> list[dict[str, Any]]:
    store = _require_store()
    return store.suggest(q, category=category, limit=limit)


@router.get("/v1/umls/concept/{cui}")
def concept(cui: str) -> dict[str, Any]:
    store = _require_store()
    concept = store.concepts.get(cui)
    if not concept:
        raise HTTPException(status_code=404, detail="UMLS concept not found")
    if not isinstance(concept, dict):
        raise HTTPException(status_code=500, detail="Invalid UMLS concept payload")
    return {"cui": cui, **concept}


__all__ = ["router"]

