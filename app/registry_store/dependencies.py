"""DB wiring for registry run persistence (SQLAlchemy sessions)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def resolve_registry_store_database_url() -> str:
    """Resolve DB URL for registry run persistence.

    Priority:
    1) REGISTRY_STORE_DATABASE_URL
    2) DATABASE_URL (Alembic target / primary DB)
    3) PHI_DATABASE_URL (legacy/demo pattern)
    4) Local sqlite demo DB (matches alembic.ini default)
    """

    return (
        os.getenv("REGISTRY_STORE_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("PHI_DATABASE_URL")
        or "sqlite:///./phi_demo.db"
    )


def _engine_kwargs(url: str) -> dict:
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    return kwargs


@lru_cache(maxsize=4)
def _engine_for_url(url: str) -> Engine:
    return create_engine(url, **_engine_kwargs(url))


def get_registry_store_engine() -> Engine:
    return _engine_for_url(resolve_registry_store_database_url())


@lru_cache(maxsize=4)
def _sessionmaker_for_url(url: str):
    engine = _engine_for_url(url)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_registry_store_db() -> Iterator[Session]:
    """FastAPI dependency that yields a registry store Session."""

    url = resolve_registry_store_database_url()
    SessionLocal = _sessionmaker_for_url(url)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = [
    "get_registry_store_db",
    "get_registry_store_engine",
    "resolve_registry_store_database_url",
]

