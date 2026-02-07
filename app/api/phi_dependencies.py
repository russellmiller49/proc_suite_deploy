"""FastAPI dependencies for PHI service wiring.

Demo-only wiring that uses the insecure demo encryption and stub scrubber.
Production deployments should swap in KMS-backed encryption and a real scrubber.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Iterator

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.phi import PHIService
from app.phi.adapters import (
    DatabaseAuditLogger,
    InsecureDemoEncryptionAdapter,
    PresidioScrubber,
    StubScrubber,
)
from app.phi.adapters.fernet_encryption import FernetEncryptionAdapter

logger = logging.getLogger(__name__)


def _default_db_url() -> str:
    return os.getenv("PHI_DATABASE_URL") or os.getenv(
        "DATABASE_URL", "sqlite:///./phi_demo.db"
    )


DATABASE_URL = _default_db_url()


def _engine_kwargs(url: str) -> dict:
    """Configure engine options for SQLite vs Postgres."""

    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        # In-memory SQLite needs StaticPool to share state across connections
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    return kwargs


engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_phi_session() -> Iterator[Session]:
    """Provide a scoped SQLAlchemy session for PHI operations."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_encryption_adapter():
    mode = os.getenv("PHI_ENCRYPTION_MODE", "fernet").lower()
    if mode == "demo":
        return InsecureDemoEncryptionAdapter()
    return FernetEncryptionAdapter()


@lru_cache
def _get_scrubber():
    mode = os.getenv("PHI_SCRUBBER_MODE", "presidio").lower()
    if mode == "stub":
        return StubScrubber()
    strict = os.getenv("PHI_SCRUBBER_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        return PresidioScrubber()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "PresidioScrubber unavailable; falling back to StubScrubber "
            "(set PHI_SCRUBBER_MODE=stub to silence, or install the configured "
            "spaCy model and presidio-analyzer to enable real scrubbing).",
            extra={"error_type": type(exc).__name__},
        )
        if strict:
            raise
        # Fallback to stub if Presidio is unavailable (keeps tests/demo running)
        return StubScrubber()


_phi_session_dep = Depends(get_phi_session)


def get_phi_service(db: Session = _phi_session_dep) -> PHIService:
    """Construct a PHIService with configured adapters (no raw PHI logging)."""

    encryption = _get_encryption_adapter()
    scrubber = _get_scrubber()
    audit_logger = DatabaseAuditLogger(db)
    return PHIService(
        session=db,
        encryption=encryption,
        scrubber=scrubber,
        audit_logger=audit_logger,
    )


def get_phi_scrubber():
    """Get the PHI scrubber as a FastAPI dependency.

    Returns the cached scrubber instance, or None if unavailable.
    This allows graceful degradation when Presidio is not configured.

    Usage:
        @app.post("/endpoint")
        async def handler(phi_scrubber = Depends(get_phi_scrubber)):
            if phi_scrubber:
                result = phi_scrubber.scrub(text)
    """
    try:
        return _get_scrubber()
    except Exception:
        logger.warning("PHI scrubber unavailable for dependency injection")
        return None


__all__ = ["get_phi_service", "get_phi_session", "get_phi_scrubber", "engine", "SessionLocal"]
