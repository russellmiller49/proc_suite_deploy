"""SQLAlchemy base and shared types for PHI models.

This module keeps ORM configuration scoped to PHI tables so we can
swap storage backends in future HIPAA deployments.
"""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base


Base = declarative_base()
metadata = Base.metadata


class GUID(TypeDecorator):
    """Platform-independent GUID/UUID type for Postgres and SQLite."""

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value if dialect.name == "postgresql" else str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        return uuid.UUID(value) if not isinstance(value, uuid.UUID) else value


# Portable column types (JSONB/UUID for Postgres, JSON/String for SQLite)
UUIDType = GUID
JSONType = JSONB().with_variant(JSON(), "sqlite")

__all__ = ["Base", "metadata", "GUID", "UUIDType", "JSONType"]
