"""SQLAlchemy models for the Registry Runs persistence layer.

IMPORTANT: This store must contain scrubbed-only note text. Never persist raw PHI.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from app.phi.db import Base, JSONType, UUIDType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RegistryRun(Base):
    __tablename__ = "registry_runs"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)

    submitter_name = Column(String(255), nullable=True, index=True)

    # NOTE: MUST be scrubbed-only text.
    note_text = Column(Text, nullable=False)
    note_sha256 = Column(String(64), nullable=False, index=True)

    schema_version = Column(String(32), nullable=False)
    pipeline_config = Column(JSONType, nullable=False, default=dict)

    # Exact response payload returned by the pipeline endpoint.
    raw_response_json = Column(JSONType, nullable=False)

    # Optional reviewer-edited data from the UI.
    corrected_response_json = Column(JSONType, nullable=True)
    edited_tables_json = Column(JSONType, nullable=True)
    correction_editor_name = Column(String(255), nullable=True)
    corrected_at = Column(DateTime(timezone=True), nullable=True)

    # One feedback submission per run.
    feedback_reviewer_name = Column(String(255), nullable=True)
    feedback_rating = Column(Integer, nullable=True)
    feedback_comment = Column(Text, nullable=True)
    feedback_submitted_at = Column(DateTime(timezone=True), nullable=True)

    needs_manual_review = Column(Boolean, nullable=False, default=False, index=True)
    review_status = Column(String(50), nullable=False, default="new", index=True)

    # Provenance metadata
    kb_version = Column(String(64), nullable=True)
    kb_hash = Column(String(64), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)


__all__ = ["RegistryRun"]

