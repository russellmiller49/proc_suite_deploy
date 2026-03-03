"""SQLAlchemy models for the Registry Runs persistence layer.

IMPORTANT: This store must contain scrubbed-only note text. Never persist raw PHI.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text

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


class RegistryAppendedDocument(Base):
    __tablename__ = "registry_appended_documents"
    __table_args__ = (
        Index(
            "ix_registry_appended_documents_registry_uuid_created_at",
            "registry_uuid",
            "created_at",
        ),
        Index(
            "ix_registry_appended_documents_registry_uuid_event_type",
            "registry_uuid",
            "event_type",
        ),
    )

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)

    user_id = Column(String(255), nullable=False, index=True)
    registry_uuid = Column(UUIDType, nullable=False, index=True)

    # NOTE: MUST be scrubbed-only text.
    note_text = Column(Text, nullable=False)
    note_sha256 = Column(String(64), nullable=False, index=True)

    event_type = Column(String(64), nullable=False, default="pathology", index=True)
    document_kind = Column(String(64), nullable=False, default="pathology")
    source_type = Column(String(64), nullable=True)
    source_modality = Column(Text, nullable=True)
    event_subtype = Column(Text, nullable=True)
    event_title = Column(Text, nullable=True)
    relative_day_offset = Column(Integer, nullable=True)
    ocr_correction_applied = Column(Boolean, nullable=False, default=False)
    metadata_json = Column("metadata", JSONType, nullable=True, default=dict)
    extracted_json = Column(JSONType, nullable=True)
    aggregated_at = Column(DateTime(timezone=True), nullable=True)
    aggregation_version = Column(Integer, nullable=True)


class RegistryCaseRecord(Base):
    __tablename__ = "registry_case_records"

    registry_uuid = Column(UUIDType, primary_key=True)
    registry_json = Column(JSONType, nullable=False, default=dict)
    schema_version = Column(String(32), nullable=False, default="v3")
    version = Column(Integer, nullable=False, default=1)
    manual_overrides = Column(JSONType, nullable=False, default=dict)
    source_run_id = Column(UUIDType, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
        index=True,
    )


__all__ = ["RegistryRun", "RegistryAppendedDocument", "RegistryCaseRecord"]
