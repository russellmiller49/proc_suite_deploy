"""PHI vault and audit models (Phase 0.1).

These ORM models implement the PHI boundaries defined in the V8 migration plan:
- Raw PHI is stored only in `PHIVault.encrypted_data`.
- Downstream services use `ProcedureData.scrubbed_text` and `entity_map` (no raw PHI).
- Access is auditable via `AuditLog`.
- Physician corrections feed back into `ScrubbingFeedback` for ML tuning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.phi.db import Base, JSONType, UUIDType


class ProcessingStatus(PyEnum):
    PENDING_REVIEW = "pending_review"
    PHI_CONFIRMED = "phi_confirmed"
    PHI_REVIEWED = "phi_reviewed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AuditAction(PyEnum):
    PHI_CREATED = "phi_created"
    PHI_ACCESSED = "phi_accessed"
    PHI_DECRYPTED = "phi_decrypted"
    REVIEW_STARTED = "review_started"
    ENTITY_CONFIRMED = "entity_confirmed"
    ENTITY_UNFLAGGED = "entity_unflagged"
    ENTITY_ADDED = "entity_added"
    REVIEW_COMPLETED = "review_completed"
    SCRUBBING_FEEDBACK_APPLIED = "scrubbing_feedback_applied"
    LLM_CALLED = "llm_called"
    REIDENTIFIED = "reidentified"


class PHIVault(Base):
    """Encrypted storage for Protected Health Information."""

    __tablename__ = "phi_vault"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    encrypted_data = Column(LargeBinary, nullable=False)
    data_hash = Column(String(64), nullable=False)
    encryption_algorithm = Column(String(50), default="FERNET")
    key_version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)

    procedure_data = relationship("ProcedureData", back_populates="phi_vault", uselist=False)
    audit_logs = relationship("AuditLog", back_populates="phi_vault")


class ProcedureData(Base):
    """De-identified clinical text and processing results."""

    __tablename__ = "procedure_data"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    phi_vault_id = Column(UUIDType, ForeignKey("phi_vault.id"), nullable=False)

    scrubbed_text = Column(Text, nullable=False)
    original_text_hash = Column(String(64), nullable=False)
    entity_map = Column(JSONType, nullable=False, default=list)

    status = Column(
        Enum(ProcessingStatus, name="processingstatus"),
        default=ProcessingStatus.PENDING_REVIEW,
    )
    coding_results = Column(JSONType, nullable=True)

    document_type = Column(String(100), nullable=True)
    specialty = Column(String(100), nullable=True)

    submitted_by = Column(String(255), nullable=False)
    reviewed_by = Column(String(255), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    phi_vault = relationship("PHIVault", back_populates="procedure_data")
    audit_logs = relationship("AuditLog", back_populates="procedure_data")
    scrubbing_feedback = relationship("ScrubbingFeedback", back_populates="procedure_data")


class AuditLog(Base):
    """HIPAA-style audit trail for PHI actions."""

    __tablename__ = "audit_log"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)

    phi_vault_id = Column(UUIDType, ForeignKey("phi_vault.id"), nullable=True)
    procedure_data_id = Column(UUIDType, ForeignKey("procedure_data.id"), nullable=True)

    user_id = Column(String(255), nullable=False)
    user_email = Column(String(255), nullable=True)
    user_role = Column(String(100), nullable=True)

    action = Column(Enum(AuditAction, name="auditaction"), nullable=False)
    action_detail = Column(Text, nullable=True)

    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(255), nullable=True)
    metadata_json = Column("metadata", JSONType, nullable=True, default=dict)

    timestamp = Column(DateTime, default=datetime.utcnow)

    phi_vault = relationship("PHIVault", back_populates="audit_logs")
    procedure_data = relationship("ProcedureData", back_populates="audit_logs")


class ScrubbingFeedback(Base):
    """ML improvement data from physician corrections."""

    __tablename__ = "scrubbing_feedback"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    procedure_data_id = Column(UUIDType, ForeignKey("procedure_data.id"))

    presidio_entities = Column(JSONType, nullable=False)
    confirmed_entities = Column(JSONType, nullable=False)

    false_positives = Column(JSONType, default=list)
    false_negatives = Column(JSONType, default=list)
    true_positives = Column(Integer, default=0)

    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)

    document_type = Column(String(100), nullable=True)
    specialty = Column(String(100), nullable=True)
    reviewer_id = Column(String(255), nullable=True)
    reviewer_email = Column(String(255), nullable=True)
    reviewer_role = Column(String(100), nullable=True)
    comment = Column(Text, nullable=True)
    updated_scrubbed_text = Column(Text, nullable=True)
    updated_entity_map = Column(JSONType, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    procedure_data = relationship("ProcedureData", back_populates="scrubbing_feedback")
