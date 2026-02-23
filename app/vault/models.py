"""SQLAlchemy models for client-side encrypted vault payloads.

These tables store ciphertext and KDF metadata only. Never store plaintext PHI.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, PrimaryKeyConstraint, String, Text

from app.phi.db import Base, UUIDType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserVaultSettings(Base):
    __tablename__ = "user_vault_settings"

    user_id = Column(String(255), primary_key=True, nullable=False, index=True)
    wrapped_vmk_b64 = Column(Text, nullable=False)
    wrap_iv_b64 = Column(String(64), nullable=False)
    kdf_salt_b64 = Column(String(128), nullable=False)
    kdf_iterations = Column(Integer, nullable=False)
    kdf_hash = Column(String(32), nullable=False, default="PBKDF2-SHA256")
    crypto_version = Column(Integer, nullable=False, default=1)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class UserPatientVault(Base):
    __tablename__ = "user_patient_vault"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "registry_uuid", name="pk_user_patient_vault"),
    )

    user_id = Column(String(255), nullable=False, index=True)
    registry_uuid = Column(UUIDType, nullable=False)
    ciphertext_b64 = Column(Text, nullable=False)
    iv_b64 = Column(String(64), nullable=False)
    crypto_version = Column(Integer, nullable=False, default=1)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


__all__ = ["UserVaultSettings", "UserPatientVault"]

