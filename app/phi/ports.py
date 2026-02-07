"""Ports/interfaces for PHI encryption, scrubbing, and audit logging.

These are intentionally lightweight so the PHIService can swap
implementations for HIPAA-ready vaults or scrubbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

from app.phi.models import AuditAction


class ScrubbedEntity(TypedDict):
    placeholder: str
    entity_type: str
    original_start: int
    original_end: int


@dataclass
class ScrubResult:
    scrubbed_text: str
    entities: list[ScrubbedEntity]


@runtime_checkable
class PHIEncryptionPort(Protocol):
    """Abstraction for encrypting/decrypting PHI payloads."""

    def encrypt(self, plaintext: str) -> tuple[bytes, str, int]:
        """Return ciphertext bytes, algorithm name, and key version."""

    def decrypt(self, ciphertext: bytes, algorithm: str, key_version: int) -> str:
        """Return decrypted plaintext from ciphertext and metadata."""


@runtime_checkable
class PHIScrubberPort(Protocol):
    """Abstraction for PHI scrubbing/detection."""

    def scrub(
        self, text: str, document_type: str | None = None, specialty: str | None = None
    ) -> ScrubResult:
        """Return scrubbed text and detected PHI entities."""


@runtime_checkable
class PHIAuditLoggerPort(Protocol):
    """Abstraction for writing PHI audit log entries."""

    def log_action(
        self,
        *,
        action: AuditAction,
        phi_vault_id=None,
        procedure_data_id=None,
        user_id: str,
        user_email: str | None = None,
        user_role: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Persist a structured audit record (no raw PHI allowed)."""


__all__ = [
    "ScrubbedEntity",
    "ScrubResult",
    "PHIEncryptionPort",
    "PHIScrubberPort",
    "PHIAuditLoggerPort",
]
