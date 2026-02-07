"""PHIService orchestrates scrubbing, vaulting, and re-identification.

Only this service and PHIVault touch raw PHI; downstream services consume
scrubbed text and entity maps.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.phi.models import (
    AuditAction,
    PHIVault,
    ProcedureData,
    ProcessingStatus,
    ScrubbingFeedback,
)
from app.phi.ports import (
    PHIAuditLoggerPort,
    PHIEncryptionPort,
    PHIScrubberPort,
    ScrubResult,
    ScrubbedEntity,
)


class PHIService:
    """Core PHI workflows for preview, vaulting, and re-identification."""

    def __init__(
        self,
        session: Session,
        encryption: PHIEncryptionPort,
        scrubber: PHIScrubberPort,
        audit_logger: PHIAuditLoggerPort,
    ):
        self._session = session
        self._encryption = encryption
        self._scrubber = scrubber
        self._audit_logger = audit_logger

    def preview(
        self, *, text: str, document_type: str | None = None, specialty: str | None = None
    ) -> ScrubResult:
        """Scrub PHI from text without persisting or auditing raw content."""

        return self._scrubber.scrub(text, document_type=document_type, specialty=specialty)

    def scrub_with_manual_entities(self, *, text: str, entities: list[dict]) -> ScrubResult:
        """Generate scrubbed text based strictly on provided entities (ignoring auto-scrubber).

        Requirements:
        - Sort entities by original_start descending to prevent index shifting.
        - Generate placeholders if not provided.
        """
        # Convert dicts to ScrubbedEntity objects if needed, but we really just need to iterate
        # We will perform the replacement manually here to avoid re-triggering the auto-scrubber.

        sorted_entities = sorted(entities, key=lambda x: x["original_start"], reverse=True)
        
        scrubbed_text = list(text)
        final_entities = []

        for idx, entity in enumerate(sorted_entities):
            start = entity["original_start"]
            end = entity["original_end"]
            
            # Ensure we don't go out of bounds (basic safety)
            if start < 0 or end > len(text):
                continue
            
            placeholder = entity.get("placeholder")
            if not placeholder:
                entity_type = entity.get("entity_type", "UNKNOWN")
                placeholder = f"<{entity_type}_{len(sorted_entities) - idx}>"
            
            # Replace text
            scrubbed_text[start:end] = list(placeholder)
            
            # Store the entity record (using ScrubbedEntity or dict structure)
            final_entities.append(ScrubbedEntity(
                placeholder=placeholder,
                entity_type=entity.get("entity_type", "UNKNOWN"),
                original_start=start,
                original_end=end
            ))

        # Reconstruct string
        result_text = "".join(scrubbed_text)
        
        # Re-sort entities by start ascending for the result
        final_entities.sort(key=lambda x: x["original_start"])

        return ScrubResult(scrubbed_text=result_text, entities=final_entities)

    def vault_phi(
        self,
        *,
        raw_text: str,
        scrub_result: ScrubResult,
        submitted_by: str,
        document_type: str | None = None,
        specialty: str | None = None,
    ) -> ProcedureData:
        """Encrypt and persist raw PHI alongside scrubbed output."""

        ciphertext, algorithm, key_version = self._encryption.encrypt(raw_text)
        text_hash = self._hash_text(raw_text)

        vault = PHIVault(
            encrypted_data=ciphertext,
            data_hash=text_hash,
            encryption_algorithm=algorithm,
            key_version=key_version,
        )
        proc = ProcedureData(
            phi_vault=vault,
            scrubbed_text=scrub_result.scrubbed_text,
            original_text_hash=text_hash,
            entity_map=self._serialize_entities(scrub_result.entities),
            status=ProcessingStatus.PENDING_REVIEW,
            document_type=document_type,
            specialty=specialty,
            submitted_by=submitted_by,
        )

        self._session.add(proc)
        self._session.flush()  # ensure IDs available for audit logging

        self._audit_logger.log_action(
            action=AuditAction.PHI_CREATED,
            phi_vault_id=vault.id,
            procedure_data_id=proc.id,
            user_id=submitted_by,
            metadata={
                "document_type": document_type,
                "specialty": specialty,
                "encryption_algorithm": algorithm,
                "key_version": key_version,
            },
        )
        self._session.commit()
        return proc

    def reidentify(
        self,
        *,
        procedure_data_id,
        user_id: str,
        user_email: str | None = None,
        user_role: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """Return the original PHI for a procedure record and audit the access."""

        proc = self._session.get(ProcedureData, procedure_data_id)
        if proc is None or proc.phi_vault is None:
            raise ValueError("Procedure data not found or missing PHI vault link")

        vault = proc.phi_vault
        plaintext = self._encryption.decrypt(vault.encrypted_data, vault.encryption_algorithm, vault.key_version)

        self._audit_logger.log_action(
            action=AuditAction.REIDENTIFIED,
            phi_vault_id=vault.id,
            procedure_data_id=proc.id,
            user_id=user_id,
            user_email=user_email,
            user_role=user_role,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
        )
        self._session.commit()
        return plaintext

    def get_procedure_for_review(self, *, procedure_data_id) -> ProcedureData:
        """Return scrubbed content for review (no raw PHI)."""

        proc = self._session.get(ProcedureData, procedure_data_id)
        if proc is None:
            raise ValueError("Procedure data not found")
        return proc

    def apply_scrubbing_feedback(
        self,
        *,
        procedure_data_id,
        scrubbed_text: str,
        entities: list[ScrubbedEntity],
        reviewer_id: str,
        reviewer_email: str | None = None,
        reviewer_role: str | None = None,
        comment: str | None = None,
    ) -> ProcedureData:
        """Persist feedback, update scrubbed output, and mark as reviewed."""

        proc = self._session.get(ProcedureData, procedure_data_id)
        if proc is None or proc.phi_vault is None:
            raise ValueError("Procedure data not found or missing PHI vault")

        serialized_entities = self._serialize_entities(entities)

        feedback = ScrubbingFeedback(
            procedure_data_id=proc.id,
            presidio_entities=proc.entity_map,
            confirmed_entities=serialized_entities,
            document_type=proc.document_type,
            specialty=proc.specialty,
            reviewer_id=reviewer_id,
            reviewer_email=reviewer_email,
            reviewer_role=reviewer_role,
            comment=comment,
            updated_scrubbed_text=scrubbed_text,
            updated_entity_map=serialized_entities,
        )

        proc.scrubbed_text = scrubbed_text
        proc.entity_map = serialized_entities
        proc.status = ProcessingStatus.PHI_REVIEWED
        proc.reviewed_by = reviewer_id
        proc.reviewed_at = datetime.utcnow()

        self._session.add(feedback)
        self._session.flush()

        self._audit_logger.log_action(
            action=AuditAction.SCRUBBING_FEEDBACK_APPLIED,
            phi_vault_id=proc.phi_vault_id,
            procedure_data_id=proc.id,
            user_id=reviewer_id,
            user_email=reviewer_email,
            user_role=reviewer_role,
            metadata={
                "comment_present": bool(comment),
                "new_status": proc.status.value if hasattr(proc.status, "value") else str(proc.status),
            },
        )

        self._session.commit()
        return proc

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_entities(entities: list[ScrubbedEntity]) -> list[dict]:
        """Normalize ScrubbedEntity payloads to JSON-safe dicts."""

        serialized: list[dict] = []
        for entity in entities:
            if is_dataclass(entity):
                serialized.append(asdict(entity))
            elif isinstance(entity, dict):
                serialized.append(
                    {
                        "placeholder": entity.get("placeholder"),
                        "entity_type": entity.get("entity_type"),
                        "original_start": entity.get("original_start"),
                        "original_end": entity.get("original_end"),
                    }
                )
            else:
                serialized.append(
                    {
                        "placeholder": getattr(entity, "placeholder", None),
                        "entity_type": getattr(entity, "entity_type", None),
                        "original_start": getattr(entity, "original_start", None),
                        "original_end": getattr(entity, "original_end", None),
                    }
                )
        return serialized


__all__ = ["PHIService"]
