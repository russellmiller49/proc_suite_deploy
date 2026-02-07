"""Database-backed audit logger for PHI actions.

Writes structured audit events to the AuditLog table without storing
raw PHI content.
"""

from __future__ import annotations

from app.phi.models import AuditAction, AuditLog
from app.phi.ports import PHIAuditLoggerPort


class DatabaseAuditLogger(PHIAuditLoggerPort):
    def __init__(self, session):
        self._session = session

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
        audit = AuditLog(
            phi_vault_id=phi_vault_id,
            procedure_data_id=procedure_data_id,
            user_id=user_id,
            user_email=user_email,
            user_role=user_role,
            action=action,
            action_detail=action.value,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            metadata_json=metadata or {},
        )
        self._session.add(audit)


__all__ = ["DatabaseAuditLogger"]
