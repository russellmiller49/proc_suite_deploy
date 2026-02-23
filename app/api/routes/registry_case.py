"""Case-level canonical registry snapshot + patch APIs."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.api.auth import AuthenticatedUser, get_current_user
from app.api.readiness import require_ready
from app.registry.schema import RegistryRecord
from app.registry_store.dependencies import get_registry_store_db
from app.registry_store.models import RegistryAppendedDocument, RegistryCaseRecord
from app.vault.models import UserPatientVault


router = APIRouter(tags=["registry-case"])

_ready_dep = Depends(require_ready)
_current_user_dep = Depends(get_current_user)
_db_dep = Depends(get_registry_store_db)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _schema_version() -> str:
    return (os.getenv("REGISTRY_SCHEMA_VERSION") or "v3").strip()


def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _ensure_case_ownership(db: Session, *, user_id: str, registry_uuid: uuid.UUID) -> None:
    case_stmt: Select[tuple[UserPatientVault]] = select(UserPatientVault).where(
        UserPatientVault.user_id == user_id,
        UserPatientVault.registry_uuid == registry_uuid,
    )
    case_row = db.execute(case_stmt).scalar_one_or_none()
    if case_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Registry case not found for this user",
        )


class RegistryCaseEventSummary(BaseModel):
    append_id: str
    created_at: str
    event_type: str
    document_kind: str
    source_type: str | None = None
    relative_day_offset: int | None = None
    has_note_text: bool = False
    has_structured_data: bool = False


class RegistryCaseResponse(BaseModel):
    registry_uuid: str
    schema_version: str
    version: int
    source_run_id: str | None = None
    updated_at: str
    registry: dict[str, Any]
    events: list[RegistryCaseEventSummary] = Field(default_factory=list)


class RegistryCasePatchRequest(BaseModel):
    registry_patch: dict[str, Any] = Field(
        ...,
        description="Deep-merge patch applied to canonical registry JSON.",
    )
    expected_version: int | None = Field(
        default=None,
        description="Optional optimistic concurrency check against current case version.",
    )


def _build_case_response(
    *,
    case_record: RegistryCaseRecord,
    append_rows: list[RegistryAppendedDocument],
) -> RegistryCaseResponse:
    events = [
        RegistryCaseEventSummary(
            append_id=str(row.id),
            created_at=row.created_at.isoformat(),
            event_type=str(getattr(row, "event_type", row.document_kind or "pathology")),
            document_kind=str(row.document_kind or ""),
            source_type=row.source_type,
            relative_day_offset=getattr(row, "relative_day_offset", None),
            has_note_text=bool(str(row.note_text or "").strip()),
            has_structured_data=bool((row.metadata_json or {}).get("structured_data")),
        )
        for row in append_rows
    ]
    return RegistryCaseResponse(
        registry_uuid=str(case_record.registry_uuid),
        schema_version=str(case_record.schema_version or _schema_version()),
        version=int(case_record.version or 1),
        source_run_id=str(case_record.source_run_id) if case_record.source_run_id else None,
        updated_at=case_record.updated_at.isoformat(),
        registry=dict(case_record.registry_json or {}),
        events=events,
    )


@router.get(
    "/v1/registry/{registry_uuid}",
    response_model=RegistryCaseResponse,
    summary="Get canonical case-level registry snapshot",
)
def get_registry_case(
    registry_uuid: uuid.UUID,
    _ready: None = _ready_dep,
    current_user: AuthenticatedUser = _current_user_dep,
    db: Session = _db_dep,
) -> RegistryCaseResponse:
    _ensure_case_ownership(db, user_id=current_user.id, registry_uuid=registry_uuid)

    case_record = db.get(RegistryCaseRecord, registry_uuid)
    if case_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Canonical registry case record not found",
        )

    append_stmt: Select[tuple[RegistryAppendedDocument]] = (
        select(RegistryAppendedDocument)
        .where(
            RegistryAppendedDocument.user_id == current_user.id,
            RegistryAppendedDocument.registry_uuid == registry_uuid,
        )
        .order_by(RegistryAppendedDocument.created_at.desc())
        .limit(200)
    )
    append_rows = db.execute(append_stmt).scalars().all()
    return _build_case_response(case_record=case_record, append_rows=append_rows)


@router.patch(
    "/v1/registry/{registry_uuid}",
    response_model=RegistryCaseResponse,
    summary="Patch canonical case-level registry snapshot",
)
def patch_registry_case(
    registry_uuid: uuid.UUID,
    payload: RegistryCasePatchRequest,
    _ready: None = _ready_dep,
    current_user: AuthenticatedUser = _current_user_dep,
    db: Session = _db_dep,
) -> RegistryCaseResponse:
    _ensure_case_ownership(db, user_id=current_user.id, registry_uuid=registry_uuid)

    case_record = db.get(RegistryCaseRecord, registry_uuid)
    if case_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Canonical registry case record not found",
        )

    current_version = int(case_record.version or 1)
    if payload.expected_version is not None and payload.expected_version != current_version:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Version conflict",
                "expected_version": payload.expected_version,
                "current_version": current_version,
            },
        )

    base_registry = dict(case_record.registry_json or {})
    merged_registry = _deep_merge_dict(base_registry, payload.registry_patch)
    try:
        validated = RegistryRecord(**merged_registry)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid registry patch: {exc}",
        ) from exc

    case_record.registry_json = validated.model_dump(exclude_none=True)
    case_record.schema_version = _schema_version()
    case_record.version = current_version + 1
    case_record.updated_at = _utcnow()
    db.add(case_record)
    db.commit()
    db.refresh(case_record)

    append_stmt: Select[tuple[RegistryAppendedDocument]] = (
        select(RegistryAppendedDocument)
        .where(
            RegistryAppendedDocument.user_id == current_user.id,
            RegistryAppendedDocument.registry_uuid == registry_uuid,
        )
        .order_by(RegistryAppendedDocument.created_at.desc())
        .limit(200)
    )
    append_rows = db.execute(append_stmt).scalars().all()
    return _build_case_response(case_record=case_record, append_rows=append_rows)


__all__ = ["router"]
