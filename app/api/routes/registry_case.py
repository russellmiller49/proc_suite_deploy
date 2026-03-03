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
from app.registry.application.case_aggregator import CaseAggregator
from app.registry.schema import RegistryRecord
from app.registry_store.dependencies import get_registry_store_db
from app.registry_store.models import RegistryAppendedDocument, RegistryCaseRecord, RegistryRun
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


def _escape_pointer_token(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _collect_leaf_paths(value: Any, path: str) -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [path or "/"]
        out: list[str] = []
        for key, child in value.items():
            child_path = f"{path}/{_escape_pointer_token(key)}" if path else f"/{_escape_pointer_token(key)}"
            out.extend(_collect_leaf_paths(child, child_path))
        return out

    if isinstance(value, list):
        if not value:
            return [path or "/"]
        out: list[str] = []
        for idx, child in enumerate(value):
            child_path = f"{path}/{idx}" if path else f"/{idx}"
            out.extend(_collect_leaf_paths(child, child_path))
        return out

    return [path or "/"]


def _changed_leaf_paths(old: Any, new: Any, path: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        changed: list[str] = []
        keys = set(old.keys()) | set(new.keys())
        for key in sorted(keys):
            child_path = f"{path}/{_escape_pointer_token(key)}" if path else f"/{_escape_pointer_token(key)}"
            if key not in old:
                changed.extend(_collect_leaf_paths(new[key], child_path))
                continue
            if key not in new:
                changed.extend(_collect_leaf_paths(old[key], child_path))
                continue
            changed.extend(_changed_leaf_paths(old[key], new[key], child_path))
        return changed

    if isinstance(old, list) and isinstance(new, list):
        if old == new:
            return []
        if len(old) != len(new):
            return [path or "/"]

        changed: list[str] = []
        for idx, (left, right) in enumerate(zip(old, new, strict=False)):
            child_path = f"{path}/{idx}" if path else f"/{idx}"
            changed.extend(_changed_leaf_paths(left, right, child_path))
        return changed

    if old != new:
        return [path or "/"]
    return []


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
    id: str
    append_id: str
    created_at: str
    event_type: str
    document_kind: str
    is_synthetic: bool = False
    source_type: str | None = None
    source_modality: str | None = None
    event_subtype: str | None = None
    relative_day_offset: int | None = None
    event_title: str | None = None
    has_note_text: bool = False
    has_structured_data: bool = False
    structured_data: dict[str, Any] | None = None
    extracted_json: dict[str, Any] | None = None


class RegistryCaseResponse(BaseModel):
    registry_uuid: str
    schema_version: str
    version: int
    source_run_id: str | None = None
    updated_at: str
    registry: dict[str, Any]
    registry_json: dict[str, Any]
    manual_overrides: dict[str, Any] = Field(default_factory=dict)
    events: list[RegistryCaseEventSummary] = Field(default_factory=list)
    recent_events: list[RegistryCaseEventSummary] = Field(default_factory=list)


class RegistryCasePatchRequest(BaseModel):
    registry_patch: dict[str, Any] = Field(
        ...,
        description="Deep-merge patch applied to canonical registry JSON.",
    )
    expected_version: int | None = Field(
        default=None,
        description="Optional optimistic concurrency check against current case version.",
    )


class RegistryCaseRebuildRequest(BaseModel):
    reset_manual_overrides: bool = Field(
        default=False,
        description=(
            "If true, clear manual override locks before replaying baseline + appended events."
        ),
    )


def _build_event_summary(row: RegistryAppendedDocument) -> RegistryCaseEventSummary:
    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    structured_data = metadata.get("structured_data") if isinstance(metadata.get("structured_data"), dict) else None
    extracted_json = row.extracted_json if isinstance(row.extracted_json, dict) else None
    event_id = str(row.id)
    return RegistryCaseEventSummary(
        id=event_id,
        append_id=event_id,
        created_at=row.created_at.isoformat(),
        event_type=str(getattr(row, "event_type", row.document_kind or "pathology")),
        document_kind=str(row.document_kind or ""),
        source_type=row.source_type,
        source_modality=getattr(row, "source_modality", None),
        event_subtype=getattr(row, "event_subtype", None),
        relative_day_offset=getattr(row, "relative_day_offset", None),
        event_title=getattr(row, "event_title", None),
        has_note_text=bool(str(row.note_text or "").strip()),
        has_structured_data=bool(structured_data),
        structured_data=structured_data,
        extracted_json=extracted_json,
    )


def _safe_parse_iso_datetime(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _build_synthetic_procedure_event(
    *,
    case_record: RegistryCaseRecord,
    source_run: RegistryRun | None,
) -> RegistryCaseEventSummary | None:
    if not case_record.source_run_id:
        return None

    run_id = str(case_record.source_run_id)
    created_at: str | None = None
    has_note_text = True
    if source_run is not None:
        created_at = source_run.created_at.isoformat() if source_run.created_at else None
        has_note_text = bool(str(source_run.note_text or "").strip())

    if not created_at:
        created_at = case_record.created_at.isoformat() if case_record.created_at else _utcnow().isoformat()

    return RegistryCaseEventSummary(
        id=run_id,
        append_id=run_id,
        created_at=created_at,
        event_type="procedure_report",
        document_kind="procedure_report",
        is_synthetic=True,
        source_type="registry_run",
        source_modality=None,
        event_subtype=None,
        relative_day_offset=0,
        event_title=None,
        has_note_text=has_note_text,
        has_structured_data=False,
    )


def _build_case_response(
    *,
    case_record: RegistryCaseRecord,
    append_rows: list[RegistryAppendedDocument],
    source_run: RegistryRun | None = None,
) -> RegistryCaseResponse:
    events = [_build_event_summary(row) for row in append_rows]
    synthetic_procedure = _build_synthetic_procedure_event(case_record=case_record, source_run=source_run)
    if synthetic_procedure is not None:
        events.append(synthetic_procedure)

    events.sort(
        key=lambda ev: _safe_parse_iso_datetime(ev.created_at) or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    registry_json = dict(case_record.registry_json or {})
    manual_overrides = dict(case_record.manual_overrides or {})

    return RegistryCaseResponse(
        registry_uuid=str(case_record.registry_uuid),
        schema_version=str(case_record.schema_version or _schema_version()),
        version=int(case_record.version or 1),
        source_run_id=str(case_record.source_run_id) if case_record.source_run_id else None,
        updated_at=case_record.updated_at.isoformat(),
        registry=registry_json,
        registry_json=registry_json,
        manual_overrides=manual_overrides,
        events=events,
        recent_events=events,
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
    source_run = db.get(RegistryRun, case_record.source_run_id) if case_record.source_run_id else None
    return _build_case_response(case_record=case_record, append_rows=append_rows, source_run=source_run)


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

    changed_paths = sorted(set(_changed_leaf_paths(base_registry, merged_registry)))
    now_iso = _utcnow().isoformat()
    manual_overrides = dict(case_record.manual_overrides or {})
    for path in changed_paths:
        manual_overrides[path] = {
            "locked": True,
            "updated_at": now_iso,
            "source": "manual",
        }

    case_record.registry_json = validated.model_dump(exclude_none=True, mode="json")
    case_record.manual_overrides = manual_overrides
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
    source_run = db.get(RegistryRun, case_record.source_run_id) if case_record.source_run_id else None
    return _build_case_response(case_record=case_record, append_rows=append_rows, source_run=source_run)


@router.post(
    "/v1/registry/{registry_uuid}/rebuild",
    response_model=RegistryCaseResponse,
    summary="Rebuild canonical case snapshot from baseline run + all appended events",
)
def rebuild_registry_case(
    registry_uuid: uuid.UUID,
    payload: RegistryCaseRebuildRequest | None = None,
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

    rebuild_opts = payload or RegistryCaseRebuildRequest()

    # Reset canonical snapshot so replay starts from baseline/default rather than stale merged state.
    case_record.registry_json = RegistryRecord().model_dump(exclude_none=True, mode="json")
    case_record.schema_version = _schema_version()
    case_record.source_run_id = None
    if rebuild_opts.reset_manual_overrides:
        case_record.manual_overrides = {}
    case_record.updated_at = _utcnow()
    db.add(case_record)

    append_stmt_all: Select[tuple[RegistryAppendedDocument]] = (
        select(RegistryAppendedDocument)
        .where(
            RegistryAppendedDocument.user_id == current_user.id,
            RegistryAppendedDocument.registry_uuid == registry_uuid,
        )
        .order_by(RegistryAppendedDocument.created_at.asc(), RegistryAppendedDocument.id.asc())
    )
    append_rows_all = list(db.execute(append_stmt_all).scalars().all())
    for row in append_rows_all:
        row.aggregated_at = None
        row.aggregation_version = None
        row.extracted_json = None
        db.add(row)

    aggregator = CaseAggregator(strategy="reprocess_all")
    case_record = aggregator.aggregate(
        db=db,
        registry_uuid=registry_uuid,
        user_id=current_user.id,
    )

    db.commit()
    db.refresh(case_record)

    append_stmt_recent: Select[tuple[RegistryAppendedDocument]] = (
        select(RegistryAppendedDocument)
        .where(
            RegistryAppendedDocument.user_id == current_user.id,
            RegistryAppendedDocument.registry_uuid == registry_uuid,
        )
        .order_by(RegistryAppendedDocument.created_at.desc())
        .limit(200)
    )
    append_rows_recent = db.execute(append_stmt_recent).scalars().all()
    source_run = db.get(RegistryRun, case_record.source_run_id) if case_record.source_run_id else None
    return _build_case_response(
        case_record=case_record,
        append_rows=append_rows_recent,
        source_run=source_run,
    )


__all__ = ["router", "RegistryCaseResponse", "RegistryCaseEventSummary"]
