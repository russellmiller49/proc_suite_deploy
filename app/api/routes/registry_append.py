"""Case-level append endpoint keyed by registry_uuid."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.api.auth import AuthenticatedUser, get_current_user
from app.api.phi_dependencies import get_phi_scrubber
from app.api.phi_redaction import apply_phi_redaction
from app.api.readiness import require_ready
from app.api.routes.registry_case import RegistryCaseResponse, _build_case_response
from app.api.services.bundle_processing import count_date_like_strings
from app.registry.application.case_aggregator import CaseAggregator
from app.registry.schema import RegistryRecord
from app.registry_store.dependencies import get_registry_store_db
from app.registry_store.models import RegistryAppendedDocument, RegistryCaseRecord, RegistryRun
from app.registry_store.phi_gate import scan_text_for_phi_risk
from app.vault.models import UserPatientVault


router = APIRouter(tags=["registry-append"])
logger = logging.getLogger(__name__)

_ready_dep = Depends(require_ready)
_current_user_dep = Depends(get_current_user)
_db_dep = Depends(get_registry_store_db)
_phi_scrubber_dep = Depends(get_phi_scrubber)

_ALLOWED_EVENT_TYPES = {
    "pathology",
    "imaging",
    "clinical_update",
    "treatment_update",
    "complication",
    "procedure_addendum",
    "other",
}
_LEGACY_EVENT_TYPE_MAP = {
    "clinical_status": "clinical_update",
    "procedure": "procedure_addendum",
}


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _note_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_event_type(event_type: str | None, document_kind: str | None) -> str:
    candidate = str(event_type or document_kind or "pathology").strip().lower()
    candidate = _LEGACY_EVENT_TYPE_MAP.get(candidate, candidate)
    if candidate not in _ALLOWED_EVENT_TYPES:
        allowed = ", ".join(sorted(_ALLOWED_EVENT_TYPES | set(_LEGACY_EVENT_TYPE_MAP.keys())))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported event_type '{candidate}'. Allowed: {allowed}",
        )
    return candidate


def _enforce_absolute_date_guard(
    *,
    text: str,
    event_title: str | None,
    structured_data: dict[str, Any] | None,
    allow_absolute_dates: bool,
) -> tuple[int, int, int]:
    text_count = count_date_like_strings(text)
    title_count = count_date_like_strings(event_title or "")
    structured_payload = ""
    if isinstance(structured_data, dict) and structured_data:
        structured_payload = json.dumps(structured_data, separators=(",", ":"), sort_keys=True)
    structured_count = count_date_like_strings(structured_payload)
    total = text_count + title_count + structured_count

    if total == 0:
        return text_count, title_count, structured_count

    if not allow_absolute_dates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Append payload contains absolute date-like strings. "
                "Send relative timing only (relative_day_offset / T±N tokens)."
            ),
        )

    if not _truthy_env("PROCSUITE_ALLOW_ABSOLUTE_DATES", default=False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "allow_absolute_dates=true is blocked by server policy. "
                "Set PROCSUITE_ALLOW_ABSOLUTE_DATES=1 for explicit audited override."
            ),
        )

    return text_count, title_count, structured_count


def _ensure_case_record(db: Session, *, registry_uuid: uuid.UUID) -> RegistryCaseRecord:
    record = db.get(RegistryCaseRecord, registry_uuid)
    if record is not None:
        return record

    now = _utcnow()
    validated = RegistryRecord()
    record = RegistryCaseRecord(
        registry_uuid=registry_uuid,
        registry_json=validated.model_dump(exclude_none=True, mode="json"),
        schema_version=(os.getenv("REGISTRY_SCHEMA_VERSION") or "v3").strip(),
        version=1,
        source_run_id=None,
        manual_overrides={},
        created_at=now,
        updated_at=now,
    )
    db.add(record)
    db.flush()
    return record


class RegistryAppendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(
        None,
        description=(
            "Optional appended scrubbed note text "
            "(or raw when already_scrubbed=false). May be omitted for structured-only events."
        ),
    )
    note: str | None = Field(
        None,
        description="Legacy alias for text (kept for backward compatibility).",
    )
    already_scrubbed: bool = Field(
        True,
        description="If false, server performs PHI scrubbing before persistence (blocked in strict ZK mode).",
    )
    event_type: str | None = Field(
        None,
        description=(
            "Canonical event type "
            "(pathology, imaging, clinical_update, treatment_update, complication, procedure_addendum, other)."
        ),
    )
    structured_data: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured event payload for note-less status updates.",
    )
    source_type: str | None = Field(
        None,
        description="Optional source marker (e.g., camera_ocr, pdf_local, manual_entry).",
    )
    source_modality: str | None = Field(
        None,
        description="Optional modality marker for imaging events (e.g., ct, pet_ct, cta, mri).",
    )
    event_subtype: str | None = Field(
        None,
        description="Optional subtype marker (e.g., preop, followup, restaging, surveillance).",
    )
    event_title: str | None = Field(
        None,
        description="Optional short safe label (must not include absolute dates).",
    )
    ocr_correction_applied: bool = Field(
        False,
        description="Whether post-redaction OCR correction was applied.",
    )
    document_kind: str = Field(
        "pathology",
        max_length=64,
        description=(
            "Legacy alias for event type. New clients should send event_type; "
            "document_kind is kept for backward compatibility."
        ),
    )
    relative_day_offset: int | None = Field(
        None,
        description=(
            "Relative day offset from client-held index date. Signed integer; "
            "absolute dates must not be sent to the server."
        ),
    )
    allow_absolute_dates: bool = Field(
        False,
        description=(
            "Audited escape hatch for date-like strings. Still blocked unless "
            "PROCSUITE_ALLOW_ABSOLUTE_DATES=1."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional non-PHI metadata for append tracking.",
    )


class RegistryAppendResponse(RegistryCaseResponse):
    append_id: str
    user_id: str
    document_kind: str
    source_type: str | None = None
    ocr_correction_applied: bool = False
    note_sha256: str
    created_at: str


def _append_rows_for_case(db: Session, *, user_id: str, registry_uuid: uuid.UUID) -> list[RegistryAppendedDocument]:
    stmt: Select[tuple[RegistryAppendedDocument]] = (
        select(RegistryAppendedDocument)
        .where(
            RegistryAppendedDocument.user_id == user_id,
            RegistryAppendedDocument.registry_uuid == registry_uuid,
        )
        .order_by(RegistryAppendedDocument.created_at.desc())
        .limit(200)
    )
    return list(db.execute(stmt).scalars().all())


@router.post(
    "/v1/registry/{registry_uuid}/append",
    response_model=RegistryAppendResponse,
    summary="Append a scrubbed case event and aggregate into canonical snapshot",
)
def append_registry_document(
    registry_uuid: uuid.UUID,
    payload: RegistryAppendRequest,
    _ready: None = _ready_dep,
    current_user: AuthenticatedUser = _current_user_dep,
    db: Session = _db_dep,
    phi_scrubber=_phi_scrubber_dep,
) -> RegistryAppendResponse:
    # Ensure the target case belongs to the current user.
    case_stmt: Select[tuple[UserPatientVault]] = select(UserPatientVault).where(
        UserPatientVault.user_id == current_user.id,
        UserPatientVault.registry_uuid == registry_uuid,
    )
    case_row = db.execute(case_stmt).scalar_one_or_none()
    if case_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Registry case not found for this user",
        )

    strict_zk = _truthy_env("PROCSUITE_STRICT_ZK", default=False)
    if strict_zk and not payload.already_scrubbed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strict ZK mode requires already_scrubbed=true",
        )

    note_input = payload.text if payload.text is not None else payload.note
    note_input = str(note_input or "")
    if not note_input.strip() and payload.structured_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either text/note or structured_data is required for append events",
        )

    text_date_count, title_date_count, structured_date_count = _enforce_absolute_date_guard(
        text=note_input,
        event_title=payload.event_title,
        structured_data=payload.structured_data,
        allow_absolute_dates=bool(payload.allow_absolute_dates),
    )

    normalized_event_type = _normalize_event_type(payload.event_type, payload.document_kind)

    if payload.already_scrubbed or not note_input.strip():
        note_text = note_input
    else:
        redaction = apply_phi_redaction(note_input, phi_scrubber)
        note_text = redaction.text

    phi_risk_reasons = scan_text_for_phi_risk(note_text)
    allow_phi_risk_persist = _truthy_env("REGISTRY_RUNS_ALLOW_PHI_RISK_PERSIST")
    if phi_risk_reasons and not allow_phi_risk_persist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "PHI risk detected in scrubbed text; append rejected",
                "reasons": phi_risk_reasons,
            },
        )

    metadata_payload = dict(payload.metadata or {})
    if payload.structured_data is not None:
        metadata_payload["structured_data"] = payload.structured_data
    if payload.allow_absolute_dates and (text_date_count or title_date_count or structured_date_count):
        metadata_payload["absolute_date_override"] = {
            "enabled": True,
            "text_date_like_count": int(text_date_count),
            "title_date_like_count": int(title_date_count),
            "structured_date_like_count": int(structured_date_count),
            "audited_at": _utcnow().isoformat(),
        }

    row = RegistryAppendedDocument(
        id=uuid.uuid4(),
        user_id=current_user.id,
        registry_uuid=registry_uuid,
        note_text=note_text,
        note_sha256=_note_sha256(note_text),
        event_type=normalized_event_type,
        document_kind=normalized_event_type,
        source_type=(payload.source_type or None),
        source_modality=(payload.source_modality or None),
        event_subtype=(payload.event_subtype or None),
        event_title=(payload.event_title or None),
        relative_day_offset=payload.relative_day_offset,
        ocr_correction_applied=bool(payload.ocr_correction_applied),
        metadata_json=metadata_payload or None,
        created_at=_utcnow(),
    )

    db.add(row)
    db.flush()

    aggregate_on_append = _truthy_env("REGISTRY_AGGREGATE_ON_APPEND", default=True)
    try:
        if aggregate_on_append:
            aggregator = CaseAggregator()
            case_record = aggregator.aggregate(
                db=db,
                registry_uuid=registry_uuid,
                user_id=current_user.id,
            )
        else:
            case_record = _ensure_case_record(db, registry_uuid=registry_uuid)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception(
            "registry_append aggregation_failed registry_uuid=%s append_id=%s event_type=%s",
            registry_uuid,
            row.id,
            normalized_event_type,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to aggregate appended event",
        ) from exc

    db.commit()
    db.refresh(row)
    db.refresh(case_record)

    append_rows = _append_rows_for_case(db, user_id=current_user.id, registry_uuid=registry_uuid)
    source_run = db.get(RegistryRun, case_record.source_run_id) if case_record.source_run_id else None
    case_response = _build_case_response(case_record=case_record, append_rows=append_rows, source_run=source_run)

    response_payload = {
        **case_response.model_dump(),
        "append_id": str(row.id),
        "user_id": str(row.user_id),
        "document_kind": str(row.document_kind),
        "source_type": row.source_type,
        "ocr_correction_applied": bool(row.ocr_correction_applied),
        "note_sha256": str(row.note_sha256),
        "created_at": row.created_at.isoformat(),
    }
    return RegistryAppendResponse(**response_payload)


__all__ = ["router", "RegistryAppendRequest", "RegistryAppendResponse"]
