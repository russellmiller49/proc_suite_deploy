"""Registry Runs persistence endpoints.

These endpoints wrap the existing stateless unified pipeline with a stateful
store for:
- scrubbed note text
- raw pipeline outputs
- optional UI corrections payloads
- a single feedback submission per run

Non-negotiable: never persist raw PHI.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import Select, and_, desc, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_coding_service, get_registry_service
from app.api.phi_dependencies import get_phi_scrubber
from app.api.readiness import require_ready
from app.api.schemas import UnifiedProcessRequest, UnifiedProcessResponse
from app.api.services.unified_pipeline import run_unified_pipeline_logic
from app.coder.application.coding_service import CodingService
from app.registry.application.registry_service import RegistryService
from app.registry_store.dependencies import get_registry_store_db
from app.registry_store.models import RegistryRun
from app.registry_store.phi_gate import scan_text_for_phi_risk


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _enforce_registry_runs_enabled() -> None:
    if not _truthy_env("REGISTRY_RUNS_PERSIST_ENABLED"):
        raise HTTPException(status_code=503, detail="Registry Runs persistence is disabled")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _note_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _schema_version() -> str:
    return (os.getenv("REGISTRY_SCHEMA_VERSION") or "v3").strip()


def _pipeline_config(payload: UnifiedProcessRequest) -> dict[str, Any]:
    return {
        "already_scrubbed": bool(payload.already_scrubbed),
        "include_financials": bool(payload.include_financials),
        "explain": bool(payload.explain),
        "include_v3_event_log": bool(payload.include_v3_event_log),
        "locality": payload.locality,
        "registry_schema_version": _schema_version(),
        "registry_extraction_engine": os.getenv("REGISTRY_EXTRACTION_ENGINE", "").strip(),
        "registry_self_correct_enabled": _truthy_env("REGISTRY_SELF_CORRECT_ENABLED"),
        "procsuite_fast_mode": _truthy_env("PROCSUITE_FAST_MODE"),
        "coder_require_phi_review": _truthy_env("CODER_REQUIRE_PHI_REVIEW"),
        "created_via": "registry_runs_api",
    }


def _serialize_run(run: RegistryRun) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "submitter_name": run.submitter_name,
        "note_text": run.note_text,
        "note_sha256": run.note_sha256,
        "schema_version": run.schema_version,
        "pipeline_config": run.pipeline_config,
        "raw_response_json": run.raw_response_json,
        "corrected_response_json": run.corrected_response_json,
        "edited_tables_json": run.edited_tables_json,
        "correction_editor_name": run.correction_editor_name,
        "corrected_at": run.corrected_at.isoformat() if run.corrected_at else None,
        "feedback_reviewer_name": run.feedback_reviewer_name,
        "feedback_rating": run.feedback_rating,
        "feedback_comment": run.feedback_comment,
        "feedback_submitted_at": run.feedback_submitted_at.isoformat()
        if run.feedback_submitted_at
        else None,
        "needs_manual_review": bool(run.needs_manual_review),
        "review_status": run.review_status,
        "kb_version": run.kb_version,
        "kb_hash": run.kb_hash,
        "processing_time_ms": run.processing_time_ms,
    }


router = APIRouter(tags=["registry-runs"])

_ready_dep = Depends(require_ready)
_registry_service_dep = Depends(get_registry_service)
_coding_service_dep = Depends(get_coding_service)
_phi_scrubber_dep = Depends(get_phi_scrubber)
_registry_store_db_dep = Depends(get_registry_store_db)


class RegistryRunCreateRequest(BaseModel):
    note: str = Field(
        ...,
        description="Procedure note text (scrubbed-only when already_scrubbed=true)",
    )
    already_scrubbed: bool = Field(
        False, description="If true, server will skip PHI scrubbing and treat note as scrubbed."
    )
    locality: str = Field("00", description="Geographic locality for RVU calculations")
    include_financials: bool = Field(True, description="Whether to include RVU/payment info")
    explain: bool = Field(False, description="Include extraction evidence/rationales")
    include_v3_event_log: bool = Field(
        False,
        description="If true, include raw event-log V3 output in the persisted response payload.",
    )
    submitter_name: str | None = Field(
        None, description="Free-text submitter name (no auth yet)", max_length=255
    )


class RegistryRunCreateResponse(BaseModel):
    run_id: str
    result: UnifiedProcessResponse


class RegistryRunFeedbackRequest(BaseModel):
    reviewer_name: str = Field(..., min_length=1, max_length=255)
    rating: int = Field(..., ge=1, le=10)
    comment: str | None = Field(None, max_length=10_000)


class RegistryRunCorrectionRequest(BaseModel):
    corrected_response_json: Any | None = None
    edited_tables_json: Any | None = None
    editor_name: str | None = Field(None, max_length=255)


class RegistryRunListItem(BaseModel):
    id: str
    created_at: datetime | None
    submitter_name: str | None
    schema_version: str
    needs_manual_review: bool
    review_status: str
    note_sha256: str
    feedback_rating: int | None
    has_feedback: bool
    has_correction: bool


class RegistryRunListResponse(BaseModel):
    items: list[RegistryRunListItem]
    limit: int
    offset: int


class RegistryRunGetResponse(BaseModel):
    run: dict[str, Any]


@router.post(
    "/v1/registry/runs",
    response_model=RegistryRunCreateResponse,
    response_model_exclude_none=True,
    summary="Run unified pipeline and persist the run (scrubbed-only)",
)
async def create_registry_run(
    payload: RegistryRunCreateRequest,
    request: Request,
    response: Response,
    _ready: None = _ready_dep,
    registry_service: RegistryService = _registry_service_dep,
    coding_service: CodingService = _coding_service_dep,
    phi_scrubber=_phi_scrubber_dep,
    db: Session = _registry_store_db_dep,
) -> RegistryRunCreateResponse:
    _enforce_registry_runs_enabled()

    response.headers["X-Registry-Runs"] = "enabled"

    unified_payload = UnifiedProcessRequest(
        note=payload.note,
        already_scrubbed=payload.already_scrubbed,
        locality=payload.locality,
        include_financials=payload.include_financials,
        explain=payload.explain,
        include_v3_event_log=payload.include_v3_event_log,
    )

    result, scrubbed_note_text_used, meta = await run_unified_pipeline_logic(
        payload=unified_payload,
        request=request,
        registry_service=registry_service,
        coding_service=coding_service,
        phi_scrubber=phi_scrubber,
    )

    note_sha256 = _note_sha256(scrubbed_note_text_used)

    phi_risk_reasons = scan_text_for_phi_risk(scrubbed_note_text_used)
    allow_phi_risk_persist = _truthy_env("REGISTRY_RUNS_ALLOW_PHI_RISK_PERSIST")
    if phi_risk_reasons and not allow_phi_risk_persist:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "PHI risk detected in scrubbed text; persistence rejected",
                "reasons": phi_risk_reasons,
            },
        )

    needs_manual_review = bool(result.needs_manual_review)
    review_status = "needs_review" if needs_manual_review else "new"
    if phi_risk_reasons and allow_phi_risk_persist:
        needs_manual_review = True
        review_status = "phi_risk"

    raw_response_json = result.model_dump(exclude_none=True)
    pipeline_cfg = _pipeline_config(unified_payload)
    if phi_risk_reasons:
        pipeline_cfg["phi_risk_reasons"] = phi_risk_reasons

    run = RegistryRun(
        id=uuid.uuid4(),
        created_at=_utcnow(),
        submitter_name=(payload.submitter_name or None),
        note_text=scrubbed_note_text_used,
        note_sha256=note_sha256,
        schema_version=_schema_version(),
        pipeline_config=pipeline_cfg,
        raw_response_json=raw_response_json,
        kb_version=str(meta.get("kb_version") or "") or None,
        kb_hash=str(meta.get("kb_hash") or "") or None,
        processing_time_ms=int(meta.get("processing_time_ms") or 0) or None,
        needs_manual_review=needs_manual_review,
        review_status=review_status,
    )

    db.add(run)
    db.commit()

    return RegistryRunCreateResponse(run_id=str(run.id), result=result)


@router.post(
    "/v1/registry/runs/{run_id}/feedback",
    response_model_exclude_none=True,
    summary="Submit feedback for a run (one-time)",
)
def submit_registry_run_feedback(
    run_id: uuid.UUID,
    payload: RegistryRunFeedbackRequest,
    _ready: None = _ready_dep,
    db: Session = _registry_store_db_dep,
) -> dict[str, Any]:
    _enforce_registry_runs_enabled()

    run = db.get(RegistryRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.feedback_submitted_at is not None or run.feedback_rating is not None:
        raise HTTPException(status_code=409, detail="Feedback already submitted for this run")

    run.feedback_reviewer_name = payload.reviewer_name
    run.feedback_rating = int(payload.rating)
    run.feedback_comment = payload.comment
    run.feedback_submitted_at = _utcnow()
    run.review_status = "feedback_submitted"

    db.add(run)
    db.commit()

    return {"ok": True, "run_id": str(run.id), "review_status": run.review_status}


@router.put(
    "/v1/registry/runs/{run_id}/correction",
    response_model_exclude_none=True,
    summary="Upsert UI correction payload for a run",
)
def upsert_registry_run_correction(
    run_id: uuid.UUID,
    payload: RegistryRunCorrectionRequest,
    _ready: None = _ready_dep,
    db: Session = _registry_store_db_dep,
) -> dict[str, Any]:
    _enforce_registry_runs_enabled()

    run = db.get(RegistryRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    run.corrected_response_json = payload.corrected_response_json
    run.edited_tables_json = payload.edited_tables_json
    run.correction_editor_name = payload.editor_name
    run.corrected_at = _utcnow()
    run.review_status = "corrected"

    db.add(run)
    db.commit()

    return {"ok": True, "run_id": str(run.id), "review_status": run.review_status}


@router.get(
    "/v1/registry/runs",
    response_model=RegistryRunListResponse,
    response_model_exclude_none=True,
    summary="List registry runs (admin-lite, no auth yet)",
)
def list_registry_runs(
    _ready: None = _ready_dep,
    db: Session = _registry_store_db_dep,
    limit: int = 50,
    offset: int = 0,
    submitter_name: str | None = None,
    review_status: str | None = None,
    needs_manual_review: bool | None = None,
    has_feedback: bool | None = None,
    has_correction: bool | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
) -> RegistryRunListResponse:
    _enforce_registry_runs_enabled()

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    query: Select[Any] = select(RegistryRun)
    filters = []

    if submitter_name:
        filters.append(RegistryRun.submitter_name == submitter_name)
    if review_status:
        filters.append(RegistryRun.review_status == review_status)
    if needs_manual_review is not None:
        filters.append(RegistryRun.needs_manual_review == needs_manual_review)
    if has_feedback is not None:
        filters.append(
            RegistryRun.feedback_submitted_at.is_not(None)
            if has_feedback
            else RegistryRun.feedback_submitted_at.is_(None)
        )
    if has_correction is not None:
        filters.append(
            RegistryRun.corrected_at.is_not(None)
            if has_correction
            else RegistryRun.corrected_at.is_(None)
        )
    if created_from is not None:
        filters.append(RegistryRun.created_at >= created_from)
    if created_to is not None:
        filters.append(RegistryRun.created_at <= created_to)

    if filters:
        query = query.where(and_(*filters))

    query = query.order_by(desc(RegistryRun.created_at)).offset(offset).limit(limit)
    rows = list(db.execute(query).scalars().all())

    items = [
        RegistryRunListItem(
            id=str(r.id),
            created_at=r.created_at,
            submitter_name=r.submitter_name,
            schema_version=r.schema_version,
            needs_manual_review=bool(r.needs_manual_review),
            review_status=r.review_status,
            note_sha256=r.note_sha256,
            feedback_rating=r.feedback_rating,
            has_feedback=r.feedback_submitted_at is not None,
            has_correction=r.corrected_at is not None,
        )
        for r in rows
    ]

    return RegistryRunListResponse(items=items, limit=limit, offset=offset)


@router.get(
    "/v1/registry/runs/{run_id}",
    response_model=RegistryRunGetResponse,
    response_model_exclude_none=True,
    summary="Get a persisted registry run by id",
)
def get_registry_run(
    run_id: uuid.UUID,
    _ready: None = _ready_dep,
    db: Session = _registry_store_db_dep,
) -> RegistryRunGetResponse:
    _enforce_registry_runs_enabled()

    run = db.get(RegistryRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return RegistryRunGetResponse(run=_serialize_run(run))


@router.get(
    "/v1/registry/export",
    include_in_schema=True,
    summary="Export registry runs as JSONL (scrubbed-only)",
)
def export_registry_runs(
    _ready: None = _ready_dep,
    db: Session = _registry_store_db_dep,
    has_feedback: bool | None = None,
    has_correction: bool | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
) -> StreamingResponse:
    _enforce_registry_runs_enabled()

    query: Select[Any] = select(RegistryRun).order_by(desc(RegistryRun.created_at))
    filters = []
    if has_feedback is not None:
        filters.append(
            RegistryRun.feedback_submitted_at.is_not(None)
            if has_feedback
            else RegistryRun.feedback_submitted_at.is_(None)
        )
    if has_correction is not None:
        filters.append(
            RegistryRun.corrected_at.is_not(None)
            if has_correction
            else RegistryRun.corrected_at.is_(None)
        )
    if created_from is not None:
        filters.append(RegistryRun.created_at >= created_from)
    if created_to is not None:
        filters.append(RegistryRun.created_at <= created_to)
    if filters:
        query = query.where(and_(*filters))

    def _iter_jsonl() -> Iterator[bytes]:
        for run in db.execute(query).scalars():
            payload = _serialize_run(run)
            yield (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    headers = {"Content-Disposition": "attachment; filename=registry_runs.jsonl"}
    return StreamingResponse(_iter_jsonl(), media_type="application/x-ndjson", headers=headers)


__all__ = ["router"]
