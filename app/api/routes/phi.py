"""PHI endpoints for preview, submission, status, and re-identification.

Raw PHI stays inside PHIService/PHIVault; responses expose only scrubbed text
except for the reidentify endpoint (intended for PHI-safe UI only).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.phi_dependencies import get_phi_service, get_phi_session
from app.phi import models
from app.phi.ports import ScrubResult
from app.phi.service import PHIService

router = APIRouter(prefix="/v1/phi", tags=["phi"])
logger = logging.getLogger("phi_api")
_phi_service_dep = Depends(get_phi_service)
_phi_session_dep = Depends(get_phi_session)


class ScrubbedEntityModel(BaseModel):
    placeholder: str
    entity_type: str
    original_start: int
    original_end: int


class ScrubPreviewRequest(BaseModel):
    text: str = Field(..., description="Raw clinical text (synthetic in demo)")
    document_type: str | None = None
    specialty: str | None = None


class ScrubPreviewResponse(BaseModel):
    scrubbed_text: str
    entities: List[ScrubbedEntityModel]


class SubmitRequest(BaseModel):
    text: str = Field(..., description="Raw clinical text to vault (synthetic in demo)")
    submitted_by: str = Field(..., description="User identifier submitting PHI")
    document_type: str | None = None
    specialty: str | None = None
    confirmed_entities: List[ScrubbedEntityModel] | None = None


class SubmitResponse(BaseModel):
    procedure_id: str
    status: str
    scrubbed_text: str
    entities: List[ScrubbedEntityModel]


class StatusResponse(BaseModel):
    procedure_id: str
    status: str
    document_type: str | None = None
    specialty: str | None = None
    submitted_by: str | None = None
    created_at: str | None = None


class ProcedureReviewResponse(BaseModel):
    procedure_id: str
    status: str
    scrubbed_text: str
    entities: List[ScrubbedEntityModel]
    document_type: str | None = None
    specialty: str | None = None
    submitted_by: str | None = None
    created_at: str | None = None


class ScrubbingFeedbackRequest(BaseModel):
    scrubbed_text: str
    entities: List[ScrubbedEntityModel]
    reviewer_id: str
    reviewer_email: str | None = None
    reviewer_role: str | None = None
    comment: str | None = None


class ReidentifyRequest(BaseModel):
    procedure_id: str
    user_id: str
    user_email: str | None = None
    user_role: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None


class ReidentifyResponse(BaseModel):
    raw_text: str


def _to_entities(scrub_result: ScrubResult) -> list[ScrubbedEntityModel]:
    return [ScrubbedEntityModel(**entity) for entity in scrub_result.entities]


def _from_entity_map(entity_map) -> list[ScrubbedEntityModel]:
    if entity_map is None:
        return []
    return [ScrubbedEntityModel(**entity) for entity in entity_map]


@router.post(
    "/scrub/preview",
    response_model=ScrubPreviewResponse,
    summary="Preview PHI scrubbing (no persistence)",
)
def preview_scrub(
    payload: ScrubPreviewRequest,
    phi_service: PHIService = _phi_service_dep,
) -> ScrubPreviewResponse:
    start = time.perf_counter()
    scrub_result = phi_service.preview(
        text=payload.text,
        document_type=payload.document_type,
        specialty=payload.specialty,
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "phi_preview",
        extra={
            "duration_ms": round(duration_ms, 2),
            "entity_count": len(scrub_result.entities),
            "document_type": payload.document_type,
            "specialty": payload.specialty,
        },
    )
    return ScrubPreviewResponse(
        scrubbed_text=scrub_result.scrubbed_text,
        entities=_to_entities(scrub_result),
    )


@router.post(
    "/submit",
    response_model=SubmitResponse,
    summary="Submit and vault PHI with scrubbing",
)
def submit_phi(
    payload: SubmitRequest,
    phi_service: PHIService = _phi_service_dep,
) -> SubmitResponse:
    start = time.perf_counter()
    if payload.confirmed_entities is not None:
        # Manual override: trust the provided entities completely
        manual_entities = [entity.model_dump() for entity in payload.confirmed_entities]
        scrub_result = phi_service.scrub_with_manual_entities(
            text=payload.text,
            entities=manual_entities,
        )
        manual_override = True
    else:
        # Auto-scrub logic
        scrub_result = phi_service.preview(
            text=payload.text,
            document_type=payload.document_type,
            specialty=payload.specialty,
        )
        manual_override = False

    proc = phi_service.vault_phi(
        raw_text=payload.text,
        scrub_result=scrub_result,
        submitted_by=payload.submitted_by,
        document_type=payload.document_type,
        specialty=payload.specialty,
    )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "phi_submit",
        extra={
            "duration_ms": round(duration_ms, 2),
            "entity_count": len(scrub_result.entities),
            "procedure_id": str(proc.id),
            "document_type": payload.document_type,
            "specialty": payload.specialty,
            "manual_override": manual_override,
        },
    )
    return SubmitResponse(
        procedure_id=str(proc.id),
        status=proc.status.value if hasattr(proc.status, "value") else str(proc.status),
        scrubbed_text=scrub_result.scrubbed_text,
        entities=_to_entities(scrub_result),
    )


@router.get(
    "/status/{procedure_id}",
    response_model=StatusResponse,
    summary="Check PHI record status",
)
def get_status(
    procedure_id: str,
    db: Session = _phi_session_dep,
) -> StatusResponse:
    try:
        proc_uuid = uuid.UUID(procedure_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid procedure_id",
        ) from exc

    proc = db.get(models.ProcedureData, proc_uuid)
    if proc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Procedure not found",
        )

    return StatusResponse(
        procedure_id=str(proc.id),
        status=proc.status.value if hasattr(proc.status, "value") else str(proc.status),
        document_type=proc.document_type,
        specialty=proc.specialty,
        submitted_by=proc.submitted_by,
        created_at=proc.created_at.isoformat() if proc.created_at else None,
    )


@router.get(
    "/procedure/{procedure_id}",
    response_model=ProcedureReviewResponse,
    summary="Fetch scrubbed content for PHI review (no raw PHI)",
)
def get_procedure_for_review(
    procedure_id: str,
    phi_service: PHIService = _phi_service_dep,
) -> ProcedureReviewResponse:
    try:
        proc_uuid = uuid.UUID(procedure_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid procedure_id",
        ) from exc

    try:
        proc = phi_service.get_procedure_for_review(procedure_data_id=proc_uuid)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Procedure not found",
        ) from exc

    return ProcedureReviewResponse(
        procedure_id=str(proc.id),
        status=proc.status.value if hasattr(proc.status, "value") else str(proc.status),
        scrubbed_text=proc.scrubbed_text,
        entities=_from_entity_map(proc.entity_map),
        document_type=proc.document_type,
        specialty=proc.specialty,
        submitted_by=proc.submitted_by,
        created_at=proc.created_at.isoformat() if proc.created_at else None,
    )


@router.post(
    "/procedure/{procedure_id}/feedback",
    response_model=ProcedureReviewResponse,
    summary="Apply scrubbing feedback and mark procedure as reviewed",
)
def submit_scrubbing_feedback(
    procedure_id: str,
    payload: ScrubbingFeedbackRequest,
    phi_service: PHIService = _phi_service_dep,
) -> ProcedureReviewResponse:
    try:
        proc_uuid = uuid.UUID(procedure_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid procedure_id",
        ) from exc

    start = time.perf_counter()
    try:
        proc = phi_service.apply_scrubbing_feedback(
            procedure_data_id=proc_uuid,
            scrubbed_text=payload.scrubbed_text,
            entities=[entity.model_dump() for entity in payload.entities],
            reviewer_id=payload.reviewer_id,
            reviewer_email=payload.reviewer_email,
            reviewer_role=payload.reviewer_role,
            comment=payload.comment,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Procedure not found or missing PHI vault",
        ) from exc
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "phi_feedback",
        extra={
            "duration_ms": round(duration_ms, 2),
            "procedure_id": str(proc.id),
            "entity_count": len(proc.entity_map or []),
            "status": proc.status.value if hasattr(proc.status, "value") else str(proc.status),
        },
    )

    return ProcedureReviewResponse(
        procedure_id=str(proc.id),
        status=proc.status.value if hasattr(proc.status, "value") else str(proc.status),
        scrubbed_text=proc.scrubbed_text,
        entities=_from_entity_map(proc.entity_map),
        document_type=proc.document_type,
        specialty=proc.specialty,
        submitted_by=proc.submitted_by,
        created_at=proc.created_at.isoformat() if proc.created_at else None,
    )


@router.post(
    "/reidentify",
    response_model=ReidentifyResponse,
    summary="Reidentify raw PHI text (PHI-safe UI only)",
)
def reidentify_phi(
    payload: ReidentifyRequest,
    phi_service: PHIService = _phi_service_dep,
) -> ReidentifyResponse:
    try:
        proc_uuid = uuid.UUID(payload.procedure_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid procedure_id",
        ) from exc

    start = time.perf_counter()
    try:
        plaintext = phi_service.reidentify(
            procedure_data_id=proc_uuid,
            user_id=payload.user_id,
            user_email=payload.user_email,
            user_role=payload.user_role,
            ip_address=payload.ip_address,
            user_agent=payload.user_agent,
            request_id=payload.request_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Procedure not found or missing PHI vault",
        ) from exc
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "phi_reidentify",
        extra={
            "duration_ms": round(duration_ms, 2),
            "procedure_id": payload.procedure_id,
            "user_id": payload.user_id,
        },
    )

    return ReidentifyResponse(raw_text=plaintext)
