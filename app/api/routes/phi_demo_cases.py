"""Endpoints for non-PHI PHI demo cases backed by Supabase or in-memory."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.phi_demo_store import get_phi_demo_store

router = APIRouter(prefix="/api/v1/phi-demo", tags=["phi-demo"])


class PhiDemoCaseCreateRequest(BaseModel):
    synthetic_patient_label: str = Field(..., description="Synthetic label, e.g., 'Patient X'")
    procedure_date: str | None = Field(None, description="ISO date string (non-PHI)")
    operator_name: str | None = Field(None, description="Synthetic operator name")
    scenario_label: str | None = Field(None, description="Scenario title (non-PHI)")
    procedure_id: str | None = Field(None, description="Optional linked procedure_id (UUID)")


class PhiDemoCaseResponse(BaseModel):
    id: str
    procedure_id: str | None = None
    synthetic_patient_label: str | None = None
    procedure_date: str | None = None
    operator_name: str | None = None
    scenario_label: str | None = None
    created_at: str


class AttachProcedureRequest(BaseModel):
    procedure_id: str


def _serialize_case(case) -> PhiDemoCaseResponse:
    return PhiDemoCaseResponse(**case.to_dict())


@router.get("/cases", response_model=List[PhiDemoCaseResponse])
def list_cases() -> List[PhiDemoCaseResponse]:
    store = get_phi_demo_store()
    return [_serialize_case(c) for c in store.list_cases()]


@router.post("/cases", response_model=PhiDemoCaseResponse, status_code=201)
def create_case(payload: PhiDemoCaseCreateRequest) -> PhiDemoCaseResponse:
    store = get_phi_demo_store()
    procedure_id = None
    if payload.procedure_id:
        try:
            procedure_id = uuid.UUID(payload.procedure_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid procedure_id") from exc

    case = store.create_case(
        synthetic_patient_label=payload.synthetic_patient_label,
        procedure_date=payload.procedure_date,
        operator_name=payload.operator_name,
        scenario_label=payload.scenario_label,
        procedure_id=procedure_id,
    )
    return _serialize_case(case)


@router.put("/cases/{case_id}/procedure", response_model=PhiDemoCaseResponse)
def attach_procedure(case_id: str, payload: AttachProcedureRequest) -> PhiDemoCaseResponse:
    store = get_phi_demo_store()
    try:
        case_uuid = uuid.UUID(case_id)
        proc_uuid = uuid.UUID(payload.procedure_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid UUID") from exc

    try:
        case = store.attach_procedure(case_uuid, proc_uuid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Case not found") from exc

    return _serialize_case(case)
