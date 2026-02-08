"""Legacy registry extraction route handlers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import get_registry_service
from app.api.guards import enforce_legacy_endpoints_allowed, enforce_request_mode_override_allowed
from app.api.phi_dependencies import get_phi_scrubber
from app.api.phi_redaction import apply_phi_redaction
from app.api.readiness import require_ready
from app.api.registry_payload import shape_registry_payload
from app.api.schemas import RegistryRequest, RegistryResponse
from app.infra.executors import run_cpu
from app.registry.application.registry_service import RegistryService
from app.registry.engine import RegistryEngine

router = APIRouter(tags=["legacy-registry"])
_ready_dep = Depends(require_ready)
_registry_service_dep = Depends(get_registry_service)
_phi_scrubber_dep = Depends(get_phi_scrubber)


@router.post(
    "/v1/registry/run",
    response_model=RegistryResponse,
    response_model_exclude_none=True,
)
async def registry_run(
    req: RegistryRequest,
    request: Request,
    _ready: None = _ready_dep,
    registry_service: RegistryService = _registry_service_dep,
    phi_scrubber=_phi_scrubber_dep,
) -> RegistryResponse:
    enforce_legacy_endpoints_allowed()

    redaction = apply_phi_redaction(req.note, phi_scrubber)
    note_text = redaction.text

    mode_value = (req.mode or "").strip().lower()
    enforce_request_mode_override_allowed(mode_value)
    if mode_value == "parallel_ner":
        result = await run_cpu(
            request.app,
            registry_service.extract_fields,
            note_text,
            req.mode,
        )
        payload = shape_registry_payload(result.record, {}, codes=result.cpt_codes)
        return JSONResponse(content=payload)

    if mode_value in {"engine_only", "no_llm", "deterministic_only"}:
        result = await run_cpu(
            request.app,
            registry_service.extract_fields,
            note_text,
            "parallel_ner",
        )
        payload = shape_registry_payload(result.record, {}, codes=result.cpt_codes)
        return JSONResponse(content=payload)

    eng = RegistryEngine()
    result = await run_cpu(request.app, eng.run, note_text, explain=req.explain)
    if isinstance(result, tuple):
        record, evidence = result
    else:
        record, evidence = result, getattr(result, "evidence", {})

    payload = shape_registry_payload(record, evidence)
    return JSONResponse(content=payload)


__all__ = ["router"]
