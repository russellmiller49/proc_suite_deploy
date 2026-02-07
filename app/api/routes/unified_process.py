"""Unified Extraction-First Process Endpoint.

This module provides the `/api/v1/process` endpoint which combines:
1. PHI scrubbing (optional; skipped when `already_scrubbed=true`)
2. Extraction-first registry pipeline (`RegistryService.extract_fields`)
   - extract registry from note text (engine selected by `REGISTRY_EXTRACTION_ENGINE`)
   - deterministically derive CPT codes from the extracted `RegistryRecord`
   - optional audit/self-correction to surface omissions and review flags
3. Response shaping for the UI (codes + evidence + review status)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from app.api.dependencies import get_coding_service, get_registry_service
from app.api.phi_dependencies import get_phi_scrubber
from app.api.readiness import require_ready
from app.api.schemas import (
    UnifiedProcessRequest,
    UnifiedProcessResponse,
)
from app.api.services.unified_pipeline import run_unified_pipeline_logic
from app.coder.application.coding_service import CodingService
from app.registry.application.registry_service import RegistryService

router = APIRouter(tags=["process"])
_ready_dep = Depends(require_ready)
_registry_service_dep = Depends(get_registry_service)
_coding_service_dep = Depends(get_coding_service)
_phi_scrubber_dep = Depends(get_phi_scrubber)


@router.post(
    "/v1/process",
    response_model=UnifiedProcessResponse,
    response_model_exclude_none=True,
    summary="Unified PHI-safe extraction and coding pipeline",
)
async def unified_process(
    payload: UnifiedProcessRequest,
    request: Request,
    response: Response,
    _ready: None = _ready_dep,
    registry_service: RegistryService = _registry_service_dep,
    coding_service: CodingService = _coding_service_dep,
    phi_scrubber=_phi_scrubber_dep,
) -> UnifiedProcessResponse:
    """Run the unified extraction pipeline."""
    response.headers["X-Process-Route"] = "router"
    result, _, _ = await run_unified_pipeline_logic(
        payload=payload,
        request=request,
        registry_service=registry_service,
        coding_service=coding_service,
        phi_scrubber=phi_scrubber,
    )
    return result
