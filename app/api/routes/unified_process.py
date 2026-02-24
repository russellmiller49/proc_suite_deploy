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

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.dependencies import get_coding_service, get_registry_service
from app.api.phi_dependencies import get_phi_scrubber
from app.api.readiness import require_ready
from app.api.schemas import (
    CameraOcrCorrectionRequest,
    CameraOcrCorrectionResponse,
    UnifiedProcessRequest,
    UnifiedProcessResponse,
)
from app.api.services.unified_pipeline import run_unified_pipeline_logic
from app.coder.application.coding_service import CodingService
from app.common.exceptions import LLMError
from app.infra.executors import run_cpu
from app.registry.application.registry_service import RegistryService
from app.text_cleaning.camera_ocr_cleaner import (
    CameraOcrCleanerUnavailable,
    sanitize_camera_ocr_text,
)

router = APIRouter(tags=["process"])
logger = logging.getLogger(__name__)
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


@router.post(
    "/v1/ocr/correct",
    response_model=CameraOcrCorrectionResponse,
    response_model_exclude_none=True,
    summary="Optional post-redaction OCR correction (camera/PDF)",
)
async def correct_camera_ocr(
    payload: CameraOcrCorrectionRequest,
    request: Request,
    response: Response,
    _ready: None = _ready_dep,
) -> CameraOcrCorrectionResponse:
    """Run optional LLM cleanup for scrubbed OCR text."""
    response.headers["X-OCR-Correction-Route"] = "router"

    if not payload.already_scrubbed:
        raise HTTPException(
            status_code=400,
            detail=(
                "OCR correction requires already_scrubbed=true. "
                "Run client-side PHI redaction before this call."
            ),
        )

    try:
        result = await run_cpu(request.app, sanitize_camera_ocr_text, payload.text)
    except CameraOcrCleanerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After") or "10"
            raise HTTPException(
                status_code=503,
                detail="Upstream LLM rate limited",
                headers={"Retry-After": str(retry_after)},
            ) from exc
        raise
    except LLMError as exc:
        if "429" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="Upstream LLM rate limited",
                headers={"Retry-After": "10"},
            ) from exc
        raise
    except Exception as exc:
        logger.error("OCR correction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="OCR correction failed") from exc

    return CameraOcrCorrectionResponse(
        cleaned_text=result.cleaned_text,
        changed=bool(result.changed),
        correction_applied=bool(result.correction_applied),
        source_type=payload.source_type,
        model=result.model,
        warnings=list(result.warnings or []),
    )
