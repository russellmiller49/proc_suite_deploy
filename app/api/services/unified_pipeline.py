"""Shared unified extraction-first pipeline logic.

This is used by:
- POST /api/v1/process (stateless)
- Registry Runs persistence endpoints (stateful store around same pipeline)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from fastapi import HTTPException, Request

from app.api.adapters.response_adapter import build_v3_evidence_payload
from app.api.phi_redaction import apply_phi_redaction
from app.api.schemas import (
    CodeSuggestionSummary,
    MissingFieldPrompt,
    UnifiedProcessRequest,
    UnifiedProcessResponse,
)
from app.coder.application.coding_service import CodingService
from app.coder.phi_gating import is_phi_review_required
from app.common.exceptions import LLMError
from app.common.knowledge import knowledge_hash, knowledge_version
from app.infra.executors import run_cpu
from app.registry.application.registry_service import RegistryExtractionResult, RegistryService

logger = logging.getLogger(__name__)


async def run_unified_pipeline_logic(
    *,
    payload: UnifiedProcessRequest,
    request: Request,
    registry_service: RegistryService,
    coding_service: CodingService,
    phi_scrubber,
) -> tuple[UnifiedProcessResponse, str, dict[str, Any]]:
    """Run the unified extraction-first pipeline.

    Returns:
        (response_model, scrubbed_note_text_used, metadata)
    """

    start_time = time.time()

    # 1) PHI Redaction (if not already scrubbed)
    redaction_was_scrubbed = False
    redaction_entity_count = 0
    redaction_warning = None
    registry_v3_event_log: dict[str, Any] | None = None
    v3_event_log_warning: str | None = None

    if payload.already_scrubbed:
        note_text = payload.note
    else:
        redaction = apply_phi_redaction(payload.note, phi_scrubber)
        note_text = redaction.text
        redaction_was_scrubbed = bool(redaction.was_scrubbed)
        redaction_entity_count = int(redaction.entity_count)
        redaction_warning = redaction.warning

    # 2) Run Registry Extraction (includes CPT coding via Hybrid Orchestrator)
    try:
        result: RegistryExtractionResult = await run_cpu(
            request.app, registry_service.extract_fields, note_text
        )
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
    except ValueError as exc:
        logger.error("Unified process configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unified process failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error") from exc

    from app.coder.domain_rules.registry_to_cpt.coding_rules import derive_all_codes_with_meta
    from app.registry.schema import RegistryRecord

    record = getattr(result, "record", None)
    if record is None:
        record = RegistryRecord.model_validate(getattr(result, "mapped_fields", {}) or {})

    derived_codes, derived_rationales, derivation_warnings = derive_all_codes_with_meta(record)
    codes = list(getattr(result, "cpt_codes", None) or derived_codes)
    code_rationales = getattr(result, "code_rationales", None) or derived_rationales

    suggestions: list[CodeSuggestionSummary] = []
    base_confidence = 0.95 if result.coder_difficulty == "HIGH_CONF" else 0.80

    for code in codes:
        proc_info = coding_service.kb_repo.get_procedure_info(code)
        description = proc_info.description if proc_info else ""
        rationale = code_rationales.get(code, "")

        if result.needs_manual_review:
            review_flag = "required"
        elif result.audit_warnings:
            review_flag = "recommended"
        else:
            review_flag = "optional"

        suggestions.append(
            CodeSuggestionSummary(
                code=code,
                description=description,
                confidence=base_confidence,
                rationale=rationale,
                review_flag=review_flag,
            )
        )

    total_work_rvu: float | None = None
    estimated_payment: float | None = None
    per_code_billing: list[dict[str, Any]] = []

    if payload.include_financials and codes:
        from config.settings import CoderSettings
        from app.api.services.financials import calculate_financials

        settings = CoderSettings()
        conversion_factor = settings.cms_conversion_factor
        units_by_code: dict[str, int] = {}

        billing = getattr(record, "billing", None)
        cpt_items = []
        if isinstance(billing, dict):
            cpt_items = billing.get("cpt_codes") or []
        else:
            cpt_items = getattr(billing, "cpt_codes", None) or []
        if isinstance(cpt_items, list):
            for item in cpt_items:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or "").strip().lstrip("+")
                if not code:
                    continue
                try:
                    units_by_code[code] = int(item.get("units") or 1)
                except (TypeError, ValueError):
                    units_by_code[code] = 1
        (
            total_work_rvu,
            estimated_payment,
            per_code_billing,
            financial_warnings,
        ) = calculate_financials(
            codes=[str(c).strip() for c in codes if str(c).strip()],
            kb_repo=coding_service.kb_repo,
            conversion_factor=conversion_factor,
            units_by_code=units_by_code,
        )

    all_warnings: list[str] = []
    all_warnings.extend(getattr(result, "warnings", None) or [])
    all_warnings.extend(getattr(result, "audit_warnings", None) or [])
    all_warnings.extend(derivation_warnings)
    if payload.include_financials and codes:
        all_warnings.extend(financial_warnings or [])

    deduped_warnings: list[str] = []
    seen_warnings: set[str] = set()
    for warning in all_warnings:
        if warning in seen_warnings:
            continue
        seen_warnings.add(warning)
        deduped_warnings.append(warning)
    all_warnings = deduped_warnings

    # Optional event-log V3 payload (raw procedures[] list).
    if payload.include_v3_event_log:
        try:
            from app.registry.pipelines.v3_pipeline import run_v3_extraction

            event_log_v3 = await run_cpu(request.app, run_v3_extraction, note_text)
            if event_log_v3 is not None:
                registry_v3_event_log = event_log_v3.model_dump(exclude_none=True)
        except Exception as exc:
            logger.warning("V3 event log extraction failed: %s", exc)
            v3_event_log_warning = f"V3_EVENT_LOG_ERROR: {type(exc).__name__}"

    if v3_event_log_warning:
        all_warnings.append(v3_event_log_warning)

    evidence_payload = build_v3_evidence_payload(record=record, codes=codes)
    if payload.explain is False and not evidence_payload:
        evidence_payload = {}

    # Completeness prompts: suggested missing fields (used by UI to nudge documentation).
    try:
        from app.registry.completeness import generate_missing_field_prompts

        missing_field_prompts = [
            MissingFieldPrompt(
                group=p.group,
                path=p.path,
                label=p.label,
                severity=p.severity,
                message=p.message,
            )
            for p in generate_missing_field_prompts(record)
        ]
    except Exception:
        missing_field_prompts = []

    needs_manual_review = result.needs_manual_review
    if is_phi_review_required():
        review_status = "pending_phi_review"
        needs_manual_review = True
    elif needs_manual_review:
        review_status = "unverified"
    else:
        review_status = "finalized"

    processing_time_ms = (time.time() - start_time) * 1000

    response_model = UnifiedProcessResponse(
        registry=record.model_dump(exclude_none=True),
        registry_v3_event_log=registry_v3_event_log,
        evidence=evidence_payload,
        missing_field_prompts=missing_field_prompts,
        cpt_codes=codes,
        suggestions=suggestions,
        total_work_rvu=total_work_rvu,
        estimated_payment=estimated_payment,
        per_code_billing=per_code_billing,
        pipeline_mode="extraction_first",
        coder_difficulty=result.coder_difficulty,
        needs_manual_review=needs_manual_review,
        audit_warnings=all_warnings,
        validation_errors=getattr(result, "validation_errors", None) or [],
        kb_version=coding_service.kb_repo.version,
        policy_version="extraction_first_v1",
        processing_time_ms=round(processing_time_ms, 2),
        review_status=review_status,
    )

    meta = {
        "processing_time_ms": int(round(processing_time_ms)),
        "kb_version": knowledge_version() or "",
        "kb_hash": knowledge_hash() or "",
        "redaction_was_scrubbed": redaction_was_scrubbed,
        "redaction_entity_count": redaction_entity_count,
        "redaction_warning": redaction_warning,
    }

    return response_model, note_text, meta


__all__ = ["run_unified_pipeline_logic"]
