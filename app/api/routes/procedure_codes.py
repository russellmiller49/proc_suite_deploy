"""Procedure codes API endpoints for the clinician review workflow.

Phase 2-4 API Implementation per NEW_ARCHITECTURE.md Section 9:
- POST /procedures/{id}/codes/suggest - Trigger rule+LLM pipeline
- GET  /procedures/{id}/codes/suggest - Retrieve pending suggestions
- POST /procedures/{id}/codes/review  - Submit ReviewAction for a suggestion
- POST /procedures/{id}/codes/manual  - Add a manual code (bypasses AI)
- GET  /procedures/{id}/codes/final   - Retrieve approved FinalCode[] for billing

Phase 3-4 Registry Export:
- POST /procedures/{id}/registry/export  - Export procedure to IP Registry
- GET  /procedures/{id}/registry/preview - Preview registry entry before export

This module ensures no code reaches billing without human review
and persists reasoning for audit.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_coding_service, get_procedure_store, get_registry_service
from app.api.guards import (
    enforce_legacy_endpoints_allowed,
    enforce_request_mode_override_allowed,
)
from app.api.phi_dependencies import get_phi_session
from app.coder.application.coding_service import CodingService
from app.coder.phi_gating import is_phi_review_required, load_procedure_for_coding
from app.common.exceptions import (
    CodingError,
    KnowledgeBaseError,
    PersistenceError,
    RegistryError,
)
from app.domain.procedure_store.repository import ProcedureStore
from app.registry.application.registry_service import RegistryService
from observability.coding_metrics import CodingMetrics
from observability.logging_config import get_logger
from observability.timing import timed
from proc_schemas.coding import CodeSuggestion, FinalCode, ReviewAction
from proc_schemas.reasoning import ReasoningFields

router = APIRouter()
logger = get_logger("procedure_codes_api")
_coding_service_dep = Depends(get_coding_service)
_registry_service_dep = Depends(get_registry_service)
_procedure_store_dep = Depends(get_procedure_store)
_phi_session_dep = Depends(get_phi_session)


# ============================================================================
# Request/Response Models
# ============================================================================


class SuggestCodesRequest(BaseModel):
    """Request body for triggering code suggestions."""

    report_text: str = Field(..., description="The procedure note text to analyze")
    use_llm: bool = Field(
        True,
        description="Whether to use LLM advisor in addition to rules",
    )
    procedure_type: str = Field(
        "unknown",
        description=(
            "Procedure type classification (e.g., bronch_diagnostic, bronch_ebus, "
            "pleural, blvr)"
        ),
    )


class ReviewActionRequest(BaseModel):
    """Request body for submitting a review action."""

    suggestion_id: str = Field(..., description="ID of the CodeSuggestion being reviewed")
    action: str = Field(..., description="Review action: 'accept', 'reject', or 'modify'")
    reviewer_id: str = Field(..., description="ID of the clinician reviewer")
    notes: str | None = Field(None, description="Optional notes about the decision")
    modified_code: str | None = Field(None, description="Modified CPT code if action is 'modify'")
    modified_description: str | None = Field(
        None, description="Modified description if action is 'modify'"
    )


class ManualCodeRequest(BaseModel):
    """Request body for adding a manual code."""

    code: str = Field(..., description="CPT code to add manually")
    description: str = Field("", description="Description of the code")
    notes: str | None = Field(None, description="Reason for manual addition")
    reviewer_id: str = Field(..., description="ID of the clinician adding the code")


class SuggestCodesResponse(BaseModel):
    """Response from code suggestion endpoint."""

    model_config = {"protected_namespaces": ()}

    procedure_id: str
    suggestions: list[CodeSuggestion]
    processing_time_ms: float
    kb_version: str = ""
    policy_version: str = ""
    model_version: str = ""


class ReviewActionResponse(BaseModel):
    """Response from review action endpoint."""

    suggestion_id: str
    action: str
    final_code: FinalCode | None
    message: str


class ManualCodeResponse(BaseModel):
    """Response from manual code endpoint."""

    final_code: FinalCode
    message: str


class RegistryExportRequest(BaseModel):
    """Request body for registry export."""

    procedure_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Procedure metadata (patient info, operator, facility, etc.)",
    )
    registry_version: str = Field(
        "v2",
        description="Target registry schema version (v2 or v3)",
    )


class RegistryExportResponse(BaseModel):
    """Response from registry export endpoint."""

    procedure_id: str
    registry_id: str
    schema_version: str
    export_id: str
    export_timestamp: datetime
    status: Literal["success", "partial", "failed"]
    bundle: dict[str, Any] = Field(
        ..., description="The registry entry as JSON"
    )
    warnings: list[str] = Field(default_factory=list)


class RegistryPreviewResponse(BaseModel):
    """Response from registry preview endpoint."""

    procedure_id: str
    registry_id: str
    schema_version: str
    status: Literal["preview"]
    bundle: dict[str, Any] = Field(
        ..., description="The draft registry entry as JSON"
    )
    completeness_score: float = Field(
        ..., description="Completeness score (0.0 to 1.0)"
    )
    missing_fields: list[str] = Field(default_factory=list)
    suggested_values: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# ============================================================================
# API Endpoints
# ============================================================================


@router.post(
    "/procedures/{proc_id}/codes/suggest",
    response_model=SuggestCodesResponse,
    summary="Trigger code suggestion pipeline",
    description="Runs rule-based and optional LLM coding pipeline, persists suggestions.",
)
def suggest_codes(
    proc_id: str,
    request: SuggestCodesRequest,
    coding_service: CodingService = _coding_service_dep,
    store: ProcedureStore = _procedure_store_dep,
    phi_db=_phi_session_dep,
) -> SuggestCodesResponse:
    """Trigger rule+LLM pipeline, persist CodeSuggestion[].

    This endpoint:
    1. Runs the rule-based coding engine
    2. Optionally runs the LLM advisor
    3. Merges results using smart_hybrid policy
    4. Validates evidence in the note text
    5. Applies NCCI/MER compliance rules
    6. Persists suggestions for review
    7. Returns suggestions with reasoning/provenance
    """
    procedure_type = request.procedure_type
    require_review = is_phi_review_required()

    report_text = request.report_text
    proc_uuid: uuid.UUID | None = None
    try:
        proc_uuid = uuid.UUID(proc_id)
    except ValueError:
        proc_uuid = None

    proc = None
    if proc_uuid is not None:
        try:
            proc = load_procedure_for_coding(
                phi_db,
                proc_uuid,
                require_review=require_review,
            )
        except PermissionError as exc:
            logger.info(
                "coding_phi_gated",
                extra={
                    "procedure_id": proc_id,
                    "require_review": require_review,
                    "reason": "not_reviewed",
                },
            )
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            logger.info(
                "coding_phi_missing",
                extra={"procedure_id": proc_id, "require_review": require_review},
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    elif require_review:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid procedure_id format (PHI review requires UUID procedure IDs)."
            ),
        )

    if proc is not None:
        # Always prefer scrubbed text from reviewed procedure
        report_text = proc.scrubbed_text
        procedure_type = procedure_type or proc.document_type or "unknown"

    logger.info(
        "Suggest codes requested",
        extra={
            "procedure_id": proc_id,
            "use_llm": request.use_llm,
            "procedure_type": procedure_type,
        },
    )

    try:
        with timed("api.suggest_codes") as timing:
            # Run the full coding pipeline
            result = coding_service.generate_result(
                procedure_id=proc_id,
                report_text=report_text,
                use_llm=request.use_llm,
                procedure_type=procedure_type,
            )

        # Persist the full result for metadata access
        store.save_result(proc_id, result)

        # Persist suggestions for review endpoints
        store.save_suggestions(proc_id, result.suggestions)

        # Record metrics with procedure_type segmentation
        CodingMetrics.record_suggestions_generated(
            num_suggestions=len(result.suggestions),
            procedure_type=procedure_type,
            used_llm=request.use_llm,
        )
        CodingMetrics.record_pipeline_latency(
            latency_ms=result.processing_time_ms,
            procedure_type=procedure_type,
            used_llm=request.use_llm,
        )
        # Record LLM latency separately if LLM was used
        if result.llm_latency_ms > 0:
            CodingMetrics.record_llm_latency(
                latency_ms=result.llm_latency_ms,
                procedure_type=procedure_type,
            )

        logger.info(
            "Code suggestions generated",
            extra={
                "procedure_id": proc_id,
                "procedure_type": procedure_type,
                "num_suggestions": len(result.suggestions),
                "processing_time_ms": timing.elapsed_ms,
                "llm_latency_ms": result.llm_latency_ms,
                "kb_version": result.kb_version,
                "policy_version": result.policy_version,
                "model_version": result.model_version,
            },
        )

        return SuggestCodesResponse(
            procedure_id=proc_id,
            suggestions=result.suggestions,
            processing_time_ms=result.processing_time_ms,
            kb_version=result.kb_version,
            policy_version=result.policy_version,
            model_version=result.model_version,
        )

    except KnowledgeBaseError as e:
        logger.error(f"Knowledge base error: {e}", extra={"procedure_id": proc_id})
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge base error: {str(e)}",
        ) from e
    except CodingError as e:
        logger.error(f"Coding error: {e}", extra={"procedure_id": proc_id})
        raise HTTPException(
            status_code=500,
            detail=f"Coding pipeline error: {str(e)}",
        ) from e
    except PersistenceError as e:
        logger.error(f"Persistence error: {e}", extra={"procedure_id": proc_id})
        raise HTTPException(
            status_code=500,
            detail=f"Storage error: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error in suggest_codes: {e}",
            extra={"procedure_id": proc_id},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.get(
    "/procedures/{proc_id}/codes/suggest",
    response_model=list[CodeSuggestion],
    summary="Retrieve pending code suggestions",
    description="Returns pending code suggestions with reasoning/evidence for review.",
)
def get_suggestions(
    proc_id: str,
    store: ProcedureStore = _procedure_store_dep,
) -> list[CodeSuggestion]:
    """Retrieve pending suggestions with reasoning/evidence.

    Returns all CodeSuggestion objects that have not yet been reviewed
    for the given procedure.
    """
    suggestions = store.get_suggestions(proc_id)

    if not suggestions:
        # Check if procedure exists at all
        if not store.exists(proc_id):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No suggestions found for procedure '{proc_id}'. "
                    "Run POST /codes/suggest first."
                ),
            )

    # Filter out suggestions that have already been reviewed
    reviewed_ids = {r.suggestion_id for r in store.get_reviews(proc_id)}
    pending = [s for s in suggestions if s.suggestion_id not in reviewed_ids]

    return pending


@router.post(
    "/procedures/{proc_id}/codes/review",
    response_model=ReviewActionResponse,
    summary="Submit review action for a suggestion",
    description="Clinician submits accept/reject/modify decision for a code suggestion.",
)
def review_suggestion(
    proc_id: str,
    request: ReviewActionRequest,
    store: ProcedureStore = _procedure_store_dep,
) -> ReviewActionResponse:
    """Submit ReviewAction for a suggestion.

    This endpoint:
    1. Validates the suggestion exists
    2. Creates a ReviewAction record
    3. If accepted/modified, creates a FinalCode
    4. Persists all records for audit

    Args:
        proc_id: Procedure identifier
        request: Review action details
        store: Injected ProcedureStore

    Returns:
        ReviewActionResponse with final code if accepted/modified

    Raises:
        HTTPException: If suggestion not found or invalid action
    """
    # Validate action
    if request.action not in ("accept", "reject", "modify"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid action '{request.action}'. Must be 'accept', 'reject', or "
                "'modify'."
            ),
        )

    # Find the suggestion
    suggestions = store.get_suggestions(proc_id)
    suggestion = next(
        (s for s in suggestions if s.suggestion_id == request.suggestion_id),
        None,
    )

    if not suggestion:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Suggestion '{request.suggestion_id}' not found for procedure '{proc_id}'."
            ),
        )

    # Create the review action
    review = ReviewAction(
        suggestion_id=request.suggestion_id,
        action=request.action,
        reviewer_id=request.reviewer_id,
        notes=request.notes,
        modified_code=request.modified_code,
        modified_description=request.modified_description,
    )

    # Create final code if accepted or modified
    final_code: FinalCode | None = None
    message = ""

    if request.action == "accept":
        final_code = FinalCode(
            code=suggestion.code,
            description=suggestion.description,
            source=suggestion.source,
            reasoning=suggestion.reasoning,
            # Note: Not setting review here to avoid circular reference during serialization
            procedure_id=proc_id,
            suggestion_id=suggestion.suggestion_id,
        )
        message = f"Code {suggestion.code} accepted and added to final codes."

    elif request.action == "modify":
        if not request.modified_code:
            raise HTTPException(
                status_code=400,
                detail="modified_code is required when action is 'modify'.",
            )
        final_code = FinalCode(
            code=request.modified_code,
            description=request.modified_description or suggestion.description,
            source="manual",  # Modified codes are marked as manual
            reasoning=suggestion.reasoning,
            # Note: Not setting review here to avoid circular reference during serialization
            procedure_id=proc_id,
            suggestion_id=suggestion.suggestion_id,
        )
        message = f"Code modified from {suggestion.code} to {request.modified_code}."

    else:  # reject
        message = f"Code {suggestion.code} rejected."

    # Note: We deliberately do not create circular references between review and final_code
    # to avoid serialization issues. The relationship is maintained via suggestion_id.

    # Persist review
    store.add_review(proc_id, review)

    # Persist final code if created
    if final_code:
        store.add_final_code(proc_id, final_code)
        CodingMetrics.record_final_code_added(source=final_code.source)

    # Record review metrics
    CodingMetrics.record_review_action(
        action=request.action,
        source=suggestion.source,
    )

    # Record LLM acceptance metrics for drift monitoring (only for AI suggestions)
    if suggestion.source in ("llm", "hybrid", "rule"):
        # Get procedure_type from stored result if available
        coding_result = store.get_result(proc_id)
        procedure_type = coding_result.procedure_type if coding_result else "unknown"

        # Accepted = accept or modify (user kept the suggestion with possible changes)
        accepted = 1 if request.action in ("accept", "modify") else 0
        CodingMetrics.record_llm_acceptance(
            accepted_count=accepted,
            reviewed_count=1,
            procedure_type=procedure_type,
            source=suggestion.source,
        )

    logger.info(
        "Review action completed",
        extra={
            "procedure_id": proc_id,
            "suggestion_id": request.suggestion_id,
            "action": request.action,
            "reviewer_id": request.reviewer_id,
        },
    )

    return ReviewActionResponse(
        suggestion_id=request.suggestion_id,
        action=request.action,
        final_code=final_code,
        message=message,
    )


@router.post(
    "/procedures/{proc_id}/codes/manual",
    response_model=ManualCodeResponse,
    summary="Add a manual code",
    description="Add a code manually (bypasses AI suggestion pipeline).",
)
def add_manual_code(
    proc_id: str,
    request: ManualCodeRequest,
    store: ProcedureStore = _procedure_store_dep,
) -> ManualCodeResponse:
    """Add a manual code (bypasses AI).

    This allows clinicians to add codes that weren't suggested by the
    AI pipeline, for cases where the AI missed something or for
    specific clinical scenarios.

    Args:
        proc_id: Procedure identifier
        request: Manual code details
        store: Injected ProcedureStore

    Returns:
        ManualCodeResponse with the created FinalCode
    """
    # Create a review action for audit trail
    review = ReviewAction(
        suggestion_id="",  # No suggestion for manual codes
        action="accept",
        reviewer_id=request.reviewer_id,
        notes=request.notes or "Manually added code",
    )

    # Create reasoning for the manual code
    reasoning = ReasoningFields(
        rule_paths=["manual_addition"],
        confidence=1.0,  # Manual codes are fully trusted
    )

    # Create the final code
    # Note: Not setting review here to avoid circular reference during serialization
    final_code = FinalCode(
        code=request.code,
        description=request.description,
        source="manual",
        reasoning=reasoning,
        procedure_id=proc_id,
        suggestion_id=None,
    )

    # Note: We deliberately do not create circular references between review and final_code

    # Persist
    store.add_review(proc_id, review)
    store.add_final_code(proc_id, final_code)

    # Record metrics
    CodingMetrics.record_manual_code_added()
    CodingMetrics.record_final_code_added(source="manual")

    logger.info(
        "Manual code added",
        extra={
            "procedure_id": proc_id,
            "code": request.code,
            "reviewer_id": request.reviewer_id,
        },
    )

    return ManualCodeResponse(
        final_code=final_code,
        message=f"Manual code {request.code} added successfully.",
    )


@router.get(
    "/procedures/{proc_id}/codes/final",
    response_model=list[FinalCode],
    summary="Retrieve final approved codes",
    description="Returns all clinician-approved codes ready for billing/registry.",
)
def get_final_codes(
    proc_id: str,
    store: ProcedureStore = _procedure_store_dep,
) -> list[FinalCode]:
    """Retrieve approved FinalCode[] for billing/registry.

    Returns only codes that have been explicitly approved by a clinician,
    either through:
    - Accepting an AI suggestion
    - Modifying an AI suggestion
    - Manually adding a code

    These codes are safe to use for billing and registry export.
    """
    return store.get_final_codes(proc_id)


@router.get(
    "/procedures/{proc_id}/codes/reviews",
    response_model=list[ReviewAction],
    summary="Retrieve review history",
    description="Returns all review actions for audit purposes.",
)
def get_review_history(
    proc_id: str,
    store: ProcedureStore = _procedure_store_dep,
) -> list[ReviewAction]:
    """Retrieve all review actions for audit trail.

    This endpoint is useful for:
    - Audit purposes
    - Understanding rejection reasons
    - Training data for improving AI suggestions
    """
    return store.get_reviews(proc_id)


@router.get(
    "/procedures/{proc_id}/codes/metrics",
    summary="Get coding metrics for a procedure",
    description="Returns metrics about suggestion acceptance, rejections, etc.",
)
def get_coding_metrics(
    proc_id: str,
    store: ProcedureStore = _procedure_store_dep,
) -> dict[str, Any]:
    """Get metrics for the coding workflow.

    Returns:
        - Total suggestions
        - Accepted count
        - Rejected count
        - Modified count
        - Manual additions count
        - Acceptance rate
        - Provenance info (kb_version, policy_version, model_version)
    """
    suggestions = store.get_suggestions(proc_id)
    reviews = store.get_reviews(proc_id)
    finals = store.get_final_codes(proc_id)
    coding_result = store.get_result(proc_id)

    # Count reviews of AI suggestions (not manual additions which have empty suggestion_id)
    accepted = sum(1 for r in reviews if r.action == "accept" and r.suggestion_id)
    rejected = sum(1 for r in reviews if r.action == "reject" and r.suggestion_id)
    modified = sum(1 for r in reviews if r.action == "modify" and r.suggestion_id)
    manual = sum(1 for f in finals if f.source == "manual" and not f.suggestion_id)

    total_reviewed = accepted + rejected + modified
    acceptance_rate = (accepted + modified) / total_reviewed if total_reviewed > 0 else 0.0

    # Include provenance from the coding result
    metrics: dict[str, Any] = {
        "procedure_id": proc_id,
        "total_suggestions": len(suggestions),
        "total_reviews": total_reviewed,
        "accepted": accepted,
        "rejected": rejected,
        "modified": modified,
        "manual_additions": manual,
        "final_codes_count": len(finals),
        "acceptance_rate": round(acceptance_rate, 3),
    }

    if coding_result:
        metrics["kb_version"] = coding_result.kb_version
        metrics["policy_version"] = coding_result.policy_version
        metrics["model_version"] = coding_result.model_version
        metrics["processing_time_ms"] = coding_result.processing_time_ms

    return metrics


# ============================================================================
# Registry Export Endpoints (Phase 3-4)
# ============================================================================


@router.post(
    "/procedures/{proc_id}/registry/export",
    response_model=RegistryExportResponse,
    summary="Export procedure to registry",
    description="Creates a registry entry from final codes and procedure metadata.",
)
def export_to_registry(
    proc_id: str,
    request: RegistryExportRequest,
    registry_service: RegistryService = _registry_service_dep,
    store: ProcedureStore = _procedure_store_dep,
) -> RegistryExportResponse:
    """Export procedure data to IP Registry.

    This endpoint:
    1. Retrieves final codes from ProcedureStore
    2. Maps CPT codes to registry boolean flags
    3. Merges procedure metadata from request or external source
    4. Validates against the target registry schema
    5. Persists the registry entry

    Args:
        proc_id: Procedure identifier
        request: Export request with metadata and version
        registry_service: Injected RegistryService
        store: Injected ProcedureStore

    Returns:
        RegistryExportResponse with the registry entry

    Raises:
        HTTPException: 404 if no final codes, 500 on service error
    """
    logger.info(
        "Registry export requested",
        extra={
            "procedure_id": proc_id,
            "registry_version": request.registry_version,
        },
    )

    # Get final codes for this procedure
    final_codes = store.get_final_codes(proc_id)
    if not final_codes:
        raise HTTPException(
            status_code=404,
            detail=f"No final codes found for procedure '{proc_id}'. "
            "Complete the coding review workflow first.",
        )

    try:
        with timed("api.registry_export") as timing:
            result = registry_service.export_procedure(
                procedure_id=proc_id,
                final_codes=final_codes,
                procedure_metadata=request.procedure_metadata,
                version=request.registry_version,
            )

        # Persist the export
        export_record = {
            "registry_id": result.registry_id,
            "schema_version": result.schema_version,
            "export_id": result.export_id,
            "export_timestamp": result.export_timestamp.isoformat(),
            "status": result.status,
            "bundle": result.entry.model_dump(mode="json"),
            "warnings": result.warnings,
        }
        store.save_export(proc_id, export_record)

        # Record metrics
        CodingMetrics.record_registry_export(
            status=result.status,
            version=request.registry_version,
            latency_ms=timing.elapsed_ms,
        )

        logger.info(
            "Registry export completed",
            extra={
                "procedure_id": proc_id,
                "export_id": result.export_id,
                "status": result.status,
                "num_warnings": len(result.warnings),
                "processing_time_ms": timing.elapsed_ms,
            },
        )

        return RegistryExportResponse(
            procedure_id=proc_id,
            registry_id=result.registry_id,
            schema_version=result.schema_version,
            export_id=result.export_id,
            export_timestamp=result.export_timestamp,
            status=result.status,
            bundle=result.entry.model_dump(mode="json"),
            warnings=result.warnings,
        )

    except RegistryError as e:
        logger.error(f"Registry export error: {e}", extra={"procedure_id": proc_id})
        raise HTTPException(
            status_code=500,
            detail=f"Registry export error: {str(e)}",
        ) from e
    except PersistenceError as e:
        logger.error(f"Persistence error: {e}", extra={"procedure_id": proc_id})
        raise HTTPException(
            status_code=500,
            detail=f"Storage error: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error in registry export: {e}",
            extra={"procedure_id": proc_id},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.get(
    "/procedures/{proc_id}/registry/preview",
    response_model=RegistryPreviewResponse,
    summary="Preview registry entry",
    description="Preview the registry entry without committing the export.",
)
def preview_registry_entry(
    proc_id: str,
    registry_version: str = Query("v2", description="Schema version (v2 or v3)"),
    registry_service: RegistryService = _registry_service_dep,
    store: ProcedureStore = _procedure_store_dep,
) -> RegistryPreviewResponse:
    """Preview registry entry before export.

    Returns a draft registry entry with validation warnings.
    Does NOT persist the entry.

    This is useful for:
    - Clinician review before finalizing export
    - UI preview with editable fields
    - QA checks for completeness

    Args:
        proc_id: Procedure identifier
        registry_version: Target schema version
        registry_service: Injected RegistryService
        store: Injected ProcedureStore

    Returns:
        RegistryPreviewResponse with draft entry and completeness info
    """
    logger.info(
        "Registry preview requested",
        extra={
            "procedure_id": proc_id,
            "registry_version": registry_version,
        },
    )

    # Get final codes (preview works even with no final codes)
    final_codes = store.get_final_codes(proc_id)

    try:
        draft = registry_service.build_draft_entry(
            procedure_id=proc_id,
            final_codes=final_codes,
            procedure_metadata={},  # Preview uses empty metadata
            version=registry_version,
        )

        # Record completeness metric
        CodingMetrics.record_registry_completeness(
            score=draft.completeness_score,
            version=registry_version,
        )

        logger.info(
            "Registry preview generated",
            extra={
                "procedure_id": proc_id,
                "completeness_score": draft.completeness_score,
                "num_warnings": len(draft.warnings),
                "num_missing_fields": len(draft.missing_fields),
            },
        )

        return RegistryPreviewResponse(
            procedure_id=proc_id,
            registry_id="ip_registry",
            schema_version=registry_version,
            status="preview",
            bundle=draft.entry.model_dump(mode="json"),
            completeness_score=draft.completeness_score,
            missing_fields=draft.missing_fields,
            suggested_values=draft.suggested_values,
            warnings=draft.warnings,
        )

    except RegistryError as e:
        logger.error(f"Registry preview error: {e}", extra={"procedure_id": proc_id})
        raise HTTPException(
            status_code=500,
            detail=f"Registry preview error: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error in registry preview: {e}",
            extra={"procedure_id": proc_id},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.get(
    "/procedures/{proc_id}/registry/export",
    response_model=RegistryExportResponse,
    summary="Get existing registry export",
    description="Retrieve a previously exported registry entry.",
)
def get_registry_export(
    proc_id: str,
    store: ProcedureStore = _procedure_store_dep,
) -> RegistryExportResponse:
    """Retrieve a previously exported registry entry.

    Args:
        proc_id: Procedure identifier
        store: Injected ProcedureStore

    Returns:
        RegistryExportResponse with the stored export

    Raises:
        HTTPException: 404 if no export exists for this procedure
    """
    export_record = store.get_export(proc_id)
    if not export_record:
        raise HTTPException(
            status_code=404,
            detail=f"No registry export found for procedure '{proc_id}'. "
            "Use POST /registry/export to create one.",
        )

    return RegistryExportResponse(
        procedure_id=proc_id,
        registry_id=export_record["registry_id"],
        schema_version=export_record["schema_version"],
        export_id=export_record["export_id"],
        export_timestamp=datetime.fromisoformat(export_record["export_timestamp"]),
        status=export_record["status"],
        bundle=export_record["bundle"],
        warnings=export_record.get("warnings", []),
    )


# ============================================================================
# Unified Extraction Endpoint (PHI-Gated)
# ============================================================================


class UnifiedExtractRequest(BaseModel):
    """Request body for PHI-gated unified extraction."""

    include_financials: bool = Field(
        False, description="Whether to calculate RVU/payment estimates"
    )
    explain: bool = Field(
        False, description="Whether to include evidence spans in response"
    )
    mode: str | None = Field(
        default=None,
        description=(
            "Optional execution mode. Use 'engine_only' to disable LLM registry "
            "extraction."
        ),
    )


class UnifiedExtractResponse(BaseModel):
    """Response from PHI-gated unified extraction."""

    procedure_id: str
    status: str = Field(..., description="Status: 'success', 'partial', or 'failed'")
    registry: dict[str, Any] = Field(
        default_factory=dict, description="Extracted registry fields"
    )
    cpt_codes: list[str] = Field(
        default_factory=list, description="Derived CPT codes"
    )
    suggestions: list[CodeSuggestion] = Field(
        default_factory=list, description="Code suggestions with rationale"
    )
    total_work_rvu: float | None = Field(None, description="Total work RVU")
    estimated_payment: float | None = Field(None, description="Estimated payment")
    coder_difficulty: str = Field("", description="Difficulty classification")
    needs_manual_review: bool = Field(False, description="Whether manual review is recommended")
    audit_warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0


@router.post(
    "/procedures/{proc_id}/extract",
    response_model=UnifiedExtractResponse,
    summary="Run unified extraction on PHI-reviewed procedure",
    description=(
        "Runs the extraction-first pipeline on a PHI-reviewed procedure:\n"
        "1. Validates that the procedure has been PHI-reviewed\n"
        "2. Uses the reviewed scrubbed_text for extraction\n"
        "3. Extracts registry fields and derives CPT codes\n"
        "4. Returns combined results for clinician review\n\n"
        "Requires: CODER_REQUIRE_PHI_REVIEW=true and procedure status=PHI_REVIEWED"
    ),
)
def run_unified_extraction(
    proc_id: str,
    request: UnifiedExtractRequest,
    registry_service: RegistryService = _registry_service_dep,
    coding_service: CodingService = _coding_service_dep,
    store: ProcedureStore = _procedure_store_dep,
    phi_db=_phi_session_dep,
) -> UnifiedExtractResponse:
    """Run unified extraction-first pipeline on a PHI-reviewed procedure.

    This endpoint enforces PHI review before extraction:
    1. Loads the PHI-reviewed procedure (requires status=PHI_REVIEWED)
    2. Uses proc.scrubbed_text (no raw PHI exposure)
    3. Runs registry extraction to get structured fields
    4. Derives CPT codes deterministically from registry
    5. Optionally calculates financial estimates
    6. Persists suggestions for review workflow

    Args:
        proc_id: Procedure identifier (from /v1/phi/submit)
        request: Extraction options (financials, explain)
        registry_service: Injected RegistryService
        coding_service: Injected CodingService (for KB access)
        store: Injected ProcedureStore
        phi_db: PHI database session

    Returns:
        UnifiedExtractResponse with registry, codes, and suggestions

    Raises:
        HTTPException 403: If procedure has not been PHI-reviewed
        HTTPException 404: If procedure not found
    """
    enforce_legacy_endpoints_allowed()
    enforce_request_mode_override_allowed((request.mode or "").strip().lower())

    from config.settings import CoderSettings
    from app.coder.domain_rules.registry_to_cpt.coding_rules import derive_all_codes_with_meta

    # Always require PHI review for this endpoint
    require_review = True

    try:
        proc_uuid = uuid.UUID(proc_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid procedure_id format",
        ) from exc

    # Load the PHI-reviewed procedure
    try:
        proc = load_procedure_for_coding(phi_db, proc_uuid, require_review=require_review)
    except PermissionError as exc:
        logger.info(
            "extraction_phi_gated",
            extra={"procedure_id": proc_id, "reason": "not_reviewed"},
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        logger.info(
            "extraction_phi_missing",
            extra={"procedure_id": proc_id},
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if proc is None:
        raise HTTPException(status_code=404, detail="Procedure not found or missing scrubbed text")

    # Use the scrubbed text from the reviewed procedure
    scrubbed_text = proc.scrubbed_text

    logger.info(
        "Unified extraction requested",
        extra={
            "procedure_id": proc_id,
            "include_financials": request.include_financials,
            "text_length": len(scrubbed_text),
        },
    )

    try:
        with timed("api.unified_extraction") as timing:
            # Step 1: Registry extraction
            mode_value = (request.mode or "").strip().lower()
            if mode_value in {"engine_only", "no_llm", "deterministic_only"}:
                mode_value = "parallel_ner"

            if mode_value:
                extraction_result = registry_service.extract_fields(scrubbed_text, mode_value)
            else:
                extraction_result = registry_service.extract_fields(scrubbed_text)

            # Step 2: Derive CPT codes from registry
            record = extraction_result.record
            if record is None:
                from app.registry.schema import RegistryRecord
                record = RegistryRecord.model_validate(extraction_result.mapped_fields)

            codes, rationales, derivation_warnings = derive_all_codes_with_meta(record)

            # Build suggestions with confidence and rationale
            suggestions = []
            base_confidence = 0.95 if extraction_result.coder_difficulty == "HIGH_CONF" else 0.80

            for code in codes:
                proc_info = coding_service.kb_repo.get_procedure_info(code)
                description = proc_info.description if proc_info else ""
                rationale = rationales.get(code, "")

                # Determine review flag
                if extraction_result.needs_manual_review:
                    review_flag = "required"
                elif extraction_result.audit_warnings:
                    review_flag = "recommended"
                else:
                    review_flag = "optional"

                suggestions.append(CodeSuggestion(
                    id=f"{proc_id}_{code}",
                    code=code,
                    description=description,
                    source="extraction_first",
                    confidence=base_confidence,
                    review_flag=review_flag,
                    reasoning=ReasoningFields(
                        rule_paths=["registry_to_cpt"],
                        confidence=base_confidence,
                        rationale=rationale,
                    ),
                ))

            # Step 3: Calculate financials if requested
            total_work_rvu = None
            estimated_payment = None

            if request.include_financials and codes:
                from app.api.services.financials import calculate_financials

                settings = CoderSettings()
                conversion_factor = settings.cms_conversion_factor
                total_work_rvu, estimated_payment, _, financial_warnings = calculate_financials(
                    codes=[str(c).strip() for c in codes if str(c).strip()],
                    kb_repo=coding_service.kb_repo,
                    conversion_factor=conversion_factor,
                    units_by_code=None,
                )

        # Persist suggestions for review workflow
        store.save_suggestions(proc_id, suggestions)

        # Combine audit warnings
        all_warnings = list(extraction_result.audit_warnings or [])
        all_warnings.extend(derivation_warnings)
        if request.include_financials and codes:
            all_warnings.extend(financial_warnings or [])

        # Serialize registry
        registry_payload = record.model_dump(exclude_none=True) if record else {}

        logger.info(
            "Unified extraction completed",
            extra={
                "procedure_id": proc_id,
                "num_codes": len(codes),
                "num_suggestions": len(suggestions),
                "processing_time_ms": timing.elapsed_ms,
            },
        )

        return UnifiedExtractResponse(
            procedure_id=proc_id,
            status="success",
            registry=registry_payload,
            cpt_codes=codes,
            suggestions=suggestions,
            total_work_rvu=total_work_rvu,
            estimated_payment=estimated_payment,
            coder_difficulty=extraction_result.coder_difficulty or "",
            needs_manual_review=extraction_result.needs_manual_review,
            audit_warnings=all_warnings,
            processing_time_ms=round(timing.elapsed_ms, 2),
        )

    except Exception as e:
        logger.exception(
            f"Unified extraction error: {e}",
            extra={"procedure_id": proc_id},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Extraction error: {str(e)}",
        ) from e


# ============================================================================
# Store management (for testing)
# ============================================================================


def clear_procedure_stores(proc_id: str | None = None) -> None:
    """Clear procedure stores for a procedure or all procedures.

    This function accesses the global ProcedureStore singleton and clears data.
    Primarily used for testing to ensure clean state between tests.

    Args:
        proc_id: If provided, clear only that procedure's data.
                 If None, clear all data.
    """
    from app.api.dependencies import get_procedure_store
    store = get_procedure_store()
    store.clear_all(proc_id)
