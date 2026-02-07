"""
FastAPI Router for Procedure Suite ML Advisor Integration

This router provides endpoints for:
- Standard coding (rule engine only)
- Hybrid coding (rule engine + ML advisor)
- Advisor-only suggestions
- Coding trace retrieval
- Health and status checks

NOTE: This router is legacy/synthetic-only and not part of the PHI-gated coding
flow. Production coding goes through CodingService (/api/v1/procedures/{id}/codes/suggest)
using scrubbed text only.

Integration:
    Add to your main FastAPI app:

    from app.api.ml_advisor_router import router as ml_advisor_router
    app.include_router(ml_advisor_router, prefix="/api/v1")

Environment Variables:
    ENABLE_ML_ADVISOR: Enable ML advisor (default: false)
    ADVISOR_BACKEND: stub or gemini (default: stub)
    ENABLE_CODING_TRACE: Enable trace logging (default: true)
    GEMINI_API_KEY: API key for Gemini
    GEMINI_USE_OAUTH: Use OAuth instead of API key
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import the schema models
from app.proc_ml_advisor.schemas import (
    # Enums
    AdvisorBackend,
    CodeModifier,
    CodeRequest,
    CodeResponse,
    # Models
    CodeWithConfidence,
    CodingTrace,
    EvaluationMetrics,
    MLAdvisorInput,
    MLAdvisorSuggestion,
    NCCIWarning,
    ProcedureCategory,
    StructuredProcedureReport,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature flags (from environment)
ENABLE_ML_ADVISOR = os.getenv("ENABLE_ML_ADVISOR", "false").lower() == "true"
ENABLE_CODING_TRACE = os.getenv("ENABLE_CODING_TRACE", "true").lower() == "true"
ADVISOR_BACKEND = os.getenv("ADVISOR_BACKEND", "stub")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "v5")

# Paths
TRACE_FILE_PATH = Path(os.getenv("TRACE_FILE_PATH", "data/coding_traces.jsonl"))


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter(
    tags=["ML Advisor"],
    responses={
        500: {"description": "Internal server error"},
    },
)
_rule_codes_query = Query(
    default=[],
    description="Pre-existing rule codes to provide context",
)


# =============================================================================
# DEPENDENCIES
# =============================================================================

def get_advisor_config() -> dict[str, Any]:
    """
    Dependency that provides advisor configuration.

    Can be overridden in tests.
    """
    return {
        "enabled": ENABLE_ML_ADVISOR,
        "backend": ADVISOR_BACKEND,
        "trace_enabled": ENABLE_CODING_TRACE,
        "trace_path": TRACE_FILE_PATH,
        "pipeline_version": PIPELINE_VERSION,
    }


AdvisorConfig = Annotated[dict[str, Any], Depends(get_advisor_config)]


# =============================================================================
# MOCK IMPLEMENTATIONS (Replace with actual imports in production)
# =============================================================================

def mock_rule_engine(
    report: StructuredProcedureReport | None,
    report_text: str | None,
    procedure_category: ProcedureCategory | None,
) -> tuple[list[CodeWithConfidence], list[CodeModifier], list[NCCIWarning], bool]:
    """
    Mock rule engine implementation.

    In production, replace with the new hexagonal architecture:
        from app.coder.application.coding_service import CodingService
        from app.api.dependencies import get_coding_service
    """
    # Simulate rule-based coding
    codes = []
    modifiers = []
    warnings = []
    mer_applied = False

    # Basic bronchoscopy detection
    text = report_text or ""
    if report and report.raw_text:
        text = report.raw_text
    text_lower = text.lower()

    # Detect procedures from text
    if "bronchoscopy" in text_lower or procedure_category == ProcedureCategory.BRONCHOSCOPY:
        codes.append(CodeWithConfidence(
            code="31622",
            confidence=0.95,
            description="Diagnostic bronchoscopy",
        ))

    if "ebus" in text_lower or procedure_category == ProcedureCategory.EBUS:
        # Check station count
        station_count = 0
        if report and report.bronchoscopy:
            station_count = len(report.bronchoscopy.stations_sampled)

        if station_count >= 3:
            codes.append(CodeWithConfidence(
                code="31653",
                confidence=0.92,
                description="EBUS-guided TBNA, 3+ stations",
            ))
        elif station_count > 0:
            codes.append(CodeWithConfidence(
                code="31652",
                confidence=0.90,
                description="EBUS-guided TBNA, 1-2 stations",
            ))

    if "bal" in text_lower or "lavage" in text_lower:
        codes.append(CodeWithConfidence(
            code="31625",
            confidence=0.88,
            description="Bronchoscopy with BAL",
        ))

    if "thoracentesis" in text_lower or procedure_category == ProcedureCategory.PLEURAL:
        # Check for imaging guidance
        if "ultrasound" in text_lower or (
            report and report.pleural and report.pleural.imaging_guidance
        ):
            codes.append(CodeWithConfidence(
                code="32555",
                confidence=0.95,
                description="Thoracentesis with imaging guidance",
            ))
        else:
            codes.append(CodeWithConfidence(
                code="32554",
                confidence=0.90,
                description="Thoracentesis without imaging guidance",
            ))

        # Check for bilateral
        if report and report.pleural and report.pleural.laterality == "bilateral":
            modifiers.append(CodeModifier(
                modifier="-50",
                reason="Bilateral procedure",
            ))

    # Apply MER if multiple bronchoscopy codes
    bronch_codes = [c for c in codes if c.code.startswith("316")]
    if len(bronch_codes) > 1:
        mer_applied = True
        modifiers.append(CodeModifier(
            modifier="-51",
            reason="Multiple endoscopy procedures",
        ))

    # Add NCCI warning if relevant
    if any(c.code == "31622" for c in codes) and any(
        c.code in ["31652", "31653"] for c in codes
    ):
        warnings.append(NCCIWarning(
            warning_id="ncci_31622_ebus",
            codes_involved=["31622", "31652/31653"],
            message="31622 typically bundled with EBUS codes per NCCI",
            severity="info",
            resolution="May be separately billable with modifier -59 for distinct service",
        ))

    return codes, modifiers, warnings, mer_applied


def mock_ml_advisor(
    input_data: MLAdvisorInput,
    backend: str = "stub",
) -> MLAdvisorSuggestion:
    """
    Mock ML advisor implementation.

    In production, replace with:
        from app.proc_ml_advisor import get_ml_advice
    """
    start_time = time.time()

    if backend == "stub":
        # Stub just echoes rule codes
        return MLAdvisorSuggestion(
            candidate_codes=input_data.autocode_codes.copy(),
            code_confidence={code: 0.5 for code in input_data.autocode_codes},
            explanation="ML advisor not configured (stub mode)",
            model_name="stub",
            latency_ms=(time.time() - start_time) * 1000,
        )

    # Simulate Gemini response
    # In production, this would call the actual Gemini API
    text_lower = input_data.report_text.lower()
    candidate_codes = input_data.autocode_codes.copy()
    code_confidence = {code: 0.85 for code in candidate_codes}
    additions = []
    removals = []

    # Suggest additional codes based on text analysis
    if "navigation" in text_lower and "31627" not in candidate_codes:
        candidate_codes.append("31627")
        code_confidence["31627"] = 0.70
        additions.append("31627")

    if "biopsy" in text_lower and "31625" not in candidate_codes:
        candidate_codes.append("31625")
        code_confidence["31625"] = 0.65
        additions.append("31625")

    explanation = None
    if additions:
        explanation = f"Consider adding: {', '.join(additions)}"

    return MLAdvisorSuggestion(
        candidate_codes=candidate_codes,
        code_confidence=code_confidence,
        explanation=explanation,
        additions=additions,
        removals=removals,
        model_name=f"mock-{backend}",
        latency_ms=(time.time() - start_time) * 1000,
    )


def log_trace(trace: CodingTrace, trace_path: Path) -> bool:
    """
    Log coding trace to JSONL file.

    In production, replace with:
        from app.analysis.coding_trace import log_coding_trace
    """
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")
        return True
    except Exception as e:
        logger.warning(f"Failed to log trace: {e}")
        return False


# =============================================================================
# RESPONSE MODELS (API-specific)
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    advisor_enabled: bool
    advisor_backend: str
    trace_enabled: bool
    pipeline_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AdvisorStatusResponse(BaseModel):
    """Advisor status response."""
    enabled: bool
    backend: str
    available_backends: list[str]
    gemini_configured: bool
    trace_path: str


class TraceListResponse(BaseModel):
    """Response containing list of traces."""
    traces: list[CodingTrace]
    total: int
    limit: int
    offset: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get(
    "/ml-advisor/health",
    response_model=HealthResponse,
    summary="ML Advisor health check",
    description="Check ML advisor service health and configuration status.",
)
async def health_check(config: AdvisorConfig) -> HealthResponse:
    """
    Health check endpoint.

    Returns current configuration status for monitoring.
    """
    return HealthResponse(
        status="ok",
        advisor_enabled=config["enabled"],
        advisor_backend=config["backend"],
        trace_enabled=config["trace_enabled"],
        pipeline_version=config["pipeline_version"],
    )


@router.get(
    "/ml-advisor/status",
    response_model=AdvisorStatusResponse,
    summary="Advisor status",
    description="Get detailed ML advisor configuration status.",
)
async def advisor_status(config: AdvisorConfig) -> AdvisorStatusResponse:
    """
    Get ML advisor configuration status.

    Useful for debugging and admin dashboards.
    """
    gemini_configured = bool(
        os.getenv("GEMINI_API_KEY") or
        os.getenv("GEMINI_USE_OAUTH", "").lower() == "true"
    )

    return AdvisorStatusResponse(
        enabled=config["enabled"],
        backend=config["backend"],
        available_backends=[b.value for b in AdvisorBackend],
        gemini_configured=gemini_configured,
        trace_path=str(config["trace_path"]),
    )


@router.post(
    "/ml-advisor/code",
    response_model=CodeResponse,
    summary="Code a procedure",
    description="Generate CPT/HCPCS codes for a procedure report using the rule engine.",
)
async def code_procedure(
    request: CodeRequest,
    config: AdvisorConfig,
) -> CodeResponse:
    """
    Standard coding endpoint (rule engine only).

    This endpoint runs the deterministic rule engine without ML advisor.
    Use /code_with_advisor to include ML suggestions.
    """
    trace_id = str(uuid.uuid4())

    try:
        # Run rule engine
        codes, modifiers, warnings, mer_applied = mock_rule_engine(
            report=request.structured_report,
            report_text=request.report_text,
            procedure_category=request.procedure_category,
        )

        # Build response
        response = CodeResponse(
            final_codes=[c.code for c in codes],
            codes=codes,
            modifiers=[m.modifier for m in modifiers],
            ncci_warnings=[w.message for w in warnings],
            mer_applied=mer_applied,
            trace_id=trace_id,
        )

        # Log trace if enabled
        if config["trace_enabled"]:
            trace = CodingTrace(
                trace_id=trace_id,
                report_id=(
                    request.structured_report.report_id if request.structured_report else None
                ),
                report_text=request.report_text or "",
                structured_report=(
                    request.structured_report.model_dump() if request.structured_report else {}
                ),
                procedure_category=(
                    request.procedure_category.value if request.procedure_category else None
                ),
                autocode_codes=[c.code for c in codes],
                autocode_confidence={c.code: c.confidence for c in codes},
                ncci_warnings=[w.message for w in warnings],
                mer_applied=mer_applied,
                source="api.code",
                pipeline_version=config["pipeline_version"],
            )
            log_trace(trace, config["trace_path"])

        return response

    except Exception as e:
        logger.error(f"Coding error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coding failed: {str(e)}",
        ) from e


@router.post(
    "/ml-advisor/code_with_advisor",
    response_model=CodeResponse,
    summary="Code with ML advisor",
    description="Generate codes using rule engine and ML advisor suggestions.",
)
async def code_with_advisor(
    request: CodeRequest,
    config: AdvisorConfig,
    include_advisor: bool = Query(
        default=True,
        description="Include ML advisor suggestions (can be disabled per-request)",
    ),
) -> CodeResponse:
    """
    Hybrid coding endpoint (rule engine + ML advisor).

    This endpoint runs both the deterministic rule engine and the ML advisor,
    returning combined results. In v1, final_codes always equals rule_codes.

    The ML advisor suggestions are for QA/review purposes only and do not
    automatically modify the final codes.
    """
    trace_id = str(uuid.uuid4())

    try:
        # Run rule engine
        codes, modifiers, warnings, mer_applied = mock_rule_engine(
            report=request.structured_report,
            report_text=request.report_text,
            procedure_category=request.procedure_category,
        )

        rule_code_list = [c.code for c in codes]

        # Run ML advisor if enabled and requested
        advisor_suggestion = None
        if config["enabled"] and include_advisor:
            # Prepare advisor input
            advisor_input = MLAdvisorInput(
                trace_id=trace_id,
                report_id=(
                    request.structured_report.report_id if request.structured_report else None
                ),
                report_text=request.report_text or (
                    request.structured_report.raw_text if request.structured_report else ""
                ),
                structured_report=(
                    request.structured_report.model_dump() if request.structured_report else {}
                ),
                autocode_codes=rule_code_list,
                procedure_category=request.procedure_category,
            )

            # Get advisor suggestions
            advisor_suggestion = mock_ml_advisor(
                advisor_input,
                backend=config["backend"],
            )

        # Build response
        response = CodeResponse(
            final_codes=rule_code_list,  # Rules win in v1
            codes=codes,
            modifiers=[m.modifier for m in modifiers],
            ncci_warnings=[w.message for w in warnings],
            mer_applied=mer_applied,
            trace_id=trace_id,
        )

        # Add advisor data if available
        if advisor_suggestion:
            response.advisor_suggestions = advisor_suggestion.code_confidence
            response.advisor_explanation = advisor_suggestion.explanation
            response.disagreements = advisor_suggestion.disagreements

        # Log trace if enabled
        if config["trace_enabled"]:
            trace = CodingTrace(
                trace_id=trace_id,
                report_id=(
                    request.structured_report.report_id if request.structured_report else None
                ),
                report_text=request.report_text or "",
                structured_report=(
                    request.structured_report.model_dump() if request.structured_report else {}
                ),
                procedure_category=(
                    request.procedure_category.value if request.procedure_category else None
                ),
                autocode_codes=rule_code_list,
                autocode_confidence={c.code: c.confidence for c in codes},
                ncci_warnings=[w.message for w in warnings],
                mer_applied=mer_applied,
                advisor_candidate_codes=(
                    advisor_suggestion.candidate_codes if advisor_suggestion else []
                ),
                advisor_code_confidence=(
                    advisor_suggestion.code_confidence if advisor_suggestion else {}
                ),
                advisor_explanation=(
                    advisor_suggestion.explanation if advisor_suggestion else None
                ),
                advisor_disagreements=(
                    advisor_suggestion.disagreements if advisor_suggestion else []
                ),
                advisor_model=advisor_suggestion.model_name if advisor_suggestion else None,
                advisor_latency_ms=(
                    advisor_suggestion.latency_ms if advisor_suggestion else None
                ),
                source="api.code_with_advisor",
                pipeline_version=config["pipeline_version"],
            )
            log_trace(trace, config["trace_path"])

        return response

    except Exception as e:
        logger.error(f"Hybrid coding error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coding failed: {str(e)}",
        ) from e


@router.post(
    "/ml-advisor/suggest",
    response_model=MLAdvisorSuggestion,
    summary="Get advisor suggestions only",
    description="Get ML advisor suggestions without running the rule engine.",
)
async def advisor_suggest(
    request: CodeRequest,
    config: AdvisorConfig,
    rule_codes: list[str] = _rule_codes_query,
) -> MLAdvisorSuggestion:
    """
    Advisor-only endpoint.

    Get ML advisor suggestions without running the rule engine.
    Useful for A/B testing or when rule codes are already known.
    """
    if not config["enabled"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML advisor is not enabled",
        )

    trace_id = str(uuid.uuid4())

    try:
        # Prepare advisor input
        advisor_input = MLAdvisorInput(
            trace_id=trace_id,
            report_id=(
                request.structured_report.report_id if request.structured_report else None
            ),
            report_text=request.report_text or (
                request.structured_report.raw_text if request.structured_report else ""
            ),
            structured_report=(
                request.structured_report.model_dump() if request.structured_report else {}
            ),
            autocode_codes=rule_codes,
            procedure_category=request.procedure_category,
        )

        # Get advisor suggestions
        suggestion = mock_ml_advisor(
            advisor_input,
            backend=config["backend"],
        )

        return suggestion

    except Exception as e:
        logger.error(f"Advisor error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advisor failed: {str(e)}",
        ) from e


@router.get(
    "/ml-advisor/traces",
    response_model=TraceListResponse,
    summary="List coding traces",
    description="Retrieve coding traces for analysis and debugging.",
)
async def list_traces(
    config: AdvisorConfig,
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum traces to return",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of traces to skip",
    ),
    source: str | None = Query(
        default=None,
        description="Filter by source",
    ),
    has_disagreements: bool | None = Query(
        default=None,
        description="Filter by disagreement presence",
    ),
) -> TraceListResponse:
    """
    List coding traces for analysis.

    Supports pagination and filtering.
    """
    trace_path = config["trace_path"]

    if not trace_path.exists():
        return TraceListResponse(traces=[], total=0, limit=limit, offset=offset)

    try:
        all_traces = []
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    trace = CodingTrace(**data)

                    # Apply filters
                    if source and trace.source != source:
                        continue
                    if has_disagreements is not None:
                        has_dis = bool(trace.advisor_disagreements)
                        if has_disagreements != has_dis:
                            continue

                    all_traces.append(trace)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Skipping malformed trace: {e}")

        # Sort by timestamp descending (most recent first)
        all_traces.sort(key=lambda t: t.timestamp, reverse=True)

        # Paginate
        total = len(all_traces)
        traces = all_traces[offset:offset + limit]

        return TraceListResponse(
            traces=traces,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(f"Error reading traces: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read traces: {str(e)}",
        ) from e


@router.get(
    "/ml-advisor/traces/{trace_id}",
    response_model=CodingTrace,
    summary="Get a specific trace",
    description="Retrieve a single coding trace by ID.",
)
async def get_trace(
    trace_id: str,
    config: AdvisorConfig,
) -> CodingTrace:
    """
    Get a specific coding trace by ID.
    """
    trace_path = config["trace_path"]

    if not trace_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace {trace_id} not found",
        )

    try:
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("trace_id") == trace_id:
                        return CodingTrace(**data)
                except json.JSONDecodeError:
                    continue

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace {trace_id} not found",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading trace: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read trace: {str(e)}",
        ) from e


@router.get(
    "/ml-advisor/traces/export",
    summary="Export traces as JSONL",
    description="Download all traces as a JSONL file.",
)
async def export_traces(
    config: AdvisorConfig,
    source: str | None = Query(default=None, description="Filter by source"),
) -> StreamingResponse:
    """
    Export traces as JSONL for offline analysis.
    """
    trace_path = config["trace_path"]

    if not trace_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No traces found",
        )

    def generate():
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                if source:
                    try:
                        data = json.loads(line.strip())
                        if data.get("source") != source:
                            continue
                    except json.JSONDecodeError:
                        continue
                yield line

    filename = f"coding_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


@router.get(
    "/ml-advisor/metrics",
    response_model=EvaluationMetrics,
    summary="Get evaluation metrics",
    description="Calculate evaluation metrics from coding traces.",
)
async def get_metrics(
    config: AdvisorConfig,
) -> EvaluationMetrics:
    """
    Calculate evaluation metrics from coding traces.

    Returns agreement rates, code coverage, and accuracy metrics
    (if human-reviewed codes are available).
    """
    trace_path = config["trace_path"]

    if not trace_path.exists():
        return EvaluationMetrics()

    try:
        total_traces = 0
        traces_with_advisor = 0
        traces_with_final = 0
        full_agreement = 0
        advisor_suggested_extras = 0
        advisor_suggested_removals = 0
        unique_rule_codes: set[str] = set()
        unique_advisor_codes: set[str] = set()

        # For precision/recall
        rule_tp, rule_fp, rule_fn = 0, 0, 0
        advisor_tp, advisor_fp, advisor_fn = 0, 0, 0

        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    total_traces += 1

                    rule_codes = set(data.get("autocode_codes", []))
                    advisor_codes = set(data.get("advisor_candidate_codes", []))
                    final_codes = data.get("final_codes")

                    unique_rule_codes.update(rule_codes)

                    if advisor_codes:
                        traces_with_advisor += 1
                        unique_advisor_codes.update(advisor_codes)

                        if rule_codes == advisor_codes:
                            full_agreement += 1

                        if advisor_codes - rule_codes:
                            advisor_suggested_extras += 1

                        disagreements = data.get("advisor_disagreements", [])
                        if any(c in rule_codes for c in disagreements):
                            advisor_suggested_removals += 1

                    if final_codes is not None:
                        traces_with_final += 1
                        final_set = set(final_codes)

                        rule_tp += len(rule_codes & final_set)
                        rule_fp += len(rule_codes - final_set)
                        rule_fn += len(final_set - rule_codes)

                        advisor_tp += len(advisor_codes & final_set)
                        advisor_fp += len(advisor_codes - final_set)
                        advisor_fn += len(final_set - advisor_codes)

                except json.JSONDecodeError:
                    continue

        # Calculate precision/recall
        rule_precision = None
        rule_recall = None
        advisor_precision = None
        advisor_recall = None

        if traces_with_final > 0:
            if rule_tp + rule_fp > 0:
                rule_precision = rule_tp / (rule_tp + rule_fp)
            if rule_tp + rule_fn > 0:
                rule_recall = rule_tp / (rule_tp + rule_fn)
            if advisor_tp + advisor_fp > 0:
                advisor_precision = advisor_tp / (advisor_tp + advisor_fp)
            if advisor_tp + advisor_fn > 0:
                advisor_recall = advisor_tp / (advisor_tp + advisor_fn)

        return EvaluationMetrics(
            total_traces=total_traces,
            traces_with_advisor=traces_with_advisor,
            traces_with_final=traces_with_final,
            full_agreement=full_agreement,
            advisor_suggested_extras=advisor_suggested_extras,
            advisor_suggested_removals=advisor_suggested_removals,
            unique_rule_codes=len(unique_rule_codes),
            unique_advisor_codes=len(unique_advisor_codes),
            rule_precision=rule_precision,
            rule_recall=rule_recall,
            advisor_precision=advisor_precision,
            advisor_recall=advisor_recall,
        )

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate metrics: {str(e)}",
        ) from e
