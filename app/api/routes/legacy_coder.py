"""Legacy raw-text coder route handlers."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.coder_adapter import convert_coding_result_to_coder_output
from app.api.dependencies import get_coding_service
from app.api.guards import enforce_legacy_endpoints_allowed
from app.api.schemas import CoderRequest, CoderResponse, HybridPipelineMetadata
from app.coder.application.coding_service import CodingService
from app.coder.phi_gating import is_phi_review_required
from app.coder.schema import CodeDecision, CoderOutput
from app.infra.executors import run_cpu
from config.settings import CoderSettings

router = APIRouter(tags=["legacy-coder"])
_coding_service_dep = Depends(get_coding_service)


@router.post("/v1/coder/run", response_model=CoderResponse)
async def coder_run(
    req: CoderRequest,
    request: Request,
    mode: str | None = None,
    coding_service: CodingService = _coding_service_dep,
) -> CoderResponse:
    """Legacy raw-text coder shim (non-PHI)."""
    enforce_legacy_endpoints_allowed()
    require_review = is_phi_review_required()
    procedure_id = str(uuid.uuid4())
    report_text = req.note

    if require_review:
        raise HTTPException(
            status_code=400,
            detail=(
                "Direct coding on raw text is disabled; submit via /v1/phi and review "
                "before coding."
            ),
        )

    mode_value = (mode or req.mode or "").strip().lower()

    if req.use_ml_first:
        use_llm_fallback = mode_value != "rules_only"
        return await _run_ml_first_pipeline(
            request,
            report_text,
            req.locality,
            coding_service,
            use_llm_fallback=use_llm_fallback,
        )

    use_llm = mode_value != "rules_only"

    result = await run_cpu(
        request.app,
        coding_service.generate_result,
        procedure_id=procedure_id,
        report_text=report_text,
        use_llm=use_llm,
        procedure_type=None,
    )

    return convert_coding_result_to_coder_output(
        result=result,
        kb_repo=coding_service.kb_repo,
        locality=req.locality,
    )


async def _run_ml_first_pipeline(
    request: Request,
    report_text: str,
    locality: str,
    coding_service: CodingService,
    *,
    use_llm_fallback: bool = True,
) -> CoderResponse:
    """Run the ML-first hybrid pipeline (SmartHybridOrchestrator)."""
    from app.coder.application.smart_hybrid_policy import build_hybrid_orchestrator

    def _run_hybrid() -> Any:
        if use_llm_fallback:
            orchestrator = build_hybrid_orchestrator()
            return orchestrator.get_codes(report_text)

        from app.coder.adapters.llm.noop_advisor import NoOpLLMAdvisorAdapter

        orchestrator = build_hybrid_orchestrator(llm_advisor=NoOpLLMAdvisorAdapter())
        result = orchestrator.get_codes(report_text)
        result.metadata = dict(result.metadata or {})
        result.metadata["llm_called"] = False
        result.metadata["llm_disabled"] = True
        if result.source == "hybrid_llm_fallback":
            result.source = "ml_rules_no_llm"
        return result

    result = await run_cpu(request.app, _run_hybrid)

    code_decisions = []
    for cpt in result.codes:
        proc_info = coding_service.kb_repo.get_procedure_info(cpt)
        desc = proc_info.description if proc_info else ""
        code_decisions.append(
            CodeDecision(
                cpt=cpt,
                description=desc,
                confidence=1.0,
                modifiers=[],
                rationale=f"Source: {result.source}",
            )
        )

    financials = None
    if code_decisions:
        from app.coder.schema import FinancialSummary, PerCodeBilling

        per_code_billing: list[PerCodeBilling] = []
        total_work_rvu = 0.0
        total_facility_payment = 0.0
        conversion_factor = CoderSettings().cms_conversion_factor

        for decision in code_decisions:
            proc_info = coding_service.kb_repo.get_procedure_info(decision.cpt)
            if proc_info:
                work_rvu = proc_info.work_rvu
                total_rvu = proc_info.total_facility_rvu
                payment = total_rvu * conversion_factor

                total_work_rvu += work_rvu
                total_facility_payment += payment

                per_code_billing.append(
                    PerCodeBilling(
                        cpt_code=decision.cpt,
                        description=decision.description,
                        modifiers=decision.modifiers,
                        work_rvu=work_rvu,
                        total_facility_rvu=total_rvu,
                        facility_payment=payment,
                        allowed_facility_rvu=total_rvu,
                        allowed_facility_payment=payment,
                    )
                )

        if per_code_billing:
            financials = FinancialSummary(
                conversion_factor=conversion_factor,
                locality=locality,
                per_code=per_code_billing,
                total_work_rvu=total_work_rvu,
                total_facility_payment=total_facility_payment,
                total_nonfacility_payment=0.0,
            )

    hybrid_metadata = HybridPipelineMetadata(
        difficulty=result.difficulty.value,
        source=result.source,
        llm_used=result.metadata.get("llm_called", False),
        ml_candidates=result.metadata.get("ml_candidates", []),
        fallback_reason=result.metadata.get("reason_for_fallback"),
        rules_error=result.metadata.get("rules_error"),
    )

    return CoderOutput(
        codes=code_decisions,
        financials=financials,
        warnings=[],
        explanation=None,
        hybrid_metadata=hybrid_metadata.model_dump(),
    )


__all__ = ["router"]
