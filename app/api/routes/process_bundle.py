"""ZK multi-document bundle processing endpoint.

This endpoint is additive and intended for a client-side temporal translation flow:
- The browser scrubs PHI and replaces absolute dates with relative `T±N` tokens.
- The server enforces a strict "no absolute dates" guardrail and runs extraction per doc.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.agents.aggregator.timeline_aggregator import BundleDocInput, aggregate_entity_ledger
from app.agents.relation_extraction.llm_proposer import propose_relations_ml
from app.agents.relation_extraction.shadow_mode import merge_relations_shadow_mode
from app.api.dependencies import get_coding_service, get_registry_service
from app.api.phi_dependencies import get_phi_scrubber
from app.api.readiness import require_ready
from app.api.schemas import (
    BundleDocResponse,
    ProcessBundleRequest,
    ProcessBundleResponse,
    UnifiedProcessRequest,
)
from app.api.services.bundle_processing import (
    count_date_like_strings,
    extract_doc_t_offset_days,
    strip_system_header,
)
from app.api.services.unified_pipeline import run_unified_pipeline_logic
from app.coder.application.coding_service import CodingService
from app.registry.application.registry_service import RegistryService

router = APIRouter(tags=["process"])
_ready_dep = Depends(require_ready)
_registry_service_dep = Depends(get_registry_service)
_coding_service_dep = Depends(get_coding_service)
_phi_scrubber_dep = Depends(get_phi_scrubber)


def _timeline_summary(docs: list[BundleDocResponse]) -> dict[str, Any]:
    offsets_by_role: dict[str, int] = {}
    role_seq: dict[str, int] = {}
    follow_up_offsets: list[int] = []

    for doc in docs:
        role = doc.timepoint_role.value
        offset = doc.doc_t_offset_days
        if offset is None:
            continue
        if role == "FOLLOW_UP":
            follow_up_offsets.append(int(offset))
        prev_seq = role_seq.get(role)
        if prev_seq is None or int(doc.seq) < int(prev_seq):
            offsets_by_role[role] = int(offset)
            role_seq[role] = int(doc.seq)

    imaging = offsets_by_role.get("PREOP_IMAGING")
    index_proc = offsets_by_role.get("INDEX_PROCEDURE")
    pathology = offsets_by_role.get("PATHOLOGY")

    time_to_diagnosis_days: int | None = None
    if imaging is not None and pathology is not None:
        time_to_diagnosis_days = int(pathology - imaging)

    time_to_treatment_days: int | None = None
    if pathology is not None and index_proc is not None:
        time_to_treatment_days = int(index_proc - pathology)

    follow_up_offsets_sorted = sorted(set(follow_up_offsets))
    follow_up_intervals: list[int] = []
    for prev, cur in zip(follow_up_offsets_sorted, follow_up_offsets_sorted[1:], strict=False):
        follow_up_intervals.append(int(cur - prev))

    return {
        "doc_offsets_by_role": offsets_by_role,
        "time_to_diagnosis_days": time_to_diagnosis_days,
        "time_to_treatment_days": time_to_treatment_days,
        "follow_up_offsets": follow_up_offsets_sorted,
        "follow_up_interval_lengths": follow_up_intervals,
    }


@router.post(
    "/v1/process_bundle",
    response_model=ProcessBundleResponse,
    response_model_exclude_none=True,
    summary="Process a zero-knowledge multi-document bundle (relative timelines only)",
)
async def process_bundle(
    payload: ProcessBundleRequest,
    request: Request,
    response: Response,
    _ready: None = _ready_dep,
    registry_service: RegistryService = _registry_service_dep,
    coding_service: CodingService = _coding_service_dep,
    phi_scrubber=_phi_scrubber_dep,
) -> ProcessBundleResponse:
    start_time = time.time()
    response.headers["X-Process-Route"] = "bundle_router"

    # Guardrail: reject absolute date-like strings (PHI leak) in any document.
    for doc in payload.documents:
        leaks = count_date_like_strings(doc.text)
        if leaks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Bundle document contains raw date-like strings. "
                    "Apply client-side temporal translation (T±N tokens) before sending."
                ),
            )

    docs_out: list[BundleDocResponse] = []
    for doc in sorted(payload.documents, key=lambda d: int(d.seq)):
        doc_offset_days = extract_doc_t_offset_days(doc.text)
        clean_text = strip_system_header(doc.text)
        unified_req = UnifiedProcessRequest(
            note=clean_text,
            already_scrubbed=payload.already_scrubbed,
            locality=payload.locality,
            include_financials=payload.include_financials,
            explain=payload.explain,
            include_v3_event_log=payload.include_v3_event_log,
        )
        result, _, _ = await run_unified_pipeline_logic(
            payload=unified_req,
            request=request,
            registry_service=registry_service,
            coding_service=coding_service,
            phi_scrubber=phi_scrubber,
        )

        docs_out.append(
            BundleDocResponse(
                timepoint_role=doc.timepoint_role,
                seq=doc.seq,
                doc_t_offset_days=doc_offset_days,
                result=result,
            )
        )

    processing_time_ms = (time.time() - start_time) * 1000.0
    ledger = aggregate_entity_ledger(
        [
            BundleDocInput(
                timepoint_role=doc.timepoint_role.value,
                seq=doc.seq,
                doc_t_offset_days=doc.doc_t_offset_days,
                registry=doc.result.registry,
            )
            for doc in docs_out
        ]
    )
    confidence_threshold = float(os.getenv("RELATIONS_ML_CONFIDENCE_THRESHOLD", "0.85"))
    ml_result = propose_relations_ml(
        ledger=ledger,
        relations_heuristic=ledger.link_proposals,
    )
    shadow = merge_relations_shadow_mode(
        relations_heuristic=ledger.link_proposals,
        relations_ml=ml_result.relations_ml,
        confidence_threshold=confidence_threshold,
    )
    relations_warnings = list(ml_result.warnings) + list(shadow.warnings)
    relations_metrics = {
        "ml": ml_result.metrics,
        "merge": shadow.metrics,
    }
    return ProcessBundleResponse(
        zk_patient_id=payload.zk_patient_id,
        episode_id=payload.episode_id,
        documents=docs_out,
        timeline=_timeline_summary(docs_out),
        entity_ledger=ledger,
        relations_heuristic=shadow.relations_heuristic,
        relations_ml=shadow.relations_ml,
        relations=shadow.relations,
        relations_warnings=relations_warnings,
        relations_metrics=relations_metrics,
        processing_time_ms=processing_time_ms,
    )


__all__ = ["router"]
