"""Interactive reporting route handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.dependencies import get_registry_service
from app.api.phi_dependencies import get_phi_scrubber
from app.api.phi_redaction import apply_phi_redaction
from app.api.readiness import require_ready
from app.api.schemas import (
    QuestionsRequest,
    QuestionsResponse,
    RenderRequest,
    RenderResponse,
    SeedFromTextRequest,
    SeedFromTextResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.infra.executors import run_cpu
from app.registry.application.registry_service import RegistryService
from app.reporting import (
    BundleJsonPatchError,
    MissingFieldIssue,
    ProcedureBundle,
    apply_bundle_json_patch,
    build_questions,
)
from app.reporting.engine import (
    ReporterEngine,
    _load_procedure_order,
    apply_bundle_patch,
    apply_patch_result,
    build_procedure_bundle_from_extraction,
    default_schema_registry,
    default_template_registry,
)
from app.reporting.inference import InferenceEngine
from app.reporting.macro_registry import get_macro_registry
from app.reporting.normalization.normalize import normalize_bundle
from app.reporting.validation import ValidationEngine

router = APIRouter(tags=["reporting"])
_logger = logging.getLogger(__name__)

_ready_dep = Depends(require_ready)
_registry_service_dep = Depends(get_registry_service)
_phi_scrubber_dep = Depends(get_phi_scrubber)


def _verify_bundle(
    bundle: ProcedureBundle,
    *,
    debug_notes: list[dict[str, Any]] | None = None,
) -> tuple[
    ProcedureBundle,
    list[MissingFieldIssue],
    list[str],
    list[str],
    list[str],
]:
    normalized = normalize_bundle(bundle)
    bundle = normalized.bundle
    if debug_notes is not None:
        debug_notes.append(
            {
                "type": "normalization",
                "notes": [
                    {
                        "kind": note.kind,
                        "path": note.path,
                        "message": note.message,
                        "source": note.source,
                    }
                    for note in normalized.notes
                ],
            }
        )
    templates = default_template_registry()
    schemas = default_schema_registry()
    inference = InferenceEngine()
    inference_result = inference.infer_bundle(bundle)
    bundle = apply_patch_result(bundle, inference_result)
    validator = ValidationEngine(templates, schemas)
    issues = validator.list_missing_critical_fields(bundle)
    warnings = validator.apply_warn_if_rules(bundle)
    suggestions = validator.list_suggestions(bundle)
    return bundle, issues, warnings, suggestions, inference_result.notes


def _render_bundle_markdown(
    bundle: ProcedureBundle,
    *,
    issues: list[MissingFieldIssue],
    warnings: list[str],
    strict: bool,
    embed_metadata: bool,
    debug_notes: list[dict[str, Any]] | None = None,
) -> str:
    templates = default_template_registry()
    schemas = default_schema_registry()
    engine = ReporterEngine(
        templates,
        schemas,
        procedure_order=_load_procedure_order(),
        render_style="builder",
        macro_registry=get_macro_registry(),
    )
    try:
        structured = engine.compose_report_with_metadata(
            bundle,
            strict=strict,
            embed_metadata=embed_metadata,
            validation_issues=issues,
            warnings=warnings,
        )
    except ValueError as exc:
        message = str(exc)
        if not strict:
            raise
        if not (
            message.startswith("Style validation failed:")
            or message.startswith("Missing required fields for")
        ):
            raise
        if debug_notes is not None:
            debug_notes.append(
                {
                    "type": "strict_fallback",
                    "error": message,
                    "action": "fallback_to_non_strict_preview",
                }
            )
        _logger.warning(
            "Strict report render failed; falling back to non-strict preview",
            extra={"error": message},
        )
        structured = engine.compose_report_with_metadata(
            bundle,
            strict=False,
            embed_metadata=embed_metadata,
            validation_issues=issues,
            warnings=warnings,
        )
    return structured.text


def _debug_template_selection(bundle: ProcedureBundle) -> dict[str, Any]:
    templates = default_template_registry()
    macros = get_macro_registry()
    procedures: list[dict[str, Any]] = []
    for proc in bundle.procedures:
        metas = templates.find_for_procedure(proc.proc_type, proc.cpt_candidates)
        procedures.append(
            {
                "proc_id": proc.proc_id or proc.schema_id,
                "proc_type": proc.proc_type,
                "cpt_candidates": [str(code) for code in (proc.cpt_candidates or [])],
                "template_ids": [meta.id for meta in metas],
                "macro_exists": macros.maybe_get(proc.proc_type) is not None,
            }
        )
    return {"type": "selection", "procedures": procedures}


def _apply_render_patch(
    bundle: ProcedureBundle,
    req: RenderRequest,
) -> ProcedureBundle:
    patch_payload = req.patch
    if not patch_payload:
        return bundle
    if isinstance(patch_payload, list):
        ops: list[dict[str, Any]] = []
        for op in patch_payload:
            if isinstance(op, BaseModel):
                ops.append(op.model_dump(exclude_none=False))
            else:
                ops.append(dict(op))
        return apply_bundle_json_patch(bundle, ops)
    return apply_bundle_patch(bundle, patch_payload)


def _apply_seed_metadata(bundle: ProcedureBundle, metadata: dict[str, Any]) -> ProcedureBundle:
    if not metadata:
        return bundle

    def _as_text(value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

    payload = bundle.model_dump(exclude_none=False)
    encounter = payload.get("encounter") or {}

    indication = _as_text(metadata.get("indication_text") or metadata.get("indication"))
    if indication:
        payload["indication_text"] = indication

    preop = _as_text(metadata.get("preop_diagnosis_text") or metadata.get("preop_diagnosis"))
    if preop:
        payload["preop_diagnosis_text"] = preop

    postop = _as_text(metadata.get("postop_diagnosis_text") or metadata.get("postop_diagnosis"))
    if postop:
        payload["postop_diagnosis_text"] = postop

    impression = _as_text(
        metadata.get("impression_plan") or metadata.get("plan") or metadata.get("disposition")
    )
    if impression:
        payload["impression_plan"] = impression

    attending = _as_text(metadata.get("attending"))
    if attending:
        encounter["attending"] = attending
    location = _as_text(metadata.get("location"))
    if location:
        encounter["location"] = location
    date_value = _as_text(metadata.get("date") or metadata.get("procedure_date"))
    if date_value:
        encounter["date"] = date_value
    encounter_id = _as_text(metadata.get("encounter_id"))
    if encounter_id:
        encounter["encounter_id"] = encounter_id

    payload["encounter"] = encounter
    return ProcedureBundle.model_validate(payload)


@router.post("/report/verify", response_model=VerifyResponse)
async def report_verify(req: VerifyRequest) -> VerifyResponse:
    bundle = build_procedure_bundle_from_extraction(req.extraction)
    bundle, issues, warnings, suggestions, notes = _verify_bundle(bundle)
    return VerifyResponse(
        bundle=bundle,
        issues=issues,
        warnings=warnings,
        suggestions=suggestions,
        inference_notes=notes,
    )


@router.post("/report/questions", response_model=QuestionsResponse)
async def report_questions(req: QuestionsRequest) -> QuestionsResponse:
    bundle, issues, warnings, suggestions, notes = _verify_bundle(req.bundle)
    questions = build_questions(bundle, issues)
    return QuestionsResponse(
        bundle=bundle,
        issues=issues,
        warnings=warnings,
        suggestions=suggestions,
        inference_notes=notes,
        questions=questions,
    )


@router.post("/report/seed_from_text", response_model=SeedFromTextResponse)
async def report_seed_from_text(
    req: SeedFromTextRequest,
    request: Request,
    _ready: None = _ready_dep,
    registry_service: RegistryService = _registry_service_dep,
    phi_scrubber=_phi_scrubber_dep,
) -> SeedFromTextResponse:
    debug_enabled = bool(req.debug or req.include_debug)
    debug_notes: list[dict[str, Any]] | None = [] if debug_enabled else None

    redaction = apply_phi_redaction(req.text, phi_scrubber)
    note_text = redaction.text

    extraction_result = await run_cpu(request.app, registry_service.extract_fields, note_text)
    bundle = build_procedure_bundle_from_extraction(extraction_result.record, source_text=note_text)
    bundle = _apply_seed_metadata(bundle, req.metadata)
    if not bundle.free_text_hint:
        bundle_payload = bundle.model_dump(exclude_none=False)
        bundle_payload["free_text_hint"] = note_text
        bundle = ProcedureBundle.model_validate(bundle_payload)

    bundle, issues, warnings, suggestions, notes = _verify_bundle(bundle, debug_notes=debug_notes)
    missing_field_prompts: list[dict[str, object]] = []
    try:
        from app.registry.completeness import generate_missing_field_prompts

        completeness_prompts = generate_missing_field_prompts(extraction_result.record)
        if completeness_prompts:
            missing_field_prompts = [
                {
                    "group": prompt.group,
                    "path": prompt.path,
                    "label": prompt.label,
                    "severity": prompt.severity,
                    "message": prompt.message,
                }
                for prompt in completeness_prompts
            ]
            suggestions = list(suggestions or [])
            for prompt in completeness_prompts:
                suggestions.append(
                    f"Completeness ({prompt.severity}): {prompt.label} — {prompt.message}"
                )
    except Exception:
        missing_field_prompts = []
    if debug_notes is not None:
        debug_notes.append(_debug_template_selection(bundle))
    questions = build_questions(bundle, issues)
    markdown = _render_bundle_markdown(
        bundle,
        issues=issues,
        warnings=warnings,
        strict=req.strict,
        embed_metadata=False,
        debug_notes=debug_notes,
    )
    return SeedFromTextResponse(
        bundle=bundle,
        markdown=markdown,
        issues=issues,
        warnings=warnings,
        inference_notes=notes,
        suggestions=suggestions,
        questions=questions,
        missing_field_prompts=missing_field_prompts,
        debug_notes=debug_notes if debug_enabled else None,
    )


@router.post("/report/render", response_model=RenderResponse)
async def report_render(req: RenderRequest) -> RenderResponse:
    debug_enabled = bool(req.debug or req.include_debug)
    debug_notes: list[dict[str, Any]] | None = [] if debug_enabled else None

    bundle = req.bundle
    try:
        bundle = _apply_render_patch(bundle, req)
    except BundleJsonPatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    bundle, issues, warnings, suggestions, notes = _verify_bundle(bundle, debug_notes=debug_notes)
    if debug_notes is not None:
        debug_notes.append(_debug_template_selection(bundle))
    markdown = _render_bundle_markdown(
        bundle,
        issues=issues,
        warnings=warnings,
        strict=req.strict,
        embed_metadata=req.embed_metadata,
        debug_notes=debug_notes,
    )
    return RenderResponse(
        bundle=bundle,
        markdown=markdown,
        issues=issues,
        warnings=warnings,
        inference_notes=notes,
        suggestions=suggestions,
        debug_notes=debug_notes if debug_enabled else None,
    )


__all__ = ["router"]
