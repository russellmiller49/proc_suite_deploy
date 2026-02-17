"""Interactive reporting route handlers."""

from __future__ import annotations

import logging
import os
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


def _apply_reporter_completeness_uplift(record: Any, note_text: str) -> Any:
    """Best-effort uplift for reporter addendum values (age/ASA/ECOG + EBUS detail)."""
    if not note_text or record is None:
        return record

    try:
        from app.registry.schema import RegistryRecord
    except Exception:
        return record

    is_dict = isinstance(record, dict)
    if not isinstance(record, RegistryRecord) and not is_dict:
        return record

    try:
        import re

        from app.registry.deterministic_extractors import run_deterministic_extractors
        from app.registry.heuristics.linear_ebus_station_detail import (
            apply_linear_ebus_station_detail_heuristics,
        )
    except Exception:
        return record

    seed = run_deterministic_extractors(note_text)
    if not isinstance(seed, dict):
        seed = {}

    data = record.model_dump(exclude_none=False) if isinstance(record, RegistryRecord) else record
    changed = False

    # Patient demographics
    age = seed.get("patient_age")
    gender = seed.get("gender")
    if age is not None or gender:
        demo = data.get("patient_demographics") or {}
        if not isinstance(demo, dict):
            demo = {}
        if age is not None and demo.get("age_years") is None:
            demo["age_years"] = age
            changed = True
        if gender and not demo.get("gender"):
            g = str(gender).strip()
            if g.lower() == "m":
                g = "Male"
            elif g.lower() == "f":
                g = "Female"
            demo["gender"] = g
            changed = True
        if changed:
            data["patient_demographics"] = demo

        patient = data.get("patient") or {}
        if not isinstance(patient, dict):
            patient = {}
        patient_changed = False
        if age is not None and patient.get("age") is None:
            patient["age"] = age
            patient_changed = True
        if gender and not patient.get("sex"):
            g = str(gender).strip()
            g_lower = g.lower()
            if g_lower in {"male", "m"}:
                g = "M"
            elif g_lower in {"female", "f"}:
                g = "F"
            else:
                g = "O"
            patient["sex"] = g
            patient_changed = True
        if patient_changed:
            data["patient"] = patient
            changed = True

    # ASA: explicit-only
    asa_val = seed.get("asa_class")
    if asa_val is not None and re.search(r"(?i)\bASA\b", note_text):
        clinical = data.get("clinical_context") or {}
        if not isinstance(clinical, dict):
            clinical = {}
        if clinical.get("asa_class") is None:
            clinical["asa_class"] = asa_val
            data["clinical_context"] = clinical
            changed = True

        risk = data.get("risk_assessment") or {}
        if not isinstance(risk, dict):
            risk = {}
        if risk.get("asa_class") is None:
            risk["asa_class"] = asa_val
            data["risk_assessment"] = risk
            changed = True

    # ECOG/Zubrod: explicit-only
    ecog_score = seed.get("ecog_score")
    ecog_text = seed.get("ecog_text")
    if (ecog_score is not None or ecog_text) and re.search(r"(?i)\b(?:ECOG|Zubrod)\b", note_text):
        clinical = data.get("clinical_context") or {}
        if not isinstance(clinical, dict):
            clinical = {}
        if clinical.get("ecog_score") is None and not clinical.get("ecog_text"):
            if ecog_score is not None:
                clinical["ecog_score"] = ecog_score
                changed = True
            elif isinstance(ecog_text, str) and ecog_text.strip():
                clinical["ecog_text"] = ecog_text.strip()
                changed = True
        if changed:
            data["clinical_context"] = clinical

    if is_dict:
        # Also set flat compat fields used by reporter bundle builder.
        if age is not None and data.get("patient_age") in (None, "", [], {}):
            data["patient_age"] = age
        if gender and data.get("gender") in (None, "", [], {}):
            g = str(gender).strip()
            if g.lower() == "m":
                g = "Male"
            elif g.lower() == "f":
                g = "Female"
            data["gender"] = g
        if asa_val is not None and data.get("asa_class") in (None, "", [], {}):
            data["asa_class"] = asa_val
        if ecog_score is not None and data.get("ecog_score") in (None, "", [], {}):
            data["ecog_score"] = ecog_score
        if ecog_text and data.get("ecog_text") in (None, "", [], {}):
            data["ecog_text"] = ecog_text

        # Apply linear EBUS per-station detail extraction to dict payloads.
        try:
            from app.registry.processing.linear_ebus_stations_detail import (
                extract_linear_ebus_stations_detail,
            )

            parsed = extract_linear_ebus_stations_detail(note_text)
            if parsed:
                granular = data.get("granular_data") or {}
                if not isinstance(granular, dict):
                    granular = {}
                existing_raw = granular.get("linear_ebus_stations_detail")
                existing = (
                    [dict(item) for item in existing_raw if isinstance(item, dict)]
                    if isinstance(existing_raw, list)
                    else []
                )
                by_station: dict[str, dict[str, Any]] = {}
                order: list[str] = []
                for item in existing:
                    station = str(item.get("station") or "").strip()
                    if not station:
                        continue
                    if station not in by_station:
                        order.append(station)
                    by_station[station] = item
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    station = str(item.get("station") or "").strip()
                    if not station:
                        continue
                    existing_item = by_station.get(station)
                    if existing_item is None:
                        by_station[station] = dict(item)
                        order.append(station)
                        continue
                    for key, value in item.items():
                        if key == "station":
                            continue
                        if value in (None, "", [], {}):
                            continue
                        if existing_item.get(key) in (None, "", [], {}):
                            existing_item[key] = value
                    by_station[station] = existing_item
                merged = [by_station[s] for s in order if s in by_station]
                if merged:
                    granular["linear_ebus_stations_detail"] = merged
                    stations = [str(item.get("station")) for item in merged if item.get("station")]
                    if stations and data.get("linear_ebus_stations") in (None, "", [], {}):
                        data["linear_ebus_stations"] = stations
                    if stations and data.get("ebus_stations_sampled") in (None, "", [], {}):
                        data["ebus_stations_sampled"] = stations
                    data["granular_data"] = granular
        except Exception:
            pass
        return data

    record_out = record
    if changed:
        try:
            record_out = RegistryRecord(**data)
        except Exception:
            record_out = record

    # Apply linear EBUS per-station detail extraction (uses note text).
    try:
        record_out, _ = apply_linear_ebus_station_detail_heuristics(note_text, record_out)
    except Exception:
        pass

    return record_out


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

    redaction = apply_phi_redaction(
        req.text,
        phi_scrubber,
        already_scrubbed=bool(req.already_scrubbed),
    )
    note_text = redaction.text

    seed_strategy = os.getenv("REPORTER_SEED_STRATEGY", "registry_extract_fields").strip().lower()
    llm_strict_raw = os.getenv("REPORTER_SEED_LLM_STRICT", "0").strip().lower()
    llm_strict = llm_strict_raw in {"1", "true", "yes", "y"}

    seed_warnings: list[str] = []
    seed_record_for_completeness = None
    seed_text_for_bundle = note_text
    extraction_source: Any = None

    if debug_notes is not None:
        debug_notes.append(
            {
                "type": "seed_strategy",
                "strategy": seed_strategy,
                "already_scrubbed": bool(req.already_scrubbed),
                "redaction": {
                    "was_scrubbed": bool(redaction.was_scrubbed),
                    "entity_count": int(redaction.entity_count),
                    "warning": redaction.warning,
                },
            }
        )

    if seed_strategy == "llm_findings":
        try:
            from app.reporting.llm_findings import (
                ReporterFindingsError,
                build_record_payload_for_reporting,
                seed_registry_record_from_llm_findings,
            )

            seed = seed_registry_record_from_llm_findings(note_text)
            seed_warnings.extend(list(seed.warnings or []))
            seed_record_for_completeness = seed.record
            seed_text_for_bundle = seed.masked_prompt_text
            extraction_source = build_record_payload_for_reporting(seed)
            if debug_notes is not None:
                debug_notes.append(
                    {
                        "type": "llm_findings_seed",
                        "accepted_findings": int(seed.accepted_findings),
                        "dropped_findings": int(seed.dropped_findings),
                        "needs_review": bool(seed.needs_review),
                        "derived_cpt_count": int(len(seed.cpt_codes or [])),
                    }
                )
        except ReporterFindingsError as exc:
            seed_warnings.append(
                f"REPORTER_SEED_FALLBACK: llm_findings_failed ({type(exc).__name__})"
            )
            if llm_strict:
                raise HTTPException(status_code=502, detail="LLM findings seeding failed") from exc
            seed_strategy = "registry_extract_fields"
        except Exception as exc:  # noqa: BLE001
            seed_warnings.append(
                f"REPORTER_SEED_FALLBACK: llm_findings_failed ({type(exc).__name__})"
            )
            if llm_strict:
                raise HTTPException(status_code=502, detail="LLM findings seeding failed") from exc
            seed_strategy = "registry_extract_fields"

    extraction_result = None
    if seed_strategy != "llm_findings":
        extraction_result = await run_cpu(request.app, registry_service.extract_fields, note_text)
        extraction_source = extraction_result.record
        seed_record_for_completeness = extraction_result.record
        seed_text_for_bundle = note_text

    # Apply completeness addendum uplift for reporter flow (age/ASA/ECOG + EBUS detail).
    seed_record_for_completeness = _apply_reporter_completeness_uplift(
        seed_record_for_completeness,
        note_text,
    )
    extraction_source = _apply_reporter_completeness_uplift(extraction_source, note_text)

    bundle = build_procedure_bundle_from_extraction(
        extraction_source,
        source_text=seed_text_for_bundle,
    )
    bundle = _apply_seed_metadata(bundle, req.metadata)
    if not bundle.free_text_hint:
        bundle_payload = bundle.model_dump(exclude_none=False)
        bundle_payload["free_text_hint"] = seed_text_for_bundle
        bundle = ProcedureBundle.model_validate(bundle_payload)

    bundle, issues, warnings, suggestions, notes = _verify_bundle(bundle, debug_notes=debug_notes)
    warnings = list(seed_warnings) + list(warnings or [])
    missing_field_prompts: list[dict[str, object]] = []
    try:
        from app.registry.completeness import generate_missing_field_prompts

        completeness_prompts = generate_missing_field_prompts(seed_record_for_completeness)
        if completeness_prompts:
            missing_field_prompts = [
                {
                    "group": prompt.group,
                    "path": prompt.path,
                    "target_path": prompt.target_path,
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
