"""Case event -> canonical snapshot aggregation service."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.common.llm import LLMService
from app.registry.aggregation.llm_models import ClinicalFollowupLLM, ImagingSnapshotLLM, PathologyResultsLLM
from app.registry.aggregation.clinical.extract_clinical_update import extract_clinical_update_event
from app.registry.aggregation.clinical.patch_clinical_update import patch_clinical_update
from app.registry.aggregation.imaging.extract_ct import extract_ct_event
from app.registry.aggregation.imaging.extract_pet_ct import extract_pet_ct_event
from app.registry.aggregation.imaging.patch_imaging import patch_imaging_update
from app.registry.aggregation.pathology.extract_pathology import extract_pathology_event
from app.registry.aggregation.pathology.extract_pathology_results import extract_pathology_results
from app.registry.aggregation.pathology.patch_pathology import patch_pathology_update
from app.registry.aggregation.pathology.patch_pathology_results import patch_pathology_results
from app.registry.aggregation.sanitize import compact_text
from app.registry.application.pathology_extraction import apply_pathology_extraction
from app.registry.schema import RegistryRecord
from app.registry_store.models import RegistryAppendedDocument, RegistryCaseRecord, RegistryRun


logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _aggregation_strategy() -> str:
    strategy = os.getenv("REGISTRY_CASE_AGGREGATION_STRATEGY", "pending_only")
    normalized = strategy.strip().lower()
    if normalized in {"pending_only", "reprocess_all"}:
        return normalized
    return "pending_only"


def _append_extraction_mode() -> str:
    raw = os.getenv("REGISTRY_APPEND_EXTRACTION_MODE", "auto")
    normalized = str(raw or "").strip().lower()
    if normalized in {"auto", "deterministic", "hybrid"}:
        return normalized
    return "auto"


def _llm_is_configured() -> bool:
    """Return True only when a *real* LLM is configured (avoid DeterministicStubLLM mismatches)."""

    provider = (os.getenv("LLM_PROVIDER", "gemini") or "gemini").strip().lower()
    if provider == "openai_compat":
        if _truthy_env("OPENAI_OFFLINE", default=False):
            return False
        if not os.getenv("OPENAI_API_KEY"):
            return False
        model = (os.getenv("OPENAI_MODEL_STRUCTURER") or os.getenv("OPENAI_MODEL") or "").strip()
        return bool(model)

    if provider == "gemini":
        if _truthy_env("GEMINI_OFFLINE", default=False):
            return False
        return bool(os.getenv("GEMINI_API_KEY"))

    return False


def _should_attempt_llm_fallback() -> bool:
    mode = _append_extraction_mode()
    if mode == "deterministic":
        return False
    # auto/hybrid: only when actually configured
    return _llm_is_configured()


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _merge_fill_missing(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if value is None:
            continue
        if _is_missing_scalar(out.get(key)):
            out[key] = value
    return out


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _llm_extract_pathology_results(note_text: str) -> dict[str, Any] | None:
    if not (note_text or "").strip():
        return None

    system_prompt = (
        "You extract structured fields from SCRUBBED pathology report text.\n"
        "Return a JSON object only. Use null when a field is not explicitly documented.\n"
        "Do not infer or guess. Do not introduce identifiers.\n"
        "For pathology_result_date, only return an ISO date string (YYYY-MM-DD) if explicitly stated."
    )
    user_prompt = (
        "Extract:\n"
        "- final_diagnosis (string|null)\n"
        "- final_staging (string|null)\n"
        "- microbiology_results (string|null)\n"
        "- pathology_result_date (YYYY-MM-DD|null)\n\n"
        f"TEXT:\n{note_text}"
    )

    llm = LLMService(task="structurer")
    result = llm.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=PathologyResultsLLM,
        temperature=0.0,
    )
    data = result.model_dump(exclude_none=True, mode="json")
    if not isinstance(data, dict) or not data:
        return None

    # Scrub/compact long free-text values. Preserve ISO date.
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if key == "pathology_result_date":
            cleaned[key] = value.strip()
        elif key == "final_staging":
            cleaned[key] = compact_text(value, max_chars=80) or None
        else:
            cleaned[key] = compact_text(value, max_chars=300) or None
    return {k: v for k, v in cleaned.items() if v is not None}


def _llm_extract_clinical_followup(note_text: str) -> dict[str, Any] | None:
    if not (note_text or "").strip():
        return None

    system_prompt = (
        "You extract structured follow-up status fields from SCRUBBED clinical notes.\n"
        "Return a JSON object only. Use null when not explicitly documented.\n"
        "Do not infer, and do not guess dates or identifiers.\n"
        "Allowed disease_status values: Progression, Stable, Response, Mixed, Indeterminate, or null."
    )
    user_prompt = (
        "Extract:\n"
        "- hospital_admission (boolean|null)\n"
        "- icu_admission (boolean|null)\n"
        "- deceased (boolean|null)\n"
        "- disease_status (Progression|Stable|Response|Mixed|Indeterminate|null)\n\n"
        f"TEXT:\n{note_text}"
    )

    llm = LLMService(task="structurer")
    result = llm.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ClinicalFollowupLLM,
        temperature=0.0,
    )
    data = result.model_dump(exclude_none=True, mode="json")
    return data if isinstance(data, dict) and data else None


def _llm_extract_imaging_snapshot(note_text: str) -> dict[str, Any] | None:
    if not (note_text or "").strip():
        return None

    system_prompt = (
        "You extract a minimal imaging summary from SCRUBBED imaging report text.\n"
        "Return a JSON object only. Use null when not explicitly documented.\n"
        "Do not infer. Allowed response values: Progression, Stable, Response, Mixed, Indeterminate, or null."
    )
    user_prompt = (
        "Extract:\n"
        "- response (Progression|Stable|Response|Mixed|Indeterminate|null)\n"
        "- overall_impression_text (string|null)\n\n"
        f"TEXT:\n{note_text}"
    )

    llm = LLMService(task="structurer")
    result = llm.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ImagingSnapshotLLM,
        temperature=0.0,
    )
    data = result.model_dump(exclude_none=True, mode="json")
    if not isinstance(data, dict) or not data:
        return None
    cleaned: dict[str, Any] = {}
    if isinstance(data.get("response"), str) and data["response"].strip():
        cleaned["response"] = data["response"].strip()
    if isinstance(data.get("overall_impression_text"), str) and data["overall_impression_text"].strip():
        cleaned["overall_impression_text"] = compact_text(str(data["overall_impression_text"]), max_chars=220) or None
    return {k: v for k, v in cleaned.items() if v is not None}


def _safe_row_text(row: RegistryAppendedDocument) -> str:
    note = str(row.note_text or "")
    if note.strip():
        return note
    metadata = row.metadata_json or {}
    structured = metadata.get("structured_data") if isinstance(metadata, dict) else None
    if isinstance(structured, dict) and structured:
        parts = [f"{k}={v}" for k, v in sorted(structured.items())]
        return "; ".join(parts)
    return ""


def _default_registry_json() -> dict[str, Any]:
    validated = RegistryRecord()
    return validated.model_dump(exclude_none=True, mode="json")


class CaseAggregator:
    """Deterministic case event aggregator with field-level lock filtering."""

    def __init__(self, *, strategy: str | None = None) -> None:
        self.strategy = (strategy or _aggregation_strategy()).strip().lower()

    def aggregate(
        self,
        *,
        db: Session,
        registry_uuid: uuid.UUID,
        user_id: str | None = None,
    ) -> RegistryCaseRecord:
        case_record = db.get(RegistryCaseRecord, registry_uuid)
        if case_record is None:
            now = _utcnow()
            case_record = RegistryCaseRecord(
                registry_uuid=registry_uuid,
                registry_json=_default_registry_json(),
                schema_version=(os.getenv("REGISTRY_SCHEMA_VERSION") or "v3").strip(),
                version=1,
                source_run_id=None,
                manual_overrides={},
                created_at=now,
                updated_at=now,
            )
            db.add(case_record)
            db.flush()

        registry_json = dict(case_record.registry_json or {})
        manual_overrides = dict(case_record.manual_overrides or {})
        bootstrap_changed = False

        if not case_record.source_run_id:
            latest = self._find_latest_registry_run_snapshot(db=db, registry_uuid=registry_uuid)
            if latest is not None:
                run_id, run_registry = latest
                merged_registry = _deep_merge_dict(run_registry, registry_json)
                if merged_registry != registry_json:
                    registry_json = merged_registry
                    bootstrap_changed = True
                if case_record.source_run_id != run_id:
                    case_record.source_run_id = run_id
                    bootstrap_changed = True
                if bootstrap_changed:
                    case_record.registry_json = registry_json

        events = self._load_events(db=db, registry_uuid=registry_uuid, user_id=user_id)
        if not events:
            if bootstrap_changed:
                case_record.version = int(case_record.version or 1) + 1
                case_record.updated_at = _utcnow()
                db.add(case_record)
            return case_record

        changed = bootstrap_changed
        processed_ids: list[str] = []

        for event_row in events:
            extracted = self._extract_event(event_row)
            event_row.extracted_json = extracted

            event_changed, qa_flags = self._apply_patch(
                registry_json,
                extracted=extracted,
                event_row=event_row,
                manual_overrides=manual_overrides,
            )

            try:
                validated = RegistryRecord(**registry_json)
                registry_json = validated.model_dump(exclude_none=True, mode="json")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Case aggregation validation failed for event {event_row.id}"
                ) from exc

            changed = changed or event_changed
            event_row.aggregated_at = _utcnow()
            processed_ids.append(str(event_row.id))
            if qa_flags:
                logger.info(
                    "case_aggregator event_qa_flags event_id=%s event_type=%s qa_flags=%s",
                    event_row.id,
                    event_row.event_type,
                    ",".join(sorted(set(qa_flags))),
                )

        if changed:
            case_record.registry_json = registry_json
            case_record.version = int(case_record.version or 1) + 1
            case_record.updated_at = _utcnow()

        final_version = int(case_record.version or 1)
        for event_row in events:
            event_row.aggregation_version = final_version
            db.add(event_row)

        db.add(case_record)
        logger.info(
            "case_aggregator aggregate_complete registry_uuid=%s processed_events=%d changed=%s",
            registry_uuid,
            len(processed_ids),
            changed,
        )
        return case_record

    def _find_latest_registry_run_snapshot(
        self,
        *,
        db: Session,
        registry_uuid: uuid.UUID,
    ) -> tuple[uuid.UUID, dict[str, Any]] | None:
        target = str(registry_uuid).strip().lower()
        if not target:
            return None

        # RegistryRun doesn't store registry_uuid as a dedicated column yet; scan recent runs.
        stmt: Select[tuple[RegistryRun]] = (
            select(RegistryRun)
            .order_by(RegistryRun.created_at.desc(), RegistryRun.id.desc())
            .limit(500)
        )
        runs = list(db.execute(stmt).scalars().all())
        for run in runs:
            raw = run.raw_response_json if isinstance(run.raw_response_json, dict) else {}
            candidates = [raw]
            maybe_result = raw.get("result") if isinstance(raw, dict) else None
            if isinstance(maybe_result, dict):
                candidates.append(maybe_result)

            for payload in candidates:
                candidate_uuid = str(payload.get("registry_uuid") or "").strip().lower()
                if candidate_uuid != target:
                    continue

                candidate_registry = payload.get("registry")
                if not isinstance(candidate_registry, dict):
                    candidate_registry = payload.get("registry_json")
                if not isinstance(candidate_registry, dict):
                    continue

                try:
                    validated = RegistryRecord(**candidate_registry).model_dump(
                        exclude_none=True,
                        mode="json",
                    )
                except Exception:  # noqa: BLE001
                    continue

                return run.id, validated

        return None

    def _load_events(
        self,
        *,
        db: Session,
        registry_uuid: uuid.UUID,
        user_id: str | None,
    ) -> list[RegistryAppendedDocument]:
        stmt: Select[tuple[RegistryAppendedDocument]] = select(RegistryAppendedDocument).where(
            RegistryAppendedDocument.registry_uuid == registry_uuid
        )
        if user_id:
            stmt = stmt.where(RegistryAppendedDocument.user_id == user_id)

        if self.strategy != "reprocess_all":
            stmt = stmt.where(RegistryAppendedDocument.aggregated_at.is_(None))

        stmt = stmt.order_by(RegistryAppendedDocument.created_at.asc(), RegistryAppendedDocument.id.asc())
        return list(db.execute(stmt).scalars().all())

    def _extract_event(self, row: RegistryAppendedDocument) -> dict[str, Any]:
        event_type = str(row.event_type or "other").strip().lower()
        note_text = _safe_row_text(row)
        metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        structured = metadata.get("structured_data") if isinstance(metadata, dict) else None
        use_llm = _should_attempt_llm_fallback()

        if event_type == "pathology":
            payload = extract_pathology_event(note_text)
            results = extract_pathology_results(note_text)

            results_update: dict[str, Any] = {}
            if isinstance(results.get("pathology_results_update"), dict):
                results_update.update(results["pathology_results_update"])

            # Deterministic biomarker/histology extraction (fills into pathology_results.*).
            try:
                record_out, _warnings = apply_pathology_extraction(RegistryRecord(), note_text)
                record_dump = record_out.model_dump(exclude_none=True, mode="json")
                pr = record_dump.get("pathology_results")
                if isinstance(pr, dict) and pr:
                    results_update.update(pr)
            except Exception:  # noqa: BLE001
                payload.setdefault("qa_flags", []).append("pathology_biomarker_extract_failed")

            payload["pathology_results_update"] = results_update
            payload["qa_flags"] = sorted(set(list(payload.get("qa_flags") or []) + list(results.get("qa_flags") or [])))

            if use_llm:
                update = payload.get("pathology_results_update")
                if not isinstance(update, dict):
                    update = {}
                needs_llm = not bool(str(update.get("final_diagnosis") or "").strip())
                if needs_llm:
                    try:
                        llm_update = _llm_extract_pathology_results(note_text)
                        if isinstance(llm_update, dict) and llm_update:
                            payload["pathology_results_update"] = _merge_fill_missing(update, llm_update)
                            payload.setdefault("qa_flags", []).append("llm_pathology_fallback")
                    except Exception:  # noqa: BLE001
                        payload.setdefault("qa_flags", []).append("llm_pathology_failed")

            payload["event_type"] = event_type
            return payload

        if event_type == "imaging":
            source_modality = str(row.source_modality or "").strip().lower()
            if "pet" in source_modality:
                payload = extract_pet_ct_event(
                    note_text,
                    relative_day_offset=row.relative_day_offset,
                    event_subtype=row.event_subtype,
                )
            else:
                payload = extract_ct_event(
                    note_text,
                    relative_day_offset=row.relative_day_offset,
                    event_subtype=row.event_subtype,
                )

            if use_llm:
                snapshot = payload.get("imaging_snapshot")
                if isinstance(snapshot, dict):
                    response = str(snapshot.get("response") or "").strip().lower()
                    impression = str(snapshot.get("overall_impression_text") or "").strip()
                    if response == "indeterminate" and not impression:
                        try:
                            llm_update = _llm_extract_imaging_snapshot(note_text)
                            if isinstance(llm_update, dict) and llm_update:
                                # Treat "Indeterminate" as missing for response, since we're already in a fallback path.
                                if isinstance(llm_update.get("response"), str) and llm_update["response"].strip():
                                    snapshot["response"] = llm_update["response"].strip()
                                snapshot = _merge_fill_missing(snapshot, llm_update)
                                payload["imaging_snapshot"] = snapshot
                                payload.setdefault("qa_flags", []).append("llm_imaging_fallback")
                        except Exception:  # noqa: BLE001
                            payload.setdefault("qa_flags", []).append("llm_imaging_failed")

            payload["event_type"] = event_type
            payload["source_modality"] = source_modality or "ct"
            return payload

        clinical_type = {
            "clinical_update": "clinical_update",
            "treatment_update": "treatment_update",
            "complication": "complication",
            "procedure_addendum": "other",
            "procedure": "other",
            "other": "other",
        }.get(event_type)

        if clinical_type is not None:
            if isinstance(structured, dict) and structured:
                disease_status = structured.get("disease_status")
                if not isinstance(disease_status, str) or not disease_status.strip():
                    disease_status = None
                mapped = {
                    "clinical_update": {
                        "relative_day_offset": int(row.relative_day_offset or 0),
                        "update_type": clinical_type,
                        "hospital_admission": structured.get("hospital_admission"),
                        "icu_admission": structured.get("icu_admission"),
                        "deceased": structured.get("deceased"),
                        "disease_status": disease_status,
                        "summary_text": compact_text(str(structured.get("comment") or ""), max_chars=220) or None,
                        "qa_flags": [],
                    },
                    "qa_flags": [],
                }
                mapped["event_type"] = event_type
                return mapped

            payload = extract_clinical_update_event(
                note_text,
                update_type=clinical_type,
                relative_day_offset=row.relative_day_offset,
            )

            if use_llm:
                update = payload.get("clinical_update")
                if isinstance(update, dict):
                    has_followup = any(
                        update.get(k) is not None
                        for k in ("hospital_admission", "icu_admission", "deceased", "disease_status")
                    )
                    if not has_followup and (note_text or "").strip():
                        try:
                            llm_update = _llm_extract_clinical_followup(note_text)
                            if isinstance(llm_update, dict) and llm_update:
                                payload["clinical_update"] = _merge_fill_missing(update, llm_update)
                                payload.setdefault("qa_flags", []).append("llm_clinical_followup_fallback")
                        except Exception:  # noqa: BLE001
                            payload.setdefault("qa_flags", []).append("llm_clinical_followup_failed")

            payload["event_type"] = event_type
            return payload

        return {
            "event_type": event_type,
            "qa_flags": ["unsupported_event_type"],
        }

    def _apply_patch(
        self,
        registry_json: dict[str, Any],
        *,
        extracted: dict[str, Any],
        event_row: RegistryAppendedDocument,
        manual_overrides: dict[str, Any] | None,
    ) -> tuple[bool, list[str]]:
        event_type = str(event_row.event_type or "other").strip().lower()
        event_id = str(event_row.id)

        if event_type == "pathology":
            changed1, flags1 = patch_pathology_update(
                registry_json,
                extracted=extracted,
                event_id=event_id,
                relative_day_offset=event_row.relative_day_offset,
                manual_overrides=manual_overrides,
            )
            changed2, flags2 = patch_pathology_results(
                registry_json,
                extracted=extracted,
                manual_overrides=manual_overrides,
            )
            return (changed1 or changed2), sorted(set(flags1 + flags2))

        if event_type == "imaging":
            return patch_imaging_update(
                registry_json,
                extracted=extracted,
                event_id=event_id,
                relative_day_offset=event_row.relative_day_offset,
                event_subtype=event_row.event_subtype,
                event_title=event_row.event_title,
                source_modality=event_row.source_modality,
                manual_overrides=manual_overrides,
            )

        if event_type in {
            "clinical_update",
            "treatment_update",
            "complication",
            "procedure_addendum",
            "procedure",
            "other",
        }:
            return patch_clinical_update(
                registry_json,
                extracted=extracted,
                event_id=event_id,
                event_title=event_row.event_title,
                manual_overrides=manual_overrides,
            )

        return False, list(extracted.get("qa_flags") or ["unsupported_event_type"])


def get_case_aggregator() -> CaseAggregator:
    return CaseAggregator()


__all__ = ["CaseAggregator", "get_case_aggregator"]
