from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from app.common.path_redaction import repo_relative_path, sanitize_path_fields
from app.registry.deterministic_extractors import extract_tbna_conventional
from app.reporting.quality_gate import BLOCKER_CODES
from ml.scripts.generate_reporter_gold_dataset import (
    CRITICAL_FLAG_EXACT,
    CRITICAL_FLAG_PREFIXES,
    collect_performed_flags,
)


REPORTER_SEED_EVAL_SCHEMA_VERSION = "procedure_suite.reporter_seed_eval.v1"
REPORTER_SEED_FALLBACK_SCHEMA_VERSION = "procedure_suite.reporter_seed_fallback_summary.v1"

REQUIRED_SECTION_HEADERS = [
    "INTERVENTIONAL PULMONOLOGY OPERATIVE REPORT",
    "INDICATION FOR OPERATION",
    "CONSENT",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
    "PROCEDURE",
    "ANESTHESIA",
    "MONITORING",
    "COMPLICATIONS",
    "PROCEDURE IN DETAIL",
    "IMPRESSION / PLAN",
]

DROP_REASON_KEYS = (
    "evidence_substring_fail",
    "keyword_fail",
    "anatomy_fail",
    "action_intent_fail",
    "procedure_key_not_allowlisted",
    "other",
)


@dataclass(frozen=True)
class ReporterEvalRow:
    id: str
    prompt_text: str
    completion_canonical: str


@dataclass
class ReporterEvalCaseOutput:
    markdown: str
    warnings: list[str]
    quality_flags: list[dict[str, Any]]
    needs_review: bool
    render_fallback_used: bool
    render_fallback_reason: str | None = None
    render_fallback_category: str | None = None
    render_fallback_details: dict[str, Any] | None = None
    accepted_findings: int = 0
    dropped_findings: int = 0
    drop_reason_counts: dict[str, int] | None = None
    error_code: str | None = None


def datetime_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_rows(path: Path, *, prompt_field: str) -> list[ReporterEvalRow]:
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        payload = load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            rows = [item for item in payload.get("cases") or [] if isinstance(item, dict)]
        elif isinstance(payload, list):
            rows = [item for item in payload if isinstance(item, dict)]
        else:
            raise ValueError(f"Unsupported reporter eval input format: {path}")

    out: list[ReporterEvalRow] = []
    for idx, row in enumerate(rows, start=1):
        prompt = str(row.get(prompt_field) or "").strip()
        if not prompt:
            prompt = str(row.get("prompt_text") or "").strip()
        completion = str(row.get("completion_canonical") or "").strip()
        if not completion:
            completion = str(row.get("ideal_output") or "").strip()
        if not completion and row.get("completion_canonical_path"):
            completion_path = Path(str(row["completion_canonical_path"]))
            if not completion_path.is_absolute():
                completion_path = path.parent / completion_path
            completion = completion_path.read_text(encoding="utf-8").strip()
        row_id = str(row.get("id") or f"row_{idx}")
        if not prompt or not completion:
            continue
        out.append(
            ReporterEvalRow(
                id=row_id,
                prompt_text=prompt,
                completion_canonical=completion,
            )
        )
    return out


def maybe_subsample(rows: list[ReporterEvalRow], max_cases: int, seed: int) -> list[ReporterEvalRow]:
    if max_cases <= 0 or len(rows) <= max_cases:
        return rows
    import random

    rng = random.Random(seed)
    picked = rng.sample(rows, max_cases)
    picked.sort(key=lambda item: item.id)
    return picked


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines)


def similarity_ratio(reference: str, candidate: str) -> float:
    return SequenceMatcher(None, normalize_text(reference), normalize_text(candidate)).ratio()


def missing_sections(report_text: str, *, required_headers: list[str] | None = None) -> list[str]:
    upper = (report_text or "").upper()
    headers = required_headers or REQUIRED_SECTION_HEADERS
    return [header for header in headers if header.upper() not in upper]


def _is_critical_flag(path: str) -> bool:
    return path in CRITICAL_FLAG_EXACT or path.startswith(CRITICAL_FLAG_PREFIXES)


def extract_flags_and_cpt(note_text: str, registry_service: Any) -> tuple[set[str], set[str]]:
    result = registry_service.extract_fields_extraction_first(note_text)
    record_data = result.record.model_dump(exclude_none=True)
    flags = collect_performed_flags(record_data)
    peripheral_tbna_flag = "procedures_performed.peripheral_tbna.performed"
    if peripheral_tbna_flag in flags:
        deterministic_tbna = extract_tbna_conventional(note_text)
        deterministic_has_peripheral = bool(
            (deterministic_tbna.get("peripheral_tbna") or {}).get("performed") is True
        )
        if not deterministic_has_peripheral:
            flags.discard(peripheral_tbna_flag)
    cpt = {str(code) for code in (result.cpt_codes or [])}
    return flags, cpt


def cpt_jaccard(gold: set[str], pred: set[str]) -> float:
    union = gold | pred
    if not union:
        return 1.0
    return float(len(gold & pred) / len(union))


def flag_f1(gold: set[str], pred: set[str]) -> tuple[float, int, int]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    if prec + rec == 0:
        return 0.0, fp, fn
    return (2 * prec * rec / (prec + rec)), fp, fn


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _extract_drop_reason_code(warning: str) -> str | None:
    text = str(warning or "").strip()
    if not text.startswith("LLM_FINDINGS_DROPPED:"):
        return None
    remainder = text.split(":", 1)[1].strip()
    if not remainder:
        return "other"
    return remainder.split()[0].strip() or "other"


def _categorize_drop_reason(code: str) -> str:
    reason = str(code or "").strip()
    if reason in {"missing_evidence_quote", "evidence_too_short"}:
        return "evidence_substring_fail"
    if reason == "keyword_missing":
        return "keyword_fail"
    if reason == "anatomy_not_in_evidence":
        return "anatomy_fail"
    if reason == "missing_action_intent":
        return "action_intent_fail"
    if reason == "invalid_procedure_key":
        return "procedure_key_not_allowlisted"
    return "other"


def drop_reason_counts(warnings: list[str] | None) -> dict[str, int]:
    counts = {key: 0 for key in DROP_REASON_KEYS}
    for warning in warnings or []:
        reason_code = _extract_drop_reason_code(warning)
        if not reason_code:
            continue
        bucket = _categorize_drop_reason(reason_code)
        counts[bucket] = int(counts.get(bucket, 0)) + 1
    return counts


def evaluate_seed_path(
    *,
    rows: list[ReporterEvalRow],
    seed_path: str,
    run_case: Callable[[ReporterEvalRow], ReporterEvalCaseOutput],
    registry_service: Any,
    input_path: str,
    output_path: str | None,
    prompt_field: str,
    environment_defaults_applied: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    required_headers: list[str] | None = None,
    forbidden_artifact_tokens: tuple[str, ...] = ("```",),
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    sim_scores: list[float] = []
    cpt_scores: list[float] = []
    f1_scores: list[float] = []
    seed_warning_counts: list[float] = []
    accepted_counts: list[float] = []
    dropped_counts: list[float] = []
    aggregate_drop_reasons = {key: 0 for key in DROP_REASON_KEYS}
    aggregate_fallback_reasons: dict[str, int] = {}
    full_shell_count = 0
    failures = 0
    critical_extra_cases = 0
    forbidden_artifact_cases = 0
    strict_fallback_cases = 0
    blocker_cases = 0
    complication_contradiction_cases = 0
    unsupported_invasive_add_cases = 0
    anatomy_mismatch_cases = 0
    generic_pathology_plan_cases = 0

    for row in rows:
        try:
            case_out = run_case(row)
            markdown = str(case_out.markdown or "")
            if not markdown.strip():
                raise RuntimeError("Empty markdown output")

            sim = similarity_ratio(row.completion_canonical, markdown)
            sim_scores.append(sim)

            miss = missing_sections(markdown, required_headers=required_headers)
            if not miss:
                full_shell_count += 1

            gold_flags, gold_cpt = extract_flags_and_cpt(row.completion_canonical, registry_service)
            pred_flags, pred_cpt = extract_flags_and_cpt(markdown, registry_service)

            cpt_score = cpt_jaccard(gold_cpt, pred_cpt)
            f1, fp, fn = flag_f1(gold_flags, pred_flags)
            critical_extra = sorted(flag for flag in (pred_flags - gold_flags) if _is_critical_flag(flag))
            critical_predicted = sorted(flag for flag in pred_flags if _is_critical_flag(flag))
            if critical_extra:
                critical_extra_cases += 1

            forbidden_hits = [token for token in forbidden_artifact_tokens if token and token in markdown]
            if forbidden_hits:
                forbidden_artifact_cases += 1

            if case_out.render_fallback_used:
                strict_fallback_cases += 1
                fallback_reason = str(case_out.render_fallback_reason or "unknown")
                aggregate_fallback_reasons[fallback_reason] = int(aggregate_fallback_reasons.get(fallback_reason, 0)) + 1

            cpt_scores.append(cpt_score)
            f1_scores.append(f1)
            seed_warning_counts.append(float(len(case_out.warnings or [])))
            accepted_counts.append(float(case_out.accepted_findings))
            dropped_counts.append(float(case_out.dropped_findings))

            reason_counts = dict(case_out.drop_reason_counts or {})
            for key in DROP_REASON_KEYS:
                aggregate_drop_reasons[key] = int(aggregate_drop_reasons.get(key, 0)) + int(reason_counts.get(key, 0))

            quality_flag_codes = []
            has_blocker = False
            for item in case_out.quality_flags or []:
                if isinstance(item, dict) and item.get("code"):
                    code = str(item["code"])
                    quality_flag_codes.append(code)
                    if code in BLOCKER_CODES:
                        has_blocker = True
            if has_blocker:
                blocker_cases += 1
            if "COMPLICATIONS_NONE_CONTRADICTION" in quality_flag_codes:
                complication_contradiction_cases += 1
            if "UNSUPPORTED_INVASIVE_PROCEDURE_ADDED" in quality_flag_codes:
                unsupported_invasive_add_cases += 1
            if "ANATOMY_SITE_DRIFT" in quality_flag_codes:
                anatomy_mismatch_cases += 1
            if "SPECIMEN_PLAN_MISMATCH" in quality_flag_codes:
                generic_pathology_plan_cases += 1

            per_case.append(
                {
                    "id": row.id,
                    "seed_path": seed_path,
                    "text_similarity": round(sim, 4),
                    "missing_sections": miss,
                    "cpt_jaccard": round(cpt_score, 4),
                    "performed_flag_f1": round(f1, 4),
                    "critical_extra_flags": critical_extra,
                    "critical_predicted_flags": critical_predicted,
                    "predicted_cpt_codes": sorted(pred_cpt),
                    "flag_false_positive_count": fp,
                    "flag_false_negative_count": fn,
                    "forbidden_artifact_hits": forbidden_hits,
                    "render_fallback_used": bool(case_out.render_fallback_used),
                    "render_fallback_reason": case_out.render_fallback_reason,
                    "render_fallback_category": case_out.render_fallback_category,
                    "render_fallback_details": dict(case_out.render_fallback_details or {}),
                    "seed_warning_count": int(len(case_out.warnings or [])),
                    "quality_flag_codes": sorted(set(quality_flag_codes)),
                    "needs_review": bool(case_out.needs_review),
                    "accepted_findings": int(case_out.accepted_findings),
                    "dropped_findings": int(case_out.dropped_findings),
                    "drop_reason_counts": {key: int(reason_counts.get(key, 0)) for key in DROP_REASON_KEYS},
                    "error": None,
                    "error_code": case_out.error_code,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            per_case.append(
                {
                    "id": row.id,
                    "seed_path": seed_path,
                    "text_similarity": 0.0,
                    "missing_sections": required_headers or REQUIRED_SECTION_HEADERS,
                    "cpt_jaccard": 0.0,
                    "performed_flag_f1": 0.0,
                    "critical_extra_flags": [],
                    "critical_predicted_flags": [],
                    "predicted_cpt_codes": [],
                    "flag_false_positive_count": 0,
                    "flag_false_negative_count": 0,
                    "forbidden_artifact_hits": [],
                    "render_fallback_used": False,
                    "render_fallback_reason": None,
                    "render_fallback_category": None,
                    "render_fallback_details": {},
                    "seed_warning_count": 0,
                    "quality_flag_codes": [],
                    "needs_review": False,
                    "accepted_findings": 0,
                    "dropped_findings": 0,
                    "drop_reason_counts": {key: 0 for key in DROP_REASON_KEYS},
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_code": None,
                }
            )

    total = len(rows)
    successful = total - failures

    summary = {
        "total_cases": total,
        "successful_cases": successful,
        "failed_cases": failures,
        "avg_text_similarity": round(_avg(sim_scores), 4),
        "required_section_coverage": round((full_shell_count / successful) if successful else 0.0, 4),
        "avg_cpt_jaccard": round(_avg(cpt_scores), 4),
        "avg_performed_flag_f1": round(_avg(f1_scores), 4),
        "critical_extra_flag_rate": round((critical_extra_cases / successful) if successful else 0.0, 4),
        "strict_render_fallback_rate": round((strict_fallback_cases / successful) if successful else 0.0, 4),
        "forbidden_artifact_case_rate": round((forbidden_artifact_cases / successful) if successful else 0.0, 4),
        "blocker_rate": round((blocker_cases / successful) if successful else 0.0, 4),
        "complication_contradiction_rate": round((complication_contradiction_cases / successful) if successful else 0.0, 4),
        "unsupported_procedure_add_rate": round((unsupported_invasive_add_cases / successful) if successful else 0.0, 4),
        "anatomy_mismatch_rate": round((anatomy_mismatch_cases / successful) if successful else 0.0, 4),
        "generic_pathology_plan_rate": round((generic_pathology_plan_cases / successful) if successful else 0.0, 4),
        "avg_seed_warning_count": round(_avg(seed_warning_counts), 4),
        "avg_accepted_findings": round(_avg(accepted_counts), 4),
        "avg_dropped_findings": round(_avg(dropped_counts), 4),
        "drop_reason_counts": aggregate_drop_reasons,
        "fallback_reason_counts": {key: int(aggregate_fallback_reasons[key]) for key in sorted(aggregate_fallback_reasons)},
    }

    return {
        "schema_version": REPORTER_SEED_EVAL_SCHEMA_VERSION,
        "kind": "reporter_seed_eval",
        "seed_path": seed_path,
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "prompt_field": prompt_field,
        "row_count": total,
        "created_at": datetime_now_iso(),
        "environment_defaults_applied": environment_defaults_applied or {},
        "metadata": metadata or {},
        "summary": summary,
        "per_case": per_case,
        "failures": [item for item in per_case if item.get("error")],
    }


def load_seed_fixture(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Reporter seed fixture must be an object with cases[]: {path}")
    out: dict[str, dict[str, Any]] = {}
    for case in payload.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            continue
        out[case_id] = case
    return out


def _fallback_rows_by_reason(report: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in report.get("per_case") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("render_fallback_used"):
            continue
        case_id = str(item.get("id") or "").strip()
        if not case_id:
            continue
        reason = str(item.get("render_fallback_reason") or "unknown")
        grouped.setdefault(reason, []).append(case_id)
    for reason in list(grouped):
        grouped[reason] = sorted(set(grouped[reason]))
    return grouped


def build_seed_path_fallback_reason_report(
    *,
    left_report: dict[str, Any],
    right_report: dict[str, Any],
    left_path: str | None,
    right_path: str | None,
) -> dict[str, Any]:
    left_grouped = _fallback_rows_by_reason(left_report)
    right_grouped = _fallback_rows_by_reason(right_report)
    reasons: list[dict[str, Any]] = []

    for reason in sorted(set(left_grouped) | set(right_grouped)):
        left_cases = list(left_grouped.get(reason, []))
        right_cases = list(right_grouped.get(reason, []))
        reasons.append(
            {
                "reason": reason,
                "left_count": len(left_cases),
                "right_count": len(right_cases),
                "delta": len(right_cases) - len(left_cases),
                "left_cases": left_cases,
                "right_cases": right_cases,
            }
        )

    return {
        "schema_version": REPORTER_SEED_FALLBACK_SCHEMA_VERSION,
        "kind": "reporter_seed_fallback_summary",
        "created_at": datetime_now_iso(),
        "left_seed_path": left_report.get("seed_path"),
        "right_seed_path": right_report.get("seed_path"),
        "left_report_path": repo_relative_path(left_path),
        "right_report_path": repo_relative_path(right_path),
        "counts": {
            "total_cases": max(
                len(left_report.get("per_case") or []),
                len(right_report.get("per_case") or []),
            ),
            "left_fallback_cases": sum(len(cases) for cases in left_grouped.values()),
            "right_fallback_cases": sum(len(cases) for cases in right_grouped.values()),
            "reason_bucket_count": len(reasons),
        },
        "reasons": reasons,
    }


def maybe_write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_path_fields(payload)
    path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
