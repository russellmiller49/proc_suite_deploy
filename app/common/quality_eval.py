from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.common.path_redaction import repo_relative_path, sanitize_path_fields


QUALITY_EVAL_SCHEMA_VERSION = "procedure_suite.quality_eval.v1"


def datetime_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_code(code: str) -> str:
    raw = (code or "").strip()
    if not raw:
        return ""
    return raw.lstrip("+").strip()


def get_path(obj: Any, path: str) -> Any:
    current = obj
    for part in (path or "").split("."):
        if not part:
            continue
        key = part
        index: int | None = None
        if "[" in part and part.endswith("]"):
            key, _, remainder = part.partition("[")
            try:
                index = int(remainder[:-1])
            except ValueError:
                return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
        if index is not None:
            if not isinstance(current, list) or index >= len(current):
                return None
            current = current[index]
    return current


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_unified_quality_corpus(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unified quality corpus must be a JSON object: {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Unified quality corpus missing cases[]: {path}")
    return payload


def detect_input_format(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "reporter_gold_jsonl"
    if path.is_dir():
        return "legacy_golden_dir"
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return "unified_quality_corpus"
    if isinstance(payload, list):
        return "legacy_golden_file"
    if isinstance(payload, dict):
        for key in ("entries", "records", "data"):
            if isinstance(payload.get(key), list):
                return "legacy_golden_file"
    raise ValueError(f"Unable to detect supported evaluation input format for {path}")


def evaluate_extraction_expectations(
    *,
    case: dict[str, Any],
    record_dict: dict[str, Any],
    predicted_codes: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    expectations = case.get("extraction_expectations") or {}
    case_id = str(case.get("id") or "unknown_case")
    tags = [str(tag) for tag in (case.get("tags") or []) if str(tag)]
    normalized_codes = sorted({code for code in (normalize_code(item) for item in predicted_codes) if code})
    warning_text = [str(item) for item in warnings if str(item)]
    failures: list[dict[str, Any]] = []

    for code in expectations.get("must_have_codes") or []:
        normalized = normalize_code(str(code))
        if normalized and normalized not in normalized_codes:
            failures.append(
                {
                    "type": "missing_code",
                    "message": f"missing required code {normalized}",
                    "expected": normalized,
                    "actual": normalized_codes,
                }
            )

    for code in expectations.get("must_not_have_codes") or []:
        normalized = normalize_code(str(code))
        if normalized and normalized in normalized_codes:
            failures.append(
                {
                    "type": "unexpected_code",
                    "message": f"found forbidden code {normalized}",
                    "expected": None,
                    "actual": normalized,
                }
            )

    checked_fields: dict[str, Any] = {}
    for path, expected in (expectations.get("must_have_fields") or {}).items():
        actual = get_path(record_dict, str(path))
        checked_fields[str(path)] = actual
        if actual != expected:
            failures.append(
                {
                    "type": "field_mismatch",
                    "message": f"{path} expected {expected!r} but got {actual!r}",
                    "field": str(path),
                    "expected": expected,
                    "actual": actual,
                }
            )

    for path, forbidden in (expectations.get("must_not_have_fields") or {}).items():
        actual = get_path(record_dict, str(path))
        checked_fields[str(path)] = actual
        if actual == forbidden:
            failures.append(
                {
                    "type": "forbidden_field_value",
                    "message": f"{path} unexpectedly matched forbidden value {forbidden!r}",
                    "field": str(path),
                    "expected": f"!= {forbidden!r}",
                    "actual": actual,
                }
            )

    for snippet in expectations.get("must_have_warnings_substrings") or []:
        needle = str(snippet)
        if needle and not any(needle.lower() in warning.lower() for warning in warning_text):
            failures.append(
                {
                    "type": "missing_warning_substring",
                    "message": f"missing warning substring {needle!r}",
                    "expected": needle,
                    "actual": warning_text,
                }
            )

    for snippet in expectations.get("must_not_have_warnings_substrings") or []:
        needle = str(snippet)
        if needle and any(needle.lower() in warning.lower() for warning in warning_text):
            failures.append(
                {
                    "type": "unexpected_warning_substring",
                    "message": f"found forbidden warning substring {needle!r}",
                    "expected": None,
                    "actual": warning_text,
                }
            )

    return {
        "id": case_id,
        "tags": tags,
        "status": "failed" if failures else "passed",
        "metrics": {
            "code_count": len(normalized_codes),
            "warning_count": len(warning_text),
        },
        "actual": {
            "predicted_codes": normalized_codes,
            "checked_fields": checked_fields,
            "warnings": warning_text,
        },
        "failures": failures,
    }


def build_standard_report(
    *,
    kind: str,
    input_path: str,
    output_path: str | None,
    source_format: str,
    corpus_name: str,
    per_case: list[dict[str, Any]],
    summary_metrics: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sorted_cases = sorted(per_case, key=lambda item: str(item.get("id") or ""))
    failures: list[dict[str, Any]] = []
    for case in sorted_cases:
        for failure in case.get("failures") or []:
            failure_entry = {
                "id": case.get("id"),
                "tags": case.get("tags") or [],
            }
            failure_entry.update(failure)
            failures.append(failure_entry)

    total_cases = len(sorted_cases)
    passed_cases = sum(1 for case in sorted_cases if case.get("status") == "passed")
    failed_cases = total_cases - passed_cases
    pass_rate = round((passed_cases / total_cases), 4) if total_cases else 0.0

    summary = {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": pass_rate,
        "metrics": summary_metrics or {},
    }

    if kind == "reporter":
        metrics = summary["metrics"]
        summary["successful_cases"] = metrics.get("successful_cases", passed_cases)
        summary["avg_similarity"] = metrics.get("avg_similarity", 0.0)
        summary["min_similarity"] = metrics.get("min_similarity", 0.0)
        summary["generated_full_shell_rate"] = metrics.get("generated_full_shell_rate", 0.0)
    elif kind == "extraction":
        metrics = summary["metrics"]
        summary["exact_code_match_cases"] = metrics.get("exact_code_match_cases", passed_cases)
        summary["exact_code_match_rate"] = metrics.get("exact_code_match_rate", pass_rate)

    return {
        "schema_version": QUALITY_EVAL_SCHEMA_VERSION,
        "kind": kind,
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "source_format": source_format,
        "corpus_name": corpus_name,
        "created_at": datetime_now_iso(),
        "runtime": runtime or {},
        "summary": summary,
        "per_case": sorted_cases,
        "failures": failures,
    }


def configure_offline_quality_eval_env() -> None:
    env_overrides = {
        "PROCSUITE_SKIP_DOTENV": "1",
        "PROCSUITE_SKIP_WARMUP": "1",
        "ENABLE_UMLS_LINKER": "false",
        "PROCSUITE_PIPELINE_MODE": "extraction_first",
        "REGISTRY_EXTRACTION_ENGINE": "parallel_ner",
        "REGISTRY_SCHEMA_VERSION": "v3",
        "REGISTRY_AUDITOR_SOURCE": "raw_ml",
        "REGISTRY_USE_STUB_LLM": "1",
        "REGISTRY_SELF_CORRECT_ENABLED": "0",
        "REPORTER_DISABLE_LLM": "1",
        "OPENAI_OFFLINE": "1",
        "GEMINI_OFFLINE": "1",
        "QA_REPORTER_ALLOW_SIMPLE_FALLBACK": "0",
    }
    for key, value in env_overrides.items():
        os.environ[key] = value


def maybe_write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_path_fields(report)
    path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def iter_legacy_golden_entries(path: Path, pattern: str) -> list[tuple[str, dict[str, Any]]]:
    paths: list[Path] = []
    if path.is_file():
        paths = [path]
    elif path.exists() and path.is_dir():
        paths = sorted(path.glob(pattern))
    output: list[tuple[str, dict[str, Any]]] = []
    for item in paths:
        payload = load_json(item)
        entries: list[dict[str, Any]]
        if isinstance(payload, list):
            entries = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            entries = []
            for key in ("entries", "records", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    entries = [row for row in value if isinstance(row, dict)]
                    break
            else:
                raise ValueError(f"Unrecognized fixture JSON shape in {item}")
        else:
            raise ValueError(f"Unrecognized fixture JSON shape in {item}")
        for entry in entries:
            output.append((str(item), entry))
    return output
