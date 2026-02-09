from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from app.reporting.metadata import MissingFieldIssue
from app.reporting.util.path_access import get_path

if TYPE_CHECKING:
    from app.reporting.engine import SchemaRegistry, TemplateMeta, TemplateRegistry
    from proc_schemas.clinical.common import ProcedureBundle, ProcedureInput


class WarnIfConfig(BaseModel):
    op: Literal["lt", "lte", "eq", "gt", "gte"]
    value: Any
    message: str | None = None


class ConsistencyCheckConfig(BaseModel):
    target: str
    message: str


class FieldConfig(BaseModel):
    path: str
    required: bool = False
    critical: bool = False
    warn_if: WarnIfConfig | None = None
    consistency_check: ConsistencyCheckConfig | None = None

    @classmethod
    def from_template(cls, path: str, data: dict[str, Any]) -> "FieldConfig":
        payload = dict(data or {})
        validation_block = payload.pop("validation", {}) or {}
        if "warn_if" in validation_block and "warn_if" not in payload:
            payload["warn_if"] = validation_block.get("warn_if")
        if "consistency_check" in validation_block and "consistency_check" not in payload:
            payload["consistency_check"] = validation_block.get("consistency_check")
        payload["path"] = path
        return cls.model_validate(payload)


def _coerce_model_data(proc: "ProcedureInput", schema_registry: "SchemaRegistry") -> dict[str, Any]:
    try:
        model_cls = schema_registry.get(proc.schema_id)
    except KeyError:
        return _normalize_payload(proc.data)
    try:
        model = proc.data if isinstance(proc.data, BaseModel) else model_cls.model_validate(proc.data or {})
        return model.model_dump(exclude_none=False)
    except Exception:
        return _normalize_payload(proc.data)


def _normalize_payload(payload: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, BaseModel):
        return payload.model_dump(exclude_none=False)
    if isinstance(payload, dict):
        return payload
    try:
        return dict(payload)  # type: ignore[arg-type]
    except Exception:
        return {}


def _expand_list_paths(payload: Any, field_path: str) -> list[str]:
    paths = [field_path]
    while any("[]" in p for p in paths):
        expanded: list[str] = []
        for path in paths:
            if "[]" not in path:
                expanded.append(path)
                continue
            head, tail = path.split("[]", 1)
            key = head.rstrip(".")
            remainder = tail.lstrip(".")
            value = get_path(payload, key)
            indices = range(len(value)) if isinstance(value, list) and value else range(1)
            suffix = f".{remainder}" if remainder else ""
            for idx in indices:
                expanded.append(f"{key}[{idx}]{suffix}")
        paths = expanded
    return paths


def _compare(value: Any, op: str, target: Any) -> bool:
    try:
        if op == "lt":
            return value < target
        if op == "lte":
            return value <= target
        if op == "eq":
            return value == target
        if op == "gt":
            return value > target
        if op == "gte":
            return value >= target
    except Exception:
        return False
    return False


class ValidationEngine:
    def __init__(self, template_registry: "TemplateRegistry", schema_registry: "SchemaRegistry") -> None:
        self.templates = template_registry
        self.schemas = schema_registry

    def _field_configs(self, meta: "TemplateMeta") -> dict[str, FieldConfig]:
        if getattr(meta, "field_configs", None):
            return meta.field_configs  # type: ignore[return-value]
        configs: dict[str, FieldConfig] = {}
        for path in getattr(meta, "required_fields", []) or []:
            configs[path] = FieldConfig(path=path, required=True, critical=True)
        for path in getattr(meta, "critical_fields", []):
            configs.setdefault(path, FieldConfig(path=path, required=True, critical=True))
        for path in getattr(meta, "recommended_fields", []):
            configs.setdefault(path, FieldConfig(path=path, required=True, critical=False))
        return configs

    def _collect_missing_and_suggestions(self, bundle: "ProcedureBundle") -> tuple[list[MissingFieldIssue], list[str]]:
        issues: list[MissingFieldIssue] = []
        suggestions: list[str] = []
        acknowledged = {k: set(v) for k, v in (bundle.acknowledged_omissions or {}).items()}

        for proc in bundle.procedures:
            metas = self.templates.find_for_procedure(proc.proc_type, proc.cpt_candidates)
            if not metas:
                continue
            payload = _coerce_model_data(proc, self.schemas)
            proc_id = proc.proc_id or proc.schema_id
            acknowledged_fields = acknowledged.get(proc_id, set())

            for meta in metas:
                field_configs = self._field_configs(meta)
                # Required / critical -> warning issues
                for path, config in field_configs.items():
                    if config.required or config.critical:
                        for expanded_path in _expand_list_paths(payload, path):
                            if expanded_path in acknowledged_fields:
                                continue
                            value = get_path(payload, expanded_path)
                            if value in (None, "", [], {}):
                                message = f"Add {expanded_path} for {meta.label or proc.proc_type}"
                                issues.append(
                                    MissingFieldIssue(
                                        proc_id=proc_id,
                                        proc_type=proc.proc_type,
                                        template_id=meta.id,
                                        field_path=expanded_path,
                                        severity="warning",
                                        message=message,
                                    )
                                )
                                suggestions.append(message)

                # Optional fields -> suggestions only
                optional_paths = set(getattr(meta, "optional_fields", []) or [])
                for path in field_configs:
                    if path in optional_paths:
                        continue
                    if not field_configs[path].required and not field_configs[path].critical:
                        optional_paths.add(path)
                for path in optional_paths:
                    for expanded_path in _expand_list_paths(payload, path):
                        if expanded_path in acknowledged_fields:
                            continue
                        value = get_path(payload, expanded_path)
                        if value in (None, "", [], {}):
                            suggestions.append(f"Consider adding {expanded_path} for {meta.label or proc.proc_type}")

        # Deduplicate suggestions
        deduped_suggestions: list[str] = []
        seen = set()
        for s in suggestions:
            if s not in seen:
                deduped_suggestions.append(s)
                seen.add(s)
        return issues, deduped_suggestions

    def list_missing_critical_fields(self, bundle: "ProcedureBundle") -> list[MissingFieldIssue]:
        issues, _ = self._collect_missing_and_suggestions(bundle)
        return issues

    def list_suggestions(self, bundle: "ProcedureBundle") -> list[str]:
        _, suggestions = self._collect_missing_and_suggestions(bundle)
        return suggestions

    def apply_warn_if_rules(self, bundle: "ProcedureBundle") -> list[str]:
        warnings: list[str] = []
        for proc in bundle.procedures:
            metas = self.templates.find_for_procedure(proc.proc_type, proc.cpt_candidates)
            if not metas:
                continue
            payload = _coerce_model_data(proc, self.schemas)
            for meta in metas:
                field_configs = self._field_configs(meta)
                for path, config in field_configs.items():
                    expanded_paths = _expand_list_paths(payload, path)
                    for expanded_path in expanded_paths:
                        value = get_path(payload, expanded_path)
                        if config.warn_if and value not in (None, "", [], {}):
                            if _compare(value, config.warn_if.op, config.warn_if.value):
                                message = config.warn_if.message or f"{meta.label or proc.proc_type}: {expanded_path} triggered warn_if"
                                if message not in warnings:
                                    warnings.append(message)
                        if config.consistency_check:
                            target_val = get_path(payload, config.consistency_check.target)
                            if target_val not in (None, "", [], {}) and value in (None, "", [], {}):
                                msg = config.consistency_check.message
                                if msg not in warnings:
                                    warnings.append(msg)
        return warnings


__all__ = [
    "FieldConfig",
    "WarnIfConfig",
    "ConsistencyCheckConfig",
    "ValidationEngine",
    "MissingFieldIssue",
]
