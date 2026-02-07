from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from .logging_utils import IssueLogger

NULL_LIKE = {"", "n/a", "na", "none", "unknown", "unspecified"}
BOOL_MAP = {
    "yes": True,
    "true": True,
    "1": True,
    "no": False,
    "false": False,
    "0": False,
}

SAFE_DEFAULTS = {"data_entry_status": "Incomplete"}

ENUM_ALIASES = {
    "gender": {
        "male": "M",
        "m": "M",
        "female": "F",
        "f": "F",
        "woman": "F",
        "man": "M",
        "unknown": "Other",
    },
    "sedation_type": {
        "conscious sedation": "Moderate",
        "moderate sedation": "Moderate",
        "minimal": "Moderate",
        "mac": "Monitored Anesthesia Care",
        "monitored anesthesia care": "Monitored Anesthesia Care",
        "general anesthesia": "General",
        "ga": "General",
        "local": "Local",
    },
    "airway_type": {
        "ett": "ETT",
        "endotracheal": "ETT",
        "native": "Native",
        "trach": "Tracheostomy",
        "tracheostomy": "Tracheostomy",
    },
    "procedure_setting": {
        "or": "OR",
        "operating room": "OR",
        "bronchoscopy suite": "Bronchoscopy Suite",
        "pleural suite": "Pleural Suite",
    },
}


class SchemaNormalizer:
    """Validate and normalize registry entries using the JSON schema."""

    def __init__(self, schema_path: str | Path) -> None:
        path = Path(schema_path)
        if not path.exists():
            raise FileNotFoundError(f"Registry schema not found at {path}")
        self.schema = json.loads(path.read_text())
        self.validator = Draft7Validator(self.schema)
        self.properties: dict[str, dict[str, Any]] = self.schema.get("properties", {})
        self.required_fields = set(self.schema.get("required", []))
        self.allowed_fields = set(self.properties.keys())
        self.enum_values: dict[str, set[str]] = {}
        self.nullable_fields: dict[str, bool] = {}
        for name, prop in self.properties.items():
            enum_values = prop.get("enum")
            if enum_values:
                self.enum_values[name] = set(enum_values)
            self.nullable_fields[name] = _allows_null(prop)

    def normalize_entry(self, raw_entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> dict[str, Any]:
        entry = copy.deepcopy(raw_entry)
        self._ensure_evidence(entry, entry_id, logger)
        self._remove_unknown_fields(entry, entry_id, logger)
        self._coerce_types(entry, entry_id, logger)
        self._canonicalize_enums(entry, entry_id, logger)
        self._apply_required_defaults(entry, entry_id, logger)
        self.validate_entry(entry, entry_id, logger)
        return entry

    def validate_entry(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> bool:
        errors = list(self.validator.iter_errors(entry))
        for error in errors:
            path = ".".join(str(part) for part in error.path) or "root"
            logger.log(
                entry_id=entry_id,
                issue_type="schema_validation_failure",
                severity="error",
                action="flagged_for_manual",
                field=path,
                details={"message": error.message},
            )
        return not errors

    def _ensure_evidence(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        evidence = entry.get("evidence")
        if evidence is None:
            entry["evidence"] = {}
            return
        if not isinstance(evidence, dict):
            normalized = {"original_evidence": evidence}
            entry["evidence"] = normalized
            logger.log(
                entry_id=entry_id,
                issue_type="evidence_normalized",
                severity="info",
                action="auto_fixed",
                field="evidence",
                details={"old": evidence, "new": normalized},
            )

    def _remove_unknown_fields(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        for key in list(entry.keys()):
            if key in self.allowed_fields or key == "evidence":
                continue
            value = entry.pop(key)
            evidence = entry.setdefault("evidence", {})
            if isinstance(evidence, dict):
                evidence[key] = value
            logger.log(
                entry_id=entry_id,
                issue_type="unknown_field_removed",
                severity="info",
                action="auto_fixed",
                field=key,
                details={"old": value, "new": None},
            )

    def _coerce_types(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        for field in list(entry.keys()):
            if field == "evidence":
                continue
            prop = self.properties.get(field)
            if not prop:
                continue
            value = entry[field]
            if value is None:
                continue
            allowed_types = _allowed_types(prop)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.lower() in NULL_LIKE and self.nullable_fields.get(field, False):
                    entry[field] = None
                    logger.log(
                        entry_id=entry_id,
                        issue_type="null_like_value_cleared",
                        severity="info",
                        action="auto_fixed",
                        field=field,
                        details={"old": value, "new": None},
                    )
                    continue
            if "boolean" in allowed_types and isinstance(value, str):
                normalized = BOOL_MAP.get(value.strip().lower())
                if normalized is not None:
                    entry[field] = normalized
                    logger.log(
                        entry_id=entry_id,
                        issue_type="boolean_coerced",
                        severity="info",
                        action="auto_fixed",
                        field=field,
                        details={"old": value, "new": normalized},
                    )
                    continue
            if "integer" in allowed_types:
                coerced = _coerce_int(value)
                if coerced is not None:
                    if coerced != value:
                        entry[field] = coerced
                        logger.log(
                            entry_id=entry_id,
                            issue_type="integer_coerced",
                            severity="info",
                            action="auto_fixed",
                            field=field,
                            details={"old": value, "new": coerced},
                        )
                    continue
            if "number" in allowed_types:
                coerced = _coerce_float(value)
                if coerced is not None:
                    if coerced != value:
                        entry[field] = coerced
                        logger.log(
                            entry_id=entry_id,
                            issue_type="number_coerced",
                            severity="info",
                            action="auto_fixed",
                            field=field,
                            details={"old": value, "new": coerced},
                        )
                    continue
            if "array" in allowed_types and not isinstance(value, list):
                entry[field] = [value]
                logger.log(
                    entry_id=entry_id,
                    issue_type="array_wrapped",
                    severity="info",
                    action="auto_fixed",
                    field=field,
                    details={"old": value, "new": [value]},
                )

    def _canonicalize_enums(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        for field, allowed in self.enum_values.items():
            value = entry.get(field)
            if value is None:
                continue
            if isinstance(value, str) and value in allowed:
                continue
            normalized = self._normalize_enum(field, value)
            if normalized:
                entry[field] = normalized
                if normalized != value:
                    logger.log(
                        entry_id=entry_id,
                        issue_type="enum_normalized",
                        severity="info",
                        action="auto_fixed",
                        field=field,
                        details={"old": value, "new": normalized},
                    )
                continue
            if self.nullable_fields.get(field, False):
                entry[field] = None
                logger.log(
                    entry_id=entry_id,
                    issue_type="enum_invalid",
                    severity="warn",
                    action="flagged_for_manual",
                    field=field,
                    details={"old": value, "new": None},
                )
            else:
                logger.log(
                    entry_id=entry_id,
                    issue_type="enum_invalid",
                    severity="error",
                    action="flagged_for_manual",
                    field=field,
                    details={"old": value, "new": None},
                )

    def _apply_required_defaults(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        for field in self.required_fields:
            if entry.get(field) is not None:
                continue
            default = SAFE_DEFAULTS.get(field)
            if default is not None:
                entry[field] = default
                logger.log(
                    entry_id=entry_id,
                    issue_type="required_default_applied",
                    severity="info",
                    action="auto_fixed",
                    field=field,
                    details={"old": None, "new": default},
                )
            else:
                logger.log(
                    entry_id=entry_id,
                    issue_type="schema_missing_required",
                    severity="error",
                    action="flagged_for_manual",
                    field=field,
                    details={"message": "Required field missing"},
                )

    def _normalize_enum(self, field: str, value: Any) -> str | None:
        allowed = self.enum_values.get(field)
        if not allowed:
            return None
        text = str(value).strip()
        if text in allowed:
            return text
        # Filter out None values when building lowercase lookup
        lower_lookup = {option.lower(): option for option in allowed if option is not None}
        lower = text.lower()
        if lower in lower_lookup:
            return lower_lookup[lower]
        alias_map = ENUM_ALIASES.get(field, {})
        mapped = alias_map.get(lower)
        if mapped and mapped in allowed:
            return mapped
        return None


def _allowed_types(prop: dict[str, Any]) -> set[str]:
    typ = prop.get("type")
    if isinstance(typ, list):
        return set(typ)
    if typ:
        return {typ}
    if prop.get("enum"):
        return {"string"}
    return set()


def _allows_null(prop: dict[str, Any]) -> bool:
    typ = prop.get("type")
    if isinstance(typ, list):
        return "null" in typ
    return False


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None
