from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

Severity = Literal["info", "warn", "error"]
Action = Literal["auto_fixed", "flagged_for_manual"]


@dataclass(slots=True)
class IssueLogEntry:
    entry_id: str
    issue_type: str
    severity: Severity
    action: Action
    field: str | None
    details: dict[str, Any] | str


class IssueLogger:
    """Collects structured issues across cleaning passes."""

    def __init__(self) -> None:
        self._entries: list[IssueLogEntry] = []

    @property
    def entries(self) -> list[IssueLogEntry]:
        return list(self._entries)

    def log(
        self,
        *,
        entry_id: str,
        issue_type: str,
        severity: Severity,
        action: Action,
        details: dict[str, Any] | str,
        field: str | None = None,
    ) -> None:
        extracted_field = field
        if extracted_field is None and isinstance(details, dict):
            extracted_field = details.get("field")
        cleaned_details = details
        if isinstance(details, dict) and "field" in details:
            cleaned_details = {k: v for k, v in details.items() if k != "field"}
        self._entries.append(
            IssueLogEntry(
                entry_id=entry_id,
                issue_type=issue_type,
                severity=severity,
                action=action,
                field=extracted_field if extracted_field is not None else None,
                details=cleaned_details,
            )
        )

    def extend(self, entries: Iterable[IssueLogEntry]) -> None:
        self._entries.extend(entries)

    def write_csv(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["entry_id", "issue_type", "severity", "action", "field", "details"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(
                    {
                        "entry_id": entry.entry_id,
                        "issue_type": entry.issue_type,
                        "severity": entry.severity,
                        "action": entry.action,
                        "field": entry.field or "",
                        "details": _serialize_details(entry.details),
                    }
                )

    def write_json(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "entry_id": entry.entry_id,
                "issue_type": entry.issue_type,
                "severity": entry.severity,
                "action": entry.action,
                "field": entry.field,
                "details": entry.details,
            }
            for entry in self._entries
        ]
        path.write_text(json.dumps(payload, indent=2))

    def summarize_by_action(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {
            "auto_fixed": defaultdict(int),
            "flagged_for_manual": defaultdict(int),
        }
        for entry in self._entries:
            summary[entry.action][entry.issue_type] += 1
        return {key: dict(value) for key, value in summary.items()}

    def error_entry_ids(self) -> set[str]:
        return {entry.entry_id for entry in self._entries if entry.severity == "error"}


def derive_entry_id(entry: dict[str, Any], index: int) -> str:
    """Return a stable identifier composed of MRN, date, and record index."""

    mrn = _clean_identifier(str(entry.get("patient_mrn") or "unknown"))
    proc_date = _clean_identifier(str(entry.get("procedure_date") or "unknown"))
    return f"{mrn}_{proc_date}_{index:05d}"


def _serialize_details(details: dict[str, Any] | str) -> str:
    if isinstance(details, str):
        return details
    try:
        return json.dumps(details, sort_keys=True)
    except TypeError:
        return str(details)


def _clean_identifier(value: str) -> str:
    value = value.strip() or "unknown"
    value = re.sub(r"\s+", "-", value)
    return value.replace(",", "-")
