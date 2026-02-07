from __future__ import annotations

from typing import Any

from .logging_utils import IssueLogger

TRAINEE_TITLES = {"fellow", "resident", "medical student"}
PNEUMO_INTERVENTIONS = {"aspiration", "chest tube", "surgery", "o2", "oxygen", "observation"}
NON_COMPLICATION = {"none", "n/a", "na", "unspecified", "no"}


class ConsistencyChecker:
    """Deterministic cross-field corrections (pass 3)."""

    def apply(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        self._fix_trainee_presence(entry, entry_id, logger)
        self._sync_sedation_reversal(entry, entry_id, logger)
        self._sync_pleurodesis(entry, entry_id, logger)
        self._sync_pneumothorax(entry, entry_id, logger)
        self._check_disposition(entry, entry_id, logger)

    def _fix_trainee_presence(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        has_trainee = bool(entry.get("fellow_name"))
        assistant_role = str(entry.get("assistant_role") or "").strip().lower()
        if assistant_role in TRAINEE_TITLES:
            has_trainee = True
        trainee_present = entry.get("trainee_present")
        if has_trainee and not trainee_present:
            entry["trainee_present"] = True
            logger.log(
                entry_id=entry_id,
                issue_type="trainee_present_corrected",
                severity="info",
                action="auto_fixed",
                field="trainee_present",
                details={"old": trainee_present, "new": True},
            )

    def _sync_sedation_reversal(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        reversal_agent = entry.get("sedation_reversal_agent")
        reversal_given = entry.get("sedation_reversal_given")
        if reversal_agent and not reversal_given:
            entry["sedation_reversal_given"] = True
            logger.log(
                entry_id=entry_id,
                issue_type="sedation_reversal_aligned",
                severity="info",
                action="auto_fixed",
                field="sedation_reversal_given",
                details={"old": reversal_given, "new": True},
            )
        if reversal_given and not reversal_agent:
            logger.log(
                entry_id=entry_id,
                issue_type="reversal_agent_missing",
                severity="warn",
                action="flagged_for_manual",
                field="sedation_reversal_agent",
                details="Reversal documented but agent missing",
            )

    def _sync_pleurodesis(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        agent = entry.get("pleurodesis_agent")
        performed = entry.get("pleurodesis_performed")
        if agent and not performed:
            entry["pleurodesis_performed"] = True
            logger.log(
                entry_id=entry_id,
                issue_type="pleurodesis_flag_corrected",
                severity="info",
                action="auto_fixed",
                field="pleurodesis_performed",
                details={"old": performed, "new": True},
            )
        if performed and not agent:
            logger.log(
                entry_id=entry_id,
                issue_type="pleurodesis_agent_missing",
                severity="warn",
                action="flagged_for_manual",
                field="pleurodesis_agent",
                details="Pleurodesis marked performed but agent missing",
            )

    def _sync_pneumothorax(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        intervention = entry.get("pneumothorax_intervention")
        pneumothorax = entry.get("pneumothorax")
        interventions = intervention if isinstance(intervention, list) else [intervention]
        inferred = False
        for item in interventions:
            text = _normalize_value(item)
            if text and text.lower() in PNEUMO_INTERVENTIONS:
                inferred = True
                break
        if inferred and not pneumothorax:
            entry["pneumothorax"] = True
            logger.log(
                entry_id=entry_id,
                issue_type="pneumothorax_inferred",
                severity="info",
                action="auto_fixed",
                field="pneumothorax",
                details={"old": pneumothorax, "new": True, "intervention": intervention},
            )
        if pneumothorax and not inferred:
            logger.log(
                entry_id=entry_id,
                issue_type="pneumothorax_intervention_missing",
                severity="warn",
                action="flagged_for_manual",
                field="pneumothorax_intervention",
                details="Pneumothorax recorded without intervention",
            )

    def _check_disposition(self, entry: dict[str, Any], entry_id: str, logger: IssueLogger) -> None:
        bleeding = _normalize_value(entry.get("bleeding_severity"))
        hypoxia = _normalize_value(entry.get("hypoxia_respiratory_failure"))
        disposition = _normalize_value(entry.get("disposition"))
        has_bleeding = bool(bleeding and bleeding.lower() not in NON_COMPLICATION)
        has_hypoxia = bool(hypoxia and hypoxia.lower() not in NON_COMPLICATION)
        if (has_bleeding or has_hypoxia) and not disposition:
            logger.log(
                entry_id=entry_id,
                issue_type="disposition_missing_after_complication",
                severity="warn",
                action="flagged_for_manual",
                field="disposition",
                details={"bleeding": bleeding, "hypoxia": hypoxia},
            )
        elif not disposition:
            logger.log(
                entry_id=entry_id,
                issue_type="disposition_missing",
                severity="warn",
                action="flagged_for_manual",
                field="disposition",
                details="Disposition missing",
            )


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value)
