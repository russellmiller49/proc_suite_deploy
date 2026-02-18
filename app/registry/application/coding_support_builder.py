from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from config.settings import CoderSettings
from app.coder.adapters.persistence.csv_kb_adapter import JsonKnowledgeBaseAdapter
from app.domain.knowledge_base.repository import KnowledgeBaseRepository
from app.registry.schema import RegistryRecord


_CPT_RE = re.compile(r"\b(\d{5})\b")


@lru_cache(maxsize=1)
def get_kb_repo() -> KnowledgeBaseRepository:
    settings = CoderSettings()
    return JsonKnowledgeBaseAdapter(settings.kb_path)


def _evidence_items_for_prefix(
    record: RegistryRecord,
    prefix: str,
    *,
    max_items: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (billing_evidence_items, note_spans) for evidence keys matching prefix."""
    evidence = getattr(record, "evidence", None)
    if not isinstance(evidence, dict) or not evidence:
        return [], []

    billing_items: list[dict[str, Any]] = []
    note_spans: list[dict[str, Any]] = []

    for key, spans in evidence.items():
        if not isinstance(key, str) or not key:
            continue
        if key != prefix and not key.startswith(prefix + "."):
            continue
        if not isinstance(spans, list):
            continue

        for span in spans:
            text = getattr(span, "text", None) if not isinstance(span, dict) else span.get("text") or span.get("quote")
            start = getattr(span, "start", None) if not isinstance(span, dict) else span.get("start") or span.get("start_char")
            end = getattr(span, "end", None) if not isinstance(span, dict) else span.get("end") or span.get("end_char")
            confidence = getattr(span, "confidence", None) if not isinstance(span, dict) else span.get("confidence")
            if text is None or start is None or end is None:
                continue

            try:
                start_val = int(start)
                end_val = int(end)
            except (TypeError, ValueError):
                continue

            billing_items.append(
                {
                    "source": "registry_span",
                    "text": str(text),
                    "span": [start_val, end_val],
                    "confidence": float(confidence) if confidence is not None else 1.0,
                }
            )
            note_spans.append({"start": start_val, "end": end_val, "snippet": str(text)})

            if len(billing_items) >= max_items:
                break
        if len(billing_items) >= max_items:
            break

    return billing_items, note_spans


def default_evidence_prefixes_for_code(code: str) -> list[str]:
    """Heuristic mapping from CPT -> registry evidence prefixes."""
    return {
        "31573": ["procedures_performed.therapeutic_injection"],
        "31600": ["procedures_performed.percutaneous_tracheostomy"],
        "31615": ["established_tracheostomy_route"],
        "31622": ["procedures_performed.diagnostic_bronchoscopy"],
        "31623": ["procedures_performed.brushings"],
        "31624": ["procedures_performed.bal"],
        "31625": ["procedures_performed.endobronchial_biopsy"],
        "31626": ["granular_data.navigation_targets", "procedures_performed.navigational_bronchoscopy"],
        "31627": ["procedures_performed.navigational_bronchoscopy", "granular_data.navigation_targets"],
        "31628": ["procedures_performed.transbronchial_biopsy", "procedures_performed.transbronchial_cryobiopsy"],
        "31629": ["procedures_performed.peripheral_tbna", "procedures_performed.tbna_conventional"],
        "31630": ["procedures_performed.airway_dilation", "granular_data.dilation_targets"],
        "31632": ["procedures_performed.transbronchial_biopsy", "procedures_performed.transbronchial_cryobiopsy"],
        "31633": ["procedures_performed.peripheral_tbna"],
        "31634": ["granular_data.blvr_chartis_measurements", "procedures_performed.blvr"],
        "31635": ["procedures_performed.foreign_body_removal"],
        "31636": ["procedures_performed.airway_stent"],
        "31637": ["procedures_performed.airway_stent"],
        "31638": ["procedures_performed.airway_stent_revision", "procedures_performed.airway_stent"],
        "31640": ["procedures_performed.mechanical_debulking"],
        "31641": [
            "procedures_performed.thermal_ablation",
            "procedures_performed.cryotherapy",
            "procedures_performed.peripheral_ablation",
            "procedures_performed.bpf_sealant",
        ],
        "31645": ["procedures_performed.therapeutic_aspiration"],
        "31646": ["procedures_performed.therapeutic_aspiration"],
        "31647": ["procedures_performed.blvr", "granular_data.blvr_valve_placements"],
        "31648": ["procedures_performed.blvr", "granular_data.blvr_valve_placements"],
        "31649": ["procedures_performed.blvr", "granular_data.blvr_valve_placements"],
        "31651": ["procedures_performed.blvr", "granular_data.blvr_valve_placements"],
        "31652": ["procedures_performed.linear_ebus", "procedures_performed.linear_ebus.node_events"],
        "31653": ["procedures_performed.linear_ebus", "procedures_performed.linear_ebus.node_events"],
        "31654": ["procedures_performed.radial_ebus"],
        "31660": ["procedures_performed.bronchial_thermoplasty"],
        "31661": ["procedures_performed.bronchial_thermoplasty"],
        "32550": ["pleural_procedures.ipc", "pleural_procedures.indwelling_pleural_catheter"],
        "32552": ["pleural_procedures.ipc", "pleural_procedures.indwelling_pleural_catheter"],
        "32551": ["pleural_procedures.chest_tube"],
        "32554": ["pleural_procedures.thoracentesis"],
        "32555": ["pleural_procedures.thoracentesis"],
        "32556": ["pleural_procedures.chest_tube"],
        "32557": ["pleural_procedures.chest_tube"],
        "32558": ["pleural_procedures.chest_tube"],
        "32560": ["pleural_procedures.pleurodesis"],
        "32561": ["pleural_procedures.fibrinolytic_therapy"],
        "32562": ["pleural_procedures.fibrinolytic_therapy"],
        "32601": ["pleural_procedures.medical_thoracoscopy"],
        "32609": ["pleural_procedures.medical_thoracoscopy"],
        "43237": ["procedures_performed.eus_b"],
        "43238": ["procedures_performed.eus_b"],
        "76604": ["procedures_performed.chest_ultrasound"],
        "76536": ["procedures_performed.neck_ultrasound"],
        "99152": ["sedation"],
        "99153": ["sedation"],
    }.get(str(code).strip().lstrip("+"), [])


def build_traceability_for_code(
    *,
    record: RegistryRecord,
    code: str,
    max_items: int = 6,
) -> tuple[list[str] | None, list[dict[str, Any]] | None]:
    """Return (derived_from, evidence) for billing.cpt_codes[*]."""
    prefixes = default_evidence_prefixes_for_code(code)
    if not prefixes:
        return None, None

    for prefix in prefixes:
        evidence_items, _note_spans = _evidence_items_for_prefix(record, prefix, max_items=max_items)
        if evidence_items:
            return [prefix], evidence_items

    return prefixes, None


def _parse_warnings_into_rule_applications(
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[dict[str, Any]]]]:
    """Return (rules_applied, dropped_codes, per_code_qa_flags)."""
    rules_applied: list[dict[str, Any]] = []
    dropped_codes: dict[str, str] = {}
    per_code_qa_flags: dict[str, list[dict[str, Any]]] = {}

    def _add_rule(
        *,
        rule_id: str,
        rule_type: str,
        outcome: str,
        codes_affected: list[str],
        details: str,
        description: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "outcome": outcome,
            "codes_affected": codes_affected,
            "details": details,
        }
        if description is not None:
            payload["description"] = description
        rules_applied.append(payload)

    for warning in warnings:
        text = str(warning or "").strip()
        if not text:
            continue

        # TBNA bundled into EBUS sampling
        if text.startswith("Suppressed 31629:"):
            dropped_codes["31629"] = text
            _add_rule(
                rule_id="tbna_with_ebus_tbna",
                rule_type="bundling",
                outcome="dropped",
                codes_affected=["31629"],
                details=text,
            )
            continue

        # Chartis bundling vs valves (same lobe)
        if text.startswith("Suppressed 31634 (Chartis):"):
            dropped_codes["31634"] = text
            _add_rule(
                rule_id="chartis_bundling",
                rule_type="bundling",
                outcome="dropped",
                codes_affected=["31634"],
                details=text,
            )
            continue

        if "31634 (Chartis) distinct from valve lobe" in text:
            per_code_qa_flags.setdefault("31634", []).append(
                {
                    "severity": "warning",
                    "rule_id": "chartis_bundling",
                    "message": text,
                }
            )
            _add_rule(
                rule_id="chartis_bundling",
                rule_type="documentation",
                outcome="flagged",
                codes_affected=["31634"],
                details=text,
            )
            continue

        # Moderate sedation time threshold / documentation gating
        if "99152/99153" in text and ("suppressing" in text or "not deriving" in text):
            dropped_codes.setdefault("99152", text)
            dropped_codes.setdefault("99153", text)
            _add_rule(
                rule_id="moderate_sedation",
                rule_type="documentation",
                outcome="dropped",
                codes_affected=["99152", "99153"],
                details=text,
            )
            continue

        # Modifier prompts
        if "requires Modifier 59" in text and "31629" in text:
            per_code_qa_flags.setdefault("31629", []).append(
                {
                    "severity": "warning",
                    "rule_id": "tbna_with_ebus_tbna",
                    "message": text,
                }
            )
            _add_rule(
                rule_id="tbna_with_ebus_tbna",
                rule_type="documentation",
                outcome="flagged",
                codes_affected=["31629"],
                details=text,
            )
            continue

        # Diagnostic bronchoscopy bundled
        if text.startswith("Diagnostic bronchoscopy present but bundled"):
            dropped_codes["31622"] = text
            _add_rule(
                rule_id="diagnostic_with_surgical",
                rule_type="bundling",
                outcome="dropped",
                codes_affected=["31622"],
                details=text,
            )
            continue

        # Dilation bundled into destruction/excision
        if text.startswith("31630 (dilation) bundled into"):
            dropped_codes["31630"] = text
            _add_rule(
                rule_id="dilation_with_destruction",
                rule_type="local",
                outcome="dropped",
                codes_affected=["31630"],
                details=text,
            )
            continue

        # Destruction bundled into excision (default same-lesion assumption)
        if text.startswith("31641 (destruction) bundled into 31640"):
            dropped_codes["31641"] = text
            per_code_qa_flags.setdefault("31640", []).append(
                {
                    "severity": "warning",
                    "rule_id": "excision_with_destruction",
                    "message": text,
                }
            )
            _add_rule(
                rule_id="excision_with_destruction",
                rule_type="bundling",
                outcome="dropped",
                codes_affected=["31641"],
                details=text,
            )
            continue

        if text.startswith("31641 requires Modifier 59"):
            per_code_qa_flags.setdefault("31641", []).append(
                {
                    "severity": "warning",
                    "rule_id": "excision_with_destruction",
                    "message": text,
                }
            )
            _add_rule(
                rule_id="excision_with_destruction",
                rule_type="documentation",
                outcome="flagged",
                codes_affected=["31641"],
                details=text,
            )
            continue

        # Otherwise, treat as an informational global rule note.
        codes = sorted(set(_CPT_RE.findall(text)))
        if codes:
            for code in codes:
                per_code_qa_flags.setdefault(code, []).append(
                    {"severity": "info", "rule_id": None, "message": text}
                )
        rules_applied.append(
            {
                "rule_id": None,
                "rule_type": "local",
                "outcome": "informational",
                "codes_affected": codes or None,
                "details": text,
            }
        )

    return rules_applied, dropped_codes, per_code_qa_flags


def build_coding_support_payload(
    *,
    record: RegistryRecord,
    codes: list[str],
    code_units: dict[str, int] | None = None,
    code_rationales: dict[str, str] | None = None,
    derivation_warnings: list[str] | None = None,
    kb_repo: KnowledgeBaseRepository | None = None,
) -> dict[str, Any]:
    """Build the optional `coding_support` payload for the IP registry schema."""
    kb = kb_repo or get_kb_repo()
    rationales = code_rationales or {}
    warnings = list(derivation_warnings or [])

    rules_applied, dropped_codes, per_code_qa_flags = _parse_warnings_into_rule_applications(warnings)

    selected = [str(c).strip().lstrip("+") for c in (codes or []) if str(c).strip()]
    dropped = sorted(set(dropped_codes.keys()) - set(selected))
    all_codes = selected + dropped

    lines: list[dict[str, Any]] = []
    per_code: list[dict[str, Any]] = []

    for idx, code in enumerate(all_codes, start=1):
        proc = kb.get_procedure_info(code)
        is_add_on = kb.is_addon_code(code)
        description = proc.description if proc else None
        units = int((code_units or {}).get(code, 1) or 1)

        selection_status = "selected" if code in selected else "dropped"
        selection_reason = (
            rationales.get(code)
            if selection_status == "selected"
            else dropped_codes.get(code) or "dropped by rule"
        )

        modifiers: list[str] | None = None
        if code == "31629" and any("Modifier 59" in w for w in warnings):
            modifiers = ["59"]
        if code == "31641" and any(str(w).startswith("31641 requires Modifier 59") for w in warnings):
            modifiers = sorted(set((modifiers or []) + ["59"]))

        evidence_items: list[dict[str, Any]] = []
        note_spans: list[dict[str, Any]] = []
        for prefix in default_evidence_prefixes_for_code(code):
            ev_items, spans = _evidence_items_for_prefix(record, prefix)
            evidence_items.extend(ev_items)
            note_spans.extend(spans)
            if evidence_items:
                break

        rule_refs = sorted(
            {
                str(rule.get("rule_id"))
                for rule in rules_applied
                if rule.get("rule_id") and code in (rule.get("codes_affected") or [])
            }
        )

        qa_flags = per_code_qa_flags.get(code)

        lines.append(
            {
                "sequence": idx,
                "code": code,
                "description": description,
                "modifiers": modifiers,
                "units": units,
                "role": "add_on" if is_add_on else "primary",
                "selection_status": selection_status,
                "selection_reason": selection_reason,
                "is_add_on": is_add_on,
                "source": "model",
                "note_spans": note_spans or None,
            }
        )

        per_code.append(
            {
                "code": code,
                "summary": selection_reason,
                "documentation_evidence": [
                    {"snippet": item.get("text"), "span": {"start": item["span"][0], "end": item["span"][1]}}
                    for item in evidence_items
                ]
                or None,
                "rule_refs": rule_refs or None,
                "qa_flags": qa_flags or None,
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "version": "coding_support.v1",
        "generated_at": generated_at,
        "generator": "registry_to_cpt",
        "knowledge_base_version": f"ip_coding_billing.v{kb.version}",
        "coding_summary": {"lines": lines or None},
        "coding_rationale": {
            "per_code": per_code or None,
            "rules_applied": rules_applied or None,
            "global_comments": warnings or None,
        },
    }


__all__ = [
    "build_coding_support_payload",
    "build_traceability_for_code",
    "default_evidence_prefixes_for_code",
    "get_kb_repo",
]
