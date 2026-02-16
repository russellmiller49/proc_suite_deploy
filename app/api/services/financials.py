"""Financial calculations for CPT code sets.

Used by extraction-first endpoints to compute:
- total work RVUs
- per-code payment estimates
- Multiple Endoscopy Rule (MER) reductions for bronchoscopy endoscopy families
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.common.knowledge import get_knowledge
from app.domain.knowledge_base.repository import KnowledgeBaseRepository


def _multiple_endoscopy_family_codes() -> set[str]:
    kb = get_knowledge()
    policies = kb.get("policies")
    if not isinstance(policies, dict):
        return set()
    policy = policies.get("multiple_endoscopy_rule")
    if not isinstance(policy, dict):
        return set()
    applies = policy.get("applies_to_family")
    if not isinstance(applies, list):
        return set()
    return {str(code).strip().lstrip("+") for code in applies if str(code).strip()}


def calculate_financials(
    *,
    codes: list[str],
    kb_repo: KnowledgeBaseRepository,
    conversion_factor: float,
    units_by_code: Mapping[str, int] | None = None,
) -> tuple[float | None, float | None, list[dict[str, Any]], list[str]]:
    """Compute financial summary for a CPT code list.

    Returns:
        (total_work_rvu, estimated_payment, per_code_billing, warnings)
    """

    if not codes:
        return None, None, [], []

    per_code_billing: list[dict[str, Any]] = []
    total_work = 0.0

    unit_map = units_by_code or {}

    # Build base line items.
    for code in codes:
        proc_info = kb_repo.get_procedure_info(code)
        if not proc_info:
            continue

        normalized = str(code).strip().lstrip("+")
        units = int(unit_map.get(normalized, unit_map.get(str(code).strip(), 1)) or 1)
        units = max(units, 1)

        work_rvu = float(proc_info.work_rvu) * units
        total_rvu = float(proc_info.total_facility_rvu) * units
        base_payment = total_rvu * float(conversion_factor)

        total_work += work_rvu

        per_code_billing.append(
            {
                "cpt_code": normalized,
                "description": proc_info.description,
                "units": units,
                "work_rvu": work_rvu,
                "total_facility_rvu": total_rvu,
                # May be adjusted by the multiple endoscopy rule below.
                "facility_payment": round(base_payment, 2),
                "_base_facility_payment": base_payment,
            }
        )

    # Apply the bronchoscopy multiple endoscopy rule (payment reduction) when applicable.
    warnings: list[str] = []
    family = _multiple_endoscopy_family_codes()

    def in_family(item: dict[str, Any]) -> bool:
        code = str(item.get("cpt_code") or "").strip().lstrip("+")
        return bool(code) and code in family and not kb_repo.is_addon_code(code)

    family_items = [item for item in per_code_billing if in_family(item)]
    if len(family_items) > 1:
        primary_item = max(
            family_items,
            key=lambda it: float(it.get("_base_facility_payment") or 0.0),
        )
        primary_code = str(primary_item.get("cpt_code") or "")

        reduced_codes: list[str] = []
        for item in family_items:
            code = str(item.get("cpt_code") or "")
            base_payment = float(item.get("_base_facility_payment") or 0.0)
            item["facility_payment_before_multiple_endoscopy"] = round(base_payment, 2)

            if code == primary_code:
                item["multiple_endoscopy_role"] = "primary"
                item["multiple_endoscopy_reduction"] = 0.0
                continue

            item["multiple_endoscopy_role"] = "secondary"
            adjusted = base_payment * 0.5
            item["facility_payment"] = round(adjusted, 2)
            item["multiple_endoscopy_reduction"] = round(base_payment - adjusted, 2)
            reduced_codes.append(code)

        warnings.append(
            "MULTIPLE_ENDOSCOPY_RULE: "
            f"primary={primary_code}; reduced={','.join(reduced_codes)}; reduction=50%"
        )

    # Strip internal fields and compute totals from displayed values (avoids penny mismatches).
    total_payment = 0.0
    for item in per_code_billing:
        total_payment += float(item.get("facility_payment") or 0.0)
        item.pop("_base_facility_payment", None)

    total_work_rvu = round(total_work, 2) if per_code_billing else None
    estimated_payment = round(total_payment, 2) if per_code_billing else None

    return total_work_rvu, estimated_payment, per_code_billing, warnings
