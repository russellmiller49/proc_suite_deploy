"""Build structured audit-compare reports for extraction-first registry pipeline."""

from __future__ import annotations

from typing import Iterable, Sequence

from ml.lib.ml_coder.predictor import CaseClassification, CodePrediction
from app.registry.audit.audit_types import (
    AuditCompareReport,
    AuditConfigSnapshot,
    AuditPrediction,
)
from app.registry.audit.raw_ml_auditor import RawMLAuditConfig


def _index_preds(preds: Iterable[CodePrediction]) -> dict[str, CodePrediction]:
    return {p.cpt: p for p in preds}


_CPT_EQUIVALENCE_GROUPS: tuple[set[str], ...] = (
    # Mutually exclusive alternatives or "more specific" variants that should not trigger audit omissions.
    {"31652", "31653"},  # linear EBUS station bucket
    {"32554", "32555"},  # thoracentesis (no imaging vs ultrasound guidance)
    {"32556", "32557"},  # pleural drain (no imaging vs imaging guidance)
    {"31640", "31641"},  # tumor excision vs destruction (choose one per site)
)


def _equivalent_cpt_map() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for group in _CPT_EQUIVALENCE_GROUPS:
        for code in group:
            mapping[code] = set(group) - {code}
    return mapping


_CPT_EQUIVALENTS = _equivalent_cpt_map()


def _covered_by_equivalent(*, predicted_cpt: str, derived_set: set[str]) -> bool:
    equivalents = _CPT_EQUIVALENTS.get(predicted_cpt)
    return bool(equivalents and (derived_set & equivalents))


def build_audit_compare_report(
    *,
    derived_codes: Sequence[str],
    cfg: RawMLAuditConfig,
    ml_case: CaseClassification | None = None,
    audit_preds: Sequence[CodePrediction] | None = None,
    warnings: list[str] | None = None,
) -> AuditCompareReport:
    """Create a structured compare report between deterministic and RAW-ML audit sets.

    This function is reporting-only:
    - It must not mutate the RegistryRecord or derived codes.
    - It must not auto-merge ML outputs into the deterministic output.
    """
    derived_list = list(derived_codes)
    derived_set = set(derived_list)

    snapshot = AuditConfigSnapshot(
        use_buckets=cfg.use_buckets,
        top_k=cfg.top_k,
        min_prob=cfg.min_prob,
        self_correct_min_prob=cfg.self_correct_min_prob,
    )

    report_warnings = list(warnings or [])

    if not ml_case or not audit_preds:
        return AuditCompareReport(
            derived_codes=derived_list,
            ml_audit_codes=[],
            missing_in_derived=[],
            missing_in_ml=sorted(derived_set),
            high_conf_omissions=[],
            ml_difficulty=None,
            config=snapshot,
            warnings=report_warnings,
        )

    high_conf_set = {p.cpt for p in ml_case.high_conf}
    gray_zone_set = {p.cpt for p in ml_case.gray_zone}
    audit_index = _index_preds(audit_preds)

    ml_audit_codes: list[AuditPrediction] = []
    for pred in audit_preds:
        bucket: str | None
        if pred.cpt in high_conf_set:
            bucket = "HIGH_CONF"
        elif pred.cpt in gray_zone_set:
            bucket = "GRAY_ZONE"
        else:
            bucket = "PREDICTIONS"
        ml_audit_codes.append(AuditPrediction(cpt=pred.cpt, prob=float(pred.prob), bucket=bucket))

    audit_set = {p.cpt for p in ml_audit_codes}
    missing_in_ml = sorted(derived_set - audit_set)

    missing_in_derived = [
        p
        for p in ml_audit_codes
        if p.cpt not in derived_set and not _covered_by_equivalent(predicted_cpt=p.cpt, derived_set=derived_set)
    ]

    high_conf_omissions = [
        p
        for p in missing_in_derived
        if p.bucket == "HIGH_CONF" or p.prob >= cfg.self_correct_min_prob
    ]

    # Ensure we always have prob populated for audit codes (defensive).
    for pred in missing_in_derived:
        if pred.prob == 0.0 and pred.cpt in audit_index:
            # This should not happen; record for debugging if it does.
            report_warnings.append(f"Missing prob for ML audit code {pred.cpt}; defaulted to 0.0")

    return AuditCompareReport(
        derived_codes=derived_list,
        ml_audit_codes=ml_audit_codes,
        missing_in_derived=missing_in_derived,
        missing_in_ml=missing_in_ml,
        high_conf_omissions=high_conf_omissions,
        ml_difficulty=ml_case.difficulty.value,
        config=snapshot,
        warnings=report_warnings,
    )


__all__ = ["build_audit_compare_report"]
