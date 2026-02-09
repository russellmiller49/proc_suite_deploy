from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class BillingCodeItem(TypedDict, total=False):
    code: str
    cpt: str
    CPT: str


class RegistryBillingPayload(TypedDict, total=False):
    cpt_codes: list[Any]


class RegistryRecordCompat(TypedDict, total=False):
    """Subset of RegistryRecord fields used by compat enrichment."""

    billing: RegistryBillingPayload
    cpt_codes: list[str]
    procedures_performed: dict[str, Any]

