"""Minimal NCCI gatekeeper for deterministic bundling."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Set

from config.settings import KnowledgeSettings

NCCI_BUNDLED_REASON_PREFIX = "ncci:bundled_into:"

_CODE_RE = re.compile(r"^\d{5}$")


def _normalize_cpt(code: object) -> str | None:
    """Normalize a CPT-like string for NCCI comparisons.

    - Strip whitespace
    - Strip leading '+' (add-on marker)
    - Require 5-digit numeric strings (skip anything else)
    """
    if not isinstance(code, str):
        return None
    cleaned = code.strip()
    if not cleaned:
        return None
    if cleaned.startswith("+"):
        cleaned = cleaned[1:].strip()
    return cleaned if _CODE_RE.match(cleaned) else None


def _modifier_allowed_from_rule(rule: dict[str, Any]) -> bool | None:
    indicator = rule.get("modifier_indicator")
    if isinstance(indicator, str) and indicator in {"0", "1"}:
        return indicator == "1"
    if "modifier_allowed" in rule:
        return bool(rule.get("modifier_allowed", False))
    return None


def _canonical_pair_key(primary: str, secondary: str) -> tuple[str, str]:
    return (primary, secondary)


def merge_ncci_sources(
    *,
    kb_document: dict[str, Any] | None,
    external_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge NCCI pairs from the internal KB and an external PTP file.

    Precedence: external pairs override internal KB pairs for the same (primary, secondary).
    """
    kb_doc: dict[str, Any] = kb_document or {}
    ext: dict[str, Any] = external_cfg or {}

    merged: dict[tuple[str, str], dict[str, Any]] = {}

    # 1) Internal KB pairs (lower precedence)
    for raw in kb_doc.get("ncci_pairs", []) or []:
        if not isinstance(raw, dict):
            continue
        primary = _normalize_cpt(raw.get("primary"))
        secondary = _normalize_cpt(raw.get("secondary"))
        if not primary or not secondary:
            continue
        modifier_allowed = bool(raw.get("modifier_allowed", False))
        key = _canonical_pair_key(primary, secondary)
        merged[key] = {
            "column1": primary,
            "column2": secondary,
            "modifier_indicator": "1" if modifier_allowed else "0",
            "modifier_allowed": modifier_allowed,
            "reason": raw.get("reason"),
            "source": "kb",
        }

    # 2) External file pairs (higher precedence)
    for raw in ext.get("pairs", []) or []:
        if not isinstance(raw, dict):
            continue
        primary = _normalize_cpt(raw.get("column1") or raw.get("primary"))
        secondary = _normalize_cpt(raw.get("column2") or raw.get("secondary"))
        if not primary or not secondary:
            continue

        modifier_allowed = _modifier_allowed_from_rule(raw)
        if modifier_allowed is None:
            modifier_allowed = False

        key = _canonical_pair_key(primary, secondary)
        merged[key] = {
            "column1": primary,
            "column2": secondary,
            "modifier_indicator": "1" if modifier_allowed else "0",
            "modifier_allowed": modifier_allowed,
            "reason": raw.get("reason"),
            "source": "external",
        }

    # Emit in stable order for deterministic diffs/logging
    merged_pairs = [merged[key] for key in sorted(merged.keys())]
    out: dict[str, Any] = {"pairs": merged_pairs}
    if isinstance(ext.get("source_year"), int):
        out["source_year"] = ext["source_year"]
    return out


@lru_cache()
def load_ncci_ptp(path: str | Path | None = None) -> Dict[str, Any]:
    """Load NCCI procedure-to-procedure rules."""
    cfg_path = Path(path) if path is not None else KnowledgeSettings().ncci_path
    with cfg_path.open(encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    # Merge in internal KB pairs (if present); external overrides internal on conflict.
    from app.common.knowledge import get_knowledge

    kb_doc = get_knowledge(KnowledgeSettings().kb_path)
    return merge_ncci_sources(kb_document=kb_doc, external_cfg=data)


class NCCIResult:
    """Result of applying NCCI bundling rules."""

    def __init__(self, allowed: Set[str], bundled: Dict[str, str]):
        self.allowed = allowed
        self.bundled = bundled


class NCCIEngine:
    """Applies simple PTP bundling rules."""

    def __init__(self, ptp_cfg: Dict[str, Any] | None = None):
        cfg = ptp_cfg or load_ncci_ptp()
        self._pairs: list[tuple[str, str, bool]] = []
        for raw in cfg.get("pairs", []) or []:
            if not isinstance(raw, dict):
                continue
            c1 = _normalize_cpt(raw.get("column1") or raw.get("primary"))
            c2 = _normalize_cpt(raw.get("column2") or raw.get("secondary"))
            if not c1 or not c2:
                continue
            modifier_allowed = _modifier_allowed_from_rule(raw)
            if modifier_allowed is None:
                modifier_allowed = False
            self._pairs.append((c1, c2, modifier_allowed))

    def apply(self, codes: Set[str]) -> NCCIResult:
        allowed = set(codes)
        bundled: Dict[str, str] = {}

        canonical: dict[str, str | None] = {code: _normalize_cpt(code) for code in codes}
        by_canonical: dict[str, list[str]] = {}
        for original, canon in canonical.items():
            if canon is None:
                continue
            by_canonical.setdefault(canon, []).append(original)

        canonical_present = set(by_canonical.keys())

        for primary, secondary, modifier_allowed in self._pairs:
            if modifier_allowed:
                continue
            if primary not in canonical_present or secondary not in canonical_present:
                continue

            for original_secondary in by_canonical.get(secondary, []):
                allowed.discard(original_secondary)
                bundled[original_secondary] = primary

        return NCCIResult(allowed=allowed, bundled=bundled)


__all__ = [
    "NCCIEngine",
    "NCCIResult",
    "load_ncci_ptp",
    "NCCI_BUNDLED_REASON_PREFIX",
    "merge_ncci_sources",
]
