"""Bundling and edit logic for the coder pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.common import knowledge
from app.common.rules_engine import ncci

from .schema import BundleDecision, CodeDecision, DetectedIntent

SEDATION_CODES = {"99152", "99153"}
NAV_RULE = "nav_required"
RADIAL_RULE = "radial_requires_tblb"
RADIAL_LINEAR_RULE = "radial_linear_exclusive"
SEDATION_RULE = "sedation_blocker"
STENT_RULE = "stent_bundling"
DISTINCT_RULE = "distinct_site_modifier"
DIAGNOSTIC_RULE = "surgical_includes_diagnostic"


@dataclass
class RuleConfig:
    navigation_required: set[str]
    radial_requires_tblb: bool
    radial_linear_exclusive: bool
    radial_codes: set[str]
    linear_codes: set[str]
    sedation_blockers: set[str]
    stent_codes: set[str]
    dilation_codes: set[str]
    diagnostic_codes: set[str]
    surgical_codes: set[str]
    mutually_exclusive: list[tuple[set[str], str]]


_RULE_CONFIG: RuleConfig | None = None
_KNOWLEDGE_REF: str | None = None


def apply_rules(
    codes: Sequence[CodeDecision], intents: Sequence[DetectedIntent]
) -> tuple[list[CodeDecision], list[BundleDecision], list[str]]:
    """Apply bundling, exclusivity, and NCCI edits."""

    config = _get_rule_config()
    working = list(codes)
    actions: list[BundleDecision] = []
    warnings: list[str] = []

    working, nav_actions = _enforce_navigation_requirement(working, intents, config)
    actions.extend(nav_actions)

    working, radial_actions = _enforce_radial_requirements(working, intents, config)
    actions.extend(radial_actions)

    working, exclusive_actions = _enforce_radial_linear_exclusive(working, config)
    actions.extend(exclusive_actions)

    working, sedation_actions, sedation_warnings = _resolve_sedation_conflicts(working, intents, config)
    actions.extend(sedation_actions)
    warnings.extend(sedation_warnings)

    working, stent_actions = _resolve_stent_dilation(working, config)
    actions.extend(stent_actions)

    working, diag_actions = _resolve_diagnostic_with_surgical(working, config)
    actions.extend(diag_actions)

    working, excision_actions = _resolve_mutually_exclusive(working, config)
    actions.extend(excision_actions)

    return working, actions, warnings


def _enforce_navigation_requirement(
    codes: list[CodeDecision], intents: Sequence[DetectedIntent], config: RuleConfig
) -> tuple[list[CodeDecision], list[BundleDecision]]:
    required = config.navigation_required
    if not required:
        return codes, []
    nav_present = any(intent.intent == "navigation" for intent in intents)
    if nav_present:
        return codes, []

    filtered: list[CodeDecision] = []
    actions: list[BundleDecision] = []
    for code in codes:
        if code.cpt in required:
            actions.append(
                BundleDecision(
                    pair=(code.cpt, "NAV"),
                    action=f"drop {code.cpt}",
                    reason="Navigation add-on requires documentation of navigation start",
                    rule=NAV_RULE,
                )
            )
            continue
        filtered.append(code)
    return filtered, actions


def _enforce_radial_requirements(
    codes: list[CodeDecision], intents: Sequence[DetectedIntent], config: RuleConfig
) -> tuple[list[CodeDecision], list[BundleDecision]]:
    if not config.radial_requires_tblb:
        return list(codes), []
    has_tblb = any(intent.intent == "tblb_lobe" for intent in intents)
    filtered: list[CodeDecision] = []
    actions: list[BundleDecision] = []
    for code in codes:
        if code.cpt in config.radial_codes:
            if has_tblb or (code.context or {}).get("peripheral_target"):
                filtered.append(code)
                continue
            actions.append(
                BundleDecision(
                    pair=(code.cpt, "PERIPH"),
                    action=f"drop {code.cpt}",
                    reason="Radial add-on reserved for peripheral lesion sampling",
                    rule=RADIAL_RULE,
                )
            )
            continue
        filtered.append(code)
    return filtered, actions


def _enforce_radial_linear_exclusive(
    codes: list[CodeDecision], config: RuleConfig
) -> tuple[list[CodeDecision], list[BundleDecision]]:
    if not config.radial_linear_exclusive:
        return list(codes), []

    has_linear = any(code.cpt in config.linear_codes for code in codes)
    if not has_linear:
        return list(codes), []

    filtered: list[CodeDecision] = []
    actions: list[BundleDecision] = []
    for code in codes:
        if code.cpt in config.radial_codes:
            if (code.context or {}).get("peripheral_target"):
                filtered.append(code)
                continue
            actions.append(
                BundleDecision(
                    pair=("LINEAR", code.cpt),
                    action=f"drop {code.cpt}",
                    reason="Radial add-on not allowed when linear EBUS performed",
                    rule=RADIAL_LINEAR_RULE,
                )
            )
            continue
        filtered.append(code)
    return filtered, actions


def _resolve_sedation_conflicts(
    codes: list[CodeDecision], intents: Sequence[DetectedIntent], config: RuleConfig
) -> tuple[list[CodeDecision], list[BundleDecision], list[str]]:
    if not config.sedation_blockers:
        return list(codes), [], []

    if not any(intent.intent in config.sedation_blockers for intent in intents):
        return list(codes), [], []

    filtered: list[CodeDecision] = []
    actions: list[BundleDecision] = []
    warnings: list[str] = []
    for code in codes:
        if code.cpt in SEDATION_CODES:
            actions.append(
                BundleDecision(
                    pair=(code.cpt, "ANES"),
                    action=f"drop {code.cpt}",
                    reason="Sedation not billed when separate anesthesia present",
                    rule=SEDATION_RULE,
                )
            )
            if not warnings:
                warnings.append("Sedation removed because anesthesia professional documented")
            continue
        filtered.append(code)
    return filtered, actions, warnings


def _resolve_stent_dilation(codes: list[CodeDecision], config: RuleConfig) -> tuple[list[CodeDecision], list[BundleDecision]]:
    filtered: list[CodeDecision] = []
    actions: list[BundleDecision] = []
    stents = [code for code in codes if code.cpt in config.stent_codes]

    for code in codes:
        if code.cpt not in config.dilation_codes:
            filtered.append(code)
            continue
        site = (code.context or {}).get("site")
        matching_stent = next(
            (stent for stent in stents if (stent.context or {}).get("site") == site and site),
            None,
        )
        distinct_doc = bool((code.context or {}).get("distinct"))
        if not distinct_doc and site:
            distinct_doc = all(
                (stent.context or {}).get("site") != site for stent in stents if (stent.context or {}).get("site")
            )
        if matching_stent:
            actions.append(
                BundleDecision(
                    pair=(matching_stent.cpt, code.cpt),
                    action=f"drop {code.cpt}",
                    reason="Stent placement bundles dilation when same segment",
                    rule=STENT_RULE,
                )
            )
            continue
        can_modify = bool(stents) and ncci.allow_with_modifier(stents[0].cpt, code.cpt)
        if can_modify:
            if distinct_doc:
                code.context.setdefault("needs_distinct_modifier", True)
                actions.append(
                    BundleDecision(
                        pair=(stents[0].cpt, code.cpt),
                        action=f"allow {code.cpt} with modifier",
                        reason="Distinct airway segment",
                        rule=DISTINCT_RULE,
                    )
                )
                code.rule_trace.append(DISTINCT_RULE)
            else:
                actions.append(
                    BundleDecision(
                        pair=(stents[0].cpt, code.cpt),
                        action=f"drop {code.cpt}",
                        reason="Modifier requires explicit documentation of distinct site",
                        rule=STENT_RULE,
                    )
                )
                continue
        filtered.append(code)
    return filtered, actions


def _resolve_diagnostic_with_surgical(
    codes: list[CodeDecision], config: RuleConfig
) -> tuple[list[CodeDecision], list[BundleDecision]]:
    has_surgical = any(code.cpt in config.surgical_codes for code in codes)
    if not has_surgical:
        return codes, []

    filtered: list[CodeDecision] = []
    actions: list[BundleDecision] = []
    for code in codes:
        if code.cpt in config.diagnostic_codes:
            actions.append(
                BundleDecision(
                    pair=(code.cpt, "SURG"),
                    action=f"drop {code.cpt}",
                    reason="Diagnostic bronchoscopy bundled with surgical procedures",
                    rule=DIAGNOSTIC_RULE,
                )
            )
            continue
        filtered.append(code)
    return filtered, actions


def _resolve_mutually_exclusive(
    codes: list[CodeDecision], config: RuleConfig
) -> tuple[list[CodeDecision], list[BundleDecision]]:
    if not config.mutually_exclusive:
        return list(codes), []

    filtered: list[CodeDecision] = []
    actions: list[BundleDecision] = []
    for code in codes:
        drop = False
        for code_set, keep in config.mutually_exclusive:
            if code.cpt in code_set and code.cpt != keep:
                if any(existing.cpt == keep for existing in codes):
                    actions.append(
                        BundleDecision(
                            pair=(keep, code.cpt),
                            action=f"drop {code.cpt}",
                            reason=f"{keep} supersedes {code.cpt}",
                            rule=f"mutually_exclusive:{keep}",
                        )
                    )
                    drop = True
                    break
        if not drop:
            filtered.append(code)
    return filtered, actions


def _get_rule_config() -> RuleConfig:
    global _RULE_CONFIG, _KNOWLEDGE_REF
    knowledge_hash = knowledge.knowledge_hash()
    if _RULE_CONFIG is not None and knowledge_hash == _KNOWLEDGE_REF:
        return _RULE_CONFIG

    bundling = knowledge.bundling_rules()
    nav_entry = bundling.get("navigation_required", {}) or {}
    radial_entry = bundling.get("radial_requires_tblb", {}) or {}
    radial_linear_entry = bundling.get("radial_linear_exclusive", {}) or {}
    stent_entry = bundling.get("stent_dilation_same_segment", {}) or {}
    diagnostic_entry = bundling.get("diagnostic_with_surgical", {}) or {}
    if "sedation_blockers" in bundling:
        sedation_blockers = set(bundling.get("sedation_blockers", []))
    else:
        sedation_blockers = {"anesthesia"}

    navigation_required = set(nav_entry.get("codes", []))
    radial_requires_tblb = bool(radial_entry) or bool(radial_entry.get("requires_tblb", False))
    radial_codes = set(radial_entry.get("radial_codes", [])) or set(radial_linear_entry.get("radial_codes", []))
    linear_codes = set(radial_linear_entry.get("linear_codes", []))
    stent_codes = set(stent_entry.get("stent_codes", []))
    dilation_codes = set(stent_entry.get("dilation_codes", []))
    diagnostic_codes = set(diagnostic_entry.get("drop_codes", []))
    surgical_codes = set(diagnostic_entry.get("therapeutic_codes", []))

    mutually_exclusive: list[tuple[set[str], str]] = []
    exclusive_entry = bundling.get("31640_vs_31641_same_site")
    if isinstance(exclusive_entry, dict):
        paired = exclusive_entry.get("paired", [])
        keep = exclusive_entry.get("dominant")
        if paired and keep:
            mutually_exclusive.append((set(paired), keep))

    config = RuleConfig(
        navigation_required=navigation_required,
        radial_requires_tblb=radial_requires_tblb,
        radial_linear_exclusive=bool(radial_linear_entry),
        radial_codes=radial_codes or set(["+31654"]),
        linear_codes=linear_codes or set(["31652", "31653"]),
        sedation_blockers=sedation_blockers,
        stent_codes=stent_codes or {"31631", "31636", "+31637"},
        dilation_codes=dilation_codes or {"31630"},
        diagnostic_codes=diagnostic_codes or {"31622"},
        surgical_codes=surgical_codes or {
            "31627",
            "31628",
            "+31632",
            "31629",
            "+31633",
            "31630",
            "31635",
            "31636",
            "31652",
            "31653",
            "+31654",
            "31638",
            "31641",
        },
        mutually_exclusive=mutually_exclusive,
    )

    _configure_ncci(knowledge.ncci_pairs())
    _RULE_CONFIG = config
    _KNOWLEDGE_REF = knowledge_hash
    return config


def _configure_ncci(pairs: Sequence[dict]) -> None:
    edits: list[ncci.NCCIEdit] = []
    for entry in pairs:
        primary = entry.get("primary")
        secondary = entry.get("secondary")
        if not primary or not secondary:
            continue
        edits.append(
            ncci.NCCIEdit(
                primary=primary,
                secondary=secondary,
                modifier_allowed=bool(entry.get("modifier_allowed")),
                reason=entry.get("reason", ""),
            )
        )
    ncci.replace_pairs(edits)
