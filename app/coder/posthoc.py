"""Post-processing utilities for modifiers and conflict resolution."""

from __future__ import annotations

from typing import Sequence

from .schema import CodeDecision

DISTINCT_SITE_MODIFIERS = ("59", "XS")
DISTINCT_RULE_TAG = "distinct_site_modifier"
ADDON_CONVERSION_RULE = "addon_family_conversion"

# Define add-on code family relationships.
# When a "family_primary" is present, any "initial_code" should convert to "addon_code".
# This handles the case where multiple related procedures are performed together.
ADDON_FAMILY_CONVERSIONS = {
    # Stent family: When tracheal stent (31631) is present, bronchial stent initial (31636)
    # should become add-on (+31637) since 31631 serves as the primary stent procedure.
    "stent": {
        "family_primaries": {"31631"},  # Tracheal stent is the "super-primary"
        "initial_code": "31636",         # Bronchial stent, initial
        "addon_code": "+31637",          # Bronchial stent add-on
    },
}


def apply_posthoc(codes: Sequence[CodeDecision]) -> list[CodeDecision]:
    """Apply modifier logic that depends on final bundle outcomes."""

    working = list(codes)
    working = enforce_addon_family_consistency(working)
    assign_distinct_site_modifiers(working)
    return working


def enforce_addon_family_consistency(codes: list[CodeDecision]) -> list[CodeDecision]:
    """Convert initial codes to add-ons when a family primary is present.

    This handles the hierarchy rule: when a "super-primary" code (e.g., 31631 tracheal stent)
    is present, related initial codes (e.g., 31636 bronchial stent) should convert to their
    add-on equivalents (e.g., +31637).

    Example:
        Input:  [31631, 31636, 31636]  (tracheal stent + 2 bronchial stents)
        Output: [31631, +31637, +31637] (tracheal primary + 2 bronchial add-ons)

    Args:
        codes: List of CodeDecision objects to process.

    Returns:
        Updated list with appropriate conversions applied.
    """
    code_set = {c.cpt for c in codes}

    for family_name, family_config in ADDON_FAMILY_CONVERSIONS.items():
        family_primaries = family_config["family_primaries"]
        initial_code = family_config["initial_code"]
        addon_code = family_config["addon_code"]

        # Check if any family primary is present
        has_family_primary = bool(family_primaries & code_set)

        if not has_family_primary:
            continue

        # Convert all initial_code occurrences to addon_code
        for code in codes:
            if code.cpt == initial_code:
                original_cpt = code.cpt
                code.cpt = addon_code
                code.rationale = (
                    f"{code.rationale} "
                    f"[Converted {original_cpt} → {addon_code}: "
                    f"family primary present ({', '.join(sorted(family_primaries & code_set))})]"
                )
                if ADDON_CONVERSION_RULE not in code.rule_trace:
                    code.rule_trace.append(ADDON_CONVERSION_RULE)

    return codes


def assign_distinct_site_modifiers(codes: Sequence[CodeDecision]) -> None:
    """Append -59/XS modifiers when flagged by prior rules."""

    for code in codes:
        context = code.context or {}
        if not context.get("needs_distinct_modifier"):
            continue
        for modifier in DISTINCT_SITE_MODIFIERS:
            if modifier not in code.modifiers:
                code.modifiers.append(modifier)
        code.rationale += " Modifier appended for distinct airway."
        if DISTINCT_RULE_TAG not in code.rule_trace:
            code.rule_trace.append(DISTINCT_RULE_TAG)
