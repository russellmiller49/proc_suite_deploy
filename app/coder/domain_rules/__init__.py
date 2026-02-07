"""Domain rules for code hierarchy and family consistency.

NOTE: This module is now a package to support submodules (e.g. registry_to_cpt).

This module provides standalone functions for enforcing CPT code family rules
that can be used by both the legacy engine and the new hexagonal architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# Define add-on code family relationships.
# When a "family_primary" is present, any "initial_code" should convert to "addon_code".
ADDON_FAMILY_CONVERSIONS: Dict[str, Dict[str, object]] = {
    # Stent family: When tracheal stent (31631) is present, bronchial stent initial (31636)
    # should become add-on (+31637) since 31631 serves as the primary stent procedure.
    "stent": {
        "family_primaries": {"31631"},  # Tracheal stent is the "super-primary"
        "initial_code": "31636",         # Bronchial stent, initial
        "addon_code": "+31637",          # Bronchial stent add-on
    },
}

# NCCI bundling pairs that should ALWAYS be enforced (even for high-confidence extractions).
# These are "hard" bundles where the secondary code is included in the primary.
# Format: (primary, secondary) -> reason
EBUS_ASPIRATION_BUNDLES: Dict[Tuple[str, str], str] = {
    ("31652", "31645"): "Therapeutic aspiration is bundled into EBUS-TBNA (1-2 stations)",
    ("31652", "31646"): "Subsequent aspiration is bundled into EBUS-TBNA (1-2 stations)",
    ("31653", "31645"): "Therapeutic aspiration is bundled into EBUS-TBNA (3+ stations)",
    ("31653", "31646"): "Subsequent aspiration is bundled into EBUS-TBNA (3+ stations)",
}

# Thoracentesis bundling with tunneled pleural catheter (32550)
# Per NCCI: thoracentesis is bundled into tunneled catheter placement
THORACENTESIS_IPC_BUNDLES: Dict[Tuple[str, str], str] = {
    ("32550", "32554"): "Thoracentesis without imaging bundled into tunneled pleural catheter placement",
    ("32550", "32555"): "Thoracentesis with imaging bundled into tunneled pleural catheter placement",
}

# Tumor excision vs destruction - mutually exclusive for same lesion
# 31640 (excision) bundled into 31641 (destruction) when same lesion
TUMOR_BUNDLES: Dict[Tuple[str, str], str] = {
    ("31641", "31640"): "Tumor excision (31640) bundled into destruction (31641) for same lesion",
}


@dataclass
class FamilyConversionResult:
    """Result of applying family consistency rules to a code set."""

    converted_codes: List[str]
    conversions: List[Tuple[str, str, str]]  # (original, new, reason)
    warnings: List[str]


def apply_addon_family_rules(codes: List[str]) -> FamilyConversionResult:
    """Apply add-on family consistency rules to a list of CPT codes.

    When a "family primary" code is present, related initial codes should
    convert to their add-on equivalents.

    Example:
        Input:  ["31631", "31636", "31636"]
        Output: ["31631", "+31637", "+31637"]

    Args:
        codes: List of CPT code strings.

    Returns:
        FamilyConversionResult with converted codes and conversion details.
    """
    code_set = set(codes)
    converted: List[str] = []
    conversions: List[Tuple[str, str, str]] = []
    warnings: List[str] = []

    for code in codes:
        new_code = code
        conversion_reason = None

        for family_name, family_config in ADDON_FAMILY_CONVERSIONS.items():
            family_primaries: Set[str] = family_config["family_primaries"]  # type: ignore
            initial_code: str = family_config["initial_code"]  # type: ignore
            addon_code: str = family_config["addon_code"]  # type: ignore

            # Check if this code should be converted
            if code == initial_code:
                # Check if any family primary is present
                primaries_present = family_primaries & code_set
                if primaries_present:
                    new_code = addon_code
                    conversion_reason = (
                        f"Converted {code} to {addon_code}: "
                        f"family primary {', '.join(sorted(primaries_present))} present"
                    )
                    conversions.append((code, addon_code, conversion_reason))
                    break

        converted.append(new_code)

    return FamilyConversionResult(
        converted_codes=converted,
        conversions=conversions,
        warnings=warnings,
    )


@dataclass
class BundlingResult:
    """Result of applying bundling rules to a code set."""

    kept_codes: List[str]
    removed_codes: List[str]
    bundle_reasons: List[Tuple[str, str, str]]  # (primary, removed, reason)


def apply_ebus_aspiration_bundles(codes: List[str]) -> BundlingResult:
    """Apply EBUS-Aspiration bundling rules.

    When EBUS-TBNA codes (31652/31653) are present, aspiration codes (31645/31646)
    should be bundled (removed) as the aspiration is inherent to the EBUS procedure.

    Args:
        codes: List of CPT code strings.

    Returns:
        BundlingResult with kept codes, removed codes, and reasons.
    """
    code_set = set(codes)
    kept: List[str] = []
    removed: List[str] = []
    reasons: List[Tuple[str, str, str]] = []

    for code in codes:
        should_remove = False
        removal_reason = ""

        for (primary, secondary), reason in EBUS_ASPIRATION_BUNDLES.items():
            if code == secondary and primary in code_set:
                should_remove = True
                removal_reason = reason
                reasons.append((primary, code, reason))
                break

        if should_remove:
            removed.append(code)
        else:
            kept.append(code)

    return BundlingResult(
        kept_codes=kept,
        removed_codes=removed,
        bundle_reasons=reasons,
    )


def apply_thoracentesis_ipc_bundles(codes: List[str]) -> BundlingResult:
    """Apply Thoracentesis-IPC bundling rules.

    When tunneled pleural catheter placement (32550) is present, thoracentesis
    codes (32554/32555) should be bundled (removed) as the thoracentesis is
    inherent to the IPC placement procedure.

    Args:
        codes: List of CPT code strings.

    Returns:
        BundlingResult with kept codes, removed codes, and reasons.
    """
    code_set = set(codes)
    kept: List[str] = []
    removed: List[str] = []
    reasons: List[Tuple[str, str, str]] = []

    for code in codes:
        should_remove = False

        for (primary, secondary), reason in THORACENTESIS_IPC_BUNDLES.items():
            if code == secondary and primary in code_set:
                should_remove = True
                reasons.append((primary, code, reason))
                break

        if should_remove:
            removed.append(code)
        else:
            kept.append(code)

    return BundlingResult(
        kept_codes=kept,
        removed_codes=removed,
        bundle_reasons=reasons,
    )


def apply_tumor_bundles(codes: List[str]) -> BundlingResult:
    """Apply tumor excision/destruction bundling rules.

    When tumor destruction (31641) is present, tumor excision (31640)
    should be bundled (removed) for the same lesion as destruction
    is the higher-value procedure.

    Args:
        codes: List of CPT code strings.

    Returns:
        BundlingResult with kept codes, removed codes, and reasons.
    """
    code_set = set(codes)
    kept: List[str] = []
    removed: List[str] = []
    reasons: List[Tuple[str, str, str]] = []

    for code in codes:
        should_remove = False

        for (primary, secondary), reason in TUMOR_BUNDLES.items():
            if code == secondary and primary in code_set:
                should_remove = True
                reasons.append((primary, code, reason))
                break

        if should_remove:
            removed.append(code)
        else:
            kept.append(code)

    return BundlingResult(
        kept_codes=kept,
        removed_codes=removed,
        bundle_reasons=reasons,
    )


def apply_all_ncci_bundles(codes: List[str]) -> BundlingResult:
    """Apply all NCCI bundling rules in sequence.

    This function applies:
    1. EBUS-Aspiration bundles (31645/31646 into 31652/31653)
    2. Thoracentesis-IPC bundles (32554/32555 into 32550)
    3. Tumor excision/destruction bundles (31640 into 31641)

    Args:
        codes: List of CPT code strings.

    Returns:
        BundlingResult with kept codes, removed codes, and all reasons.
    """
    all_removed: List[str] = []
    all_reasons: List[Tuple[str, str, str]] = []
    current_codes = list(codes)

    # Apply EBUS-Aspiration bundles
    ebus_result = apply_ebus_aspiration_bundles(current_codes)
    all_removed.extend(ebus_result.removed_codes)
    all_reasons.extend(ebus_result.bundle_reasons)
    current_codes = ebus_result.kept_codes

    # Apply Thoracentesis-IPC bundles
    thoracentesis_result = apply_thoracentesis_ipc_bundles(current_codes)
    all_removed.extend(thoracentesis_result.removed_codes)
    all_reasons.extend(thoracentesis_result.bundle_reasons)
    current_codes = thoracentesis_result.kept_codes

    # Apply Tumor bundles
    tumor_result = apply_tumor_bundles(current_codes)
    all_removed.extend(tumor_result.removed_codes)
    all_reasons.extend(tumor_result.bundle_reasons)
    current_codes = tumor_result.kept_codes

    return BundlingResult(
        kept_codes=current_codes,
        removed_codes=all_removed,
        bundle_reasons=all_reasons,
    )


def count_sampled_ebus_stations(text: str) -> int:
    """Count EBUS stations that were actually SAMPLED (not just inspected).

    This function differentiates between stations that were merely visualized/inspected
    versus those that were actively sampled (biopsy, FNA, needle aspiration).

    Sampling indicators:
        - "sampled", "biopsy", "biopsied", "FNA", "needle aspiration"
        - "passes", "pass", "aspirate", "aspirated"

    Inspection-only indicators (should NOT count):
        - "inspected", "assessed", "visualized", "normal appearing"
        - "no sampling", "not sampled"
        - Lists under "Sites Inspected:" headers

    Args:
        text: The procedure note text.

    Returns:
        Number of stations that were actually sampled.
    """
    import re

    text_lower = text.lower()

    # Station pattern - must have "station" prefix to avoid false matches
    # Matches: "station 7", "station 11L", "station 4R"
    station_with_prefix = re.compile(
        r"station\s*(\d{1,2}[LRlr]?)",
        re.IGNORECASE
    )

    # Sampling context patterns - must be near station mention
    sampling_keywords = [
        r"sampl(?:ed|ing)",
        r"biops(?:y|ied|ies)",
        r"fna\b",
        r"fine\s*needle\s*aspir",
        r"needle\s*aspir",
        r"tbna\b",
        r"\d+\s*pass(?:es)?",  # "3 passes", "4 passes"
        r"aspirat(?:ed|e|ion)",
        r"cytology",
        r"specimen",
        r"rose\s+(?:adequate|positive|negative)",  # ROSE assessment indicates sampling
    ]
    sampling_pattern = re.compile(
        r"|".join(sampling_keywords),
        re.IGNORECASE
    )

    # Inspection-only patterns (negates sampling in that context)
    inspection_headers = [
        r"sites?\s+inspect",
        r"lymph\s+nodes?\s+inspect",
        r"stations?\s+inspect",
        r"inspect(?:ed|ing)\s*:",
    ]
    inspection_header_pattern = re.compile(
        r"|".join(inspection_headers),
        re.IGNORECASE
    )

    inspection_negators = [
        r"inspect(?:ed|ing)",
        r"assess(?:ed|ing)",
        r"visualiz(?:ed|ing)",
        r"normal\s*appear",
        r"unremarkable",
        r"no\s+(?:mass|lesion|abnormal)",
        r"not\s+sampl",
        r"no\s+sampl",
        r"without\s+sampl",
    ]
    inspection_negator_pattern = re.compile(
        r"|".join(inspection_negators),
        re.IGNORECASE
    )

    sampled_stations: Set[str] = set()

    # Split text into sentences/lines for context analysis
    # Use more aggressive splitting to separate inspection lists from sampling statements
    lines = re.split(r'[.!?\n]+', text_lower)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are inspection headers or lists
        if inspection_header_pattern.search(line):
            continue

        # Skip lines with inspection-only language and NO sampling language
        has_inspection_negator = bool(inspection_negator_pattern.search(line))
        has_sampling_context = bool(sampling_pattern.search(line))

        if has_inspection_negator and not has_sampling_context:
            continue

        # Only count stations that appear WITH sampling context
        if has_sampling_context:
            # Extract station numbers from this line
            for match in station_with_prefix.finditer(line):
                station = match.group(1).upper()
                sampled_stations.add(station)

    return len(sampled_stations)


def determine_ebus_code(sampled_station_count: int) -> str:
    """Determine the appropriate EBUS code based on sampled station count.

    Args:
        sampled_station_count: Number of stations that were actually sampled.

    Returns:
        "31652" for 1-2 stations, "31653" for 3+ stations, or "" for 0.
    """
    if sampled_station_count == 0:
        return ""
    elif sampled_station_count <= 2:
        return "31652"  # 1-2 nodal stations
    else:
        return "31653"  # 3+ nodal stations
