"""Derivation and validation helpers for granular registry data.

This module contains the non-model logic that previously lived in
`app.registry.schema_granular` and now operates on types defined in
`app.registry.schema.granular_models`.
"""

from __future__ import annotations

import re
from typing import Any

from app.registry.schema.granular_models import EBUSStationDetail, EnhancedRegistryGranularData


def validate_ebus_consistency(
    stations_detail: list[EBUSStationDetail] | None,
    stations_sampled: list[str] | None,
) -> list[str]:
    """Validate that EBUS station detail matches sampled stations list.

    Returns list of validation error messages (empty if valid).
    """
    errors = []

    if not stations_detail and not stations_sampled:
        return errors

    if stations_detail and stations_sampled:
        detail_stations = {s.station for s in stations_detail if s.sampled is True or s.sampled is None}
        sampled_set = set(stations_sampled)

        missing = sampled_set - detail_stations
        extra = detail_stations - sampled_set

        if missing:
            errors.append(f"Stations in sampled list but missing detail: {sorted(missing)}")
        if extra:
            errors.append(f"Stations with detail but not in sampled list: {sorted(extra)}")

    elif stations_sampled and not stations_detail:
        errors.append("stations_sampled populated but no stations_detail provided")

    return errors


def derive_aggregate_fields(granular: EnhancedRegistryGranularData) -> dict[str, Any]:
    """Derive aggregate fields from granular data for backward compatibility.

    Returns a dict of aggregate fields that can be merged into the main registry record.
    """
    derived = {}

    # Derive linear_ebus_stations from detail
    if granular.linear_ebus_stations_detail:
        derived["linear_ebus_stations"] = [
            s.station for s in granular.linear_ebus_stations_detail if s.sampled is True or s.sampled is None
        ]

        # Count total passes
        total_passes = sum(s.number_of_passes or 0 for s in granular.linear_ebus_stations_detail)
        if total_passes:
            derived["ebus_total_passes"] = total_passes

        # Get first ROSE result (for legacy compatibility)
        rose_results = [s.rose_result for s in granular.linear_ebus_stations_detail if s.rose_result]
        if rose_results:
            derived["ebus_rose_result"] = rose_results[0]

    # Derive nav fields
    if granular.navigation_targets:
        derived["nav_targets_count"] = len(granular.navigation_targets)
        confirmed = sum(1 for t in granular.navigation_targets if t.tool_in_lesion_confirmed)
        derived["nav_til_confirmed_count"] = confirmed

    # Derive BLVR fields
    if granular.blvr_valve_placements:
        derived["blvr_number_of_valves"] = len(granular.blvr_valve_placements)
        lobes = set(v.target_lobe for v in granular.blvr_valve_placements)
        if len(lobes) == 1:
            derived["blvr_target_lobe"] = list(lobes)[0]

    # Derive cryobiopsy specimen count
    if granular.cryobiopsy_sites:
        total = sum(s.number_of_biopsies or 0 for s in granular.cryobiopsy_sites)
        if total:
            derived["cryo_specimens_count"] = total

    return derived


def derive_procedures_from_granular(
    granular_data: dict[str, Any] | None,
    existing_procedures: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Derive top-level procedures_performed fields from granular data.

    This function ensures that granular data is reflected in the top-level
    procedures_performed structure. It also generates validation warnings
    for inconsistencies.

    Args:
        granular_data: The granular_data dict from the registry record
        existing_procedures: The existing procedures_performed dict

    Returns:
        Tuple of (updated procedures_performed dict, list of validation warnings)
    """
    if not granular_data:
        return existing_procedures or {}, []

    procedures = dict(existing_procedures) if existing_procedures else {}
    warnings: list[str] = []

    def _normalize_sampling_tool(tool: str) -> str | None:
        valid_tools = {"Needle", "Forceps", "Brush", "Cryoprobe", "NeedleInNeedle"}

        s = str(tool).strip()
        if not s:
            return None
        s_lower = s.lower()

        # Accept already-valid enum values (case-insensitive)
        for valid in valid_tools:
            if s_lower == valid.lower():
                return valid

        # Map common variations to schema enums
        token = "".join(ch for ch in s_lower if ch.isalnum())
        if token in {"needleinneedle", "nin"} or "needle-in-needle" in s_lower or "needle in needle" in s_lower:
            return "NeedleInNeedle"
        if "tbna" in s_lower or "needle" in s_lower:
            return "Needle"
        if "forceps" in s_lower or "forcep" in s_lower:
            return "Forceps"
        if "brush" in s_lower:
            return "Brush"
        if "cryo" in s_lower:
            return "Cryoprobe"

        # Unknown / invalid tools (e.g., "BAL") must not propagate into the strict enum field.
        return None

    def _extract_station_tokens(text: str) -> list[str]:
        """Extract IASLC station tokens like 4R, 7, 11L from free text."""
        import re

        matches = re.findall(
            r"\b(2R|2L|3p|4R|4L|7|10R|10L|11R|11L|12R|12L)\b", text, flags=re.IGNORECASE
        )
        normalized: list[str] = []
        for m in matches:
            token = m.upper()
            if token not in normalized:
                normalized.append(token)
        return normalized

    # ==========================================================================
    # 1. Derive transbronchial_cryobiopsy from cryobiopsy_sites
    # ==========================================================================
    cryobiopsy_sites = granular_data.get("cryobiopsy_sites", [])
    if cryobiopsy_sites:
        cryo = procedures.get("transbronchial_cryobiopsy") or {}
        if not cryo.get("performed"):
            cryo["performed"] = True

        # Sum biopsies across all sites
        total_biopsies = sum(site.get("number_of_biopsies", 0) or 0 for site in cryobiopsy_sites)
        if total_biopsies and cryo.get("number_of_samples") in (None, "", 0):
            cryo["number_of_samples"] = total_biopsies

        # Get probe size (use first site's if uniform)
        probe_sizes = [site.get("probe_size_mm") for site in cryobiopsy_sites if site.get("probe_size_mm")]
        if probe_sizes and cryo.get("probe_size_mm") in (None, ""):
            cryo["probe_size_mm"] = probe_sizes[0]

        # Get freeze time (use first site's)
        freeze_times = [
            site.get("freeze_time_seconds") for site in cryobiopsy_sites if site.get("freeze_time_seconds")
        ]
        if freeze_times and cryo.get("freeze_time_seconds") in (None, ""):
            cryo["freeze_time_seconds"] = freeze_times[0]

        # Get locations
        locations = [
            f"{site.get('lobe', '')} {site.get('segment', '')}".strip()
            for site in cryobiopsy_sites
            if site.get("lobe")
        ]
        if locations and not (cryo.get("locations_biopsied") or []):
            cryo["locations_biopsied"] = locations

        procedures["transbronchial_cryobiopsy"] = cryo

        # Clear incorrect transbronchial_biopsy.forceps_type = "Cryoprobe"
        tbbx = procedures.get("transbronchial_biopsy")
        if tbbx and tbbx.get("forceps_type") == "Cryoprobe":
            tbbx["forceps_type"] = None

    # ==========================================================================
    # 2. Derive radial_ebus.performed from navigation_targets
    # ==========================================================================
    navigation_targets = granular_data.get("navigation_targets", [])
    if navigation_targets:
        # Check if any target used radial EBUS
        any_rebus = any(target.get("rebus_used") or target.get("rebus_view") for target in navigation_targets)

        radial_ebus = procedures.get("radial_ebus") or {}
        if any_rebus:
            if not radial_ebus.get("performed"):
                radial_ebus["performed"] = True
            if not radial_ebus.get("probe_position"):
                # Get probe position from first target with a view
                for target in navigation_targets:
                    if target.get("rebus_view"):
                        radial_ebus["probe_position"] = target["rebus_view"]
                        break
            procedures["radial_ebus"] = radial_ebus
        elif radial_ebus.get("probe_position") and not radial_ebus.get("performed"):
            # probe_position is set but performed is not - fix it
            radial_ebus["performed"] = True
            procedures["radial_ebus"] = radial_ebus
            warnings.append("radial_ebus.probe_position was set but performed was null - auto-set performed=true")

    # ==========================================================================
    # 3. Derive linear_ebus.stations_sampled from linear_ebus_stations_detail
    # ==========================================================================
    linear_ebus_detail = granular_data.get("linear_ebus_stations_detail", [])
    if linear_ebus_detail:
        linear_ebus = procedures.get("linear_ebus") or {}

        # Get sampled stations
        sampled_stations = [
            station.get("station")
            for station in linear_ebus_detail
            if station.get("sampled") is True or station.get("sampled") is None
        ]

        if sampled_stations:
            existing_sampled = linear_ebus.get("stations_sampled") or []
            if not isinstance(existing_sampled, list):
                existing_sampled = []
            existing_norm = [str(s).strip() for s in existing_sampled if str(s).strip()]
            merged = list(existing_norm)
            for st in sampled_stations:
                if not st:
                    continue
                st_norm = str(st).strip()
                if st_norm and st_norm not in merged:
                    merged.append(st_norm)
            if merged:
                linear_ebus["stations_sampled"] = merged

        if not linear_ebus.get("performed"):
            linear_ebus["performed"] = True

        procedures["linear_ebus"] = linear_ebus

    # ==========================================================================
    # 4. Derive BAL, brushings from specimens_collected
    # ==========================================================================
    specimens = granular_data.get("specimens_collected", [])
    if specimens:
        # EBUS-TBNA specimens can serve as backup evidence for linear_ebus
        ebus_tbna_specimens = [s for s in specimens if s.get("source_procedure") == "EBUS-TBNA"]
        if ebus_tbna_specimens:
            linear_ebus = procedures.get("linear_ebus") or {}
            if not linear_ebus.get("performed"):
                linear_ebus["performed"] = True
            if not linear_ebus.get("stations_sampled"):
                stations: list[str] = []
                for spec in ebus_tbna_specimens:
                    loc = spec.get("source_location") or ""
                    stations.extend(_extract_station_tokens(str(loc)))
                if stations:
                    # preserve order while de-duping
                    deduped: list[str] = []
                    for st in stations:
                        if st not in deduped:
                            deduped.append(st)
                    linear_ebus["stations_sampled"] = deduped
            procedures["linear_ebus"] = linear_ebus

        # BAL
        bal_specimens = [
            s for s in specimens if s.get("source_procedure") in ("BAL", "Bronchoalveolar lavage", "mini-BAL")
        ]
        if bal_specimens:
            bal = procedures.get("bal") or {}
            if not bal.get("performed"):
                bal["performed"] = True
                # Get location from first BAL specimen
                for spec in bal_specimens:
                    loc = spec.get("source_location")
                    if loc and loc != "BAL":
                        bal["location"] = loc
                        break
                procedures["bal"] = bal

        # Bronchial wash
        wash_specimens = [s for s in specimens if s.get("source_procedure") == "Bronchial wash"]
        if wash_specimens:
            bronchial_wash = procedures.get("bronchial_wash") or {}
            if not bronchial_wash.get("performed"):
                bronchial_wash["performed"] = True
            if not bronchial_wash.get("location"):
                bronchial_wash["location"] = wash_specimens[0].get("source_location")
            procedures["bronchial_wash"] = bronchial_wash

        # Brushings
        brushing_specimens = [s for s in specimens if s.get("source_procedure") == "Brushing"]
        if brushing_specimens:
            brushings = procedures.get("brushings") or {}
            if not brushings.get("performed"):
                brushings["performed"] = True
                # Get locations
                locations = [s.get("source_location") for s in brushing_specimens if s.get("source_location")]
                if locations:
                    brushings["locations"] = locations
                # Count samples
                total_samples = sum(s.get("specimen_count", 1) or 1 for s in brushing_specimens)
                brushings["number_of_samples"] = total_samples
                procedures["brushings"] = brushings

        # Endobronchial biopsy
        ebx_specimens = [s for s in specimens if s.get("source_procedure") == "Endobronchial biopsy"]
        if ebx_specimens:
            ebx = procedures.get("endobronchial_biopsy") or {}
            if not ebx.get("performed"):
                ebx["performed"] = True
            if not ebx.get("locations"):
                ebx_locations = [s.get("source_location") for s in ebx_specimens if s.get("source_location")]
                if ebx_locations:
                    ebx["locations"] = ebx_locations
            if not ebx.get("number_of_samples"):
                ebx_samples = sum((s.get("specimen_count") or 0) for s in ebx_specimens)
                if ebx_samples:
                    ebx["number_of_samples"] = ebx_samples
            procedures["endobronchial_biopsy"] = ebx

        # Transbronchial biopsy (including navigation-guided biopsy specimens)
        tbbx_specimens = [
            s for s in specimens if s.get("source_procedure") in ("Transbronchial biopsy", "Navigation biopsy")
        ]
        if tbbx_specimens:
            tbbx = procedures.get("transbronchial_biopsy") or {}
            if not tbbx.get("performed"):
                tbbx["performed"] = True
            if not tbbx.get("locations"):
                tbbx_locations = [s.get("source_location") for s in tbbx_specimens if s.get("source_location")]
                if tbbx_locations:
                    tbbx["locations"] = tbbx_locations
            if not tbbx.get("number_of_samples"):
                tbbx_samples = sum((s.get("specimen_count") or 0) for s in tbbx_specimens)
                if tbbx_samples:
                    tbbx["number_of_samples"] = tbbx_samples
            procedures["transbronchial_biopsy"] = tbbx

        # Navigation biopsy specimens imply navigation was performed
        if any(s.get("source_procedure") == "Navigation biopsy" for s in specimens):
            nav_bronch = procedures.get("navigational_bronchoscopy") or {}
            if not nav_bronch.get("performed"):
                nav_bronch["performed"] = True
            procedures["navigational_bronchoscopy"] = nav_bronch

    # ==========================================================================
    # 5. Derive navigational_bronchoscopy.sampling_tools_used from navigation_targets
    # ==========================================================================
    if navigation_targets:
        def _nav_primary_location_text() -> str | None:
            for t in navigation_targets:
                loc = t.get("target_location_text")
                if not isinstance(loc, str):
                    continue
                loc_clean = loc.strip()
                lower = loc_clean.lower()
                if (
                    "||" in lower
                    or lower.startswith("pt:")
                    or lower.startswith("patient:")
                    or re.search(r"(?i)\b(?:mrn|dob)\b\s*:", loc_clean)
                ):
                    continue
                if loc_clean:
                    return loc_clean
            return None

        def _nav_primary_location_parts() -> dict[str, str] | None:
            bronchus_re = re.compile(r"\b([LR]B\d{1,2})\b", re.IGNORECASE)
            for t in navigation_targets:
                loc = t.get("target_location_text")
                if not isinstance(loc, str):
                    continue
                loc_clean = loc.strip()
                lower = loc_clean.lower()
                if (
                    "||" in lower
                    or lower.startswith("pt:")
                    or lower.startswith("patient:")
                    or re.search(r"(?i)\b(?:mrn|dob)\b\s*:", loc_clean)
                ):
                    continue

                parts: dict[str, str] = {}
                lobe = t.get("target_lobe")
                if isinstance(lobe, str) and lobe.strip():
                    parts["lobe"] = lobe.strip()
                segment = t.get("target_segment")
                if isinstance(segment, str) and segment.strip():
                    parts["segment"] = segment.strip()
                match = bronchus_re.search(loc_clean)
                if match:
                    parts["bronchus"] = match.group(1).upper()
                if parts:
                    return parts
            return None

        def _shorten_segment(segment: str) -> str:
            value = (segment or "").strip()
            value = re.sub(r"(?i)\bsegment\b", "", value).strip()
            value = re.sub(r"\s+", " ", value).strip(" -")
            return value

        def _nav_target_lobe_label(target: dict[str, Any]) -> str | None:
            lobe = target.get("target_lobe")
            if isinstance(lobe, str) and lobe.strip():
                return lobe.strip()
            loc = str(target.get("target_location_text") or "")
            for token in ("RUL", "RML", "RLL", "LUL", "LLL"):
                if re.search(rf"(?i)\b{token}\b", loc):
                    return token
            if re.search(r"(?i)\blingula\b", loc):
                return "Lingula"
            return None

        nav_bronch = procedures.get("navigational_bronchoscopy") or {}
        if not nav_bronch.get("performed"):
            nav_bronch["performed"] = True
        existing_tools = nav_bronch.get("sampling_tools_used") or []

        # Collect all tools from all targets and union with any existing list
        all_tools: set[str] = set()

        for t in existing_tools:
            if t:
                norm = _normalize_sampling_tool(t)
                if norm:
                    all_tools.add(norm)

        for target in navigation_targets:
            tools = target.get("sampling_tools_used", []) or []
            for tool in tools:
                if tool:
                    norm = _normalize_sampling_tool(tool)
                    if norm:
                        all_tools.add(norm)

        if all_tools:
            nav_bronch["sampling_tools_used"] = sorted(all_tools)
        else:
            nav_bronch.pop("sampling_tools_used", None)

        procedures["navigational_bronchoscopy"] = nav_bronch

        # Up-propagate needle sampling / biopsy / brushings from target-level sampling evidence
        peripheral_tbna_performed = (procedures.get("peripheral_tbna") or {}).get("performed") is True
        tbbx_performed = (procedures.get("transbronchial_biopsy") or {}).get("performed") is True
        brushings_performed = (procedures.get("brushings") or {}).get("performed") is True
        cryo_performed = (procedures.get("transbronchial_cryobiopsy") or {}).get("performed") is True

        has_needle = peripheral_tbna_performed or "Needle" in all_tools or any(
            (t.get("number_of_needle_passes") or 0) > 0 for t in navigation_targets
        )
        has_forceps = tbbx_performed or "Forceps" in all_tools or any(
            (t.get("number_of_forceps_biopsies") or 0) > 0 for t in navigation_targets
        )
        has_brush = brushings_performed or "Brush" in all_tools
        has_cryo = cryo_performed or "Cryoprobe" in all_tools or any(
            (t.get("number_of_cryo_biopsies") or 0) > 0 for t in navigation_targets
        )

        if has_needle:
            peripheral_tbna = procedures.get("peripheral_tbna") or {}
            if not peripheral_tbna.get("performed"):
                peripheral_tbna["performed"] = True

            existing_targets = peripheral_tbna.get("targets_sampled") or []
            is_placeholder = len(existing_targets) == 1 and str(existing_targets[0]).strip().lower() in {"lung mass"}
            targets = [
                t.get("target_location_text")
                for t in navigation_targets
                if t.get("target_location_text")
                and (
                    (t.get("number_of_needle_passes") or 0) > 0
                    or any("needle" in str(x).lower() for x in (t.get("sampling_tools_used") or ()))
                )
            ]
            if not targets:
                targets = [t.get("target_location_text") for t in navigation_targets if t.get("target_location_text")]

            if not existing_targets or is_placeholder:
                peripheral_tbna["targets_sampled"] = targets or ["Lung Mass"]
            elif targets:
                existing_clean = [str(x) for x in existing_targets if str(x).strip()]
                if len(existing_clean) == 1 and len(targets) > 1:
                    existing_lower = existing_clean[0].strip().lower()
                    if existing_lower and any(
                        existing_lower in str(candidate).strip().lower() for candidate in targets if candidate
                    ):
                        peripheral_tbna["targets_sampled"] = targets
                    else:
                        merged = list(existing_clean)
                        for candidate in targets:
                            if not candidate:
                                continue
                            if candidate not in merged:
                                merged.append(candidate)
                        if merged != existing_targets:
                            peripheral_tbna["targets_sampled"] = merged
                else:
                    merged = list(existing_clean)
                    for candidate in targets:
                        if not candidate:
                            continue
                        if candidate not in merged:
                            merged.append(candidate)
                    if merged != existing_targets:
                        peripheral_tbna["targets_sampled"] = merged

            procedures["peripheral_tbna"] = peripheral_tbna
        else:
            # If peripheral TBNA is already asserted elsewhere, backfill targets from
            # navigation targets even when tool counts are missing (common in LLM-only runs).
            peripheral_tbna = procedures.get("peripheral_tbna") or {}
            if peripheral_tbna.get("performed") is True:
                existing_targets = peripheral_tbna.get("targets_sampled") or []
                is_placeholder = (
                    len(existing_targets) == 1 and str(existing_targets[0]).strip().lower() in {"lung mass"}
                )
                if not existing_targets or is_placeholder:
                    primary_loc = _nav_primary_location_text()
                    if primary_loc:
                        peripheral_tbna["targets_sampled"] = [primary_loc]
                        procedures["peripheral_tbna"] = peripheral_tbna
                        warnings.append(
                            f"BACKFILL: Assigned target '{primary_loc}' to peripheral_tbna.targets_sampled"
                        )
                    else:
                        primary_parts = _nav_primary_location_parts()
                        if primary_parts and primary_parts.get("lobe"):
                            lobe = primary_parts["lobe"]
                            peripheral_tbna["targets_sampled"] = [lobe]
                            procedures["peripheral_tbna"] = peripheral_tbna
                            warnings.append(
                                f"BACKFILL: Assigned target '{lobe}' to peripheral_tbna.targets_sampled"
                            )

        if has_forceps:
            tbbx = procedures.get("transbronchial_biopsy") or {}
            if not tbbx.get("performed"):
                tbbx["performed"] = True
            if not tbbx.get("locations"):
                tbbx_locations = [
                    (
                        f"{_nav_target_lobe_label(t)} {_shorten_segment(str(t.get('target_segment') or ''))}".strip()
                        if _nav_target_lobe_label(t) and str(t.get("target_segment") or "").strip()
                        else (_nav_target_lobe_label(t) or t.get("target_location_text"))
                    )
                    for t in navigation_targets
                    if (_nav_target_lobe_label(t) or t.get("target_location_text"))
                    and (
                        (t.get("number_of_forceps_biopsies") or 0) > 0
                        or any("forceps" in str(x).lower() for x in (t.get("sampling_tools_used") or ()))
                    )
                ]
                if not tbbx_locations:
                    primary_loc = _nav_primary_location_text()
                    primary_parts = _nav_primary_location_parts()
                    loc_label = None
                    if primary_parts and primary_parts.get("lobe") and primary_parts.get("segment"):
                        loc_label = f"{primary_parts['lobe']} {_shorten_segment(primary_parts['segment'])}".strip()
                    elif primary_parts and primary_parts.get("lobe"):
                        loc_label = primary_parts.get("lobe")
                    elif primary_loc:
                        loc_label = primary_loc
                    if loc_label:
                        tbbx_locations = [loc_label]
                        warnings.append(
                            f"BACKFILL: Assigned target '{loc_label}' to transbronchial_biopsy.locations"
                        )
                if tbbx_locations:
                    tbbx["locations"] = tbbx_locations
            if not tbbx.get("number_of_samples"):
                total_biopsies = sum((t.get("number_of_forceps_biopsies") or 0) for t in navigation_targets)
                if total_biopsies:
                    tbbx["number_of_samples"] = total_biopsies
            procedures["transbronchial_biopsy"] = tbbx
        else:
            # If transbronchial biopsy is asserted elsewhere, backfill locations from navigation
            # targets when missing (common when the note introduces a target once and later
            # biopsy sentences omit the lobe/segment).
            tbbx = procedures.get("transbronchial_biopsy") or {}
            if tbbx.get("performed") is True and not (tbbx.get("locations") or []):
                primary_loc = _nav_primary_location_text()
                primary_parts = _nav_primary_location_parts()
                loc_label = None
                if primary_parts and primary_parts.get("lobe") and primary_parts.get("segment"):
                    loc_label = f"{primary_parts['lobe']} {_shorten_segment(primary_parts['segment'])}".strip()
                elif primary_parts and primary_parts.get("lobe"):
                    loc_label = primary_parts.get("lobe")
                elif primary_loc:
                    loc_label = primary_loc
                if loc_label:
                    tbbx["locations"] = [loc_label]
                    procedures["transbronchial_biopsy"] = tbbx
                    warnings.append(
                        f"BACKFILL: Assigned target '{loc_label}' to transbronchial_biopsy.locations"
                    )

        if has_brush:
            brushings = procedures.get("brushings") or {}
            if not brushings.get("performed"):
                brushings["performed"] = True
            bronchus_re = re.compile(r"\b([LR]B\d{1,2})\b", re.IGNORECASE)
            desired_tokens: list[str] = []
            seen_tokens: set[str] = set()

            for target in navigation_targets:
                lobe = _nav_target_lobe_label(target)
                if lobe and lobe not in seen_tokens:
                    desired_tokens.append(lobe)
                    seen_tokens.add(lobe)

                loc = target.get("target_location_text")
                if not isinstance(loc, str) or not loc.strip():
                    continue
                match = bronchus_re.search(loc)
                if not match:
                    continue
                bronchus = match.group(1).upper()
                if bronchus and bronchus not in seen_tokens:
                    desired_tokens.append(bronchus)
                    seen_tokens.add(bronchus)

            existing_locations = brushings.get("locations") or []
            existing_tokens = [str(x) for x in existing_locations if isinstance(x, str) and x.strip()]
            existing_all_tokens = existing_tokens and all((" " not in x) and (len(x) <= 12) for x in existing_tokens)

            if existing_tokens and existing_all_tokens and desired_tokens:
                merged = list(existing_tokens)
                for token in desired_tokens:
                    if token not in merged:
                        merged.append(token)
                if merged != existing_locations:
                    brushings["locations"] = merged
            elif not existing_locations:
                if desired_tokens:
                    brushings["locations"] = desired_tokens
                    warnings.append(f"BACKFILL: Assigned target '{desired_tokens}' to brushings.locations")
                else:
                    primary_loc = _nav_primary_location_text()
                    if primary_loc:
                        brushings["locations"] = [primary_loc]
                        warnings.append(f"BACKFILL: Assigned target '{primary_loc}' to brushings.locations")
            procedures["brushings"] = brushings
        else:
            # If brushings are asserted elsewhere, backfill locations from navigation targets when missing.
            brushings = procedures.get("brushings") or {}
            if brushings.get("performed") is True and not (brushings.get("locations") or []):
                primary_loc = _nav_primary_location_text()
                primary_parts = _nav_primary_location_parts()
                tokens = []
                if primary_parts:
                    lobe = primary_parts.get("lobe")
                    bronchus = primary_parts.get("bronchus")
                    if lobe:
                        tokens.append(lobe)
                    if bronchus and bronchus not in tokens:
                        tokens.append(bronchus)
                if tokens:
                    brushings["locations"] = tokens
                    procedures["brushings"] = brushings
                    warnings.append(f"BACKFILL: Assigned target '{tokens}' to brushings.locations")
                elif primary_loc:
                    brushings["locations"] = [primary_loc]
                    procedures["brushings"] = brushings
                    warnings.append(f"BACKFILL: Assigned target '{primary_loc}' to brushings.locations")

        if has_cryo:
            cryo = procedures.get("transbronchial_cryobiopsy") or {}
            if not cryo.get("performed"):
                cryo["performed"] = True
            cryo_locations = [
                t.get("target_location_text")
                for t in navigation_targets
                if t.get("target_location_text")
                and (
                    (t.get("number_of_cryo_biopsies") or 0) > 0
                    or any("cryo" in str(x).lower() for x in (t.get("sampling_tools_used") or ()))
                )
            ]
            if not cryo_locations:
                cryo_locations = [t.get("target_location_text") for t in navigation_targets if t.get("target_location_text")]

            existing_locations = cryo.get("locations_biopsied") or []
            if not existing_locations:
                if cryo_locations:
                    cryo["locations_biopsied"] = cryo_locations
            elif cryo_locations:
                existing_clean = [str(x) for x in existing_locations if str(x).strip()]
                if len(existing_clean) == 1 and len(cryo_locations) > 1:
                    existing_lower = existing_clean[0].strip().lower()
                    if existing_lower and any(
                        existing_lower in str(candidate).strip().lower() for candidate in cryo_locations if candidate
                    ):
                        cryo["locations_biopsied"] = cryo_locations
                    else:
                        merged = list(existing_clean)
                        for candidate in cryo_locations:
                            if not candidate:
                                continue
                            if candidate not in merged:
                                merged.append(candidate)
                        if merged != existing_locations:
                            cryo["locations_biopsied"] = merged
                else:
                    merged = list(existing_clean)
                    for candidate in cryo_locations:
                        if not candidate:
                            continue
                        if candidate not in merged:
                            merged.append(candidate)
                    if merged != existing_locations:
                        cryo["locations_biopsied"] = merged

            if cryo.get("number_of_samples") in (None, "", 0):
                total = sum((t.get("number_of_cryo_biopsies") or 0) for t in navigation_targets)
                if total:
                    cryo["number_of_samples"] = total
            procedures["transbronchial_cryobiopsy"] = cryo
        else:
            # If cryobiopsy is asserted elsewhere, backfill biopsy locations from navigation
            # targets even when tool counts are missing (common in LLM-only runs).
            cryo = procedures.get("transbronchial_cryobiopsy") or {}
            if cryo.get("performed") is True and not (cryo.get("locations_biopsied") or []):
                cryo_locations = [
                    t.get("target_location_text") for t in navigation_targets if t.get("target_location_text")
                ]
                if cryo_locations:
                    cryo["locations_biopsied"] = cryo_locations
                    procedures["transbronchial_cryobiopsy"] = cryo

    # ==========================================================================
    # 5.1 Normalize TBNA: keep nodal TBNA separate from peripheral targets
    # ==========================================================================
    tbna = procedures.get("tbna_conventional") or {}
    peripheral_tbna = procedures.get("peripheral_tbna") or {}
    tbna_sites = tbna.get("stations_sampled") or []

    station_token_re = re.compile(
        r"^(?:2R|2L|3P|4R|4L|5|7|8|9|10R|10L|11R(?:S|I)?|11L(?:S|I)?|12R|12L)$",
        re.IGNORECASE,
    )
    has_non_station_site = any(
        (str(site).strip() and not station_token_re.match(str(site).strip()))
        for site in tbna_sites
        if site is not None
    )

    if tbna.get("performed") is True and has_non_station_site:
        # Treat free-text/non-station sites as peripheral TBNA targets.
        if not peripheral_tbna.get("performed"):
            peripheral_tbna["performed"] = True
        if not (peripheral_tbna.get("targets_sampled") or []):
            peripheral_tbna["targets_sampled"] = [str(s) for s in tbna_sites if s]
        procedures["peripheral_tbna"] = peripheral_tbna
        procedures.pop("tbna_conventional", None)

    # If both peripheral TBNA and linear EBUS are present, suppress conventional nodal TBNA
    # unless explicitly supported elsewhere (prevents phantom tbna_conventional alongside EBUS).
    if (
        (procedures.get("peripheral_tbna") or {}).get("performed") is True
        and (procedures.get("linear_ebus") or {}).get("performed") is True
        and (procedures.get("tbna_conventional") or {}).get("performed") is True
    ):
        procedures.pop("tbna_conventional", None)

    # ==========================================================================
    # 6. Derive BLVR performed from blvr_valve_placements
    # ==========================================================================
    blvr_valves = granular_data.get("blvr_valve_placements", [])
    if blvr_valves:
        blvr = procedures.get("blvr") or {}
        if not blvr.get("performed"):
            blvr["performed"] = True
        blvr.setdefault("procedure_type", "Valve placement")
        if not blvr.get("number_of_valves"):
            blvr["number_of_valves"] = len(blvr_valves)
        if not blvr.get("valve_sizes"):
            sizes = [v.get("valve_size") for v in blvr_valves if v.get("valve_size")]
            if sizes:
                blvr["valve_sizes"] = sizes
        if not blvr.get("segments_treated"):
            segments = [v.get("segment") for v in blvr_valves if v.get("segment")]
            if segments:
                blvr["segments_treated"] = segments
        if not blvr.get("target_lobe"):
            lobes = {v.get("target_lobe") for v in blvr_valves if v.get("target_lobe")}
            if len(lobes) == 1:
                blvr["target_lobe"] = next(iter(lobes))
        if not blvr.get("valve_type"):
            valve_types = {v.get("valve_type") for v in blvr_valves if v.get("valve_type")}
            if len(valve_types) == 1:
                blvr["valve_type"] = next(iter(valve_types))
        procedures["blvr"] = blvr

    # ==========================================================================
    # 7. Derive CAO-related performed flags from cao_interventions_detail
    # ==========================================================================
    cao_details = granular_data.get("cao_interventions_detail", [])
    if cao_details:
        modalities: list[str] = []
        stent_any = False
        secretions_drained_any = False
        for detail in cao_details:
            if detail.get("stent_placed_at_site"):
                stent_any = True
            if detail.get("secretions_drained"):
                secretions_drained_any = True
            for app in detail.get("modalities_applied") or []:
                mod = app.get("modality")
                if mod:
                    modalities.append(str(mod).lower())

        if modalities:
            if any("balloon" in m or "dilation" in m for m in modalities):
                airway_dilation = procedures.get("airway_dilation") or {}
                airway_dilation["performed"] = True
                procedures["airway_dilation"] = airway_dilation

            mechanical_method: str | None = None
            if any("cryoextraction" in m for m in modalities):
                mechanical_method = "Cryoextraction"
            elif any("microdebrider" in m for m in modalities):
                mechanical_method = "Microdebrider"
            elif any("rigid coring" in m for m in modalities):
                mechanical_method = "Rigid coring"
            elif any("mechanical debulking" in m for m in modalities):
                mechanical_method = "Forceps debulking"

            if mechanical_method:
                mechanical = procedures.get("mechanical_debulking") or {}
                mechanical["performed"] = True
                if not mechanical.get("method"):
                    mechanical["method"] = mechanical_method
                if not mechanical.get("location"):
                    locations = sorted(
                        {
                            str(detail.get("location")).strip()
                            for detail in cao_details
                            if detail.get("location") and str(detail.get("location")).strip()
                        }
                    )
                    if locations:
                        mechanical["location"] = ", ".join(locations)
                procedures["mechanical_debulking"] = mechanical

            if any(m.startswith("apc") or "electrocautery" in m or "laser" in m for m in modalities):
                thermal_ablation = procedures.get("thermal_ablation") or {}
                thermal_ablation["performed"] = True
                procedures["thermal_ablation"] = thermal_ablation

            if any("cryo" in m for m in modalities):
                cryotherapy = procedures.get("cryotherapy") or {}
                cryotherapy["performed"] = True
                procedures["cryotherapy"] = cryotherapy

            if any("suction" in m or "aspirat" in m for m in modalities) or secretions_drained_any:
                aspiration = procedures.get("therapeutic_aspiration") or {}
                aspiration["performed"] = True
                procedures["therapeutic_aspiration"] = aspiration

        if stent_any:
            stent = procedures.get("airway_stent") or {}
            if not stent.get("performed"):
                stent["performed"] = True
            procedures["airway_stent"] = stent

    # ==========================================================================
    # 8. Derive outcomes.procedure_completed and complications
    # ==========================================================================
    # This is done at a higher level since outcomes is a separate top-level field

    # ==========================================================================
    # Validation warnings for inconsistencies
    # ==========================================================================

    # Check: cryobiopsy_sites present but transbronchial_cryobiopsy not set
    if cryobiopsy_sites:
        cryo_proc = procedures.get("transbronchial_cryobiopsy")
        if not cryo_proc or not cryo_proc.get("performed"):
            warnings.append(
                "granular_data.cryobiopsy_sites is populated but "
                "procedures_performed.transbronchial_cryobiopsy.performed was not set"
            )

    # Check: navigation_target.rebus_used but radial_ebus.performed not set
    if navigation_targets:
        any_rebus = any(target.get("rebus_used") or target.get("rebus_view") for target in navigation_targets)
        radial_ebus = procedures.get("radial_ebus")
        if any_rebus and (not radial_ebus or not radial_ebus.get("performed")):
            warnings.append("navigation_targets has rebus_used=true but radial_ebus.performed was not set")

    # Check: performed=True but required detail missing (e.g., EBUS without stations)
    linear_ebus = procedures.get("linear_ebus") or {}
    if linear_ebus.get("performed") is True and not (linear_ebus.get("stations_sampled") or []):
        warnings.append("procedures_performed.linear_ebus.performed=true but stations_sampled is empty/missing")

    tbna = procedures.get("tbna_conventional") or {}
    if tbna.get("performed") is True and not (tbna.get("stations_sampled") or []):
        warnings.append("procedures_performed.tbna_conventional.performed=true but stations_sampled is empty/missing")

    peripheral_tbna = procedures.get("peripheral_tbna") or {}
    if peripheral_tbna.get("performed") is True and not (peripheral_tbna.get("targets_sampled") or []):
        warnings.append("procedures_performed.peripheral_tbna.performed=true but targets_sampled is empty/missing")

    bronchial_wash = procedures.get("bronchial_wash") or {}
    if bronchial_wash.get("performed") is True and not bronchial_wash.get("location"):
        warnings.append("procedures_performed.bronchial_wash.performed=true but location is missing")

    brushings = procedures.get("brushings") or {}
    if brushings.get("performed") is True and not (brushings.get("locations") or []):
        warnings.append("procedures_performed.brushings.performed=true but locations is empty/missing")

    tbbx = procedures.get("transbronchial_biopsy") or {}
    if tbbx.get("performed") is True and not (tbbx.get("locations") or []):
        warnings.append("procedures_performed.transbronchial_biopsy.performed=true but locations is empty/missing")

    return procedures, warnings


__all__ = ["validate_ebus_consistency", "derive_aggregate_fields", "derive_procedures_from_granular"]
