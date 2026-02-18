"""Deterministic CPT code derivation from RegistryRecord only.

This module is used by the extraction-first pipeline:
  note_text -> RegistryRecord extraction -> deterministic RegistryRecord → CPT

Non-negotiable constraint:
Rules here must accept ONLY RegistryRecord and must not parse raw note text.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.registry.schema import RegistryRecord


_BLVR_PLACEMENT_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"valves?\b[^.\n]{0,80}\b(?:deploy(?:ed|ment)?|insert(?:ed|ion)?|placement|placing|place\b|(?<!well\s)(?<!previously\s)(?<!prior\s)(?<!already\s)placed\b)"
    r"|"
    r"(?:deploy(?:ed|ment)?|insert(?:ed|ion)?|placement|placing|place\b|(?<!well\s)(?<!previously\s)(?<!prior\s)(?<!already\s)placed\b)[^.\n]{0,80}\bvalves?\b"
    r")",
    re.IGNORECASE,
)
_BLVR_REMOVAL_CONTEXT_RE = re.compile(
    r"\bvalve\b[^.\n]{0,80}\b(?:remov|retriev|extract|explant)\w*\b",
    re.IGNORECASE,
)
_BLVR_VALVE_SIZE_HINT_RE = re.compile(
    r"(?is)\b(?:spiration|zephyr)\b.{0,80}\bsize\b",
)
_FIBRINOLYSIS_SUBSEQUENT_TOKEN_RE = re.compile(
    r"\b(?:subsequent|day\s*2|day\s*3|dose\s*#?\s*2|dose\s*#?\s*3|32562)\b",
    re.IGNORECASE,
)
_CHEST_TUBE_INSERTION_DATE_RE = re.compile(
    r"\bdate\s+of\s+(?:the\s+)?chest\s+tube\s+insertion\b",
    re.IGNORECASE,
)
_DATE_TOKEN_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
_IASLC_STATION_TOKEN_RE = re.compile(
    r"^(?:2R|2L|3P|4R|4L|5|7|8|9|10R|10L|11R(?:S|I)?|11L(?:S|I)?|12R|12L)$",
    re.IGNORECASE,
)
_GENERIC_PERIPHERAL_TBNA_TARGET_RE = re.compile(
    r"(?i)^\s*(?:lung|pulmonary)\s+(?:nodule|mass|lesion)s?\s*$",
)
_SEGMENT_TOKEN_RE = re.compile(r"(?i)\b(?:RB|LB)\d{1,2}\b")
_SIDED_LOBE_RE = re.compile(r"(?i)\b(?:right|left)\s+(?:upper|middle|lower)\s+lobe\b")


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _performed(obj: Any) -> bool:
    performed = _get(obj, "performed")
    return performed is True


def _proc(record: RegistryRecord, name: str) -> Any:
    procedures = _get(record, "procedures_performed")
    if procedures is None:
        return None
    if isinstance(procedures, dict):
        return procedures.get(name)
    return _get(procedures, name)


def _pleural(record: RegistryRecord, name: str) -> Any:
    pleural = _get(record, "pleural_procedures")
    if pleural is None:
        return None
    if isinstance(pleural, dict):
        return pleural.get(name)
    return _get(pleural, name)


def _stations_sampled(record: RegistryRecord) -> tuple[list[str], str]:
    linear = _proc(record, "linear_ebus")
    if linear is None:
        return ([], "none")

    qualifying_actions = {"needle_aspiration", "core_biopsy", "forceps_biopsy"}
    node_events = _get(linear, "node_events")
    if isinstance(node_events, (list, tuple)) and node_events:
        sampled: list[str] = []
        for event in node_events:
            action = _get(event, "action")
            if action not in qualifying_actions:
                continue
            station = _get(event, "station")
            if station is None:
                continue
            station_clean = str(station).upper().strip()
            if station_clean:
                sampled.append(station_clean)
        return (sampled, "node_events")

    stations = _get(linear, "stations_sampled")
    if not stations:
        return ([], "none")

    return ([str(s).upper().strip() for s in stations if s], "stations_sampled")


def _navigation_targets(record: RegistryRecord) -> list[Any]:
    granular = _get(record, "granular_data")
    targets = _get(granular, "navigation_targets") if granular is not None else None
    if not targets:
        return []
    return list(targets)


def _fiducial_marker_placed(record: RegistryRecord) -> bool:
    # Check explicit fiducial_placement procedure first
    fiducial_proc = _proc(record, "fiducial_placement")
    if _performed(fiducial_proc):
        return True

    # Check granular navigation targets (fallback)
    for target in _navigation_targets(record):
        placed = _get(target, "fiducial_marker_placed")
        details = _get(target, "fiducial_marker_details")
        if placed is True:
            return True
        if details is not None and str(details).strip():
            return True
    return False


def _stent_action_is_removal(action: Any) -> bool:
    if action is None:
        return False
    text = str(action).strip().lower()
    return bool(text) and ("remov" in text or "retriev" in text or "explant" in text or "extract" in text)


def _stent_action_is_placement(action: Any) -> bool:
    if action is None:
        return False
    text = str(action).strip().lower()
    return bool(text) and ("placement" in text or "revision" in text or "reposition" in text)


def _lobe_tokens(values: list[str]) -> set[str]:
    lobes: set[str] = set()
    for value in values:
        upper = value.upper()
        for token in ("RUL", "RML", "RLL", "LUL", "LLL", "LINGULA"):
            if token in upper:
                lobes.add("Lingula" if token == "LINGULA" else token)
    return lobes


def _is_nodal_target_value(value: str) -> bool:
    upper = value.upper().strip()
    if not upper:
        return False
    if _IASLC_STATION_TOKEN_RE.match(upper):
        return True
    if "STATION" in upper:
        return True
    if re.search(r"\b(?:LYMPH\s+NODE|NODAL|NODE)\b", upper):
        return True
    return False


def _has_distinct_peripheral_tbna_target(target_values: list[str]) -> bool:
    """Require explicit, non-nodal anatomic detail for EBUS+31629 unbundling."""
    for raw_target in target_values:
        target = str(raw_target or "").strip()
        if not target:
            continue
        if _is_nodal_target_value(target):
            continue
        if _GENERIC_PERIPHERAL_TBNA_TARGET_RE.match(target):
            continue

        upper = target.upper()
        has_anatomic_detail = bool(
            _lobe_tokens([target])
            or _SEGMENT_TOKEN_RE.search(target)
            or _SIDED_LOBE_RE.search(target)
            or "SEGMENT" in upper
            or "SUBSEGMENT" in upper
        )
        if has_anatomic_detail:
            return True

    return False


def _airway_site_tokens(value: Any) -> set[str]:
    """Extract coarse airway anatomic tokens from a free-text location string."""
    if value is None:
        return set()
    raw = str(value).strip()
    if not raw:
        return set()
    upper = raw.upper()

    tokens = _lobe_tokens([upper])

    if "TRACHEA" in upper or "TRACHEAL" in upper:
        tokens.add("Trachea")
    if "CARINA" in upper:
        tokens.add("Carina")
    if "BRONCHUS INTERMEDIUS" in upper or re.search(r"\bBI\b", upper):
        tokens.add("BI")
    if "RIGHT MAIN" in upper or re.search(r"\bRMS\b", upper):
        tokens.add("RMS")
    if "LEFT MAIN" in upper or re.search(r"\bLMS\b", upper):
        tokens.add("LMS")

    return tokens


def _normalize_lobe(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in {"RUL", "RML", "RLL", "LUL", "LLL"}:
        return upper
    if upper == "LINGULA":
        return "Lingula"
    return None


def _blvr_valve_lobes(record: RegistryRecord, *, blvr_proc: Any | None) -> set[str]:
    lobes: set[str] = set()
    granular = _get(record, "granular_data")
    placements = _get(granular, "blvr_valve_placements") if granular is not None else None
    if isinstance(placements, (list, tuple)):
        for placement in placements:
            lobe = _normalize_lobe(_get(placement, "target_lobe"))
            if lobe:
                lobes.add(lobe)

    # Fallback to the aggregate blvr.target_lobe when the registry indicates a valve procedure
    if lobes:
        return lobes
    if blvr_proc is None:
        return lobes
    procedure_type = _get(blvr_proc, "procedure_type")
    if procedure_type not in {"Valve placement", "Valve removal"}:
        return lobes
    if not _performed(blvr_proc):
        return lobes
    lobe = _normalize_lobe(_get(blvr_proc, "target_lobe"))
    if lobe:
        lobes.add(lobe)
    return lobes


def _blvr_chartis_lobes(record: RegistryRecord, *, blvr_proc: Any | None) -> set[str]:
    lobes: set[str] = set()
    granular = _get(record, "granular_data")
    measurements = _get(granular, "blvr_chartis_measurements") if granular is not None else None
    if isinstance(measurements, (list, tuple)):
        for measurement in measurements:
            lobe = _normalize_lobe(_get(measurement, "lobe_assessed"))
            if lobe:
                lobes.add(lobe)

    # Fallback to aggregate collateral_ventilation_assessment + target_lobe
    if blvr_proc is None:
        return lobes
    cv = _get(blvr_proc, "collateral_ventilation_assessment")
    cv_text = str(cv).lower() if cv is not None else ""
    if not cv_text.strip():
        return lobes
    # 31634 covers Chartis assessment and other balloon occlusion/endobronchial blocker workflows.
    if not re.search(
        r"(?i)\b(?:chartis|balloon\s+occlusion|serial\s+occlusion|endobronchial\s+blocker|uniblocker|arndt|ardnt|fogarty)\b",
        cv_text,
    ):
        return lobes
    lobe = _normalize_lobe(_get(blvr_proc, "target_lobe"))
    if lobe:
        lobes.add(lobe)
    return lobes


def _evidence_text_for_prefixes(record: RegistryRecord, prefixes: tuple[str, ...]) -> str:
    evidence = _get(record, "evidence")
    if not isinstance(evidence, dict):
        return ""

    chunks: list[str] = []
    for key, spans in evidence.items():
        if not isinstance(key, str):
            continue
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if not spans:
            continue
        for span in spans:
            if isinstance(span, str):
                text = span
            elif isinstance(span, dict):
                text = span.get("text") or span.get("quote") or ""
            else:
                text = getattr(span, "text", "") or getattr(span, "quote", "") or ""
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def _time_to_minutes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)

    try:
        from datetime import time as dt_time

        if isinstance(value, dt_time):
            return int(value.hour) * 60 + int(value.minute)
    except Exception:
        # Defensive: don't let weird typing issues break deterministic CPT derivation.
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = text.split(":")
        if len(parts) < 2:
            return None
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            return None
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None
        return hour * 60 + minute

    return None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None

    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", text)
    if not match:
        return None
    try:
        month = int(match.group(1))
        day_val = int(match.group(2))
        year = int(match.group(3))
    except ValueError:
        return None

    if year < 100:
        year = year + 2000 if year < 70 else year + 1900
    try:
        return date(year, month, day_val)
    except ValueError:
        return None


def _extract_chest_tube_insertion_date(record: RegistryRecord) -> date | None:
    evidence_text = _evidence_text_for_prefixes(record, ("",))
    if not evidence_text:
        return None

    match = _CHEST_TUBE_INSERTION_DATE_RE.search(evidence_text)
    if not match:
        return None

    window = evidence_text[match.end() : match.end() + 80]
    date_match = _DATE_TOKEN_RE.search(window) or _DATE_TOKEN_RE.search(evidence_text)
    if not date_match:
        return None
    return _parse_date(date_match.group(1))


def _sedation_intraservice_minutes(record: RegistryRecord) -> int | None:
    sedation = _get(record, "sedation")
    if sedation is None:
        return None

    minutes = _get(sedation, "intraservice_minutes")
    if isinstance(minutes, int):
        return minutes
    if isinstance(minutes, float):
        return int(minutes)
    if isinstance(minutes, str) and minutes.strip().isdigit():
        try:
            return int(minutes.strip())
        except ValueError:
            return None

    start = _time_to_minutes(_get(sedation, "start_time"))
    end = _time_to_minutes(_get(sedation, "end_time"))
    if start is None or end is None:
        return None
    if end < start:
        end += 24 * 60
    return end - start


def _normalize_pleural_side(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("r"):
        return "Right"
    if text.startswith("l"):
        return "Left"
    return None


def _dilation_in_distinct_lobe_from_destruction(record: RegistryRecord) -> bool:
    """Check if dilation was performed in a different lobe than destruction.

    Returns True only if granular data proves distinct anatomic locations.
    If granular data is missing, returns False (assume bundled).
    """
    granular = _get(record, "granular_data")
    if not granular:
        return False

    # Get lobes for dilation
    dilation_lobes: set[str] = set()
    dilation_targets = _get(granular, "dilation_targets") or []
    for target in dilation_targets:
        lobe = _get(target, "lobe")
        if lobe:
            dilation_lobes.add(str(lobe).upper())

    # Get lobes for destruction/ablation
    destruction_lobes: set[str] = set()
    ablation_targets = _get(granular, "ablation_targets") or []
    for target in ablation_targets:
        lobe = _get(target, "lobe")
        if lobe:
            destruction_lobes.add(str(lobe).upper())

    # If no granular data for either, assume bundled
    if not dilation_lobes or not destruction_lobes:
        return False

    # Check for distinct lobes (dilation in lobe not used for destruction)
    return bool(dilation_lobes - destruction_lobes)


def _eus_b_has_sampling(record: RegistryRecord, eus_proc: Any) -> bool:
    """Return True when EUS-B sampling/FNA is supported by structured fields or evidence."""
    sites = _get(eus_proc, "sites_sampled")
    if isinstance(sites, (list, tuple)):
        if any(str(site).strip() for site in sites):
            return True

    passes = _get(eus_proc, "passes")
    if isinstance(passes, (int, float)) and int(passes) > 0:
        return True

    evidence_text = _evidence_text_for_prefixes(record, ("procedures_performed.eus_b",))
    if not evidence_text:
        return False

    has_sampling = bool(
        re.search(
            r"(?i)\b(?:fna|tbna|sampled|sampling|needle\s+aspirat(?:ion|ed)?|aspirat(?:ed|ion)|biops(?:y|ies|ied)|passes?)\b",
            evidence_text,
        )
    )
    has_inspection_only = bool(
        re.search(
            r"(?i)\b(?:inspection\s+only|identified|visualized|evaluated|inspected)\b",
            evidence_text,
        )
        and not re.search(r"(?i)\b(?:sampled|sampling|fna|tbna|biops|aspirat)\b", evidence_text)
    )
    return has_sampling and not has_inspection_only


def derive_all_codes_with_meta(
    record: RegistryRecord,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Return (codes, rationales, warnings)."""
    codes: list[str] = []
    rationales: dict[str, str] = {}
    warnings: list[str] = []
    header_code_text = _evidence_text_for_prefixes(record, ("code_evidence",))

    # --- Airway management ---
    if _performed(_proc(record, "intubation")):
        codes.append("31500")
        rationales["31500"] = "intubation.performed=true"

    # --- Bronchoscopy family ---
    diagnostic = _proc(record, "diagnostic_bronchoscopy")
    if _performed(diagnostic):
        interventional_names = [
            "bal",
            "brushings",
            "endobronchial_biopsy",
            "tbna_conventional",
            "linear_ebus",
            "radial_ebus",
            "navigational_bronchoscopy",
            "transbronchial_biopsy",
            "transbronchial_cryobiopsy",
            "therapeutic_aspiration",
            "foreign_body_removal",
            "airway_dilation",
            "airway_stent",
            "mechanical_debulking",
            "thermal_ablation",
            "cryotherapy",
            "blvr",
            "bronchial_thermoplasty",
            "whole_lung_lavage",
            "rigid_bronchoscopy",
        ]
        if any(_performed(_proc(record, name)) for name in interventional_names):
            warnings.append("Diagnostic bronchoscopy present but bundled into another bronchoscopic procedure")
        else:
            codes.append("31622")
            rationales["31622"] = "diagnostic_bronchoscopy.performed=true and no interventional bronchoscopy procedures"

    if _performed(_proc(record, "brushings")):
        codes.append("31623")
        rationales["31623"] = "brushings.performed=true"

    if _performed(_proc(record, "bal")):
        codes.append("31624")
        rationales["31624"] = "bal.performed=true"

    if _performed(_proc(record, "endobronchial_biopsy")):
        codes.append("31625")
        rationales["31625"] = "endobronchial_biopsy.performed=true"

    # Transbronchial lung biopsy (31628) and cryobiopsy are billed under 31628,
    # with add-on 31632 for additional lobes when documented.
    tbbx = _proc(record, "transbronchial_biopsy")
    cryo_tbbx = _proc(record, "transbronchial_cryobiopsy")
    if _performed(tbbx) or _performed(cryo_tbbx):
        codes.append("31628")
        if _performed(tbbx):
            rationales["31628"] = "transbronchial_biopsy.performed=true"
        else:
            rationales["31628"] = "transbronchial_cryobiopsy.performed=true"

        # Additional lobe add-on (31632) requires multi-lobe locations.
        location_values: list[str] = []

        def _extend_locations(proc_obj: Any) -> None:
            if proc_obj is None:
                return
            raw_locations = (
                _get(proc_obj, "locations")
                or _get(proc_obj, "locations_biopsied")
                or _get(proc_obj, "sites")
            )
            if isinstance(raw_locations, (list, tuple)):
                for item in raw_locations:
                    if item:
                        location_values.append(str(item))
            elif raw_locations:
                location_values.append(str(raw_locations))

        _extend_locations(tbbx)
        _extend_locations(cryo_tbbx)

        # Prefer structured cryobiopsy site detail when available.
        granular = _get(record, "granular_data")
        cryo_sites = _get(granular, "cryobiopsy_sites") if granular is not None else None
        if isinstance(cryo_sites, (list, tuple)):
            for site in cryo_sites:
                lobe = _get(site, "lobe")
                if lobe:
                    location_values.append(str(lobe))

        lobes = _lobe_tokens([v for v in location_values if v])
        if len(lobes) >= 2:
            codes.append("31632")
            rationales["31632"] = f"transbronchial biopsy spans lobes={sorted(lobes)}"

    # Conventional (non-EBUS) TBNA (31629) with add-on 31633 for additional lobes.
    tbna_nodal = _proc(record, "tbna_conventional")
    peripheral_tbna = _proc(record, "peripheral_tbna")

    ebus_stations, station_source = _stations_sampled(record)
    ebus_station_count = len(set(s for s in ebus_stations if s))
    has_ebus_sampling = _performed(_proc(record, "linear_ebus")) and ebus_station_count > 0

    added_31629 = False
    if _performed(peripheral_tbna):
        targets = _get(peripheral_tbna, "targets_sampled") or []
        target_values = [str(x).strip() for x in targets if x and str(x).strip()]
        has_distinct_target = _has_distinct_peripheral_tbna_target(target_values)

        if has_ebus_sampling and not has_distinct_target:
            warnings.append(
                "Suppressed 31629: peripheral_tbna present with EBUS-TBNA but targets_sampled lack a clearly distinct, non-nodal anatomic target"
            )
        else:
            codes.append("31629")
            rationales["31629"] = "peripheral_tbna.performed=true"
            added_31629 = True
            if has_ebus_sampling:
                warnings.append(
                    "31629 requires Modifier 59 when peripheral TBNA is distinct from EBUS-TBNA sampling"
                )
                rationales["31629"] = (
                    "peripheral_tbna.performed=true (distinct site from EBUS-TBNA; Modifier 59)"
                )
    elif _performed(tbna_nodal):
        # NCCI: Conventional nodal TBNA is bundled into EBUS-TBNA when EBUS sampling is performed.
        if has_ebus_sampling:
            warnings.append(
                "Suppressed 31629: tbna_conventional bundled into EBUS-TBNA (31652/31653)"
            )
        else:
            codes.append("31629")
            rationales["31629"] = "tbna_conventional.performed=true"
            added_31629 = True

    if added_31629 and _performed(peripheral_tbna):
        targets = _get(peripheral_tbna, "targets_sampled") or []
        lobes = _lobe_tokens([str(x) for x in targets if x])
        if len(lobes) >= 2:
            codes.append("31633")
            rationales["31633"] = f"peripheral_tbna.targets_sampled spans lobes={sorted(lobes)}"

    # Linear EBUS TBNA (31652/31653) based on station count.
    if _performed(_proc(record, "linear_ebus")):
        station_count = ebus_station_count
        if station_count >= 3:
            codes.append("31653")
            rationales["31653"] = (
                f"linear_ebus.performed=true and sampled_station_count={station_count} (>=3) "
                f"from {station_source}"
            )
        elif station_count in (1, 2):
            codes.append("31652")
            rationales["31652"] = (
                f"linear_ebus.performed=true and sampled_station_count={station_count} (1-2) "
                f"from {station_source}"
            )
        else:
            if station_source == "node_events":
                warnings.append(
                    "linear_ebus.performed=true but node_events contains no qualifying sampling actions; "
                    "cannot derive 31652/31653"
                )
            else:
                warnings.append(
                    "linear_ebus.performed=true but stations_sampled missing/empty; cannot derive 31652/31653"
                )

        elastography_used = _get(_proc(record, "linear_ebus"), "elastography_used") is True
        elastography_pattern = _get(_proc(record, "linear_ebus"), "elastography_pattern")
        if elastography_used or (isinstance(elastography_pattern, str) and elastography_pattern.strip()):
            target_stations: set[str] = set()
            node_events = _get(_proc(record, "linear_ebus"), "node_events")
            if isinstance(node_events, (list, tuple)):
                for event in node_events:
                    station = _get(event, "station")
                    if station:
                        target_stations.add(str(station).upper().strip())

            stations_detail = _get(_proc(record, "linear_ebus"), "stations_detail")
            if isinstance(stations_detail, (list, tuple)):
                for detail in stations_detail:
                    if isinstance(detail, dict):
                        station = detail.get("station")
                    else:
                        station = _get(detail, "station")
                    if station:
                        target_stations.add(str(station).upper().strip())

            if not target_stations:
                stations = _get(_proc(record, "linear_ebus"), "stations_sampled")
                if isinstance(stations, (list, tuple)):
                    for station in stations:
                        if station:
                            target_stations.add(str(station).upper().strip())

            target_count = len({s for s in target_stations if s})
            if target_count <= 0:
                # Guardrail: don't bill elastography if the record has no extracted targets.
                # Allow a conservative fallback only when evidence mentions elastography on
                # explicit node/station tokens (e.g., "11L", "Station 7").
                evidence_text = _evidence_text_for_prefixes(
                    record,
                    ("procedures_performed.linear_ebus", "procedures_performed.linear_ebus.node_events", "code_evidence"),
                )
                has_station_token = bool(
                    re.search(
                        r"(?i)\b(2R|2L|4R|4L|5|7|8|9|10R|10L|11R(?:S|I)?|11L(?:S|I)?)\b",
                        evidence_text or "",
                    )
                )
                has_elastography_token = bool(re.search(r"(?i)\belastograph\w*\b", evidence_text or ""))
                if has_station_token and has_elastography_token:
                    target_count = 1
                else:
                    warnings.append(
                        "Suppressed 76982/76983: elastography indicated but no EBUS targets/stations extracted."
                    )
                    target_count = 0

            if target_count <= 0:
                # Nothing billable without target evidence.
                pass
            else:
                # CPT 76982 is first target lesion; 76983 is each additional target lesion.
                codes.append("76982")
                rationales["76982"] = f"linear_ebus elastography used (targets={target_count})"
                if target_count >= 2:
                    addon_units = min(target_count - 1, 2)
                    codes.append("76983")
                    rationales["76983"] = (
                        f"linear_ebus elastography used and targets={target_count} (units={addon_units}; MUE cap=2)"
                    )

                # Parenchyma elastography (76981) is distinct from target-lesion elastography (76982/76983).
                # When EBUS elastography is documented on lymph nodes/targets, suppress 76981 even if the
                # header lists it (common templating artifact).
                if "76981" in (header_code_text or ""):
                    warnings.append(
                        "Suppressed 76981: header lists parenchyma elastography but EBUS elastography targets derive 76982/76983."
                    )

    # Radial EBUS (add-on code for peripheral lesion localization)
    if _performed(_proc(record, "radial_ebus")):
        codes.append("31654")
        rationales["31654"] = "radial_ebus.performed=true"
        # Optional documentation QA: radial EBUS is typically used as an adjunct to peripheral sampling/therapy.
        has_peripheral_sampling = any(
            _performed(_proc(record, name))
            for name in (
                "transbronchial_biopsy",
                "transbronchial_cryobiopsy",
                "peripheral_tbna",
                "brushings",
                "peripheral_ablation",
                "bal",
                "therapeutic_aspiration",
            )
        )
        if not has_peripheral_sampling:
            warnings.append(
                "Radial EBUS documented without extracted peripheral sampling/therapy; verify clinical context."
            )

    # Navigation add-on
    if _performed(_proc(record, "navigational_bronchoscopy")):
        codes.append("31627")
        rationales["31627"] = "navigational_bronchoscopy.performed=true"

    # Fiducial marker placement (navigation add-on)
    if _fiducial_marker_placed(record):
        codes.append("31626")
        rationales["31626"] = "granular_data.navigation_targets indicates fiducial marker placement"

    # Imaging adjuncts commonly documented in robotic/navigation cases
    equipment = _get(record, "equipment")
    cbct_used = _get(equipment, "cbct_used") is True
    nav_platform = _get(equipment, "navigation_platform")
    augmented = _get(equipment, "augmented_fluoroscopy") is True

    # 77012: CT guidance (used here as a proxy for documented cone-beam CT guidance in navigation cases)
    if cbct_used and (_performed(_proc(record, "navigational_bronchoscopy")) or nav_platform):
        codes.append("77012")
        rationales["77012"] = "equipment.cbct_used=true"

    # 76377: 3D rendering / reconstruction (requires explicit 3D rendering/reconstruction evidence)
    if augmented and (cbct_used or nav_platform) and (_performed(_proc(record, "navigational_bronchoscopy")) or nav_platform):
        codes.append("76377")
        rationales["76377"] = "equipment.augmented_fluoroscopy=true (3D reconstruction/rendering documented)"

    # Therapeutic aspiration
    if _performed(_proc(record, "therapeutic_aspiration")):
        if "31652" in codes or "31653" in codes:
            warnings.append(
                "Suppressed 31645: therapeutic aspiration is bundled into EBUS-TBNA (31652/31653) per NCCI (no modifier allowed)."
            )
        else:
            evidence = _evidence_text_for_prefixes(record, ("code_evidence",)) + "\n" + _evidence_text_for_prefixes(
                record,
                (
                    "procedures_performed.therapeutic_aspiration.is_subsequent",
                    "procedures_performed.therapeutic_aspiration",
                ),
            )
            is_subsequent = bool(
                re.search(
                    r"(?i)\b31646\b|\bsubsequent\s+aspirat|\brepeat\s+aspirat|\bsubsequent\s+episode\b",
                    evidence or "",
                )
            )
            if is_subsequent:
                codes.append("31646")
                rationales["31646"] = "therapeutic_aspiration.performed=true and subsequent episode indicated"
            else:
                codes.append("31645")
                rationales["31645"] = "therapeutic_aspiration.performed=true"

    # Therapeutic instillation/injection (31573)
    if _performed(_proc(record, "therapeutic_injection")):
        codes.append("31573")
        rationales["31573"] = "therapeutic_injection.performed=true"

    # Foreign body removal
    if _performed(_proc(record, "foreign_body_removal")):
        codes.append("31635")
        rationales["31635"] = "foreign_body_removal.performed=true"

    # Airway dilation
    if _performed(_proc(record, "airway_dilation")):
        codes.append("31630")
        rationales["31630"] = "airway_dilation.performed=true"

    # Airway stent
    stent = _proc(record, "airway_stent")
    foreign_body = _proc(record, "foreign_body_removal")
    foreign_body_performed = _performed(foreign_body)
    if stent is not None:
        action = _get(stent, "action")
        action_text = str(action).strip().lower() if action is not None else ""
        removal_flag = _get(stent, "airway_stent_removal") is True

        assessment_only = action_text.startswith("assessment")
        revision_action = "revision" in action_text or "reposition" in action_text
        placement_action = "placement" in action_text
        removal_action = action_text.startswith("remov") or _stent_action_is_removal(action) or removal_flag

        stent_evidence = _evidence_text_for_prefixes(
            record,
            (
                "procedures_performed.airway_stent",
                "code_evidence",
            ),
        ) or ""
        has_history_cue = bool(
            re.search(r"(?i)\b(?:known|existing|prior|previous|history\s+of)\b", stent_evidence)
        )
        has_placement_verb = bool(
            re.search(
                r"(?i)\b(?:place(?:d|ment)|deploy(?:ed|ment)?|insert(?:ed|ion)?|implant(?:ed|ation)?|position(?:ed|ing)?)\b",
                stent_evidence,
            )
        )

        if assessment_only:
            pass
        elif placement_action and not revision_action and not removal_action:
            codes.append("31636")
            rationales["31636"] = "airway_stent.action indicates placement"
        elif revision_action:
            # Revision/repositioning can reflect either:
            # - exchange/removal of an existing stent (31638), or
            # - immediate intraprocedural repositioning of a newly placed stent (typically bundled into 31636).
            if removal_action:
                codes.append("31638")
                rationales["31638"] = "airway_stent indicates removal/exchange"
            elif has_history_cue:
                codes.append("31638")
                rationales["31638"] = "airway_stent revision/repositioning of an existing stent (history cue present)"
            elif not has_placement_verb:
                codes.append("31638")
                rationales["31638"] = "airway_stent revision/repositioning without placement verbs (treat as existing stent revision)"
            else:
                codes.append("31636")
                rationales["31636"] = "airway_stent revision/repositioning documented without removal; bundled into placement (consider modifier 22 if significant)"
                warnings.append(
                    "Stent revision/repositioning documented without removal; coded as placement (31636). Consider modifier 22 when documentation supports increased work."
                )
        elif removal_action:
            codes.append("31638")
            rationales["31638"] = "airway_stent indicates removal/exchange"
            if foreign_body_performed and not placement_action and not revision_action:
                warnings.append(
                    "Foreign body removal was extracted alongside airway stent removal; coding as 31638 (stent removal). Review if a distinct foreign body was also removed."
                )
        elif _performed(stent):
            warnings.append(
                "airway_stent.performed=true but action is missing/ambiguous; suppressing stent placement CPT"
            )

    # Mechanical debulking (tumor excision) → 31640
    if _performed(_proc(record, "mechanical_debulking")):
        codes.append("31640")
        rationales["31640"] = "mechanical_debulking.performed=true"

    # Thermal ablation (tumor destruction) → 31641
    if _performed(_proc(record, "thermal_ablation")):
        codes.append("31641")
        rationales["31641"] = "thermal_ablation.performed=true"

    # Cryotherapy (tumor destruction) → 31641
    # Note: If both thermal_ablation and cryotherapy performed, only one 31641
    if _performed(_proc(record, "cryotherapy")) and "31641" not in codes:
        codes.append("31641")
        rationales["31641"] = "cryotherapy.performed=true"

    # Peripheral ablation (e.g., bronchoscopic microwave/RFA/cryoablation workflows).
    # Payer policies vary; map to 31641 as a conservative default when explicitly performed.
    if _performed(_proc(record, "peripheral_ablation")) and "31641" not in codes:
        codes.append("31641")
        rationales["31641"] = "peripheral_ablation.performed=true"
        warnings.append(
            "Derived 31641 from peripheral_ablation.performed=true; verify payer policy for peripheral tumor ablation vs unlisted coding."
        )

    # BPF glue/sealant intervention (unlisted in many fee schedules). Use 31641 as a placeholder primary
    # therapeutic bronchoscopy code when explicitly documented so outputs are non-empty.
    if _performed(_proc(record, "bpf_sealant")) and "31641" not in codes:
        codes.append("31641")
        rationales["31641"] = "bpf_sealant.performed=true"
        warnings.append(
            "Derived 31641 from bpf_sealant.performed=true (BPF sealant/glue intervention); consider unlisted coding per payer policy."
        )

    # Header fallback: some workflows intentionally list 31641 when cryo intervention
    # is charted as cryobiopsy text. Require both explicit header code and cryobiopsy.
    if (
        "31641" not in codes
        and re.search(r"\b31641\b", header_code_text or "")
        and _performed(_proc(record, "transbronchial_cryobiopsy"))
    ):
        codes.append("31641")
        rationales["31641"] = (
            "code_evidence lists 31641 and transbronchial_cryobiopsy.performed=true "
            "(header-explicit fallback)"
        )
        warnings.append(
            "Added 31641 from explicit procedure header code list with cryobiopsy evidence; verify payer policy for destruction vs biopsy coding."
        )

    # BLVR valve family
    blvr = _proc(record, "blvr")
    blvr_procedure_type = _get(blvr, "procedure_type")
    valve_lobes = _blvr_valve_lobes(record, blvr_proc=blvr)
    chartis_lobes = _blvr_chartis_lobes(record, blvr_proc=blvr)

    # Valve placement / removal codes
    if _performed(blvr) and (
        blvr_procedure_type == "Valve removal"
        or (
            blvr_procedure_type in {None, "Valve assessment"}
            and _BLVR_REMOVAL_CONTEXT_RE.search(
                _evidence_text_for_prefixes(
                    record,
                    ("procedures_performed.blvr", "granular_data.blvr_valve_placements"),
                )
                or ""
            )
        )
    ):
        codes.append("31648")
        rationale = (
            "blvr.procedure_type='Valve removal'"
            if blvr_procedure_type == "Valve removal"
            else "blvr.performed=true and removal language detected (fallback)"
        )
        if valve_lobes:
            rationale += f" (lobes={sorted(valve_lobes)})"
        rationales["31648"] = rationale
        if len(valve_lobes) >= 2:
            codes.append("31649")
            rationales["31649"] = f"Valve removal in multiple lobes={sorted(valve_lobes)} (add-on lobe)"

    elif _performed(blvr):
        blvr_text = " ".join(
            str(v)
            for v in (
                _get(blvr, "procedure_type"),
                _get(blvr, "target_lobe"),
                _get(blvr, "valve_type"),
                _get(blvr, "segments_treated"),
                _get(blvr, "valve_sizes"),
                _get(blvr, "number_of_valves"),
            )
            if v is not None and str(v).strip()
        )
        blvr_text += "\n" + _evidence_text_for_prefixes(
            record,
            (
                "procedures_performed.blvr",
                "granular_data.blvr_valve_placements",
            ),
        )

        has_zephyr = "zephyr" in blvr_text.lower()
        has_spiration = "spiration" in blvr_text.lower()
        if has_zephyr and has_spiration:
            warnings.append(
                "NEEDS_REVIEW: Mixed BLVR valve manufacturers detected (Zephyr + Spiration); default to Zephyr when attribution is unclear."
            )

        explicit_31647 = bool(re.search(r"\b31647\b", blvr_text))
        explicit_31651 = bool(re.search(r"\b31651\b", blvr_text))

        placement_keywords = bool(_BLVR_PLACEMENT_CONTEXT_RE.search(blvr_text))
        valve_size_hints = {m.group(0).strip().lower() for m in _BLVR_VALVE_SIZE_HINT_RE.finditer(blvr_text)}
        valve_size_hint_count = len(valve_size_hints)

        placement_signal = (
            blvr_procedure_type == "Valve placement"
            or explicit_31647
            or explicit_31651
            or valve_lobes
            or _get(blvr, "number_of_valves") not in {None, 0, "0"}
            or bool(_get(blvr, "valve_sizes"))
            or placement_keywords
            or bool(valve_size_hints)
        )

        forced_by_family = False
        families_raw = _get(record, "procedure_families") or []
        families = {str(f).strip().upper() for f in families_raw if f}
        if (
            not placement_signal
            and "BLVR" in families
            and blvr_procedure_type in {None, "Valve assessment"}
            and not chartis_lobes
        ):
            placement_signal = True
            forced_by_family = True

        if placement_signal and blvr_procedure_type != "Coil placement":
            codes.append("31647")
            if blvr_procedure_type == "Valve placement":
                rationale = "blvr.procedure_type='Valve placement'"
            elif explicit_31647 or explicit_31651:
                rationale = "blvr.performed=true and header/billing mentions BLVR CPT (fallback)"
            elif placement_keywords:
                rationale = "blvr.performed=true and valve placement language detected (fallback)"
            elif valve_size_hints:
                rationale = f"blvr.performed=true and valve size mentions detected (fallback, n={valve_size_hint_count})"
            elif forced_by_family:
                rationale = "blvr.performed=true and procedure_families includes BLVR (fallback)"
            else:
                rationale = "blvr.performed=true and placement inferred (fallback)"

            if valve_lobes:
                rationale += f" (lobes={sorted(valve_lobes)})"
            rationales["31647"] = rationale

            if len(valve_lobes) >= 2 or explicit_31651:
                codes.append("31651")
                if len(valve_lobes) >= 2:
                    rationales["31651"] = f"Valve placement in multiple lobes={sorted(valve_lobes)} (add-on lobe)"
                else:
                    rationales["31651"] = "header/billing explicitly mentions 31651 (add-on lobe)"

            if not valve_lobes and (placement_keywords or explicit_31647 or valve_size_hints):
                warnings.append(
                    "BLVR valve placement inferred but target lobe(s) missing; verify lobes to support 31651 add-on lobe billing."
                )

    # Balloon occlusion / Chartis assessment (31634).
    # - For Chartis, apply same-lobe bundling vs BLVR valve placement.
    # - For non-Chartis balloon occlusion (e.g., Uniblocker/Arndt/Fogarty for air leak/bleeding),
    #   do not suppress solely due to valve overlap (different clinical intent).
    occlusion_source: str | None = None
    balloon_occlusion = _proc(record, "balloon_occlusion")
    balloon_evidence = _evidence_text_for_prefixes(
        record,
        ("procedures_performed.balloon_occlusion", "code_evidence"),
    ).lower()
    balloon_signal = _performed(balloon_occlusion) or bool(
        re.search(
            r"(?i)\b(?:balloon\s+occlusion|serial\s+occlusion|endobronchial\s+blocker|uniblocker|arndt|ardnt|fogarty)\b",
            balloon_evidence or "",
        )
    )
    if chartis_lobes:
        occlusion_source = "Chartis"
    else:
        cv = _get(blvr, "collateral_ventilation_assessment")
        cv_text = str(cv).lower() if cv is not None else ""
        blvr_evidence = _evidence_text_for_prefixes(
            record,
            ("procedures_performed.blvr", "granular_data.blvr_chartis_measurements"),
        ).lower()
        if "chartis" in cv_text or "chartis" in blvr_evidence:
            occlusion_source = "Chartis"
        elif balloon_signal:
            occlusion_source = "Balloon occlusion"
        elif re.search(
            r"(?i)\b(?:tisseel|thrombin|fibrin\s+(?:glue|sealant)|(?:fibrin\s+)?glue)\b",
            cv_text + "\n" + blvr_evidence,
        ):
            occlusion_source = "Substance occlusion"

    if occlusion_source:
        if not chartis_lobes:
            codes.append("31634")
            rationales["31634"] = f"{occlusion_source} documented (target lobe missing)"
            warnings.append(
                f"{occlusion_source} documented but target lobe missing; verify documentation supports 31634 and consider modifiers/bundling when applicable."
            )
        elif valve_lobes and occlusion_source == "Chartis":
            overlap = chartis_lobes & valve_lobes
            distinct = chartis_lobes - valve_lobes
            if not distinct:
                warnings.append(
                    f"Suppressed 31634 (Chartis): bundled with BLVR valve procedure in same lobe(s)={sorted(overlap)}"
                )
            else:
                codes.append("31634")
                rationales["31634"] = f"Chartis documented in distinct lobe(s)={sorted(distinct)}"
                warnings.append(
                    "31634 (Chartis) distinct from valve lobe(s); consider modifier -59/-XS and ensure documentation supports distinctness"
                )
                if overlap:
                    warnings.append(
                        f"Chartis also documented in valve lobe(s)={sorted(overlap)} (bundled for those lobes)"
                    )
        else:
            codes.append("31634")
            rationales["31634"] = f"{occlusion_source} documented (lobes={sorted(chartis_lobes)})"
            if valve_lobes and occlusion_source != "Chartis":
                warnings.append(
                    f"31634 ({occlusion_source}) performed alongside valve work; ensure documentation supports separate billing and consider modifier -59/-XS when appropriate."
                )

    # Bronchial thermoplasty: 31660 initial + 31661 additional lobes.
    bt = _proc(record, "bronchial_thermoplasty")
    if _performed(bt):
        codes.append("31660")
        rationales["31660"] = "bronchial_thermoplasty.performed=true"
        areas = _get(bt, "areas_treated")
        if areas and len(areas) >= 2:
            codes.append("31661")
            rationales["31661"] = f"bronchial_thermoplasty.areas_treated_count={len(areas)} (>=2)"

    # Tracheostomy: distinguish established route vs new percutaneous trach.
    established_trach_route = _get(record, "established_tracheostomy_route") is True
    if established_trach_route:
        codes.append("31615")
        rationales["31615"] = "established_tracheostomy_route=true"

    # Percutaneous tracheostomy (new trach creation)
    pt_obj = _proc(record, "percutaneous_tracheostomy")
    pt_performed = _performed(pt_obj)
    puncture_evidence = _evidence_text_for_prefixes(
        record,
        (
            "procedures_performed.tracheal_puncture",
            "procedures_performed.percutaneous_tracheostomy",
            "code_evidence",
        ),
    )
    puncture_only = bool(
        re.search(
            r"(?i)\b31612\b|\btranstracheal\b|\bangiocat|\bpunctur\w*\b",
            puncture_evidence or "",
        )
    )

    # 31612: percutaneous tracheal puncture / transtracheal access (NOT a tracheostomy creation).
    # If puncture-only evidence is present without explicit trach-creation language, derive 31612.
    pt_evidence = _evidence_text_for_prefixes(
        record,
        ("procedures_performed.percutaneous_tracheostomy",),
    )
    explicit_trach_creation = bool(
        re.search(
            r"(?i)\bpercutaneous\s+(?:dilatational\s+)?tracheostomy\b|\bperc\s+trach\b|\btracheostomy\b[^.\n]{0,60}\b(?:performed|placed|inserted|created)\b",
            pt_evidence or "",
        )
    )
    if puncture_only and not explicit_trach_creation:
        codes.append("31612")
        rationales["31612"] = "tracheal puncture evidence present (puncture-only; not tracheostomy creation)"
        if pt_performed and not established_trach_route:
            warnings.append(
                "percutaneous_tracheostomy.performed=true but evidence indicates puncture-only; deriving 31612 (not 31600)"
            )

    # 31600: percutaneous tracheostomy creation (new trach).
    if pt_performed and "31612" not in codes:
        if established_trach_route:
            warnings.append(
                "percutaneous_tracheostomy.performed=true but established_tracheostomy_route=true; suppressing 31600"
            )
        else:
            codes.append("31600")
            rationales["31600"] = "percutaneous_tracheostomy.performed=true and established_tracheostomy_route=false"

    # Neck ultrasound (often pre-tracheostomy vascular mapping)
    if _performed(_proc(record, "neck_ultrasound")):
        codes.append("76536")
        rationales["76536"] = "neck_ultrasound.performed=true"

    # Chest ultrasound (diagnostic, real-time with documentation)
    if _performed(_proc(record, "chest_ultrasound")):
        codes.append("76604")
        rationales["76604"] = "chest_ultrasound.performed=true"

    # EUS-B (endoscopic ultrasound via bronchoscope)
    eus_b = _proc(record, "eus_b")
    if _performed(eus_b):
        if _eus_b_has_sampling(record, eus_b):
            codes.append("43238")
            rationales["43238"] = "eus_b.performed=true with sampling/FNA evidence"
        else:
            codes.append("43237")
            rationales["43237"] = "eus_b.performed=true without sampling/FNA evidence (inspection only)"

    # --- Pleural family ---
    ipc = _pleural(record, "ipc")
    if _performed(ipc):
        action = _get(ipc, "action")
        action_text = str(action).strip().lower() if action is not None else ""
        insertion_action = action == "Insertion" or action_text.startswith(("insert", "place", "plac", "deploy"))
        removal_action = action == "Removal" or action_text.startswith(("remov", "withdraw", "pull", "discontinue", "d/c", "dc", "explant"))
        if insertion_action:
            codes.append("32550")
            rationales["32550"] = "pleural_procedures.ipc.performed=true and action indicates insertion"
        elif removal_action:
            codes.append("32552")
            rationales["32552"] = "pleural_procedures.ipc.performed=true and action indicates removal"
        else:
            warnings.append(
                "pleural_procedures.ipc.performed=true but action is missing/ambiguous; suppressing 32550/32552"
            )

    thora = _pleural(record, "thoracentesis")
    if _performed(thora):
        guidance = _get(thora, "guidance")
        if guidance == "Ultrasound":
            codes.append("32555")
            rationales["32555"] = "thoracentesis.performed=true and guidance='Ultrasound'"
        else:
            codes.append("32554")
            rationales["32554"] = "thoracentesis.performed=true and guidance!='Ultrasound'"

    chest_tube = _pleural(record, "chest_tube")
    if _performed(chest_tube):
        action = _get(chest_tube, "action")
        tube_type = _get(chest_tube, "tube_type")
        tube_size_fr = _get(chest_tube, "tube_size_fr")
        guidance = _get(chest_tube, "guidance")

        action_text = str(action).strip().lower() if action is not None else ""
        if action in {"Removal", "Repositioning", "Exchange"} or action_text.startswith(
            ("remov", "reposition", "exchange", "mainten")
        ):
            warnings.append(
                f"pleural_procedures.chest_tube.action={action!r}; skipping insertion codes (bundled/not separately billable)"
            )
        elif action not in {"Insertion"} and not action_text.startswith(("insert", "place", "plac", "deploy")):
            warnings.append(
                f"pleural_procedures.chest_tube.performed=true but action={action!r} is not insertion; suppressing insertion CPT"
            )
            chest_tube = None
        else:
            thoracoscopy = _pleural(record, "medical_thoracoscopy")
            if _performed(thoracoscopy):
                thor_side = _normalize_pleural_side(_get(thoracoscopy, "side"))
                tube_side = _normalize_pleural_side(_get(chest_tube, "side"))
                contralateral = bool(thor_side and tube_side and thor_side != tube_side)
                if not contralateral:
                    warnings.append(
                        "Suppressed chest tube insertion CPT: bundled with medical thoracoscopy unless contralateral/distinct site is documented."
                    )
                    chest_tube = None
                else:
                    warnings.append(
                        "Chest tube insertion appears contralateral to medical thoracoscopy; consider modifier -59/-XS and ensure documentation supports distinctness."
                    )

        if chest_tube is not None and _performed(chest_tube) and not (
            action in {"Removal", "Repositioning", "Exchange"}
            or action_text.startswith(("remov", "reposition", "exchange", "mainten"))
        ):
            imaging = guidance in {"Ultrasound", "CT", "Fluoroscopy"}
            is_small_bore = False
            if tube_type == "Pigtail":
                is_small_bore = True
            elif isinstance(tube_size_fr, int) and tube_size_fr <= 16:
                is_small_bore = True

            if is_small_bore:
                if imaging:
                    codes.append("32557")
                    rationales["32557"] = (
                        "pleural_procedures.chest_tube.performed=true and small-bore drain with imaging guidance"
                    )
                else:
                    codes.append("32556")
                    rationales["32556"] = (
                        "pleural_procedures.chest_tube.performed=true and small-bore drain without imaging guidance"
                    )
            else:
                codes.append("32551")
                rationales["32551"] = "pleural_procedures.chest_tube.performed=true (tube thoracostomy)"

    thoracoscopy = _pleural(record, "medical_thoracoscopy")
    pleurodesis = _pleural(record, "pleurodesis")

    thoracoscopy_performed = _performed(thoracoscopy)
    pleurodesis_performed = _performed(pleurodesis)

    # Surgical thoracoscopy upgrade: thoracoscopy + pleurodesis is reported as 32650
    # (and chemical pleurodesis 32560 is bundled into the thoracoscopic pleurodesis).
    if thoracoscopy_performed:
        if pleurodesis_performed:
            codes.append("32650")
            rationales["32650"] = (
                "pleural_procedures.medical_thoracoscopy.performed=true and pleural_procedures.pleurodesis.performed=true"
            )
        else:
            biopsies_taken = _get(thoracoscopy, "biopsies_taken")
            if biopsies_taken is True:
                codes.append("32609")
                rationales["32609"] = (
                    "pleural_procedures.medical_thoracoscopy.performed=true and biopsies_taken=true"
                )
            else:
                codes.append("32601")
                rationales["32601"] = "pleural_procedures.medical_thoracoscopy.performed=true"

        adhesiolysis = _get(thoracoscopy, "adhesiolysis_performed")
        if adhesiolysis is True:
            codes.append("32653")
            rationales["32653"] = (
                "pleural_procedures.medical_thoracoscopy.performed=true and adhesiolysis_performed=true"
            )

    # Chemical pleurodesis without thoracoscopy (e.g., talc slurry via chest tube).
    if pleurodesis_performed and not thoracoscopy_performed:
        codes.append("32560")
        rationales["32560"] = "pleural_procedures.pleurodesis.performed=true"

    fibrinolytic = _pleural(record, "fibrinolytic_therapy")
    if _performed(fibrinolytic):
        number_of_doses = _get(fibrinolytic, "number_of_doses")
        header_text = _get(_get(record, "clinical_context"), "primary_indication")
        indication_text = _get(fibrinolytic, "indication")
        evidence_text = _evidence_text_for_prefixes(
            record,
            (
                "clinical_context.primary_indication",
                "pleural_procedures.fibrinolytic_therapy",
            ),
        )
        combined = " ".join(
            str(v)
            for v in (
                header_text,
                indication_text,
                evidence_text,
            )
            if v is not None and str(v).strip()
        )
        subsequent_by_token = bool(_FIBRINOLYSIS_SUBSEQUENT_TOKEN_RE.search(combined))

        subsequent_by_doses = False
        if isinstance(number_of_doses, int):
            subsequent_by_doses = number_of_doses >= 2
        else:
            try:
                subsequent_by_doses = int(str(number_of_doses)) >= 2
            except (TypeError, ValueError):
                subsequent_by_doses = False

        subsequent_day = subsequent_by_token or subsequent_by_doses

        if subsequent_day:
            codes.append("32562")
            if subsequent_by_token and not subsequent_by_doses:
                rationales["32562"] = (
                    "pleural_procedures.fibrinolytic_therapy.performed=true and subsequent-day token found in header/indication"
                )
            else:
                rationales["32562"] = (
                    "pleural_procedures.fibrinolytic_therapy.performed=true and number_of_doses>=2 (subsequent day)"
                )
        else:
            codes.append("32561")
            rationales["32561"] = "pleural_procedures.fibrinolytic_therapy.performed=true"
            procedure_date = _parse_date(_get(record, "procedure_date"))
            insertion_date = _extract_chest_tube_insertion_date(record)
            if procedure_date is not None and insertion_date is not None and insertion_date < procedure_date:
                warnings.append(
                    f"AUDIT_WARNING: 32561 (initial fibrinolysis) selected but chest tube insertion date {insertion_date.isoformat()} precedes procedure_date {procedure_date.isoformat()}; consider 32562 if this is a subsequent instillation."
                )

    # --- Sedation (moderate sedation billing) ---
    sedation = _get(record, "sedation")
    sedation_type = _get(sedation, "type")
    anesthesia_provider = _get(sedation, "anesthesia_provider")
    if sedation_type == "Moderate":
        if anesthesia_provider != "Proceduralist":
            if anesthesia_provider:
                warnings.append(
                    f"Moderate sedation present but anesthesia_provider={anesthesia_provider!r}; not deriving 99152/99153"
                )
            else:
                warnings.append(
                    "Moderate sedation present but anesthesia_provider missing; not deriving 99152/99153"
                )
        else:
            minutes = _sedation_intraservice_minutes(record)
            if minutes is None:
                warnings.append(
                    "Moderate sedation by proceduralist present but intraservice_minutes missing; not deriving 99152/99153"
                )
            elif minutes < 10:
                warnings.append(
                    f"Moderate sedation intraservice_minutes={minutes} (<10); suppressing 99152/99153"
                )
            else:
                codes.append("99152")
                rationales["99152"] = (
                    f"sedation.type='Moderate' and anesthesia_provider='Proceduralist' and intraservice_minutes={minutes}"
                )
                # Conservative: only emit 99153 when at least one full additional 15-min block exists.
                if minutes >= 30:
                    codes.append("99153")
                    rationales["99153"] = f"sedation.intraservice_minutes={minutes} (>=30 implies add-on time beyond initial 15 min)"

    # ---------------------------------------------------------------------
    # Post-processing: mutual exclusions & add-on safety
    # ---------------------------------------------------------------------
    derived = sorted(set(codes))

    # Mutually exclusive: 31652 vs 31653 (prefer 31653)
    if "31652" in derived and "31653" in derived:
        derived = [c for c in derived if c != "31652"]
        rationales.pop("31652", None)

    # Mutually exclusive: 32554 vs 32555 (prefer imaging-guided)
    if "32554" in derived and "32555" in derived:
        derived = [c for c in derived if c != "32554"]
        rationales.pop("32554", None)

    # Mutually exclusive: 32556 vs 32557 (prefer imaging-guided)
    if "32556" in derived and "32557" in derived:
        derived = [c for c in derived if c != "32556"]
        rationales.pop("32556", None)

    # Bundling: stent revision/exchange (31638) supersedes removal/placement in same site.
    if "31638" in derived:
        dropped_stent_codes = [c for c in ("31635", "31636") if c in derived]
        if dropped_stent_codes:
            derived = [c for c in derived if c not in {"31635", "31636"}]
            for c in dropped_stent_codes:
                rationales.pop(c, None)
            warnings.append(
                "CPT_CONFLICT_STENT_CYCLE: dropped 31635/31636 because 31638 covers stent revision/exchange"
            )

    # Bundling: Dilation (31630) is typically integral to stent placement/revision
    # when performed to expand the stent at the same anatomic site.
    # Default (safe): assume same site unless the record provides distinct anatomy.
    if "31630" in derived and any(c in derived for c in ("31636", "31638")):
        stent_loc = _get(_proc(record, "airway_stent"), "location")
        dilation_loc = _get(_proc(record, "airway_dilation"), "location")

        stent_tokens = _airway_site_tokens(stent_loc)
        dilation_tokens = _airway_site_tokens(dilation_loc)
        distinct_locations = bool(
            stent_tokens and dilation_tokens and stent_tokens.isdisjoint(dilation_tokens)
        )

        if not distinct_locations:
            derived = [c for c in derived if c != "31630"]
            rationales.pop("31630", None)
            stent_code = "31638" if "31638" in derived else "31636"
            warnings.append(
                f"31630 (dilation) bundled into {stent_code} (stent). If dilation was performed at a distinct site, add 31630 with modifier 59/XS."
            )

    # Bundling: Destruction (31641) is integral to Excision (31640) on the same lesion.
    # Default (safe): assume same lesion unless granular/anatomic location proves otherwise.
    if "31640" in derived and "31641" in derived:
        excision_loc = _get(_proc(record, "mechanical_debulking"), "location")
        destruction_loc = _get(_proc(record, "thermal_ablation"), "location")

        excision_lobes = _lobe_tokens([str(excision_loc)]) if excision_loc else set()
        destruction_lobes = _lobe_tokens([str(destruction_loc)]) if destruction_loc else set()
        distinct_locations = bool(excision_lobes and destruction_lobes and excision_lobes.isdisjoint(destruction_lobes))

        if distinct_locations:
            warnings.append(
                "31641 requires Modifier 59: destruction performed on distinct lesion from excision (31640)."
            )
        else:
            derived = [c for c in derived if c != "31641"]
            rationales.pop("31641", None)
            warnings.append(
                "31641 (destruction) bundled into 31640 (excision). If performed on separate lesion, add 31641 with modifier 59/XS."
            )

    # Bundling: Dilation (31630) vs Destruction (31641) / Excision (31640)
    # If destruction/excision is present, bundle dilation unless in distinct lobe
    destruction_codes = {"31641", "31640"}
    if any(c in destruction_codes for c in derived) and "31630" in derived:
        distinct_lobes = _dilation_in_distinct_lobe_from_destruction(record)
        if not distinct_lobes:
            derived = [c for c in derived if c != "31630"]
            warnings.append(
                "31630 (dilation) bundled into destruction/excision code - "
                "add granular lobe data if performed in distinct anatomic location"
            )
            rationales.pop("31630", None)

    # NCCI hard bundles (modifier not allowed): enforce KB-configured unconditional drops.
    derived = _apply_ncci_hard_bundles(derived, rationales=rationales, warnings=warnings)

    # Add-on codes require a primary bronchoscopy.
    addon_codes = {"31626", "31627", "31632", "31633", "31649", "31651", "31654", "31661", "76983"}
    primary_bronch = {
        "31615",
        "31622",
        "31623",
        "31624",
        "31625",
        "31628",
        "31629",
        "31634",
        "31635",
        "31640",
        "31641",
        "31645",
        "31647",
        "31648",
        "31652",
        "31653",
        "31660",
    }
    if not any(c in primary_bronch for c in derived):
        derived = [c for c in derived if c not in addon_codes]
        for c in addon_codes:
            rationales.pop(c, None)

    return derived, rationales, warnings


def _apply_ncci_hard_bundles(
    codes: list[str],
    *,
    rationales: dict[str, str],
    warnings: list[str],
) -> list[str]:
    """Apply unconditional NCCI bundles from the knowledge base.

    For modifier_allowed=false pairs, drop the lower-valued code (RVU-based tie-breaker)
    to guard against occasional primary/secondary ordering drift in KB data.
    """
    from app.common.knowledge import ncci_pairs, total_rvu

    active = list(codes)
    active_set = set(active)

    for pair in ncci_pairs() or []:
        if not isinstance(pair, dict):
            continue
        if bool(pair.get("modifier_allowed")):
            continue
        primary = str(pair.get("primary") or "").strip()
        secondary = str(pair.get("secondary") or "").strip()
        if not primary or not secondary:
            continue
        if primary not in active_set or secondary not in active_set:
            continue

        primary_rvu = float(total_rvu(primary) or 0.0)
        secondary_rvu = float(total_rvu(secondary) or 0.0)

        # Default to dropping KB-defined secondary, but prefer dropping the lower-valued
        # code when both RVUs are known.
        drop = secondary
        keep = primary
        if primary_rvu > 0 and secondary_rvu > 0:
            if primary_rvu < secondary_rvu:
                drop, keep = primary, secondary
            else:
                drop, keep = secondary, primary

        if drop not in active_set:
            continue

        active_set.remove(drop)
        try:
            active.remove(drop)
        except ValueError:
            pass
        rationales.pop(drop, None)

        reason = str(pair.get("reason") or "").strip()
        if reason:
            warnings.append(f"NCCI_BUNDLE: dropped {drop} (bundled into {keep}): {reason}")
        else:
            warnings.append(f"NCCI_BUNDLE: dropped {drop} (bundled into {keep})")

    return active


def derive_all_codes(record: RegistryRecord) -> list[str]:
    codes, _rationales, _warnings = derive_all_codes_with_meta(record)
    return codes


def _ebus_elastography_target_count(record: RegistryRecord) -> int:
    target_stations: set[str] = set()

    node_events = _get(_proc(record, "linear_ebus"), "node_events")
    if isinstance(node_events, (list, tuple)):
        for event in node_events:
            station = _get(event, "station")
            if station:
                target_stations.add(str(station).upper().strip())

    stations_detail = _get(_proc(record, "linear_ebus"), "stations_detail")
    if isinstance(stations_detail, (list, tuple)):
        for detail in stations_detail:
            if isinstance(detail, dict):
                station = detail.get("station")
            else:
                station = _get(detail, "station")
            if station:
                target_stations.add(str(station).upper().strip())

    if not target_stations:
        stations = _get(_proc(record, "linear_ebus"), "stations_sampled")
        if isinstance(stations, (list, tuple)):
            for station in stations:
                if station:
                    target_stations.add(str(station).upper().strip())

    count = len({s for s in target_stations if s})
    return count if count > 0 else 1


def _tbbx_lobe_count(record: RegistryRecord) -> int:
    lobes: set[str] = set()

    for proc_name in ("transbronchial_biopsy", "transbronchial_cryobiopsy"):
        proc = _proc(record, proc_name)
        if proc is None:
            continue
        raw_locations = _get(proc, "locations") or _get(proc, "locations_biopsied") or _get(proc, "sites") or []
        values: list[str] = []
        if isinstance(raw_locations, (list, tuple)):
            values = [str(x) for x in raw_locations if x]
        elif raw_locations:
            values = [str(raw_locations)]
        lobes |= _lobe_tokens(values)

    granular = _get(record, "granular_data")
    cryo_sites = _get(granular, "cryobiopsy_sites") if granular is not None else None
    if isinstance(cryo_sites, (list, tuple)):
        lobes |= _lobe_tokens([str(_get(site, "lobe")) for site in cryo_sites if _get(site, "lobe")])

    return len(lobes)


def _peripheral_tbna_lobe_count(record: RegistryRecord) -> int:
    proc = _proc(record, "peripheral_tbna")
    if proc is None:
        return 0
    targets = _get(proc, "targets_sampled") or []
    if isinstance(targets, (list, tuple)):
        values = [str(x) for x in targets if x]
    elif targets:
        values = [str(targets)]
    else:
        values = []
    return len(_lobe_tokens(values))


def derive_units_for_codes(record: RegistryRecord, codes: list[str]) -> dict[str, int]:
    """Return derived code units for add-on/multi-unit codes.

    This is intentionally small-scope: most CPTs are single-unit in our current rule set.
    """
    units: dict[str, int] = {}

    # 76983 is "each additional target lesion" (add-on); cap at 2 units (MUE).
    if "76983" in codes:
        target_count = _ebus_elastography_target_count(record)
        units["76983"] = max(1, min(target_count - 1, 2))

    # 31632 is "each additional lobe" for transbronchial biopsy (31628).
    if "31632" in codes:
        lobe_count = _tbbx_lobe_count(record)
        if lobe_count >= 2:
            units["31632"] = max(1, min(lobe_count - 1, 4))

    # 31633 is "each additional lobe" for peripheral (non-nodal) TBNA (31629).
    if "31633" in codes:
        lobe_count = _peripheral_tbna_lobe_count(record)
        if lobe_count >= 2:
            units["31633"] = max(1, min(lobe_count - 1, 4))

    # BLVR add-ons are per additional lobe.
    if "31651" in codes:
        lobes = _blvr_valve_lobes(record, blvr_proc=_proc(record, "blvr"))
        if len(lobes) >= 2:
            units["31651"] = max(1, min(len(lobes) - 1, 4))

    if "31649" in codes:
        lobes = _blvr_valve_lobes(record, blvr_proc=_proc(record, "blvr"))
        if len(lobes) >= 2:
            units["31649"] = max(1, min(len(lobes) - 1, 4))

    return units


__all__ = ["derive_all_codes", "derive_all_codes_with_meta", "derive_units_for_codes"]
