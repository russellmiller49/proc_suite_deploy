from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Literal

from app.registry.schema import RegistryRecord

PromptSeverity = Literal["required", "recommended"]

_STATION_TOKEN_RE = re.compile(r"\b(?:[1-9]|1[0-2])(?:[LR](?:[SI])?)?\b", re.IGNORECASE)


@dataclass(frozen=True)
class MissingFieldPrompt:
    """UI-facing prompt for a missing or under-documented field."""

    group: str
    path: str
    target_path: str
    label: str
    severity: PromptSeverity
    message: str


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _is_missing_value(value: object) -> bool:
    if _is_missing_scalar(value):
        return True
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _split_path(path: str) -> list[tuple[str, bool]]:
    """Return list of (segment, is_wildcard_list) segments."""
    segments: list[tuple[str, bool]] = []
    for raw in (path or "").split("."):
        token = raw.strip()
        if not token:
            continue
        if token.endswith("[*]"):
            segments.append((token[:-3], True))
        else:
            segments.append((token, False))
    return segments


def _iter_path_values(data: object, path: str) -> list[object]:
    segments = _split_path(path)
    if not segments:
        return []

    current: list[object] = [data]
    for key, wildcard in segments:
        next_values: list[object] = []
        for item in current:
            if isinstance(item, dict):
                child = item.get(key)
            else:
                child = getattr(item, key, None)

            if wildcard:
                if isinstance(child, list):
                    next_values.extend(child)
                else:
                    continue
            else:
                next_values.append(child)
        current = next_values

    return current


def _path_has_meaningful_value(
    data: dict[str, Any],
    *,
    path: str,
    missing_sentinels: set[object] | None = None,
) -> bool:
    sentinels = missing_sentinels or set()
    values = _iter_path_values(data, path)
    for value in values:
        if _is_missing_value(value):
            continue
        if value in sentinels:
            continue
        return True
    return False


def _any_path_has_meaningful_value(
    data: dict[str, Any],
    *,
    paths: Iterable[str],
    missing_sentinels_by_path: dict[str, set[object]] | None = None,
) -> bool:
    sentinels_map = missing_sentinels_by_path or {}
    for path in paths:
        sentinels = sentinels_map.get(path)
        if _path_has_meaningful_value(data, path=path, missing_sentinels=sentinels):
            return True
    return False


def _get_bool(data: dict[str, Any], path: str) -> bool | None:
    values = _iter_path_values(data, path)
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _normalize_station_token(value: object) -> str | None:
    if _is_missing_scalar(value):
        return None
    raw = str(value).strip().upper()
    if not raw:
        return None
    match = _STATION_TOKEN_RE.search(raw)
    if not match:
        compact = re.sub(r"[\s_\-]+", "", raw)
        match = _STATION_TOKEN_RE.search(compact)
    if not match:
        return None
    return match.group(0).upper()


def _station_sort_key(station: str) -> tuple[int, str]:
    m = re.match(r"^(?P<num>\d+)(?P<suffix>[A-Z]*)$", str(station).upper())
    if not m:
        return (999, str(station))
    try:
        num = int(m.group("num"))
    except (TypeError, ValueError):
        num = 999
    return (num, m.group("suffix") or "")


def _is_meaningful(value: object) -> bool:
    return not _is_missing_value(value)


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _build_ebus_station_contexts(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    contexts: dict[str, dict[str, Any]] = {}

    def ensure_ctx(station: str) -> dict[str, Any]:
        existing = contexts.get(station)
        if existing is not None:
            return existing
        created: dict[str, Any] = {
            "station": station,
            "sampled": False,
            "detail_index": None,
            "node_event_index": None,
            "has_needle_gauge": False,
            "has_passes": False,
            "has_short_axis": False,
            "has_adequacy": False,
        }
        contexts[station] = created
        return created

    linear_ebus = (
        (data.get("procedures_performed") or {}).get("linear_ebus") if isinstance(data, dict) else None
    )
    if not isinstance(linear_ebus, dict):
        linear_ebus = {}

    global_needle_gauge_present = _is_meaningful(linear_ebus.get("needle_gauge"))

    sampled_stations = linear_ebus.get("stations_sampled")
    if isinstance(sampled_stations, list):
        for token in sampled_stations:
            station = _normalize_station_token(token)
            if not station:
                continue
            ensure_ctx(station)["sampled"] = True

    node_events = linear_ebus.get("node_events")
    if isinstance(node_events, list):
        for idx, event in enumerate(node_events):
            if not isinstance(event, dict):
                continue
            station = _normalize_station_token(event.get("station"))
            if not station:
                continue
            ctx = ensure_ctx(station)
            if ctx["node_event_index"] is None:
                ctx["node_event_index"] = idx

            action = str(event.get("action") or "").strip().lower()
            if action and (
                "needle" in action
                or "tbna" in action
                or "fna" in action
                or "biops" in action
                or "aspirat" in action
                or "sample" in action
            ):
                ctx["sampled"] = True

            passes_values = [
                event.get("passes"),
                event.get("pass_count"),
                event.get("number_of_passes"),
            ]
            if any(_is_meaningful(v) for v in passes_values):
                ctx["has_passes"] = True
                ctx["sampled"] = True

            if _is_meaningful(event.get("outcome")):
                ctx["has_adequacy"] = True
                ctx["sampled"] = True

    granular = data.get("granular_data") if isinstance(data, dict) else None
    details = (granular or {}).get("linear_ebus_stations_detail") if isinstance(granular, dict) else None
    if isinstance(details, list):
        for idx, row in enumerate(details):
            if not isinstance(row, dict):
                continue
            station = _normalize_station_token(row.get("station"))
            if not station:
                continue
            ctx = ensure_ctx(station)
            if ctx["detail_index"] is None:
                ctx["detail_index"] = idx

            sampled = row.get("sampled")
            if sampled is True:
                ctx["sampled"] = True
            elif sampled is None and any(
                _is_meaningful(v)
                for v in (
                    row.get("needle_gauge"),
                    row.get("number_of_passes"),
                    row.get("rose_result"),
                    row.get("short_axis_mm"),
                )
            ):
                ctx["sampled"] = True

            if _is_meaningful(row.get("needle_gauge")):
                ctx["has_needle_gauge"] = True
            if _is_meaningful(row.get("number_of_passes")):
                ctx["has_passes"] = True
            if _is_meaningful(row.get("short_axis_mm")):
                ctx["has_short_axis"] = True
            if (
                isinstance(row.get("lymphocytes_present"), bool)
                or isinstance(row.get("rose_adequacy"), bool)
                or _is_meaningful(row.get("rose_result"))
            ):
                ctx["has_adequacy"] = True

    ordered = sorted(contexts.values(), key=lambda c: _station_sort_key(str(c.get("station") or "")))
    return ordered, global_needle_gauge_present


def generate_missing_field_prompts(record: RegistryRecord) -> list[MissingFieldPrompt]:
    """Generate "missing data" prompts from the extracted registry record.

    Philosophy:
    - Prompt only for fields that are useful for research/quality and safe to request.
    - Prefer explicit-only fields (do not infer missing values).
    - Keep prompts stable and UI-friendly (grouped, short labels, actionable messages).
    """
    data: dict[str, Any] = record.model_dump(exclude_none=False)
    evidence = getattr(record, "evidence", None)
    if not isinstance(evidence, dict):
        evidence = {}

    prompts: list[MissingFieldPrompt] = []

    def add_prompt_if_missing(
        *,
        group: str,
        path: str,
        target_path: str | None = None,
        label: str,
        severity: PromptSeverity,
        message: str,
        any_of_paths: list[str] | None = None,
        missing_sentinels_by_path: dict[str, set[object]] | None = None,
    ) -> None:
        paths = any_of_paths or [path]
        if _any_path_has_meaningful_value(
            data,
            paths=paths,
            missing_sentinels_by_path=missing_sentinels_by_path,
        ):
            return
        prompts.append(
            MissingFieldPrompt(
                group=group,
                path=path,
                target_path=target_path or path,
                label=label,
                severity=severity,
                message=message,
            )
        )

    # ---------------------------------------------------------------------
    # Global: always prompt if missing in the note/extraction
    # ---------------------------------------------------------------------
    add_prompt_if_missing(
        group="Global",
        path="patient_demographics.age_years",
        target_path="patient.age",
        label="Patient age (years)",
        severity="required",
        message="Age was not found in the note. Add age in years (e.g., “65-year-old”).",
        any_of_paths=["patient.age", "patient_demographics.age_years"],
    )

    add_prompt_if_missing(
        group="Global",
        path="clinical_context.asa_class",
        target_path="risk_assessment.asa_class",
        label="ASA class",
        severity="required",
        message="ASA class was not found in the note. Add ASA 1–6 (e.g., “ASA 3”).",
        any_of_paths=["risk_assessment.asa_class", "clinical_context.asa_class"],
    )

    add_prompt_if_missing(
        group="Global",
        path="clinical_context.ecog_score",
        label="ECOG/Zubrod performance status",
        severity="required",
        message="ECOG/Zubrod performance status was not found in the note. Add ECOG 0–4 (or a range like 0–1).",
        any_of_paths=["clinical_context.ecog_score", "clinical_context.ecog_text"],
    )

    # ---------------------------------------------------------------------
    # Procedure-specific prompts
    # ---------------------------------------------------------------------
    # Pleural: chest ultrasound
    pleural_us_performed = _get_bool(data, "pleural_procedures.chest_ultrasound.performed") is True
    if pleural_us_performed:
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.hemithorax",
            label="Chest ultrasound hemithorax",
            severity="required",
            message="Hemithorax (Right/Left/Bilateral) was not captured for chest ultrasound.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.effusion_volume",
            label="Pleural effusion volume",
            severity="required",
            message="Effusion volume (none/minimal/small/moderate/large) was not captured for chest ultrasound.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.effusion_loculations",
            label="Pleural effusion loculations",
            severity="required",
            message="Loculations (none/thin/thick) were not captured for chest ultrasound.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.effusion_echogenicity",
            label="Pleural effusion echogenicity",
            severity="recommended",
            message="Echogenicity (anechoic/hypoechoic/isoechoic/hyperechoic) was not captured for chest ultrasound.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.lung_sliding_pre",
            label="Lung sliding (pre)",
            severity="recommended",
            message="Pre-procedure lung sliding (present/absent) was not captured for chest ultrasound.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.lung_sliding_post",
            label="Lung sliding (post)",
            severity="recommended",
            message="Post-procedure lung sliding (present/absent) was not captured for chest ultrasound.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.lung_consolidation_present",
            label="Lung consolidation/atelectasis",
            severity="recommended",
            message="Consolidation/atelectasis presence was not captured for chest ultrasound.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.chest_ultrasound.pleura_characteristics",
            label="Pleura characteristics",
            severity="recommended",
            message="Pleura characteristics (normal/thick/nodular) were not captured for chest ultrasound.",
        )

    # Pleural: fibrinolytic therapy
    fibrinolysis_performed = _get_bool(data, "pleural_procedures.fibrinolytic_therapy.performed") is True
    if fibrinolysis_performed:
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.fibrinolytic_therapy.agents",
            label="Fibrinolytic agents",
            severity="required",
            message="Fibrinolytic agents were not captured. Document tPA/DNase (or other agents) when given.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.fibrinolytic_therapy.tpa_dose_mg",
            label="tPA dose (mg)",
            severity="required",
            message="tPA dose was not captured. Document dose in mg when given.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.fibrinolytic_therapy.dnase_dose_mg",
            label="DNase dose (mg)",
            severity="required",
            message="DNase dose was not captured. Document dose in mg when given.",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.fibrinolytic_therapy.number_of_doses",
            label="Number of doses",
            severity="required",
            message="Number of fibrinolytic doses was not captured. Document dose count (e.g., dose #2).",
        )
        add_prompt_if_missing(
            group="Pleural",
            path="pleural_procedures.fibrinolytic_therapy.indication",
            label="Fibrinolytic indication",
            severity="recommended",
            message="Indication for fibrinolytic therapy was not captured (e.g., empyema/complex parapneumonic).",
        )

    nav_performed = _get_bool(data, "procedures_performed.navigational_bronchoscopy.performed") is True
    if not nav_performed:
        nav_targets = _iter_path_values(data, "granular_data.navigation_targets")
        nav_performed = any(isinstance(v, list) and len(v) > 0 for v in nav_targets)

    if nav_performed:
        add_prompt_if_missing(
            group="Navigation",
            path="granular_data.navigation_targets[*].target_location_text",
            label="Navigation target location",
            severity="required",
            message="No navigation target location was captured. Document target location (lobe/segment/bronchus).",
        )
        add_prompt_if_missing(
            group="Navigation",
            path="granular_data.navigation_targets[*].lesion_size_mm",
            label="Lesion size (mm)",
            severity="recommended",
            message="Lesion size was not captured. Add a size (mm or cm) for the target lesion.",
        )
        add_prompt_if_missing(
            group="Navigation",
            path="granular_data.navigation_targets[*].ct_characteristics",
            label="CT characteristics",
            severity="recommended",
            message="CT characteristics were not captured. Add solid/part-solid/ground-glass/cavitary/calcified when documented.",
        )
        add_prompt_if_missing(
            group="Navigation",
            path="granular_data.navigation_targets[*].distance_from_pleura_mm",
            label="Distance from pleura (mm)",
            severity="recommended",
            message="Distance from pleura was not captured. Add distance (mm/cm) or state abutting pleura.",
        )
        add_prompt_if_missing(
            group="Navigation",
            path="granular_data.navigation_targets[*].pet_suv_max",
            label="PET SUV max",
            severity="recommended",
            message="SUV max was not captured. Add SUV (e.g., “SUV max 4.2”) when documented.",
        )

        # Bronchus sign exists both at clinical_context (aggregate) and per-target (granular).
        # Treat the default sentinel "Not assessed" as missing for prompting purposes.
        add_prompt_if_missing(
            group="Navigation",
            path="clinical_context.bronchus_sign",
            label="CT bronchus sign",
            severity="recommended",
            message="CT bronchus sign was not captured. Document positive/negative when explicitly stated.",
            any_of_paths=[
                "clinical_context.bronchus_sign",
                "granular_data.navigation_targets[*].bronchus_sign",
            ],
            missing_sentinels_by_path={
                "clinical_context.bronchus_sign": {"Not assessed"},
                "granular_data.navigation_targets[*].bronchus_sign": {"Not assessed"},
            },
        )

        add_prompt_if_missing(
            group="Navigation",
            path="granular_data.navigation_targets[*].registration_error_mm",
            label="Registration error (mm)",
            severity="recommended",
            message="Registration error was not captured. Document registration error (mm) if stated.",
        )

        add_prompt_if_missing(
            group="Navigation",
            path="procedures_performed.navigational_bronchoscopy.tool_in_lesion_confirmed",
            label="Tool-in-lesion confirmation",
            severity="recommended",
            message="Tool-in-lesion confirmation was not captured. Document whether tool-in-lesion was confirmed.",
            any_of_paths=[
                "procedures_performed.navigational_bronchoscopy.tool_in_lesion_confirmed",
                "granular_data.navigation_targets[*].tool_in_lesion_confirmed",
            ],
        )

        # Method: only prompt if tool-in-lesion is confirmed true but method missing.
        til_true = False
        if _get_bool(data, "procedures_performed.navigational_bronchoscopy.tool_in_lesion_confirmed") is True:
            til_true = True
        else:
            til_vals = _iter_path_values(data, "granular_data.navigation_targets[*].tool_in_lesion_confirmed")
            til_true = any(v is True for v in til_vals)

        if til_true:
            add_prompt_if_missing(
                group="Navigation",
                path="procedures_performed.navigational_bronchoscopy.confirmation_method",
                label="Tool-in-lesion confirmation method",
                severity="recommended",
                message="Tool-in-lesion method was not captured. Document rEBUS/CBCT/fluoroscopy/augmented fluoroscopy when confirmed.",
                any_of_paths=[
                    "procedures_performed.navigational_bronchoscopy.confirmation_method",
                    "granular_data.navigation_targets[*].confirmation_method",
                ],
            )

        add_prompt_if_missing(
            group="Navigation",
            path="procedures_performed.navigational_bronchoscopy.divergence_mm",
            label="CT-to-body divergence (mm)",
            severity="recommended",
            message="CT-to-body divergence/registration mismatch was not captured. Document divergence (mm) when stated.",
        )

    # Radial EBUS view classification completeness
    radial_performed = _get_bool(data, "procedures_performed.radial_ebus.performed") is True
    if not radial_performed:
        rebus_used_vals = _iter_path_values(data, "granular_data.navigation_targets[*].rebus_used")
        radial_performed = any(v is True for v in rebus_used_vals)

    if radial_performed:
        add_prompt_if_missing(
            group="Navigation",
            path="procedures_performed.radial_ebus.probe_position",
            label="Radial EBUS view classification",
            severity="recommended",
            message="Radial EBUS view was not captured. Document concentric/eccentric/adjacent/not visualized when radial EBUS is used.",
            any_of_paths=[
                "procedures_performed.radial_ebus.probe_position",
                "granular_data.navigation_targets[*].rebus_view",
            ],
        )

    # Linear EBUS per-station completeness
    ebus_performed = _get_bool(data, "procedures_performed.linear_ebus.performed") is True
    if not ebus_performed:
        stations = _iter_path_values(data, "granular_data.linear_ebus_stations_detail")
        ebus_performed = any(isinstance(v, list) and len(v) > 0 for v in stations)
    if not ebus_performed:
        node_events = _iter_path_values(data, "procedures_performed.linear_ebus.node_events")
        ebus_performed = any(isinstance(v, list) and len(v) > 0 for v in node_events)
    if not ebus_performed:
        sampled = _iter_path_values(data, "procedures_performed.linear_ebus.stations_sampled")
        ebus_performed = any(isinstance(v, list) and len(v) > 0 for v in sampled)

    if ebus_performed:
        existing_detail_rows = _iter_path_values(data, "granular_data.linear_ebus_stations_detail")
        detail_row_count = 0
        for value in existing_detail_rows:
            if isinstance(value, list):
                detail_row_count = len(value)
                break
        next_detail_idx = detail_row_count

        add_prompt_if_missing(
            group="EBUS",
            path="granular_data.linear_ebus_stations_detail[*].station",
            label="EBUS stations documented",
            severity="required",
            message="No per-station EBUS detail was captured. Document stations inspected/sampled (e.g., 4R, 7, 11L).",
        )
        station_contexts, global_needle_gauge_present = _build_ebus_station_contexts(data)
        sampled_contexts = [ctx for ctx in station_contexts if ctx.get("sampled") is True]
        station_targets = sampled_contexts or station_contexts

        for ctx in station_targets:
            station = str(ctx.get("station") or "").upper()
            if not station:
                continue
            detail_idx = _to_int_or_none(ctx.get("detail_index"))
            node_event_idx = _to_int_or_none(ctx.get("node_event_index"))
            detail_row_missing = detail_idx is None

            if detail_row_missing:
                detail_idx = next_detail_idx
                next_detail_idx += 1
                prompts.append(
                    MissingFieldPrompt(
                        group="EBUS",
                        path="granular_data.linear_ebus_stations_detail[*].station",
                        target_path=f"granular_data.linear_ebus_stations_detail[{detail_idx}].station",
                        label=f"Per-station detail row (station {station})",
                        severity="recommended",
                        message=(
                            f"Station {station} appears sampled, but no per-station detail row is present. "
                            "Add a station row to capture size, passes, and adequacy consistently."
                        ),
                    )
                )

            if not ctx.get("has_needle_gauge") and not global_needle_gauge_present:
                target_path = (
                    f"granular_data.linear_ebus_stations_detail[{detail_idx}].needle_gauge"
                    if detail_idx is not None
                    else "granular_data.linear_ebus_stations_detail[*].needle_gauge"
                )
                prompts.append(
                    MissingFieldPrompt(
                        group="EBUS",
                        path="granular_data.linear_ebus_stations_detail[*].needle_gauge",
                        target_path=target_path,
                        label=f"Needle gauge (station {station})",
                        severity="recommended",
                        message=f"Needle gauge for station {station} was not captured. Document 19/21/22/25G when stated.",
                    )
                )

            if not ctx.get("has_passes"):
                if detail_idx is not None:
                    target_path = f"granular_data.linear_ebus_stations_detail[{detail_idx}].number_of_passes"
                elif node_event_idx is not None:
                    target_path = f"procedures_performed.linear_ebus.node_events[{node_event_idx}].passes"
                else:
                    # If no detail row and no node event exists, ask for row creation only.
                    target_path = ""
                if not target_path:
                    continue
                prompts.append(
                    MissingFieldPrompt(
                        group="EBUS",
                        path="granular_data.linear_ebus_stations_detail[*].number_of_passes",
                        target_path=target_path,
                        label=f"Passes (station {station})",
                        severity="recommended",
                        message=f"Pass count for station {station} was not captured. Document number of needle passes when stated.",
                    )
                )

            if not ctx.get("has_short_axis"):
                target_path = (
                    f"granular_data.linear_ebus_stations_detail[{detail_idx}].short_axis_mm"
                    if detail_idx is not None
                    else "granular_data.linear_ebus_stations_detail[*].short_axis_mm"
                )
                prompts.append(
                    MissingFieldPrompt(
                        group="EBUS",
                        path="granular_data.linear_ebus_stations_detail[*].short_axis_mm",
                        target_path=target_path,
                        label=f"Node size short axis (station {station})",
                        severity="recommended",
                        message=f"Short-axis node size for station {station} was not captured. Document node size when stated.",
                    )
                )

            if not ctx.get("has_adequacy"):
                target_path = (
                    f"granular_data.linear_ebus_stations_detail[{detail_idx}].lymphocytes_present"
                    if detail_idx is not None
                    else "granular_data.linear_ebus_stations_detail[*].lymphocytes_present"
                )
                prompts.append(
                    MissingFieldPrompt(
                        group="EBUS",
                        path="granular_data.linear_ebus_stations_detail[*].lymphocytes_present",
                        target_path=target_path,
                        label=f"Adequacy/ROSE (station {station})",
                        severity="recommended",
                        message=(
                            f"Adequacy/ROSE for station {station} was not captured in per-station detail. "
                            "Document lymphocytes present or station-level ROSE adequacy when stated."
                        ),
                    )
                )

    # Complications: pneumothorax intervention level
    if _get_bool(data, "complications.pneumothorax.occurred") is True:
        add_prompt_if_missing(
            group="Complications",
            path="complications.pneumothorax.intervention",
            label="Pneumothorax intervention",
            severity="required",
            message="Pneumothorax intervention level was not captured. Document observation/aspiration/pigtail/chest tube/surgery when applicable.",
        )

    # Complications: Nashville bleeding grade when bleeding occurred
    if _get_bool(data, "complications.bleeding.occurred") is True:
        add_prompt_if_missing(
            group="Complications",
            path="complications.bleeding.bleeding_grade_nashville",
            label="Bleeding grade (Nashville 0–4)",
            severity="required",
            message="Bleeding grade was not captured. Document hemostasis interventions (suction/saline/epi/blocker/transfusion) to support Nashville grading.",
        )

    # De-duplicate prompts by (path, label) while preserving order.
    deduped: list[MissingFieldPrompt] = []
    seen: set[tuple[str, str]] = set()
    for prompt in prompts:
        key = (prompt.path, prompt.label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(prompt)

    return deduped


__all__ = ["MissingFieldPrompt", "PromptSeverity", "generate_missing_field_prompts"]
