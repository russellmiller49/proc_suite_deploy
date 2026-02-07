from __future__ import annotations

import re
from collections import Counter

from app.registry.schema import RegistryRecord
from app.registry.schema.ip_v3_extraction import IPRegistryV3, ProcedureEvent


_GAUGE_RE = re.compile(r"\b(19|21|22|23)\s*g\b", re.IGNORECASE)


def project_v3_to_v2(v3_registry: IPRegistryV3) -> RegistryRecord:
    """Project a V3 event-log registry into the legacy V2 RegistryRecord shape."""
    record_data: dict = {}

    procedures_performed: dict[str, dict] = {}
    pleural_procedures: dict[str, dict] = {}

    linear_ebus_stations: set[str] = set()
    ebus_stations_sampled: set[str] = set()
    ebus_needle_gauges: list[str] = []

    lesion_size_mm_candidates: list[float] = []
    lesion_long_axis_mm_candidates: list[float] = []
    lesion_short_axis_mm_candidates: list[float] = []
    lesion_craniocaudal_mm_candidates: list[float] = []
    lesion_suv_max_candidates: list[float] = []
    lesion_location_candidates: list[str] = []
    lesion_morphology_candidates: list[str] = []
    lesion_size_text_candidates: list[str] = []

    pre_obstruction_pct_candidates: list[int] = []
    post_obstruction_pct_candidates: list[int] = []
    pre_diameter_mm_candidates: list[float] = []
    post_diameter_mm_candidates: list[float] = []

    def _ensure_proc(name: str) -> dict:
        proc = procedures_performed.get(name)
        if proc is None:
            proc = {"performed": True}
            procedures_performed[name] = proc
        else:
            proc["performed"] = True
        return proc

    def _ensure_pleural(name: str) -> dict:
        proc = pleural_procedures.get(name)
        if proc is None:
            proc = {"performed": True}
            pleural_procedures[name] = proc
        else:
            proc["performed"] = True
        return proc

    def _event_type(event: ProcedureEvent) -> str:
        raw = (event.type or "").strip().lower()
        raw = raw.replace("-", "_").replace(" ", "_")
        raw = re.sub(r"[^a-z0-9_]+", "", raw)
        return raw

    def _station(event: ProcedureEvent) -> str | None:
        station = getattr(getattr(event, "target", None), "station", None)
        if not station:
            return None
        station = str(station).strip()
        return station.upper() if station else None

    def _location_str(event: ProcedureEvent) -> str | None:
        target = getattr(event, "target", None)
        if target is None:
            return None
        parts = []
        for val in (getattr(target, "lobe", None), getattr(target, "segment", None), getattr(target, "station", None)):
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        if not parts:
            return None
        return " ".join(parts)

    def _needle_gauges_from_event(event: ProcedureEvent) -> list[str]:
        gauges: list[str] = []
        for device in getattr(event, "devices", []) or []:
            if not isinstance(device, str):
                continue
            m = _GAUGE_RE.search(device)
            if m:
                gauges.append(f"{m.group(1)}G")
        return gauges

    def _pick_gauge(gauges: list[str]) -> str | None:
        if not gauges:
            return None
        counts = Counter(gauges)
        return counts.most_common(1)[0][0]

    def _unique_float(values: list[float], *, places: int = 1) -> float | None:
        if not values:
            return None
        uniques = {round(float(v), places) for v in values}
        return next(iter(uniques)) if len(uniques) == 1 else None

    def _unique_int(values: list[int]) -> int | None:
        if not values:
            return None
        uniques = {int(v) for v in values}
        return next(iter(uniques)) if len(uniques) == 1 else None

    def _unique_str(values: list[str]) -> str | None:
        cleaned = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
        uniques = {v for v in cleaned}
        return next(iter(uniques)) if len(uniques) == 1 else None

    for event in v3_registry.procedures:
        typ = _event_type(event)
        station = _station(event)
        location = _location_str(event)

        lesion = getattr(event, "lesion", None)
        if lesion is not None:
            raw = getattr(lesion, "size_mm", None)
            if isinstance(raw, (int, float)):
                lesion_size_mm_candidates.append(float(raw))
            raw = getattr(lesion, "long_axis_mm", None)
            if isinstance(raw, (int, float)):
                lesion_long_axis_mm_candidates.append(float(raw))
            raw = getattr(lesion, "short_axis_mm", None)
            if isinstance(raw, (int, float)):
                lesion_short_axis_mm_candidates.append(float(raw))
            raw = getattr(lesion, "craniocaudal_mm", None)
            if isinstance(raw, (int, float)):
                lesion_craniocaudal_mm_candidates.append(float(raw))
            raw = getattr(lesion, "suv_max", None)
            if isinstance(raw, (int, float)):
                lesion_suv_max_candidates.append(float(raw))
            raw = getattr(lesion, "location", None)
            if isinstance(raw, str) and raw.strip():
                lesion_location_candidates.append(raw)
            raw = getattr(lesion, "morphology", None)
            if isinstance(raw, str) and raw.strip():
                lesion_morphology_candidates.append(raw)
            raw = getattr(lesion, "size_text", None)
            if isinstance(raw, str) and raw.strip():
                lesion_size_text_candidates.append(raw)

        outcomes = getattr(event, "outcomes", None)
        if outcomes is not None:
            raw = getattr(outcomes, "pre_obstruction_pct", None)
            if isinstance(raw, int):
                pre_obstruction_pct_candidates.append(raw)
            raw = getattr(outcomes, "post_obstruction_pct", None)
            if isinstance(raw, int):
                post_obstruction_pct_candidates.append(raw)
            raw = getattr(outcomes, "pre_diameter_mm", None)
            if isinstance(raw, (int, float)):
                pre_diameter_mm_candidates.append(float(raw))
            raw = getattr(outcomes, "post_diameter_mm", None)
            if isinstance(raw, (int, float)):
                post_diameter_mm_candidates.append(float(raw))

        if typ in {"diagnostic", "diagnostic_bronchoscopy"}:
            _ensure_proc("diagnostic_bronchoscopy")
            continue

        if typ in {"bal"}:
            proc = _ensure_proc("bal")
            if location:
                proc.setdefault("location", location)
            continue

        if typ in {"brushing", "brushings"}:
            proc = _ensure_proc("brushings")
            if location:
                proc.setdefault("locations", [])
                if location not in proc["locations"]:
                    proc["locations"].append(location)
            continue

        if typ in {"endobronchial_biopsy", "endobronchialbiopsy"}:
            proc = _ensure_proc("endobronchial_biopsy")
            if location:
                proc.setdefault("locations", [])
                if location not in proc["locations"]:
                    proc["locations"].append(location)
            continue

        if typ in {"tbna", "tbna_conventional"}:
            proc = _ensure_proc("tbna_conventional")
            if station:
                proc.setdefault("stations_sampled", [])
                if station not in proc["stations_sampled"]:
                    proc["stations_sampled"].append(station)
            gauges = _needle_gauges_from_event(event)
            if gauges:
                proc.setdefault("needle_gauge", _pick_gauge(gauges))
            continue

        if typ in {"ebus_tbna", "linear_ebus", "ebus_inspection"}:
            proc = _ensure_proc("linear_ebus")
            if station:
                proc.setdefault("stations_sampled", [])
                if station not in proc["stations_sampled"]:
                    proc["stations_sampled"].append(station)
                linear_ebus_stations.add(station)
                ebus_stations_sampled.add(station)
            ebus_needle_gauges.extend(_needle_gauges_from_event(event))
            continue

        if typ in {"radial_ebus"}:
            _ensure_proc("radial_ebus")
            continue

        if typ in {"navigation", "navigational_bronchoscopy"}:
            _ensure_proc("navigational_bronchoscopy")
            continue

        if typ in {"tbbx", "transbronchial_biopsy"}:
            proc = _ensure_proc("transbronchial_biopsy")
            if location:
                proc.setdefault("locations", [])
                if location not in proc["locations"]:
                    proc["locations"].append(location)
            continue

        if typ in {"cryobiopsy", "transbronchial_cryobiopsy"}:
            proc = _ensure_proc("transbronchial_cryobiopsy")
            if location:
                proc.setdefault("locations_biopsied", [])
                if location not in proc["locations_biopsied"]:
                    proc["locations_biopsied"].append(location)
            continue

        if typ in {"therapeutic_aspiration"}:
            proc = _ensure_proc("therapeutic_aspiration")
            if location:
                proc.setdefault("location", location)
            continue

        if typ in {"airway_dilation", "recanalizationdilation"}:
            proc = _ensure_proc("airway_dilation")
            if location:
                proc.setdefault("location", location)
            continue

        if typ in {"airway_stent", "stent"}:
            proc = _ensure_proc("airway_stent")
            if location:
                proc.setdefault("location", location)

            stent_brand = getattr(event, "stent_material_or_brand", None)
            if isinstance(stent_brand, str) and stent_brand.strip():
                proc.setdefault("stent_brand", stent_brand.strip())

            stent_size = getattr(event, "stent_size", None)
            if isinstance(stent_size, str) and stent_size.strip():
                # Common shape: "14x40" (diameter x length, mm).
                m = re.search(r"(?P<d>\\d+(?:\\.\\d+)?)\\s*[x×]\\s*(?P<l>\\d+(?:\\.\\d+)?)", stent_size.lower())
                if m:
                    try:
                        proc.setdefault("diameter_mm", float(m.group("d")))
                        proc.setdefault("length_mm", float(m.group("l")))
                    except Exception:
                        pass
            continue

        if typ in {"blvr"}:
            proc = _ensure_proc("blvr")
            target_lobe = getattr(getattr(event, "target", None), "lobe", None)
            if isinstance(target_lobe, str) and target_lobe.strip():
                proc.setdefault("target_lobe", target_lobe.strip())
            continue

        if typ in {"thoracentesis"}:
            _ensure_pleural("thoracentesis")
            continue

        if typ in {"chest_tube"}:
            proc = _ensure_pleural("chest_tube")
            size_fr = getattr(event, "catheter_size_fr", None)
            if isinstance(size_fr, (int, float)):
                proc.setdefault("tube_size_fr", float(size_fr))
            continue

        if typ in {"ipc", "pleurx"}:
            _ensure_pleural("ipc")
            continue

        if typ in {"pleurodesis"}:
            _ensure_pleural("pleurodesis")
            continue

        if typ in {"fibrinolytic_therapy"}:
            _ensure_pleural("fibrinolytic_therapy")
            continue

    clinical_context: dict[str, object] = {}
    target_lesion: dict[str, object] = {}

    lesion_size_mm = _unique_float(lesion_size_mm_candidates)
    if lesion_size_mm is not None:
        clinical_context["lesion_size_mm"] = lesion_size_mm

    lesion_suv_max = _unique_float(lesion_suv_max_candidates)
    if lesion_suv_max is not None:
        clinical_context["suv_max"] = lesion_suv_max
        target_lesion["suv_max"] = lesion_suv_max

    long_axis_mm = _unique_float(lesion_long_axis_mm_candidates)
    if long_axis_mm is not None:
        target_lesion["long_axis_mm"] = long_axis_mm

    short_axis_mm = _unique_float(lesion_short_axis_mm_candidates)
    if short_axis_mm is not None:
        target_lesion["short_axis_mm"] = short_axis_mm

    craniocaudal_mm = _unique_float(lesion_craniocaudal_mm_candidates)
    if craniocaudal_mm is not None:
        target_lesion["craniocaudal_mm"] = craniocaudal_mm

    morphology = _unique_str(lesion_morphology_candidates)
    if morphology is not None:
        target_lesion["morphology"] = morphology

    lesion_location = _unique_str(lesion_location_candidates)
    if lesion_location is not None:
        clinical_context["lesion_location"] = lesion_location
        target_lesion["location"] = lesion_location

    size_text = _unique_str(lesion_size_text_candidates)
    if size_text is not None:
        target_lesion["size_text"] = size_text

    if target_lesion:
        clinical_context["target_lesion"] = target_lesion
    if clinical_context:
        record_data["clinical_context"] = clinical_context

    therapeutic_outcomes: dict[str, object] = {}
    pre_obstruction = _unique_int(pre_obstruction_pct_candidates)
    if pre_obstruction is not None:
        therapeutic_outcomes["pre_obstruction_pct"] = pre_obstruction
    post_obstruction = _unique_int(post_obstruction_pct_candidates)
    if post_obstruction is not None:
        therapeutic_outcomes["post_obstruction_pct"] = post_obstruction
    pre_diameter = _unique_float(pre_diameter_mm_candidates)
    if pre_diameter is not None:
        therapeutic_outcomes["pre_diameter_mm"] = pre_diameter
    post_diameter = _unique_float(post_diameter_mm_candidates)
    if post_diameter is not None:
        therapeutic_outcomes["post_diameter_mm"] = post_diameter
    if therapeutic_outcomes:
        procedures_performed.setdefault("therapeutic_outcomes", {}).update(therapeutic_outcomes)

    if procedures_performed:
        record_data["procedures_performed"] = procedures_performed
    if pleural_procedures:
        record_data["pleural_procedures"] = pleural_procedures

    if linear_ebus_stations:
        record_data["linear_ebus_stations"] = sorted(linear_ebus_stations)
    if ebus_stations_sampled:
        record_data["ebus_stations_sampled"] = sorted(ebus_stations_sampled)

    if ebus_needle_gauges and "procedures_performed" in record_data:
        gauge = _pick_gauge(ebus_needle_gauges)
        if gauge and isinstance(procedures_performed.get("linear_ebus"), dict):
            procedures_performed["linear_ebus"].setdefault("needle_gauge", gauge)

    return RegistryRecord.model_validate(record_data)

def convert_v3_to_v2(v3_registry: IPRegistryV3) -> RegistryRecord:
    """Alias for `project_v3_to_v2` (kept for clearer adapter naming)."""
    return project_v3_to_v2(v3_registry)

__all__ = ["project_v3_to_v2", "convert_v3_to_v2"]
