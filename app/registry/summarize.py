"""Helpers for generating UI-friendly summary strings for registry procedures."""

from __future__ import annotations

from typing import Any, Mapping


def generate_procedure_summary(proc_type: str, details: Mapping[str, Any]) -> str:
    """Generate a natural language summary for a single procedure dict.

    The goal is a compact, UI-ready string derived from common fields without
    leaking PHI (no note text, names, MRNs, etc.).
    """
    performed = details.get("performed")
    if performed is not True:
        return "Not Performed"

    parts: list[str] = ["Performed"]

    passes_per_station = details.get("passes_per_station")
    if isinstance(passes_per_station, int) and passes_per_station > 0:
        parts.append(f"{passes_per_station} passes")

    stations_sampled = details.get("stations_sampled")
    if isinstance(stations_sampled, list):
        stations = [str(s).strip() for s in stations_sampled if isinstance(s, str) and s.strip()]
        if stations:
            parts.append(f"stations: {', '.join(stations)}")

    location = details.get("location")
    if isinstance(location, str) and location.strip():
        parts.append(f"location: {location.strip()}")
    else:
        locations = details.get("locations")
        if isinstance(locations, list):
            loc_list = [str(s).strip() for s in locations if isinstance(s, str) and s.strip()]
            if loc_list:
                parts.append(f"location: {', '.join(loc_list)}")

    material = details.get("material")
    if isinstance(material, str) and material.strip():
        parts.append(f"material: {material.strip()}")

    number_of_samples = details.get("number_of_samples")
    if isinstance(number_of_samples, int) and number_of_samples > 0:
        parts.append(f"{number_of_samples} samples")

    volume_instilled = details.get("volume_instilled_ml")
    volume_returned = details.get("volume_returned_ml")
    if isinstance(volume_instilled, (int, float)) or isinstance(volume_returned, (int, float)):
        instilled_str = f"{int(volume_instilled)}" if isinstance(volume_instilled, (int, float)) else "?"
        returned_str = f"{int(volume_returned)}" if isinstance(volume_returned, (int, float)) else "?"
        parts.append(f"volume: {instilled_str}ml/{returned_str}ml")

    # proc_type is currently unused but reserved for future type-specific summaries.
    _ = proc_type
    return "; ".join(parts)


def add_procedure_summaries(record: dict[str, Any]) -> None:
    """Add `summary` fields to procedure dicts inside a registry record dict."""
    procedures = record.get("procedures_performed")
    if isinstance(procedures, dict):
        for proc_name, proc_data in procedures.items():
            if isinstance(proc_data, dict):
                proc_data["summary"] = generate_procedure_summary(proc_name, proc_data)

    pleural = record.get("pleural_procedures")
    if isinstance(pleural, dict):
        for proc_name, proc_data in pleural.items():
            if isinstance(proc_data, dict):
                proc_data["summary"] = generate_procedure_summary(proc_name, proc_data)


__all__ = ["generate_procedure_summary", "add_procedure_summaries"]

