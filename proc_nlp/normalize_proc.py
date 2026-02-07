"""Rule-based normalization helpers for dictations and extractor hints."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional


_STATION_RE = re.compile(r"station[s]?\s*(?P<station>[0-9]{1,2}[LR]?)", re.IGNORECASE)
_PASSES_RE = re.compile(r"(?P<count>[0-9]+)\s*(?:passes|samples|bx)", re.IGNORECASE)
_LATERALITY_MAP = {
    "right": "right",
    "rml": "right",
    "rll": "right",
    "left": "left",
    "lingula": "left",
    "bilateral": "bilateral",
}


def _infer_type(text: str, hints: Optional[Dict]) -> str:
    blob = f"{text} {(hints or {}).get('procedure_type','')}".lower()
    if "ebus" in blob or "tbna" in blob:
        return "ebus_tbna"
    if "cryobiopsy" in blob:
        return "cryobiopsy"
    if "thoracentesis" in blob:
        return "thoracentesis"
    if "pleuro" in blob:
        return "pleuroscopy"
    if "ipc" in blob:
        return "ipc"
    if "stent" in blob:
        return "stent"
    if "robotic" in blob or "ion" in blob or "monarch" in blob:
        return "robotic_nav"
    return "bronchoscopy"


def _infer_laterality(text: str, hints: Optional[Dict]) -> Optional[str]:
    for key, val in (hints or {}).items():
        if key.lower().startswith("laterality") and val:
            return str(val).lower()
    blob = text.lower()
    for token, normalized in _LATERALITY_MAP.items():
        if token in blob:
            return normalized
    return None


def _extract_stations(text: str) -> List[str]:
    stations = [m.group("station").upper() for m in _STATION_RE.finditer(text)]
    # Deduplicate but keep order
    seen = set()
    ordered = []
    for station in stations:
        if station not in seen:
            ordered.append(station)
            seen.add(station)
    return ordered


def _extract_targets(text: str) -> List[Dict]:
    stations = _extract_stations(text)
    passes = _PASSES_RE.findall(text)
    pass_count = int(passes[0]) if passes else 3
    targets = []
    for station in stations:
        specimens = {"fna": pass_count}
        targets.append({
            "lobe": None,
            "segment": station,
            "guidance": "radial_ebus" if station.isdigit() else "fluoro",
            "specimens": specimens,
        })
    return targets


def normalize_dictation(text: str, hints: Optional[Dict] = None) -> Dict:
    """Return normalized fields compatible with ProcedureCore."""
    hints = hints or {}
    proc_type = _infer_type(text, hints)
    laterality = _infer_laterality(text, hints)
    stations = _extract_stations(text)
    targets = _extract_targets(text)

    devices: Dict[str, str] = {}
    blob = text.lower()
    if "ion" in blob:
        devices["robot"] = "ion"
    if "monarch" in blob:
        devices["robot"] = "monarch"
    if "ebus" in blob:
        devices.setdefault("scope", "ebus")

    fluoro: Dict[str, str] = {}
    if "fluoro" in blob:
        fluoro["used"] = "yes"

    return {
        "type": proc_type,
        "laterality": laterality,
        "stations_sampled": stations,
        "targets": targets,
        "devices": devices,
        "fluoro": fluoro,
    }


__all__ = ["normalize_dictation"]
