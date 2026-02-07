"""Helpers for populating navigation target fiducial details."""

from __future__ import annotations

import re
from typing import Any


def apply_navigation_fiducials(data: dict[str, Any], text: str) -> bool:
    """Populate granular navigation targets when fiducial markers are documented.

    Returns True if data was modified.
    """
    if not isinstance(data, dict):
        return False

    families = {str(x) for x in (data.get("procedure_families") or []) if x}
    if families and "NAVIGATION" not in families:
        return False

    lowered = (text or "").lower()
    if "fiducial" not in lowered:
        return False

    fiducial_sentence: str | None = None
    fiducial_pattern = r"\bfiducial(?:\s+marker)?s?\b"
    for line in text.splitlines():
        if re.search(fiducial_pattern, line, re.IGNORECASE):
            fiducial_sentence = line.strip()
            break
    if fiducial_sentence is None:
        match = re.search(fiducial_pattern, text, re.IGNORECASE)
        if match:
            window = text[match.start() : match.start() + 300]
            fiducial_sentence = window.splitlines()[0].strip() if window else None
    if fiducial_sentence is None:
        return False

    fiducial_lower = fiducial_sentence.lower()
    if not re.search(r"\b(?:plac(?:ed|ement)|deploy\w*|position\w*|insert\w*)\b", fiducial_lower):
        return False
    if re.search(r"\b(?:no|not|without)\b", fiducial_lower):
        return False

    def _is_placeholder_location(value: object) -> bool:
        if value is None:
            return True
        s = str(value).strip().lower()
        if not s:
            return True
        return s in {
            "unknown",
            "unknown target",
            "target",
            "target lesion",
            "target lesion 1",
            "target lesion 2",
            "target lesion 3",
        }

    def _extract_target_location() -> str:
        for pattern in (
            r"\bengage(?:d)?\s+the\s+([^\n.]{3,200})",
            r"\bnavigate(?:d|ion)?\s+to\s+([^\n.]{3,200})",
            r"\btarget(?:ed)?\s+lesion\s+(?:is\s+)?(?:in|at)\s+([^\n.]{3,200})",
        ):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                loc = match.group(1).strip()
                if loc:
                    return loc
        return "Unknown target"

    def _extract_target_lobe(location_text: str) -> str | None:
        upper = location_text.upper()
        for token in ("RUL", "RML", "RLL", "LUL", "LLL"):
            if re.search(rf"\b{token}\b", upper):
                return token
        if "LINGULA" in upper:
            return "Lingula"
        if "LEFT UPPER" in upper:
            return "LUL"
        if "LEFT LOWER" in upper:
            return "LLL"
        if "RIGHT UPPER" in upper:
            return "RUL"
        if "RIGHT MIDDLE" in upper:
            return "RML"
        if "RIGHT LOWER" in upper:
            return "RLL"
        return None

    def _extract_lesion_size_mm() -> float | None:
        cm_match = re.search(r"\blesion\b[^.\n]{0,80}\b(\d+(?:\.\d+)?)\s*cm\b", lowered)
        if cm_match:
            try:
                return float(cm_match.group(1)) * 10.0
            except ValueError:
                return None
        mm_match = re.search(r"\blesion\b[^.\n]{0,80}\b(\d+(?:\.\d+)?)\s*mm\b", lowered)
        if mm_match:
            try:
                return float(mm_match.group(1))
            except ValueError:
                return None
        return None

    def _extract_segment(location_text: str) -> str | None:
        match = re.search(r"\((LB[^)]+)\)", location_text, re.IGNORECASE)
        if match:
            seg = match.group(1).strip()
            return seg or None
        return None

    granular_raw = data.get("granular_data")
    granular: dict[str, Any]
    if granular_raw is None:
        granular = {}
    elif isinstance(granular_raw, dict):
        granular = dict(granular_raw)
    else:
        return False

    targets_raw = granular.get("navigation_targets")
    if isinstance(targets_raw, list):
        targets = [t for t in targets_raw if isinstance(t, dict)]
    else:
        targets = []

    modified = False
    details = fiducial_sentence
    location = _extract_target_location()
    lesion_size_mm = _extract_lesion_size_mm()
    extracted_lobe = _extract_target_lobe(location)
    extracted_segment = _extract_segment(location)

    if targets:
        target0 = dict(targets[0])
        if target0.get("target_number") in (None, ""):
            target0["target_number"] = 1
            modified = True
        if _is_placeholder_location(target0.get("target_location_text")):
            target0["target_location_text"] = location
            modified = True
        if extracted_lobe is not None and target0.get("target_lobe") in (None, ""):
            target0["target_lobe"] = extracted_lobe
            modified = True
        if extracted_segment is not None and target0.get("target_segment") in (None, ""):
            target0["target_segment"] = extracted_segment
            modified = True
        if lesion_size_mm is not None and target0.get("lesion_size_mm") in (None, ""):
            target0["lesion_size_mm"] = lesion_size_mm
            modified = True
        if target0.get("fiducial_marker_placed") is not True:
            target0["fiducial_marker_placed"] = True
            modified = True
        if target0.get("fiducial_marker_details") in (None, ""):
            target0["fiducial_marker_details"] = details
            modified = True
        targets[0] = target0
    else:
        target_payload: dict[str, Any] = {
            "target_number": 1,
            "target_location_text": location,
            "fiducial_marker_placed": True,
            "fiducial_marker_details": details,
        }
        if extracted_lobe is not None:
            target_payload["target_lobe"] = extracted_lobe
        if extracted_segment is not None:
            target_payload["target_segment"] = extracted_segment
        if lesion_size_mm is not None:
            target_payload["lesion_size_mm"] = lesion_size_mm
        targets = [target_payload]
        modified = True

    if not modified:
        return False

    granular["navigation_targets"] = targets
    data["granular_data"] = granular
    return True


__all__ = ["apply_navigation_fiducials"]
