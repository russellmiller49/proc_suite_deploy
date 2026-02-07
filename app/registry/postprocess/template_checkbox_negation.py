from __future__ import annotations

import re
from typing import Any

from app.registry.schema import RegistryRecord


_CHECKBOX_NEGATIVE_DASH_RE = re.compile(r"(?im)^\s*0\s*[—\-]\s*(?P<label>.+?)\s*$")
_CHECKBOX_NEGATIVE_BRACKET_RE = re.compile(r"(?im)^\s*\[\s*\]\s*(?P<label>.+?)\s*$")
_CHECKBOX_NEGATIVE_UNICODE_RE = re.compile(r"(?im)^\s*[☐□]\s*(?P<label>.+?)\s*$")
_CHECKBOX_TOKEN_RE = re.compile(
    r"(?im)(?<!\d)(?P<val>[01])\s*[^\w\n]{0,6}\s*(?P<label>[A-Za-z][A-Za-z /()_-]{0,80})"
)


def _get_field(record_data: dict[str, Any], field_path: str) -> object:
    current: object = record_data
    for part in [p for p in field_path.split(".") if p]:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_field(record_data: dict[str, Any], field_path: str, value: object) -> bool:
    parts = [p for p in field_path.split(".") if p]
    if not parts:
        return False

    current: dict[str, Any] = record_data
    for part in parts[:-1]:
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]

    leaf = parts[-1]
    prior = current.get(leaf)
    current[leaf] = value
    return prior != value


def _wipe_object_fields(record_data: dict[str, Any], prefix: str, fields: dict[str, object]) -> bool:
    changed = False
    for field_name, value in fields.items():
        changed |= _set_field(record_data, f"{prefix}.{field_name}", value)
    return changed


def apply_template_checkbox_negation(note_text: str, record: RegistryRecord) -> tuple[RegistryRecord, list[str]]:
    """Force explicit checkbox-template negatives to False.

    Templates often encode unchecked options as:
      - "0- Chest tube"
      - "[ ] Tunneled Pleural Catheter"
      - "☐ Airway dilation"
    These must never be interpreted as performed/true.
    """
    text = note_text or ""
    if ("\n" not in text and "\r" not in text) and ("\\n" in text or "\\r" in text):
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    if not text.strip():
        return record, []

    negative_labels: list[str] = []
    for match in _CHECKBOX_NEGATIVE_DASH_RE.finditer(text):
        negative_labels.append(match.group("label") or "")
    for match in _CHECKBOX_NEGATIVE_BRACKET_RE.finditer(text):
        negative_labels.append(match.group("label") or "")
    for match in _CHECKBOX_NEGATIVE_UNICODE_RE.finditer(text):
        negative_labels.append(match.group("label") or "")
    for match in _CHECKBOX_TOKEN_RE.finditer(text):
        if (match.group("val") or "").strip() != "0":
            continue
        negative_labels.append(match.group("label") or "")

    if not negative_labels:
        return record, []

    record_data: dict[str, Any] = record.model_dump()
    warnings: list[str] = []
    changed = False

    def _positive_evidence_present(field_path: str) -> bool:
        """Return True when narrative evidence supports keeping the field True.

        Critical hierarchy of truth: explicit narrative evidence supersedes unchecked
        template checkboxes. This prevents template artifacts from forcing performed
        flags to False when the procedure is clearly documented as performed.
        """
        if not text:
            return False

        patterns_by_path: dict[str, tuple[str, ...]] = {
            "pleural_procedures.ipc.performed": (
                r"(?i)\b(?:pleurx|aspira|indwelling\s+pleural\s+catheter|tunneled\s+pleural\s+catheter|ipc)\b"
                r"[^.\n]{0,120}\b(?:placed|insert(?:ed|ion)|tunnel(?:ed|ing)|deploy(?:ed|ment)|secured)\b",
                r"(?i)\b(?:placed|insert(?:ed|ion)|tunnel(?:ed|ing)|deploy(?:ed|ment)|secured)\b"
                r"[^.\n]{0,120}\b(?:pleurx|aspira|indwelling\s+pleural\s+catheter|tunneled\s+pleural\s+catheter|ipc)\b",
            ),
            "pleural_procedures.chest_tube.performed": (
                r"(?i)\b(?:chest\s+tube|tube\s+thoracostomy|pigtail)\b"
                r"[^.\n]{0,120}\b(?:placed|insert(?:ed|ion)|introduced|advanced|connected|secured)\b",
                r"(?i)\b(?:placed|insert(?:ed|ion)|introduced|advanced|connected|secured)\b"
                r"[^.\n]{0,120}\b(?:chest\s+tube|tube\s+thoracostomy|pigtail)\b",
            ),
            "complications.pneumothorax.occurred": (
                r"(?i)\bpneumothorax\b[^.\n]{0,120}\b(?:noted|occurred|developed|present|post|small|trace)\b",
                r"(?i)\b(?:noted|occurred|developed|present|post|small|trace)\b[^.\n]{0,120}\bpneumothorax\b",
            ),
            "procedures_performed.airway_dilation.performed": (
                r"(?i)\b(?:airway|tracheal|bronchial)\b[^.\n]{0,120}\b(?:dilat(?:e|ed|ion)|balloon)\b"
                r"[^.\n]{0,80}\b(?:performed|completed|successfully)\b",
                r"(?i)\b(?:balloon)\b[^.\n]{0,120}\b(?:dilat(?:e|ed|ion))\b",
            ),
            "procedures_performed.rigid_bronchoscopy.performed": (
                r"(?i)\brigid\s+bronchoscop\w*\b[^.\n]{0,120}\b(?:performed|using|with)\b",
                r"(?i)\busing\b[^.\n]{0,120}\brigid\s+(?:scope|bronchoscop\w*)\b",
            ),
            "procedures_performed.airway_stent.performed": (
                r"(?i)\b(?:airway\s+)?stent\b[^.\n]{0,120}\b(?:deploy(?:ed|ment)|place(?:d|ment)|insert(?:ed|ion)|revis(?:ed|ion)|reposition(?:ed|ing)|adjust(?:ed|ment))\b",
                r"(?i)\b(?:deploy(?:ed|ment)|place(?:d|ment)|insert(?:ed|ion)|revis(?:ed|ion)|reposition(?:ed|ing)|adjust(?:ed|ment))\b[^.\n]{0,120}\b(?:airway\s+)?stent\b",
                r"(?i)\b(?:grasp(?:ed)?|pull(?:ed)?|move(?:d)?)\b[^.\n]{0,120}\b(?:airway\s+)?stent\b[^.\n]{0,80}\bproxim(?:al|ally)\b",
            ),
        }

        patterns = patterns_by_path.get(field_path)
        if not patterns:
            return False

        lowered = text.lower()
        # Avoid treating explicit negations ("no pneumothorax") as positive evidence.
        if field_path.endswith("pneumothorax.occurred") and re.search(r"(?i)\bno\s+pneumothorax\b", lowered):
            return False

        return any(re.search(pat, text) for pat in patterns)

    def _match_any(label: str, candidates: tuple[str, ...]) -> bool:
        lowered = (label or "").strip().lower()
        return bool(lowered) and any(candidate in lowered for candidate in candidates)

    def _force_false(path: str, *, wipe_prefix: str | None = None, wipe_fields: dict[str, object] | None = None) -> None:
        nonlocal changed
        current = _get_field(record_data, path)
        if current is not True:
            return
        if _positive_evidence_present(path):
            warnings.append(
                f"CHECKBOX_NEGATIVE_OVERRIDE: narrative evidence supports keeping {path}=true; ignoring template negative"
            )
            return
        if _set_field(record_data, path, False):
            changed = True
        if wipe_prefix and wipe_fields:
            changed |= _wipe_object_fields(record_data, wipe_prefix, wipe_fields)
        warnings.append(f"CHECKBOX_NEGATIVE: forcing {path}=false")

    for raw_label in negative_labels:
        label = (raw_label or "").strip()
        if not label:
            continue

        if _match_any(
            label,
            (
                "tunneled pleural catheter",
                "tunnelled pleural catheter",
                "indwelling pleural catheter",
                "pleurx",
                "aspira",
                "ipc",
            ),
        ):
            _force_false(
                "pleural_procedures.ipc.performed",
                wipe_prefix="pleural_procedures.ipc",
                wipe_fields={
                    "action": None,
                    "side": None,
                    "catheter_brand": None,
                    "indication": None,
                    "tunneled": None,
                },
            )
            continue

        if _match_any(label, ("chest tube", "tube thoracostomy", "pigtail")):
            _force_false(
                "pleural_procedures.chest_tube.performed",
                wipe_prefix="pleural_procedures.chest_tube",
                wipe_fields={
                    "action": None,
                    "side": None,
                    "indication": None,
                    "tube_type": None,
                    "tube_size_fr": None,
                    "guidance": None,
                },
            )
            continue

        if _match_any(label, ("pneumothorax", "ptx")):
            _force_false("complications.pneumothorax.occurred")
            continue

        if _match_any(label, ("airway dilation", "balloon dilat", "tracheal dilation", "bronchial dilation")):
            _force_false(
                "procedures_performed.airway_dilation.performed",
                wipe_prefix="procedures_performed.airway_dilation",
                wipe_fields={
                    "location": None,
                    "etiology": None,
                    "method": None,
                    "balloon_diameter_mm": None,
                    "pre_dilation_diameter_mm": None,
                    "post_dilation_diameter_mm": None,
                },
            )
            continue

        if _match_any(label, ("rigid bronchoscopy", "rigid bronchoscop", "rigid scope", "rigid optic")):
            _force_false(
                "procedures_performed.rigid_bronchoscopy.performed",
                wipe_prefix="procedures_performed.rigid_bronchoscopy",
                wipe_fields={
                    "rigid_scope_size": None,
                    "indication": None,
                    "jet_ventilation_used": None,
                },
            )
            continue

        label_lower = label.lower()
        if "stent" in label_lower and any(token in label_lower for token in ("airway", "trache", "bronch", "stent")):
            _force_false(
                "procedures_performed.airway_stent.performed",
                wipe_prefix="procedures_performed.airway_stent",
                wipe_fields={
                    "action": None,
                    "stent_type": None,
                    "stent_brand": None,
                    "diameter_mm": None,
                    "length_mm": None,
                    "location": None,
                    "indication": None,
                    "deployment_successful": None,
                    "airway_stent_removal": False,
                },
            )
            continue

    if not changed:
        return record, []

    return RegistryRecord(**record_data), warnings


__all__ = ["apply_template_checkbox_negation"]
