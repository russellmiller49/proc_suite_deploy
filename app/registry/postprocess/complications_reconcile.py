from __future__ import annotations

import re
from typing import Any

from app.common.spans import Span
from app.registry.schema import RegistryRecord


_PUNCTUATION_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")

_NEGATION_PREFIX_RE = re.compile(
    r"(?i)\b(?:no|not|without|denies|negative\s+for)\b[^.\n]{0,60}$"
)

_HEMATOMA_RE = re.compile(r"(?i)\b(?:hematoma|haematoma)\b")
_PNEUMOTHORAX_RE = re.compile(
    r"(?i)\bpneumothorax\b[^.\n]{0,120}\b(?:noted|occurred|developed|present|post|small|trace)\b"
)


def _maybe_unescape_newlines(text: str) -> str:
    raw = text or ""
    if not raw.strip():
        return raw
    if "\n" in raw or "\r" in raw:
        return raw
    if "\\n" not in raw and "\\r" not in raw:
        return raw
    return raw.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def _line_snippet(text: str, start: int, end: int, *, limit: int = 240) -> str:
    if not text:
        return ""
    s = max(0, min(len(text), start))
    e = max(0, min(len(text), end))
    line_start = text.rfind("\n", 0, s)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = text.find("\n", e)
    if line_end == -1:
        line_end = len(text)
    snippet = text[line_start:line_end].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet[:limit].rstrip()


def reconcile_complications_from_narrative(record: RegistryRecord, full_text: str) -> list[str]:
    """Ensure explicit narrative complications are not overridden by summary 'None' lines.

    Hierarchy of truth:
    - Specific narrative sections documenting a complication (e.g., "small hematoma")
      should supersede a later templated "COMPLICATIONS: None".
    """
    warnings: list[str] = []
    text = _maybe_unescape_newlines(full_text or "")
    if not text.strip():
        return warnings

    # Match in narrative text. Use simple negation guards to avoid "no pneumothorax"/"no hematoma".
    pneumothorax_match = _PNEUMOTHORAX_RE.search(text)
    hematoma_match = _HEMATOMA_RE.search(text)

    if pneumothorax_match:
        prefix = text[max(0, pneumothorax_match.start() - 80) : pneumothorax_match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            pneumothorax_match = None

    if hematoma_match:
        prefix = text[max(0, hematoma_match.start() - 80) : hematoma_match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            hematoma_match = None

    if not pneumothorax_match and not hematoma_match:
        return warnings

    record_data: dict[str, Any] = record.model_dump()
    complications = record_data.get("complications")
    if not isinstance(complications, dict):
        complications = {}

    comp_list = complications.get("complication_list")
    if not isinstance(comp_list, list):
        comp_list = []

    evidence = record_data.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    changed = False

    def _ensure_any_complication() -> None:
        nonlocal changed
        if complications.get("any_complication") is not True:
            complications["any_complication"] = True
            changed = True

    def _add_comp_list(item: str) -> None:
        nonlocal changed
        if item not in comp_list:
            comp_list.append(item)
            changed = True

    def _add_event(event_type: str, notes: str) -> None:
        nonlocal changed
        events = complications.get("events")
        if not isinstance(events, list):
            events = []
        if any(isinstance(e, dict) and str(e.get("type") or "").lower() == event_type.lower() for e in events):
            return
        events.append({"type": event_type, "notes": notes or None})
        complications["events"] = events
        changed = True

    def _add_evidence(field_key: str, match: re.Match[str]) -> None:
        nonlocal changed
        if evidence.get(field_key):
            return
        snippet = (match.group(0) or "").strip()
        if not snippet:
            snippet = _line_snippet(text, match.start(), match.end())
        if not snippet:
            return
        evidence.setdefault(field_key, []).append(
            Span(text=snippet, start=match.start(), end=match.end(), confidence=0.9)
        )
        changed = True

    if pneumothorax_match:
        _ensure_any_complication()
        _add_comp_list("Pneumothorax")
        pneumothorax = complications.get("pneumothorax")
        if not isinstance(pneumothorax, dict):
            pneumothorax = {}
        if pneumothorax.get("occurred") is not True:
            pneumothorax["occurred"] = True
            complications["pneumothorax"] = pneumothorax
            changed = True
        snippet = _line_snippet(text, pneumothorax_match.start(), pneumothorax_match.end())
        _add_event("Pneumothorax", snippet)
        _add_evidence("complications.pneumothorax.occurred", pneumothorax_match)
        warnings.append("COMPLICATION_OVERRIDE: pneumothorax mentioned in narrative; overriding summary 'None'.")

    if hematoma_match:
        _ensure_any_complication()
        _add_comp_list("Other")
        snippet = _line_snippet(text, hematoma_match.start(), hematoma_match.end())
        details = str(complications.get("other_complication_details") or "").strip()
        if not details:
            complications["other_complication_details"] = snippet or "Hematoma"
            changed = True
        elif "hematoma" not in details.lower():
            complications["other_complication_details"] = (details + "; " + (snippet or "Hematoma")).strip()
            changed = True
        _add_event("Hematoma", snippet)
        _add_evidence("complications.other_complication_details", hematoma_match)
        warnings.append("COMPLICATION_OVERRIDE: hematoma mentioned in narrative; overriding summary 'None'.")

    if not changed:
        return warnings

    complications["complication_list"] = comp_list
    record_data["complications"] = complications
    record_data["evidence"] = evidence
    updated = RegistryRecord(**record_data)
    record.__dict__.update(updated.__dict__)
    return warnings


__all__ = ["reconcile_complications_from_narrative"]

