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

_PNEUMO_WORD_RE = re.compile(r"(?i)\bpneumothorax\b")
_PNEUMO_INTERVENTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Surgery", re.compile(r"(?i)\b(?:vats|thoracotomy|surg(?:ery|ical))\b")),
    ("Heimlich valve", re.compile(r"(?i)\bheimlich\b")),
    ("Chest tube", re.compile(r"(?i)\b(?:chest\s+tube|tube\s+thoracostomy|thoracostomy\s+tube)\b")),
    ("Pigtail catheter", re.compile(r"(?i)\b(?:pigtail|small[-\s]?bore\s+(?:catheter|tube))\b")),
    ("Aspiration", re.compile(r"(?i)\b(?:needle\s+aspirat(?:ion|ed)|aspirat(?:ed|ion))\b")),
    ("Observation", re.compile(r"(?i)\b(?:observ(?:ed|ation)|managed\s+conservatively|no\s+intervention)\b")),
)

_BLEEDING_WORD_RE = re.compile(r"(?i)\b(?:bleed(?:ing)?|hemorrhag(?:e|ic)|haemorrhag(?:e|ic)|ooz(?:ing)?)\b")
_NO_BLEEDING_RE = re.compile(r"(?i)\b(?:no|without)\b[^.\n]{0,40}\b(?:bleeding|hemorrhage|haemorrhage|oozing)\b")
_NASHVILLE_GRADE_RE = re.compile(r"(?i)\bnashville\b[^.\n]{0,40}\bgrade\b\s*(?P<grade>[0-4])\b")

_SUCTION_RE = re.compile(r"(?i)\bsuction(?:ed|ing)?(?:\s+only|\s+alone)?\b")
_WEDGE_RE = re.compile(r"(?i)\bwedge(?:d|ing)?\b|\bbronchoscope\s+wedged\b")
_COLD_SALINE_RE = re.compile(r"(?i)\b(?:cold|iced)\s+saline\b|\bice\s+saline\b")
_EPI_RE = re.compile(r"(?i)\b(?:endobronchial\s+)?epi(?:nephrine)?\b")
_BALLOON_RE = re.compile(r"(?i)\b(?:balloon\s+tamponade|tamponade|fogarty)\b")
_BLOCKER_RE = re.compile(r"(?i)\b(?:endobronchial\s+blocker|arndt|blocker)\b")
_TRANSFUSION_RE = re.compile(r"(?i)\b(?:transfus(?:ion|ed)|prbc|packed\s+red\s+blood)\b")
_EMBOLIZATION_RE = re.compile(r"(?i)\bemboliz(?:ation|ed)\b")
_SURGERY_RE = re.compile(r"(?i)\bsurger(?:y|ical)\b")
_ABORT_FOR_BLEEDING_RE = re.compile(
    r"(?i)\b(?:abort(?:ed|ing)|terminate(?:d|ing)|stop(?:ped|ping))\b[^.\n]{0,120}\b(?:bleed(?:ing)?|hemorrhage|haemorrhage)\b"
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


def _first_match_with_bleeding_context(
    pattern: re.Pattern[str],
    text: str,
    *,
    context_window: int = 180,
) -> re.Match[str] | None:
    for match in pattern.finditer(text or ""):
        start = match.start()
        end = match.end()
        window = text[max(0, start - context_window) : min(len(text), end + context_window)]
        if not _BLEEDING_WORD_RE.search(window):
            continue
        prefix = text[max(0, start - 80) : start]
        if _NEGATION_PREFIX_RE.search(prefix):
            continue
        return match
    return None


def _first_match_with_pneumothorax_context(
    pattern: re.Pattern[str],
    text: str,
    *,
    context_window: int = 220,
) -> re.Match[str] | None:
    for match in pattern.finditer(text or ""):
        start = match.start()
        end = match.end()
        window = text[max(0, start - context_window) : min(len(text), end + context_window)]
        if not _PNEUMO_WORD_RE.search(window):
            continue
        prefix = text[max(0, start - 80) : start]
        if _NEGATION_PREFIX_RE.search(prefix):
            continue
        return match
    return None


def _infer_nashville_bleeding_grade(text: str) -> tuple[int | None, re.Match[str] | None]:
    """Infer Nashville bleeding grade (0-4) from explicit grade or hemostasis interventions."""
    if not text or not text.strip():
        return None, None

    explicit = _NASHVILLE_GRADE_RE.search(text)
    if explicit:
        prefix = text[max(0, explicit.start() - 80) : explicit.start()]
        if not _NEGATION_PREFIX_RE.search(prefix):
            try:
                return int(explicit.group("grade")), explicit
            except Exception:
                pass

    # Grade 4: escalation (transfusion/embolization/surgery) in bleeding context.
    for pat in (_TRANSFUSION_RE, _EMBOLIZATION_RE, _SURGERY_RE):
        match = _first_match_with_bleeding_context(pat, text)
        if match:
            return 4, match

    # Grade 3: tamponade/blocker/aborted for bleeding (bleeding context enforced).
    abort_match = _ABORT_FOR_BLEEDING_RE.search(text)
    if abort_match:
        prefix = text[max(0, abort_match.start() - 80) : abort_match.start()]
        if not _NEGATION_PREFIX_RE.search(prefix):
            return 3, abort_match

    for pat in (_BALLOON_RE, _BLOCKER_RE):
        match = _first_match_with_bleeding_context(pat, text)
        if match:
            return 3, match

    # Grade 2: wedge/cold saline/topical vasoconstrictor in bleeding context.
    for pat in (_WEDGE_RE, _COLD_SALINE_RE, _EPI_RE):
        match = _first_match_with_bleeding_context(pat, text)
        if match:
            return 2, match

    # Grade 1: suction-only hemostasis in bleeding context.
    suction_match = _first_match_with_bleeding_context(_SUCTION_RE, text)
    if suction_match:
        return 1, suction_match

    # Grade 0: explicit no-bleeding statement (only if no higher-grade evidence).
    no_bleed_match = _NO_BLEEDING_RE.search(text)
    if no_bleed_match:
        prefix = text[max(0, no_bleed_match.start() - 80) : no_bleed_match.start()]
        if not _NEGATION_PREFIX_RE.search(prefix):
            return 0, no_bleed_match

    return None, None


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
    bleeding_grade, bleeding_match = _infer_nashville_bleeding_grade(text)

    if pneumothorax_match:
        prefix = text[max(0, pneumothorax_match.start() - 80) : pneumothorax_match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            pneumothorax_match = None

    if hematoma_match:
        prefix = text[max(0, hematoma_match.start() - 80) : hematoma_match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            hematoma_match = None

    if not pneumothorax_match and not hematoma_match:
        if bleeding_grade is None:
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

    def _append_evidence(field_key: str, match: re.Match[str]) -> None:
        nonlocal changed
        snippet = (match.group(0) or "").strip()
        if not snippet:
            snippet = _line_snippet(text, match.start(), match.end())
        if not snippet:
            return
        spans = evidence.setdefault(field_key, [])
        if any(isinstance(s, Span) and s.start == match.start() and s.end == match.end() for s in spans):
            return
        spans.append(Span(text=snippet, start=match.start(), end=match.end(), confidence=0.9))
        changed = True

    if bleeding_grade is not None and bleeding_match is not None:
        bleeding = complications.get("bleeding")
        if not isinstance(bleeding, dict):
            bleeding = {}

        if bleeding.get("bleeding_grade_nashville") is None:
            bleeding["bleeding_grade_nashville"] = bleeding_grade
            changed = True
            # Conservative: treat grade > 0 as a bleeding event that occurred.
            if bleeding_grade > 0:
                bleeding["occurred"] = True
                _ensure_any_complication()
                grade_to_comp = {1: "Bleeding - Mild", 2: "Bleeding - Moderate", 3: "Bleeding - Severe", 4: "Bleeding - Severe"}
                _add_comp_list(grade_to_comp.get(int(bleeding_grade), "Bleeding - Mild"))
            elif bleeding_grade == 0:
                # Only set occurred=false when the note explicitly states "no bleeding".
                bleeding.setdefault("occurred", False)

            complications["bleeding"] = bleeding
            _add_evidence("complications.bleeding.bleeding_grade_nashville", bleeding_match)
            warnings.append(f"BLEEDING_GRADE_DERIVED: Nashville grade={bleeding_grade}")

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

        # Tier 2: intervention level (explicit-only; do not infer from tool lists).
        existing_interventions = pneumothorax.get("intervention")
        if not isinstance(existing_interventions, list):
            existing_interventions = []
        found: list[str] = []
        for label, pat in _PNEUMO_INTERVENTIONS:
            match = _first_match_with_pneumothorax_context(pat, text)
            if not match:
                continue
            if label not in found:
                found.append(label)
            _append_evidence("complications.pneumothorax.intervention", match)

        if found and not existing_interventions:
            pneumothorax["intervention"] = found
            complications["pneumothorax"] = pneumothorax
            changed = True
            warnings.append(f"PNEUMOTHORAX_INTERVENTION_DERIVED: {', '.join(found)}")

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
