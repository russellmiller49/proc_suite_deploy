from __future__ import annotations

import re
from typing import Any

from app.common.spans import Span
from app.registry.quality_signals import make_quality_signal_warning
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

_BLEEDING_WORD_RE = re.compile(r"(?i)\b(?:bleed(?:ing)?|hemorrhag(?:e|ic)|haemorrhag(?:e|ic)|ooz(?:e|ed|ing)?)\b")
_NO_BLEEDING_RE = re.compile(r"(?i)\b(?:no|without)\b[^.\n]{0,40}\b(?:bleeding|hemorrhage|haemorrhage|oozing)\b")
_NASHVILLE_GRADE_RE = re.compile(r"(?i)\bnashville\b[^.\n]{0,40}\bgrade\b\s*(?P<grade>[0-4])\b")
_COMPLICATIONS_NONE_RE = re.compile(r"(?i)\bcomplications?\s*:?\s*none\b|\bno\s+immediate\s+complications\b")
_PROCEDURAL_NONE_RE = re.compile(
    r"(?i)\bcomplications?\s*:?\s*none\s+procedural\b|\bno\s+procedural\s+complications\b"
)
_LOW_GRADE_BLEEDING_CUE_RE = re.compile(
    r"(?i)\b(?:minor|minimal|mild|trace|scant|contact|blood-tinged)\b(?:\s+\w+){0,2}\s+(?:bleeding|oozing|hemorrhag(?:e|ic))\b"
    r"|\bminor\s+ooz(?:e|ing)\b|\bmild\s+ooz(?:e|ing)\b|\boo?z(?:e|ing)\b|\bminor\s+procedural\s+hemorrhage\b"
    r"|\b(?:no|without)\s+(?:clinically\s+)?significant\s+bleeding\b"
)
_HIGH_GRADE_BLEEDING_CUE_RE = re.compile(
    r"(?i)\b(?:moderate|significant|severe|massive|brisk|active)\s+(?:bleeding|hemorrhag(?:e|ic))\b"
    r"|\bhemorrhag(?:e|ic)\b[^.\n]{0,40}\b(?:moderate|significant|severe|massive|brisk|active)\b"
)

_SUCTION_RE = re.compile(r"(?i)\bsuction(?:ed|ing)?(?:\s+only|\s+alone)?\b")
_WEDGE_RE = re.compile(r"(?i)\bwedge(?:d|ing)?\b|\bbronchoscope\s+wedged\b")
_COLD_SALINE_RE = re.compile(r"(?i)\b(?:cold|iced)\s+saline\b|\bice\s+saline\b")
_EPI_RE = re.compile(r"(?i)\b(?:endobronchial\s+)?epi(?:nephrine)?\b")
_TXA_RE = re.compile(r"(?i)\b(?:tranexamic\s+acid|txa)\b")
_BALLOON_RE = re.compile(r"(?i)\b(?:balloon\s+tamponade|tamponade|fogarty)\b")
_BLOCKER_RE = re.compile(r"(?i)\b(?:endobronchial\s+blocker|arndt|blocker)\b")
_TRANSFUSION_RE = re.compile(r"(?i)\b(?:transfus(?:ion|ed)|prbc|packed\s+red\s+blood)\b")
_EMBOLIZATION_RE = re.compile(r"(?i)\bemboliz(?:ation|ed)\b")
_SURGERY_RE = re.compile(r"(?i)\bsurger(?:y|ical)\b")
_DIRECT_PRESSURE_RE = re.compile(r"(?i)\b(?:direct\s+pressure|compression)\b")
_PROTAMINE_RE = re.compile(r"(?i)\bprotamine\b")
_ROUTINE_HEMOSTASIS_INTERVENTION_RE = re.compile(
    r"(?i)\b(?:suction(?:ed|ing)?|wedge(?:d|ing)?|bronchoscope\s+wedged|(?:cold|iced)\s+saline|"
    r"ice\s+saline|(?:endobronchial\s+)?epi(?:nephrine)?|(?:tranexamic\s+acid|txa)|direct\s+pressure|compression)\b"
)
_ROUTINE_HEMOSTASIS_RESOLUTION_RE = re.compile(
    r"(?i)\b(?:hemostasis\s+(?:was\s+)?(?:achieved|confirmed)|bleeding\s+(?:resolved|ceased|controlled)|"
    r"no\s+active\s+bleeding|(?:no|without)\s+(?:clinically\s+)?significant\s+bleeding)\b"
)
_PROPHYLACTIC_BLEEDING_SUPPRESSOR_RE = re.compile(
    r"(?i)\b(?:control\s+any\s+(?:distal\s+)?bleeding|in\s+case\s+of\s+bleeding|prevent\s+bleeding|"
    r"should\s+bleeding\s+occur|if\s+bleeding\s+occurs?|available\s+to\s+control\b[^.\n]{0,80}\bbleeding)\b"
)
_ABORT_FOR_BLEEDING_RE = re.compile(
    r"(?i)\b(?:abort(?:ed|ing)|terminate(?:d|ing)|stop(?:ped|ping))\b[^.\n]{0,120}\b(?:bleed(?:ing)?|hemorrhage|haemorrhage)\b"
)
_BLEEDING_CONTROL_RE = re.compile(
    r"(?i)\b(?:"
    r"tamponade|hemostasis|control(?:led)?|cessation|stopp?ed|halt(?:ed)?"
    r"|prevent\w*\b[^.\n]{0,80}\bblood\b[^.\n]{0,80}\b(?:soil|contaminat|spill)\w*\b"
    r"|avoid\w*\b[^.\n]{0,80}\bblood\b[^.\n]{0,80}\b(?:soil|contaminat|spill)\w*\b"
    r"|isolation\b|isolate\w*\b"
    r"|protect\w*\b[^.\n]{0,80}\b(?:contralateral|other)\b[^.\n]{0,40}\b(?:lung|airway)\b"
    r")"
)

_ARRHYTHMIA_RE = re.compile(
    r"(?i)\b(?:arrhythmia|atrial\s+fibrillation|a\.?\s*fib|afib|a\s+fib|rvr|tachyarrhythmia)\b"
)
_CARDIOVERSION_RE = re.compile(r"(?i)\bcardioversion\b")
_AIRWAY_INJURY_RE = re.compile(
    r"(?i)\b(?:trachea|airway|posterior\s+membrane|bronch(?:us|ial))\b[^.\n]{0,140}\b(?:tear|lacerat(?:ion|ed)?|injur(?:y|ed)|defect|perforat(?:ion|ed)?)\b"
    r"|\b(?:tear|lacerat(?:ion|ed)?|injur(?:y|ed)|defect|perforat(?:ion|ed)?)\b[^.\n]{0,140}\b(?:trachea|airway|posterior\s+membrane|bronch(?:us|ial))\b"
)
_ASPIRATION_RE = re.compile(
    r"(?i)\b(?:aspirat(?:ion|ed)|aspiration)\b[^.\n]{0,140}\b(?:emesis|vomit|gastric|contents?)\b"
    r"|\b(?:emesis|vomit|gastric\s+contents?)\b[^.\n]{0,140}\baspirat(?:ion|ed)?\b"
)
_DENTAL_INJURY_RE = re.compile(
    r"(?i)\b(?:tooth|teeth|dental)\b[^.\n]{0,140}\b(?:loss|lost|injur(?:y|ed)|fractur(?:e|ed)|avuls(?:ion|ed)|dislodg(?:ed|ement))\b"
    r"|\b(?:loss|lost|injur(?:y|ed)|fractur(?:e|ed)|avuls(?:ion|ed)|dislodg(?:ed|ement))\b[^.\n]{0,140}\b(?:tooth|teeth|dental)\b"
)
_CARDIAC_ARREST_RE = re.compile(
    r"(?i)\b(?:cardiac\s+arrest|pulseless\s+electrical\s+activity|pea|asystole|coded|arrested)\b"
)
_DEATH_RE = re.compile(
    r"(?i)\b(?:pronounced\s+dead|declared\s+dead|expired|death|died)\b"
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


def _sentence_window(text: str, start: int, end: int) -> str:
    raw = text or ""
    if not raw:
        return ""
    left_boundary = max(raw.rfind(".", 0, start), raw.rfind("\n", 0, start), raw.rfind(";", 0, start))
    sentence_start = left_boundary + 1 if left_boundary != -1 else 0
    right_candidates = [pos for pos in (raw.find(".", end), raw.find("\n", end), raw.find(";", end)) if pos != -1]
    sentence_end = min(right_candidates) if right_candidates else len(raw)
    return raw[sentence_start:sentence_end]


def _sentence_prefix(text: str, start: int, end: int) -> str:
    sentence = _sentence_window(text, start, end)
    if not sentence:
        return ""
    left_boundary = max((text or "").rfind(".", 0, start), (text or "").rfind("\n", 0, start), (text or "").rfind(";", 0, start))
    sentence_start = left_boundary + 1 if left_boundary != -1 else 0
    local_start = max(0, start - sentence_start)
    return sentence[:local_start]


def _has_low_ebl_context(text: str) -> bool:
    low_ebl = _low_ebl_milliliters(text)
    if low_ebl is not None and low_ebl <= 20:
        return True
    return bool(
        re.search(
            r"(?i)\b(?:minimal|low|trace|scant)\s+(?:ebl|estimated\s+blood\s+loss|blood\s+loss)\b",
            text or "",
        )
    )


def _first_match_with_bleeding_context(
    pattern: re.Pattern[str],
    text: str,
    *,
    context_window: int = 180,
) -> re.Match[str] | None:
    for match in pattern.finditer(text or ""):
        start = match.start()
        end = match.end()
        sentence = _sentence_window(text, start, end)
        if not _BLEEDING_WORD_RE.search(sentence):
            continue
        if _PROPHYLACTIC_BLEEDING_SUPPRESSOR_RE.search(sentence):
            continue
        prefix = _sentence_prefix(text, start, end)
        if not prefix:
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


def _first_unnegated_match(
    pattern: re.Pattern[str],
    text: str,
    *,
    prefix_window: int = 80,
) -> re.Match[str] | None:
    for match in pattern.finditer(text or ""):
        prefix = text[max(0, match.start() - prefix_window) : match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            continue
        return match
    return None


def _low_ebl_milliliters(text: str) -> int | None:
    match = re.search(
        r"(?i)\b(?:ebl|estimated\s+blood\s+loss|blood\s+loss)\b[^0-9]{0,20}(?P<first>\d{1,3})(?:\s*[-–]\s*(?P<second>\d{1,3}))?\s*(?:ml|cc)\b",
        text or "",
    )
    if not match:
        return None
    try:
        first = int(match.group("first"))
        second_raw = match.group("second")
        second = int(second_raw) if second_raw is not None else first
    except Exception:
        return None
    return max(first, second)


def _minor_bleeding_with_supportive_transfusion(text: str) -> bool:
    if not text or not text.strip():
        return False
    low_grade = bool(_LOW_GRADE_BLEEDING_CUE_RE.search(text))
    controlled = bool(_DIRECT_PRESSURE_RE.search(text) or re.search(r"(?i)\bcontrolled\s+with\b", text))
    low_ebl = _low_ebl_milliliters(text)
    thrombocytopenia_context = bool(
        re.search(r"(?i)\b(?:platelets?|plt)\b[^.\n]{0,40}\b(?:low|k\b|thrombocytopenia)\b", text)
    )
    return low_grade and controlled and _first_unnegated_match(_HIGH_GRADE_BLEEDING_CUE_RE, text) is None and bool(
        thrombocytopenia_context or (low_ebl is not None and low_ebl <= 20)
    )


def _routine_hemostasis_only(text: str) -> bool:
    if not text or not text.strip():
        return False
    complications_none = bool(_COMPLICATIONS_NONE_RE.search(text) or _PROCEDURAL_NONE_RE.search(text))
    low_grade = bool(
        _LOW_GRADE_BLEEDING_CUE_RE.search(text)
        or re.search(r"(?i)\b(?:small|trace|minimal)\s+amount\s+of\s+(?:bleeding|oozing)\b", text)
        or complications_none
    )
    if not low_grade:
        return False
    if not _ROUTINE_HEMOSTASIS_INTERVENTION_RE.search(text):
        return False
    if not _ROUTINE_HEMOSTASIS_RESOLUTION_RE.search(text):
        return False
    if _first_unnegated_match(_HIGH_GRADE_BLEEDING_CUE_RE, text):
        return False
    if any(
        pattern.search(text)
        for pattern in (
            _BALLOON_RE,
            _BLOCKER_RE,
            _TRANSFUSION_RE,
            _EMBOLIZATION_RE,
            _SURGERY_RE,
            _PROTAMINE_RE,
            _ABORT_FOR_BLEEDING_RE,
        )
    ):
        return False
    return True


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

    complications_none = bool(_COMPLICATIONS_NONE_RE.search(text))
    low_ebl_context = _has_low_ebl_context(text)
    explicit_high_grade = _first_unnegated_match(_HIGH_GRADE_BLEEDING_CUE_RE, text)
    if _routine_hemostasis_only(text):
        return None, None

    def _suppress_low_grade_intervention(match: re.Match[str]) -> bool:
        if not complications_none:
            return False
        sentence = _sentence_window(text, match.start(), match.end())
        if _PROPHYLACTIC_BLEEDING_SUPPRESSOR_RE.search(sentence):
            return True
        window = text[max(0, match.start() - 220) : min(len(text), match.end() + 220)]
        if explicit_high_grade and explicit_high_grade.start() >= max(0, match.start() - 220) and explicit_high_grade.end() <= min(len(text), match.end() + 220):
            return False
        if low_ebl_context and explicit_high_grade is None:
            return True
        return bool(_LOW_GRADE_BLEEDING_CUE_RE.search(sentence) or _LOW_GRADE_BLEEDING_CUE_RE.search(window))

    # Grade 4: escalation (transfusion/embolization/surgery) in bleeding context.
    for pat in (_TRANSFUSION_RE, _EMBOLIZATION_RE, _SURGERY_RE):
        match = _first_match_with_bleeding_context(pat, text)
        if match:
            if pat is _TRANSFUSION_RE and _minor_bleeding_with_supportive_transfusion(text):
                return 1, match
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
            window = text[max(0, match.start() - 160) : min(len(text), match.end() + 160)]
            # Require explicit bleeding control/tamponade language to avoid
            # over-calling grade 3 when blockers are used for non-bleeding workflows.
            if _BLEEDING_CONTROL_RE.search(window):
                return 3, match

    # Grade 2: wedge/cold saline/topical vasoconstrictor in bleeding context.
    for pat in (_WEDGE_RE, _COLD_SALINE_RE, _EPI_RE, _TXA_RE):
        match = _first_match_with_bleeding_context(pat, text)
        if match:
            if _suppress_low_grade_intervention(match):
                continue
            return 2, match

    # Grade 2: anticoagulation reversal (e.g., protamine) in bleeding context.
    protamine_match = _first_match_with_bleeding_context(_PROTAMINE_RE, text)
    if protamine_match:
        return 2, protamine_match

    # Grade 1: suction-only hemostasis in bleeding context.
    suction_match = _first_match_with_bleeding_context(_SUCTION_RE, text)
    if suction_match:
        # Common template phrase: "after suctioning blood and secretions there was no evidence of active bleeding".
        # This often reflects routine clearance rather than a bleeding complication; avoid over-calling Grade 1
        # unless the note explicitly frames suction as hemostasis (e.g., "bleeding controlled with suction").
        window = text[max(0, suction_match.start() - 220) : min(len(text), suction_match.end() + 220)]
        if re.search(r"(?i)\bno\s+(?:evidence\s+of\s+)?active\s+bleeding\b", window) and not re.search(
            r"(?i)\bcontrolled\b|\bhemostasis\b", window
        ):
            suction_match = None
        elif _suppress_low_grade_intervention(suction_match):
            suction_match = None
        else:
            return 1, suction_match

    if complications_none and low_ebl_context and explicit_high_grade is None:
        return None, None

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
    routine_hemostasis_only = _routine_hemostasis_only(text)

    # Match in narrative text. Use simple negation guards to avoid "no pneumothorax"/"no hematoma".
    pneumothorax_match = _PNEUMOTHORAX_RE.search(text)
    hematoma_match = _HEMATOMA_RE.search(text)
    bleeding_grade, bleeding_match = _infer_nashville_bleeding_grade(text)
    present_on_arrival_hemoptysis = bool(
        _PROCEDURAL_NONE_RE.search(text)
        and re.search(r"(?i)\b(?:indication|pre\s*dx|post\s*dx|diagnosis)\b[^.\n]{0,160}\bhemoptysis\b", text)
        and re.search(r"(?i)\b(?:massive|brisk|active)\s+hemoptysis\b|\bhemoptysis\b", text)
        and not re.search(
            r"(?i)\b(?:biops(?:y|ies|ied)|cryobiops(?:y|ies)|tbna|needle\s+aspiration|brushings?|debridement|tumou?r\s+debulking)\b",
            text,
        )
    )
    if present_on_arrival_hemoptysis and bleeding_grade is not None:
        bleeding_grade = None
        bleeding_match = None

    if pneumothorax_match:
        prefix = text[max(0, pneumothorax_match.start() - 80) : pneumothorax_match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            pneumothorax_match = None

    if hematoma_match:
        prefix = text[max(0, hematoma_match.start() - 80) : hematoma_match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            hematoma_match = None

    arrhythmia_match = _ARRHYTHMIA_RE.search(text)
    if arrhythmia_match:
        prefix = text[max(0, arrhythmia_match.start() - 80) : arrhythmia_match.start()]
        if _NEGATION_PREFIX_RE.search(prefix) or re.search(r"(?i)\b(?:history|prior|previous|known)\b", prefix):
            arrhythmia_match = None

    airway_injury_match = _first_unnegated_match(_AIRWAY_INJURY_RE, text)
    aspiration_match = _first_unnegated_match(_ASPIRATION_RE, text)
    dental_injury_match = _first_unnegated_match(_DENTAL_INJURY_RE, text)
    cardiac_arrest_match = _first_unnegated_match(_CARDIAC_ARREST_RE, text)
    death_match = _first_unnegated_match(_DEATH_RE, text)

    if (
        not pneumothorax_match
        and not hematoma_match
        and arrhythmia_match is None
        and airway_injury_match is None
        and aspiration_match is None
        and dental_injury_match is None
        and cardiac_arrest_match is None
        and death_match is None
    ):
        if bleeding_grade is None:
            if routine_hemostasis_only:
                warnings.append(
                    "ROUTINE_HEMOSTASIS_SUPPRESSED: routine topical hemostasis/no-complication language did not create a bleeding complication."
                )
                warnings.append(
                    make_quality_signal_warning(
                        "routine_hemostasis_bleeding_suppressed",
                        field="complications.bleeding",
                        action="suppressed",
                        detail="Routine TXA/epinephrine/cold saline/wedging hemostasis with no-complication language was not promoted to a bleeding complication.",
                        source="complications_reconcile",
                    )
                )
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

    def _add_event(event_type: str, notes: str, interventions: list[str] | None = None) -> None:
        nonlocal changed
        events = complications.get("events")
        if not isinstance(events, list):
            events = []
        if any(isinstance(e, dict) and str(e.get("type") or "").lower() == event_type.lower() for e in events):
            return
        payload: dict[str, Any] = {"type": event_type, "notes": notes or None}
        if interventions:
            payload["interventions"] = interventions
        events.append(payload)
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

    def _interventions_from_window(start: int, end: int) -> list[str]:
        window = text[max(0, start - 260) : min(len(text), end + 260)]
        interventions: list[str] = []
        if re.search(
            r"(?i)\bleft\b[^.\n]{0,80}\bchest\s+tube\b[^.\n]{0,80}\bright\b[^.\n]{0,80}\bchest\s+tube\b"
            r"|\bright\b[^.\n]{0,80}\bchest\s+tube\b[^.\n]{0,80}\bleft\b[^.\n]{0,80}\bchest\s+tube\b"
            r"|\bbilateral\s+chest\s+tubes?\b",
            window,
        ):
            interventions.append("Bilateral chest tubes")
        elif re.search(r"(?i)\b(?:chest\s+tube|pigtail)\b", window):
            interventions.append("Chest tube")
        if _ROUTINE_HEMOSTASIS_INTERVENTION_RE.search(window) or _BLEEDING_CONTROL_RE.search(window):
            interventions.append("Hemostatic measures")
        return interventions

    def _append_other_detail(match: re.Match[str], label: str) -> None:
        nonlocal changed
        snippet = _line_snippet(text, match.start(), match.end()) or label
        details = str(complications.get("other_complication_details") or "").strip()
        if not details:
            complications["other_complication_details"] = snippet
            changed = True
        elif snippet and snippet.lower() not in details.lower():
            complications["other_complication_details"] = (details + "; " + snippet).strip()
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

            # Add a bleeding event with specific interventions when documented.
            if bleeding_grade > 0:
                interventions: list[str] = []
                protamine_match = _first_match_with_bleeding_context(_PROTAMINE_RE, text)
                if protamine_match:
                    interventions.append("Protamine administration")
                    _append_evidence("complications.bleeding.intervention_required", protamine_match)
                snippet = _line_snippet(text, bleeding_match.start(), bleeding_match.end())
                _add_event("Bleeding", snippet, interventions if interventions else None)

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

    if arrhythmia_match:
        _ensure_any_complication()
        _add_comp_list("Arrhythmia")
        snippet = _line_snippet(text, arrhythmia_match.start(), arrhythmia_match.end())
        window = text[max(0, arrhythmia_match.start() - 200) : min(len(text), arrhythmia_match.end() + 200)]
        interventions: list[str] = []
        if _CARDIOVERSION_RE.search(window):
            interventions.append("Cardioversion")
        _add_event("Arrhythmia", snippet, interventions or None)
        _add_evidence("complications.complication_list", arrhythmia_match)
        if interventions:
            _append_evidence("complications.events", arrhythmia_match)
        warnings.append("COMPLICATION_OVERRIDE: arrhythmia mentioned in narrative; overriding summary 'None'.")

    if airway_injury_match:
        _ensure_any_complication()
        _add_comp_list("Other")
        snippet = _line_snippet(text, airway_injury_match.start(), airway_injury_match.end())
        _append_other_detail(airway_injury_match, "Airway injury")
        _add_event("Airway injury", snippet, _interventions_from_window(airway_injury_match.start(), airway_injury_match.end()) or None)
        _append_evidence("complications.other_complication_details", airway_injury_match)
        warnings.append("COMPLICATION_OVERRIDE: airway injury mentioned in narrative; overriding summary 'None'.")
        warnings.append(
            make_quality_signal_warning(
                "airway_injury_promoted_from_narrative",
                field="complications.other_complication_details",
                action="promoted",
                detail="Narrative airway tear/injury language was promoted to a structured complication.",
                source="complications_reconcile",
            )
        )

    if aspiration_match:
        _ensure_any_complication()
        _add_comp_list("Aspiration")
        snippet = _line_snippet(text, aspiration_match.start(), aspiration_match.end())
        _add_event("Aspiration", snippet, _interventions_from_window(aspiration_match.start(), aspiration_match.end()) or None)
        _append_evidence("complications.complication_list", aspiration_match)
        warnings.append("COMPLICATION_OVERRIDE: aspiration mentioned in narrative; overriding summary 'None'.")
        warnings.append(
            make_quality_signal_warning(
                "aspiration_promoted_from_narrative",
                field="complications.complication_list",
                action="promoted",
                detail="Narrative aspiration language was promoted to a structured complication.",
                source="complications_reconcile",
            )
        )

    if dental_injury_match:
        _ensure_any_complication()
        _add_comp_list("Other")
        snippet = _line_snippet(text, dental_injury_match.start(), dental_injury_match.end())
        _append_other_detail(dental_injury_match, "Dental injury")
        _add_event("Dental injury", snippet)
        _append_evidence("complications.other_complication_details", dental_injury_match)
        warnings.append("COMPLICATION_OVERRIDE: dental injury mentioned in narrative; overriding summary 'None'.")
        warnings.append(
            make_quality_signal_warning(
                "dental_injury_promoted_from_narrative",
                field="complications.other_complication_details",
                action="promoted",
                detail="Narrative tooth-loss/dental-injury language was promoted to a structured complication.",
                source="complications_reconcile",
            )
        )

    if cardiac_arrest_match:
        _ensure_any_complication()
        _add_comp_list("Cardiac arrest")
        snippet = _line_snippet(text, cardiac_arrest_match.start(), cardiac_arrest_match.end())
        _add_event(
            "Cardiac arrest",
            snippet,
            _interventions_from_window(cardiac_arrest_match.start(), cardiac_arrest_match.end()) or None,
        )
        _append_evidence("complications.complication_list", cardiac_arrest_match)
        warnings.append("COMPLICATION_OVERRIDE: cardiac arrest mentioned in narrative; overriding summary 'None'.")
        warnings.append(
            make_quality_signal_warning(
                "cardiac_arrest_promoted_from_narrative",
                field="complications.complication_list",
                action="promoted",
                detail="Narrative cardiac-arrest/PEA/asystole language was promoted to a structured complication.",
                source="complications_reconcile",
            )
        )

    if death_match:
        _ensure_any_complication()
        _add_comp_list("Death")
        snippet = _line_snippet(text, death_match.start(), death_match.end())
        _add_event("Death", snippet, _interventions_from_window(death_match.start(), death_match.end()) or None)
        _append_evidence("complications.complication_list", death_match)
        warnings.append("COMPLICATION_OVERRIDE: death mentioned in narrative; overriding summary 'None'.")
        warnings.append(
            make_quality_signal_warning(
                "death_promoted_from_narrative",
                field="complications.complication_list",
                action="promoted",
                detail="Narrative death/expired/declared-dead language was promoted to a structured complication.",
                source="complications_reconcile",
            )
        )

    if not changed:
        return warnings

    complications["complication_list"] = comp_list
    record_data["complications"] = complications
    record_data["evidence"] = evidence
    updated = RegistryRecord(**record_data)
    record.__dict__.update(updated.__dict__)
    return warnings


__all__ = ["reconcile_complications_from_narrative"]
