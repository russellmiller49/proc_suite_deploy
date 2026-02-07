"""Candidate expansion for ML-first fastpath.

The ML-first SmartHybridOrchestrator validates ML candidates via rules. For
high-confidence cases, ML can be "confident but incomplete" and miss rare
add-on codes that are unambiguous in text. This module expands the candidate
set using conservative, anchored, non-negated keyword rules before strict
rules validation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.coder.adapters.nlp.simple_negation_detector import get_negation_detector


_FIDUCIAL_RE = re.compile(r"\bfiducial\s+marker(?:s)?\b", re.IGNORECASE)
_FIDUCIAL_ACTION_RE = re.compile(r"\b(?:placed|deployed)\b", re.IGNORECASE)

_BALLOON_RE = re.compile(r"\bballoon\b", re.IGNORECASE)
_MUSTANG_BALLOON_RE = re.compile(r"\bmustang\s+balloon\b", re.IGNORECASE)
_DILATION_RE = re.compile(r"\bdilat", re.IGNORECASE)

_THERAPEUTIC_ASP_RE = re.compile(r"\btherapeutic\s+aspiration\b", re.IGNORECASE)

_EBUS_RE = re.compile(r"\bebus\b|endobronchial\s+ultrasound", re.IGNORECASE)
_TBNA_RE = re.compile(r"\btbna\b|transbronchial\s+needle\s+aspiration|needle\s+aspiration", re.IGNORECASE)
_SAMPLED_CONTEXT_RE = re.compile(r"\b(?:tbna|sampl|pass|needle|aspirat|biops)\w*\b", re.IGNORECASE)
_SAMPLED_NEGATION_RE = re.compile(
    r"\b(?:no|not|without)\s+(?:tbna|sampl(?:ing|ed)?|needle\s+aspirat(?:ion|ed)?|aspirat(?:ion|ed)?|biops(?:y|ies))\b"
    r"|\b(?:tbna|sampl(?:ing|ed)?|needle\s+aspirat(?:ion|ed)?|aspirat(?:ion|ed)?|biops(?:y|ies))\b.{0,40}\bnot\b.{0,40}\b(?:perform|done|obtain|take)\w*\b",
    re.IGNORECASE,
)

# Stations without an explicit "station" prefix are only accepted when they
# contain laterality (R/L) to avoid false positives from numbers (e.g., "10 cc").
_STATION_WITH_SIDE_RE = re.compile(r"\b(?:2|4|10|11|12)\s*[RL](?:s)?\b", re.IGNORECASE)
_STATION_EXPLICIT_RE = re.compile(r"\bstation\s*(?:2|4|7|10|11|12)(?:\s*)[RrLl]?(?:s)?\b", re.IGNORECASE)


def _stable_unique(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _any_nonnegated_match(note_text: str, pattern: re.Pattern[str], *, scope_chars: int = 80) -> bool:
    detector = get_negation_detector()
    for match in pattern.finditer(note_text):
        if detector.is_negated(note_text, match.start(), match.end(), scope_chars=scope_chars):
            continue
        return True
    return False


def _add_if_anchor(
    *,
    note_text: str,
    triggers: re.Pattern[str],
    required_in_context: re.Pattern[str] | None,
    scope_chars: int,
) -> bool:
    detector = get_negation_detector()
    for match in triggers.finditer(note_text):
        if detector.is_negated(note_text, match.start(), match.end(), scope_chars=scope_chars):
            continue

        if required_in_context is None:
            return True

        start = max(0, match.start() - scope_chars)
        end = min(len(note_text), match.end() + scope_chars)
        context = note_text[start:end]
        if required_in_context.search(context):
            return True

    return False


def _normalize_station_token(raw_token: str) -> str | None:
    token = raw_token.strip().upper()
    if token.startswith("STATION"):
        token = token.replace("STATION", "").strip()

    # Remove trailing punctuation/labels.
    token = re.sub(r"[^0-9A-Z]", "", token)
    if not token:
        return None

    # Normalize common suffix like "11RS" -> "11R".
    if token.endswith("RS"):
        token = token[:-1]
    if token.endswith("LS"):
        token = token[:-1]

    # Allow station 7 without laterality.
    if token == "7":
        return "7"

    match = re.fullmatch(r"(2|4|10|11|12)(R|L)", token)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return None


def _count_sampled_ebus_stations(note_text: str) -> set[str]:
    sampled: set[str] = set()

    station_matches: list[re.Match[str]] = []
    station_matches.extend(_STATION_EXPLICIT_RE.finditer(note_text))
    station_matches.extend(_STATION_WITH_SIDE_RE.finditer(note_text))

    for match in station_matches:
        start = max(0, match.start() - 120)
        end = min(len(note_text), match.end() + 200)
        context = note_text[start:end]
        if not _SAMPLED_CONTEXT_RE.search(context):
            continue
        if _SAMPLED_NEGATION_RE.search(context):
            continue

        normalized = _normalize_station_token(match.group(0))
        if normalized:
            sampled.add(normalized)

    return sampled


def expand_candidates(note_text: str, candidates: list[str]) -> list[str]:
    """Expand a CPT candidate list using conservative, anchored evidence.

    This function is designed for ML-first HIGH_CONF fastpath before strict
    rules validation.

    Adds:
    - 31626 (fiducial marker placement) when "fiducial marker" + placed/deployed
    - 31630 (airway dilation) when balloon + dilat* (or "mustang balloon")
    - 31645 (therapeutic aspiration) when explicitly documented
    - 31652/31653 (linear EBUS-TBNA) when EBUS + TBNA + sampled station evidence
    """
    note_text = note_text or ""
    expanded = _stable_unique(candidates)
    present = set(expanded)

    additions: list[str] = []

    if "31626" not in present and _add_if_anchor(
        note_text=note_text,
        triggers=_FIDUCIAL_RE,
        required_in_context=_FIDUCIAL_ACTION_RE,
        scope_chars=140,
    ):
        additions.append("31626")

    if "31630" not in present:
        has_dilation = _add_if_anchor(
            note_text=note_text,
            triggers=_MUSTANG_BALLOON_RE,
            required_in_context=None,
            scope_chars=140,
        )
        if not has_dilation:
            detector = get_negation_detector()
            for match in _BALLOON_RE.finditer(note_text):
                if detector.is_negated(note_text, match.start(), match.end(), scope_chars=120):
                    continue
                start = max(0, match.start() - 120)
                end = min(len(note_text), match.end() + 200)
                context = note_text[start:end]
                if _DILATION_RE.search(context):
                    has_dilation = True
                    break
        if has_dilation:
            additions.append("31630")

    if "31645" not in present and _add_if_anchor(
        note_text=note_text,
        triggers=_THERAPEUTIC_ASP_RE,
        required_in_context=None,
        scope_chars=120,
    ):
        additions.append("31645")

    if ("31652" not in present and "31653" not in present) and _any_nonnegated_match(
        note_text, _EBUS_RE, scope_chars=120
    ) and _any_nonnegated_match(note_text, _TBNA_RE, scope_chars=120):
        sampled_stations = _count_sampled_ebus_stations(note_text)
        if sampled_stations:
            additions.append("31653" if len(sampled_stations) >= 3 else "31652")

    for code in additions:
        if code not in present:
            expanded.append(code)
            present.add(code)

    return expanded
