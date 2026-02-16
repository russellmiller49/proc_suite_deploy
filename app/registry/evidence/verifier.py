from __future__ import annotations

import re

from rapidfuzz.fuzz import partial_ratio

from app.common.spans import Span
from app.registry.deterministic_extractors import (
    AIRWAY_DILATION_PATTERNS,
    CHEST_TUBE_PATTERNS,
    IPC_PATTERNS,
    RIGID_BRONCHOSCOPY_PATTERNS,
    THERAPEUTIC_ASPIRATION_PATTERNS,
    extract_airway_stent,
)
from app.registry.schema import RegistryRecord
from app.registry.schema.ip_v3_extraction import IPRegistryV3, ProcedureEvent


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_STENT_TOKEN_RE = re.compile(r"\bstent(?:ing|s)?\b", re.IGNORECASE)
_STENT_NEGATION_WINDOW_RE = re.compile(
    r"\b(?:"
    r"no\s+(?:appropriate|suitable)\s+landing\s+point"
    r"|no\s+landing\s+point"
    r"|no\s+appropriate\s+place\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r"|not\s+(?:safe|possible|feasible)\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r"|unable\s+to\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r"|could\s+not\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r"|decid(?:ed|ing)\b[^.\n]{0,80}\b(?:against|to\s+not)\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r"|decision\b[^.\n]{0,80}\bto\s+not\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r"|abandon(?:ed|ing)?\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r"|abort(?:ed|ing)?\b[^.\n]{0,80}\bstent(?:ing|s)?\b"
    r")\b",
    re.IGNORECASE,
)

EVIDENCE_REQUIRED: dict[str, str] = {
    # HARD: flip performed=false when unsupported
    "procedures_performed.airway_dilation.performed": "HARD",
    "pleural_procedures.chest_tube.performed": "HARD",
    "pleural_procedures.ipc.performed": "HARD",
    # airway_stent is HARD only when not "Assessment only"
    "procedures_performed.airway_stent.performed": "HARD",
    # REVIEW: keep but require manual review
    "procedures_performed.rigid_bronchoscopy.performed": "REVIEW",
}


def normalize_text(text: str) -> str:
    """Normalize text for robust substring matching.

    - lowercase
    - remove punctuation (keep a-z, 0-9)
    - collapse whitespace to single spaces
    """
    lowered = (text or "").lower()
    no_punct = _NON_ALNUM_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", no_punct).strip()
    return collapsed


def verify_registry(registry: IPRegistryV3, full_source_text: str) -> IPRegistryV3:
    """Verify and anchor event evidence quotes against the full source note text.

    For each procedure event, attempt to:
    1) Anchor the quote to exact offsets in the note text (preferred), updating
       `evidence.quote` to the exact substring from the note and populating
       `evidence.start/end`.
    2) Fall back to normalized containment verification when offsets cannot be
       determined, clearing any pre-filled offsets to avoid misleading spans.

    Events whose evidence quote cannot be verified are dropped.
    """

    from app.evidence.quote_anchor import anchor_quote
    from app.registry.schema.ip_v3_extraction import EvidenceSpan

    full_text = full_source_text or ""
    normalized_source = normalize_text(full_text)

    kept: list[ProcedureEvent] = []
    for event in registry.procedures:
        evidence = getattr(event, "evidence", None)
        quote = getattr(evidence, "quote", None) if evidence is not None else None
        quote_clean = (str(quote) if quote is not None else "").strip()
        if not quote_clean:
            continue

        anchored = anchor_quote(full_text, quote_clean)
        if anchored.span is not None:
            updated = event.model_copy(deep=True)
            if updated.evidence is None:
                updated.evidence = EvidenceSpan(
                    quote=anchored.span.text,
                    start=anchored.span.start,
                    end=anchored.span.end,
                )
            else:
                updated.evidence.quote = anchored.span.text
                updated.evidence.start = anchored.span.start
                updated.evidence.end = anchored.span.end
            kept.append(updated)
            continue

        normalized_quote = normalize_text(quote_clean)
        if normalized_quote and normalized_quote in normalized_source:
            updated = event.model_copy(deep=True)
            if updated.evidence is not None:
                updated.evidence.start = None
                updated.evidence.end = None
            kept.append(updated)

    return registry.model_copy(update={"procedures": kept})


def _normalized_contains(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return normalize_text(needle) in normalize_text(haystack)


def _verify_quote_in_text(quote: str, full_text: str, *, fuzzy_threshold: int = 85) -> bool:
    quote_clean = (quote or "").strip()
    if not quote_clean:
        return False
    if quote_clean in (full_text or ""):
        return True
    lowered_quote = quote_clean.lower()
    lowered_text = (full_text or "").lower()
    if lowered_quote in lowered_text:
        return True
    if _normalized_contains(full_text or "", quote_clean):
        return True

    normalized_quote = normalize_text(quote_clean)
    if len(normalized_quote) < 12:
        return False

    score = partial_ratio(normalized_quote, normalize_text(full_text or ""))
    return score >= fuzzy_threshold


def _evidence_texts_for_prefix(record: RegistryRecord, prefix: str) -> list[str]:
    evidence = getattr(record, "evidence", None) or {}
    if not isinstance(evidence, dict):
        return []

    texts: list[str] = []
    for key, spans in evidence.items():
        if not isinstance(key, str) or not key:
            continue
        if key != prefix and not key.startswith(prefix + "."):
            continue
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, Span):
                continue
            text = (span.text or "").strip()
            if text:
                texts.append(text)
    return texts


def _drop_evidence_prefix(record: RegistryRecord, prefix: str) -> None:
    evidence = getattr(record, "evidence", None)
    if not isinstance(evidence, dict) or not evidence:
        return
    to_drop = [k for k in evidence.keys() if isinstance(k, str) and (k == prefix or k.startswith(prefix + "."))]
    for key in to_drop:
        evidence.pop(key, None)


def _add_first_anchor_span(record: RegistryRecord, field_path: str, full_text: str, patterns: list[str]) -> bool:
    if not full_text or not patterns:
        return False
    for pat in patterns:
        match = re.search(pat, full_text, re.IGNORECASE)
        if not match:
            continue
        anchor_text = (match.group(0) or "").strip()
        if not anchor_text:
            continue
        record.evidence.setdefault(field_path, []).append(
            Span(
                text=anchor_text,
                start=int(match.start()),
                end=int(match.end()),
                confidence=0.9,
            )
        )
        return True
    return False


def _wipe_model_fields(obj: object, wipe_fields: dict[str, object]) -> None:
    if obj is None:
        return
    for name, value in wipe_fields.items():
        if hasattr(obj, name):
            setattr(obj, name, value)


def _find_therapeutic_aspiration_anchor(full_text: str) -> tuple[str, int, int] | None:
    text_lower = (full_text or "").lower()
    if not text_lower:
        return None

    def _match_negated(match: re.Match[str]) -> bool:
        start, end = match.start(), match.end()
        before = full_text[max(0, start - 30) : start]
        after = full_text[end : min(len(full_text), end + 80)]

        if re.search(r"(?i)\b(?:no|without)\b[^.\n]{0,20}$", before):
            return True

        if re.search(
            r"(?i)\b(?:not\s+(?:performed|done|attempted)|was\s+not\s+performed|declined|deferred|aborted)\b",
            after,
        ):
            return True

        return False

    for pattern in THERAPEUTIC_ASPIRATION_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if not match:
            continue
        if _match_negated(match):
            continue
        return (match.group(0).strip(), match.start(), match.end())

    contextual_patterns = [
        r"\b(?:copious|large\s+amount\s+of|thick|tenacious|purulent|bloody|blood-tinged)\s+secretions?\b[^.]{0,80}\b(?:suction(?:ed|ing)|aspirat(?:ed|ion)|cleared|remov(?:ed|al))\b",
        r"\b(?:suction(?:ed|ing)|aspirat(?:ed|ion)|cleared|remov(?:ed|al))\b[^.]{0,80}\b(?:copious|large\s+amount\s+of|thick|tenacious|purulent|bloody|blood-tinged)\s+secretions?\b",
        r"\b(?:suction(?:ed|ing)|aspirat(?:ed|ion)|cleared|remov(?:ed|al))\b[^.]{0,80}\b(?:mucus\s+plug|clot|blood)\b",
        r"\b(?:mucus|mucous|secretions?|blood|clot(?:s)?|debris|fluid|plug(?:s)?)\b[^.]{0,80}\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?)\b",
        r"\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?)\b[^.]{0,80}\b(?:mucus|mucous|secretions?|blood|clot(?:s)?|debris|fluid|plug(?:s)?)\b",
        r"\b(?:airway|airways|trachea|bronch(?:us|i)?|tracheobronchial\s+tree)\b[^.]{0,120}\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?)\b",
        r"\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?)\b[^.]{0,120}\b(?:airway|airways|trachea|bronch(?:us|i)?|tracheobronchial\s+tree)\b",
    ]
    contextual_patterns.extend(
        [
            r"\bsecretions?\b[^.]{0,120}\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?)\b",
            r"\b(?:suction(?:ed|ing)?|aspirat(?:ed|ion|ing)?|clear(?:ed|ing)?)\b[^.]{0,120}\bsecretions?\b",
        ]
    )

    for pattern in contextual_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if not match:
            continue
        if _match_negated(match):
            continue
        return (match.group(0).strip(), match.start(), match.end())

    return None


def _stent_is_negated(full_text: str) -> bool:
    """Return True when stent mentions are explicitly negated (not performed)."""
    text = full_text or ""
    if not text:
        return False

    for match in _STENT_TOKEN_RE.finditer(text):
        start = max(0, match.start() - 220)
        end = min(len(text), match.end() + 220)
        window = text[start:end]
        if _STENT_NEGATION_WINDOW_RE.search(window):
            return True

    return False


def verify_evidence_integrity(record: RegistryRecord, full_note_text: str) -> tuple[RegistryRecord, list[str]]:
    """Apply Python-side guardrails against hallucinated performed events/details.

    This is intentionally conservative: if a high-risk performed=true procedure
    cannot be supported by extractable evidence, it is flipped to performed=false
    and dependent details are wiped.
    """

    warnings: list[str] = []
    full_text = full_note_text or ""

    procedures = getattr(record, "procedures_performed", None)
    if procedures is None:
        return record, warnings

    # ------------------------------------------------------------------
    # High-risk: therapeutic aspiration (frequent false-positives)
    # ------------------------------------------------------------------
    ta = getattr(procedures, "therapeutic_aspiration", None)
    if getattr(ta, "performed", None) is True:
        prefixes = [
            "procedures_performed.therapeutic_aspiration",
            "therapeutic_aspiration",
        ]
        candidate_quotes: list[str] = []
        for prefix in prefixes:
            candidate_quotes.extend(_evidence_texts_for_prefix(record, prefix))

        verified = any(_verify_quote_in_text(q, full_text) for q in candidate_quotes)
        if not verified:
            anchor = _find_therapeutic_aspiration_anchor(full_text)
            if anchor is not None:
                anchor_text, start, end = anchor
                record.evidence.setdefault("procedures_performed.therapeutic_aspiration.performed", []).append(
                    Span(text=anchor_text, start=start, end=end)
                )
                verified = True

        if not verified:
            setattr(ta, "performed", False)
            for dependent_field in ("material", "location"):
                if hasattr(ta, dependent_field):
                    setattr(ta, dependent_field, None)
            for prefix in prefixes:
                _drop_evidence_prefix(record, prefix)
            warnings.append("WIPED_VERIFICATION_FAILED: procedures_performed.therapeutic_aspiration")

    # ------------------------------------------------------------------
    # High-risk: hallucinated percutaneous trach device name (e.g., Portex)
    # ------------------------------------------------------------------
    trach = getattr(procedures, "percutaneous_tracheostomy", None)
    device_name = getattr(trach, "device_name", None)
    if isinstance(device_name, str) and device_name.strip():
        if not _normalized_contains(full_text, device_name):
            setattr(trach, "device_name", None)
            warnings.append("WIPED_DEVICE_NAME_NOT_IN_TEXT: procedures_performed.percutaneous_tracheostomy.device_name")

    # ------------------------------------------------------------------
    # High-risk: airway stent false positives (keyword present but explicitly not performed)
    # ------------------------------------------------------------------
    stent = getattr(procedures, "airway_stent", None)
    if getattr(stent, "performed", None) is True:
        seed = extract_airway_stent(full_text) if full_text.strip() else {}
        action_supported = bool(seed.get("airway_stent", {}).get("performed") is True)
        if not action_supported and _stent_is_negated(full_text):
            setattr(stent, "performed", False)
            wipe_fields = {
                "action": None,
                "stent_type": None,
                "stent_brand": None,
                "diameter_mm": None,
                "length_mm": None,
                "location": None,
                "indication": None,
                "deployment_successful": None,
                "airway_stent_removal": False,
            }
            for field_name, value in wipe_fields.items():
                if hasattr(stent, field_name):
                    setattr(stent, field_name, value)
            prefixes = [
                "procedures_performed.airway_stent",
                "airway_stent",
            ]
            for prefix in prefixes:
                _drop_evidence_prefix(record, prefix)
            warnings.append("NEGATION_GUARD: procedures_performed.airway_stent")

    # ------------------------------------------------------------------
    # Evidence-required enforcement (HARD vs REVIEW)
    # ------------------------------------------------------------------
    pleural = getattr(record, "pleural_procedures", None)
    rigid = getattr(procedures, "rigid_bronchoscopy", None)
    airway_dilation = getattr(procedures, "airway_dilation", None)
    chest_tube = getattr(pleural, "chest_tube", None) if pleural is not None else None
    ipc = getattr(pleural, "ipc", None) if pleural is not None else None

    def _enforce_boolean(
        *,
        field_path: str,
        obj: object,
        policy: str,
        anchor_patterns: list[str],
        wipe_fields: dict[str, object] | None = None,
        skip_if: bool = False,
    ) -> None:
        nonlocal warnings
        if skip_if:
            return
        if obj is None or not hasattr(obj, "performed"):
            return
        if getattr(obj, "performed", None) is not True:
            return

        prefixes = [field_path, field_path.rsplit(".", 1)[0]]
        candidate_quotes: list[str] = []
        for prefix in prefixes:
            candidate_quotes.extend(_evidence_texts_for_prefix(record, prefix))

        verified = any(_verify_quote_in_text(q, full_text) for q in candidate_quotes)
        if not verified:
            if _add_first_anchor_span(record, field_path, full_text, anchor_patterns):
                candidate_quotes = _evidence_texts_for_prefix(record, field_path)
                verified = any(_verify_quote_in_text(q, full_text) for q in candidate_quotes)

        if verified:
            return

        if policy == "REVIEW":
            warnings.append(f"NEEDS_REVIEW: EVIDENCE_MISSING: {field_path}")
            return

        # HARD: flip performed=false + wipe dependent details
        setattr(obj, "performed", False)
        if wipe_fields:
            _wipe_model_fields(obj, wipe_fields)
        for prefix in prefixes:
            _drop_evidence_prefix(record, prefix)
        warnings.append(f"EVIDENCE_HARD_FAIL: {field_path}")

    # HARD policies
    _enforce_boolean(
        field_path="procedures_performed.airway_dilation.performed",
        obj=airway_dilation,
        policy=EVIDENCE_REQUIRED["procedures_performed.airway_dilation.performed"],
        anchor_patterns=AIRWAY_DILATION_PATTERNS,
        wipe_fields={
            "location": None,
            "etiology": None,
            "method": None,
            "balloon_diameter_mm": None,
            "pre_dilation_diameter_mm": None,
            "post_dilation_diameter_mm": None,
        },
    )
    _enforce_boolean(
        field_path="pleural_procedures.chest_tube.performed",
        obj=chest_tube,
        policy=EVIDENCE_REQUIRED["pleural_procedures.chest_tube.performed"],
        anchor_patterns=CHEST_TUBE_PATTERNS,
        wipe_fields={
            "action": None,
            "side": None,
            "indication": None,
            "tube_type": None,
            "tube_size_fr": None,
            "guidance": None,
        },
    )
    _enforce_boolean(
        field_path="pleural_procedures.ipc.performed",
        obj=ipc,
        policy=EVIDENCE_REQUIRED["pleural_procedures.ipc.performed"],
        anchor_patterns=IPC_PATTERNS,
        wipe_fields={
            "action": None,
            "side": None,
            "catheter_brand": None,
            "indication": None,
            "tunneled": None,
        },
    )

    stent_action = str(getattr(stent, "action", "") or "").strip().lower() if stent is not None else ""
    stent_assessment_only = stent_action.startswith("assessment")
    _enforce_boolean(
        field_path="procedures_performed.airway_stent.performed",
        obj=stent,
        policy=EVIDENCE_REQUIRED["procedures_performed.airway_stent.performed"],
        anchor_patterns=[r"\bairway\s+stent\b", r"\bstent\b"],
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
        skip_if=stent_assessment_only,
    )

    # REVIEW policies
    _enforce_boolean(
        field_path="procedures_performed.rigid_bronchoscopy.performed",
        obj=rigid,
        policy=EVIDENCE_REQUIRED["procedures_performed.rigid_bronchoscopy.performed"],
        anchor_patterns=RIGID_BRONCHOSCOPY_PATTERNS,
        wipe_fields={
            "rigid_scope_size": None,
            "indication": None,
            "jet_ventilation_used": None,
        },
    )

    return record, warnings


__all__ = ["normalize_text", "verify_registry", "verify_evidence_integrity"]
