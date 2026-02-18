"""Deterministic backstop extraction for central airway obstruction (CAO) site detail.

The v3 registry schema supports `registry.granular_data.cao_interventions_detail[]`
as a per-site structure for:
- location (Trachea, LMS/RMS/BI, lobar, etc.)
- pre/post obstruction percent or diameter
- modalities applied (APC, electrocautery snare, cryoextraction, balloon dilation, etc.)

Model-driven extraction can miss these fields; this module provides a conservative
regex-based extractor that only emits values explicitly supported by the note text.
"""

from __future__ import annotations

import re
from typing import Any


_CAO_HINT_RE = re.compile(
    r"(?i)\b(?:"
    r"central\s+airway|airway\s+obstruction|trache(?:a|al)|main(?:\s*|-)?stem(?:\s+obstruction)?|"
    r"debulk(?:ing)?|tumou?r\s+ablation|endoluminal\s+tumou?r|recanaliz|"
    r"airway\s+stent|y-?stent|tracheomalacia|bronchomalacia|rigid\s+bronchos"
    r")\b"
)

_CAO_LOCATION_CONTEXT_RE = re.compile(
    r"(?i)\b(?:obstruct|occlud|stenos|narrow|lesion|mass|tumou?r|granulation|"
    r"endobronchial|recanaliz|debulk|ablat|fungating|web|collapse|stent)\w*\b"
)

_POST_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"at\s+the\s+end|end\s+of\s+the\s+procedure|at\s+conclusion|finally|"
    r"post[-\s]?(?:procedure|intervention|treatment|op)|"
    r"post[-\s]?dilat\w*|post[-\s]?debulk\w*|"
    r"after\s+(?:debulk\w*|dilat\w*|ablat\w*|treat\w*|interven\w*)|"
    r"improv(?:ed|ement)\s+to|reduc(?:ed)?\s+to|decreas(?:ed)?\s+to"
    r")\b"
)

_PRE_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"prior\s+to|before|pre[-\s]?(?:procedure|intervention|treatment|op)|baseline|"
    r"initial\s+inspection|initial\s+evaluation|pre[-\s]?dilat\w*"
    r")\b"
)

_SENTENCE_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")

_REFERENCE_MEASUREMENT_PREFIX_RE = re.compile(
    r"(?i)\b(?:distance|dist\.?)\b[^.\n]{0,140}\b(?:to|from|relative\s+to)\s*$"
)

_FREEZE_TIME_SECONDS_RE = re.compile(
    r"(?i)\bfreeze(?:\s+(?:times?|time))?\b[^0-9]{0,30}"
    r"(?P<a>\d{1,3})(?:\s*(?:-|–|to)\s*(?P<b>\d{1,3}))?\s*"
    r"(?:s|sec|secs|second|seconds)\b"
)

_LOCATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Tracheostomy tube lumen", re.compile(r"(?i)\btracheostomy\s+tube\b.{0,80}?\blumen\b")),
    ("Trachea", re.compile(r"(?i)\btrachea(?:l)?\b")),
    ("Carina", re.compile(r"(?i)\b(?:main\s+carina|carina)\b")),
    ("RMS", re.compile(r"(?i)\bright\s+main(?:\s*|-)?stem\b|\bRMS\b")),
    ("LMS", re.compile(r"(?i)\bleft\s+main(?:\s*|-)?stem\b|\bLMS\b")),
    ("BI", re.compile(r"(?i)\bbronchus\s+intermedius\b|\bBI\b")),
    ("Lingula", re.compile(r"(?i)\blingula\b")),
    ("RUL", re.compile(r"(?i)\bright\s+upper\s+lobe\b|\bRUL\b")),
    ("RML", re.compile(r"(?i)\bright\s+middle\s+lobe\b|\bRML\b")),
    ("RLL", re.compile(r"(?i)\bright\s+lower\s+lobe\b|\bRLL\b")),
    ("LUL", re.compile(r"(?i)\bleft\s+upper\s+lobe\b|\bLUL\b")),
    ("LLL", re.compile(r"(?i)\bleft\s+lower\s+lobe\b|\bLLL\b")),
)

_MODALITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("APC", re.compile(r"(?i)\bapc\b|argon\s+plasma")),
    # Do not treat tool mentions ("snare") as intent; require explicit cautery/hot-snare context.
    ("Electrocautery - snare", re.compile(r"(?i)\b(?:electrocautery|cautery)\s+snare\b|\bhot\s+snare\b")),
    ("Electrocautery - knife", re.compile(r"(?i)\b(?:electrocautery|cautery)\s+(?:knife|needle\s*knife)\b")),
    ("Electrocautery - probe", re.compile(r"(?i)\b(?:electrocautery|cautery)\s+(?:probe|bicap|coag(?:ulation)?\s+probe)\b")),
    ("Cryoextraction", re.compile(r"(?i)\bcryo[-\s]*extraction\b|\bcryoextraction\b")),
    # Avoid misclassifying diagnostic cryoprobe use (cryobiopsy) as CAO cryotherapy.
    # Require explicit cryotherapy language OR therapeutic intent verbs near the cryoprobe mention.
    (
        "Cryotherapy - contact",
        re.compile(
            r"(?i)\bcryotherap\w*\b"
            r"|\bcryo\s*probe\b[^.\n]{0,80}\b(?:ablat|destroy|debulk|treat|reliev|recanaliz|devitaliz)\w*\b"
            r"|\b(?:ablat|destroy|debulk|treat|reliev|recanaliz|devitaliz)\w*\b[^.\n]{0,80}\bcryo\s*probe\b"
        ),
    ),
    ("Laser", re.compile(r"(?i)\blaser\b|nd:yag|yag\b")),
    ("Microwave", re.compile(r"(?i)\bmicrowave\b|\bmwa\b")),
    ("Microdebrider", re.compile(r"(?i)\bmicrodebrider\b")),
    ("Mechanical debulking", re.compile(r"(?i)\bmechanical\b.{0,30}\bdebulk|\bdebulk\w*\b|\bcore\s*out\b|\brig(?:id)?\s+coring\b")),
    ("Balloon dilation", re.compile(r"(?i)\bcre\s+balloon\b|\bballoon\b[^.\n]{0,80}\bdilat\w*\b|\bdilat\w*\b[^.\n]{0,80}\bballoon\b")),
    ("Suctioning", re.compile(r"(?i)\bsuction(?:ed|ing)?\b|\baspirat(?:e|ion)\b")),
    ("Iced saline lavage", re.compile(r"(?i)\b(?:cold|iced)\s+saline\b|\bsaline\s+flush")),
    ("Epinephrine instillation", re.compile(r"(?i)\bepinephrine\b")),
    ("Tranexamic acid instillation", re.compile(r"(?i)\btranexamic\b|\btxa\b")),
)

_LESION_MORPHOLOGY_RE = re.compile(
    r"(?i)\b(?:polypoid|fungating|exophytic|necrotic|granulation|web|stenos(?:is|ed))\b"
)
_LESION_COUNT_RE = re.compile(
    r"(?i)(?:\(|\b)?(?P<count>>\s*\d{1,3}|\d{1,3})\)?[^.\n]{0,40}\blesions?\b"
)

_INTRINSIC_OBSTRUCTION_RE = re.compile(
    r"(?i)\b(?:endobronchial|endoluminal|intraluminal|exophytic|tumou?r\s+ingrowth|granulation)\b"
)
_EXTRINSIC_OBSTRUCTION_RE = re.compile(
    r"(?i)\b(?:extrinsic|external)\s+compression\b|\bexternally\s+compress\w*\b|\bcompress(?:ed|ion)\b|\bmass\s+effect\b|\bbulging\b"
)
_MIXED_OBSTRUCTION_RE = re.compile(r"(?i)\bmixed\b")
_CLASSIFICATION_RE = re.compile(
    r"(?i)\b(?:myer[-\s]?cotton|cotton[-\s]?myer)\b[^.\n]{0,60}"
)

_OBSTRUCTION_PCT_AFTER_LOC_RE = re.compile(
    r"(?i)\b(?P<loc>[^.]{0,80}?)\b(?:was|were|is|are|remained|remains)?\s*"
    r"(?:only\s+)?(?:about|around|approximately|approx\.?)?\s*"
    r"(?P<pct>\d{1,3})\s*%\s*"
    r"(?:obstruct(?:ed|ion)?|occlud(?:ed|ing|e)?|stenos(?:is|ed)|narrow(?:ing|ed)?|block(?:ed|ing)?)\b"
    r"(?!\s+of\b)"
)
_OBSTRUCTION_PCT_BEFORE_LOC_RE = re.compile(
    r"(?i)\b(?:about|around|approximately|approx\.?)?\s*(?P<pct>\d{1,3})\s*%\s*"
    r"(?:obstruct(?:ion)?|obstructed|occlud(?:ed|ing|e)?|stenos(?:is|ed)|narrow(?:ing|ed)?|block(?:ed|ing)?)"
    r"(?:\s+(?:of|in|at)\s+(?:the\s+)?)"
    r"(?P<loc>[^.]{3,80})"
)
_BLOCKING_PCT_RE = re.compile(
    r"(?i)\bblocking\b[^%]{0,40}?(?P<pct>\d{1,3})\s*%\s*(?:of\s+(?:the\s+)?)?(?:airway|lumen)\b"
)
_OCCLUDING_PCT_RE = re.compile(
    r"(?i)\b(?:occluding|occluded)\b[^%]{0,40}?(?P<pct>\d{1,3})\s*%\s*"
    r"(?:of\s+(?:the\s+)?)?(?P<loc>[^.]{3,80})"
)
_PCT_OPEN_RE = re.compile(
    r"(?i)\b(?P<pct>\d{1,3})\s*%\s*(?:open|patent|recanaliz(?:ed|ation))\b"
)
_PATENCY_PCT_AFTER_WORD_RE = re.compile(
    r"(?i)\b(?:open|patent|recanaliz(?:ed|ation))\w*\b[^%]{0,40}?(?P<pct>\d{1,3})\s*%"
)
_RESIDUAL_OBSTRUCTION_PCT_RE = re.compile(
    r"(?i)\bresidual(?:\s+(?:obstruction|stenosis|narrowing))?\b[^%]{0,24}?(?P<pct>\d{1,3})\s*%"
)
_PCT_RESIDUAL_OBSTRUCTION_RE = re.compile(
    r"(?i)\b(?P<pct>\d{1,3})\s*%\s*(?:residual\s+)?(?:obstruction|stenosis|narrowing)\b"
)
_OBSTRUCTION_WORD_BEFORE_PCT_RE = re.compile(
    r"(?i)\b(?:"
    r"obstruct(?:ed|ion)?|occlud(?:ed|ing|e)?|stenos(?:is|ed)|narrow(?:ed|ing)?|block(?:ed|ing)?"
    r"|compress(?:ed|ion)"
    r")\b[^%]{0,24}?(?P<pct>\d{1,3})\s*%"
)
_COMPLETE_OBSTRUCTION_OF_RE = re.compile(
    r"(?i)\b(?:complete(?:ly)?|total)\s+(?:obstruction|occlusion)\b\s+of\s+(?:the\s+)?"
    r"(?P<loc>[^,.;\n]{3,80}?)(?=\s+(?:and|with|to|from|,|\\.|;|$))"
)
_COMPLETELY_OBSTRUCTED_RE = re.compile(
    r"(?i)\b(?:was|were|remained|remains)\s+(?:still\s+)?(?:completely|totally)\s+(?:obstructed|occluded|blocked)\b"
)

_STENT_PLACED_RE = re.compile(
    r"(?i)\b(?:stent|y-?stent)\b.{0,80}\b(?:placed|placement|deploy(?:ed)?|deployed|inserted)\b"
)
_STENT_NEGATION_CUES_RE = re.compile(
    r"(?i)\b(?:"
    r"considered|discussion|reluctan|declin|defer(?:red)?|not\s+placed|no\s+stent|"
    r"unable|not\s+possible|unsuccessful|"
    r"advocat(?:e|ing)\s+for|would\s+(?:recommend|consider)|if\s+i\s+need"
    r")\b"
)
_STENT_DEPLOY_DECISION_RE = re.compile(r"(?i)\b(?:decision|decided)\b.{0,80}\bdeploy\b")


def _maybe_unescape_newlines(text: str) -> str:
    raw = text or ""
    if not raw.strip():
        return raw
    if "\n" in raw or "\r" in raw:
        return raw
    if "\\n" not in raw and "\\r" not in raw:
        return raw
    return raw.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def _infer_location(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    for canonical, pattern in _LOCATION_PATTERNS:
        if pattern.search(raw):
            return canonical
    return None


def _infer_location_last(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    best_loc: str | None = None
    best_pos = -1
    for canonical, pattern in _LOCATION_PATTERNS:
        for match in pattern.finditer(raw):
            if match.start() >= best_pos:
                best_pos = match.start()
                best_loc = canonical
    return best_loc


def _append_modality(site: dict[str, Any], modality: str, *, meta: dict[str, Any] | None = None) -> None:
    apps = site.get("modalities_applied")
    if not isinstance(apps, list):
        apps = []
        site["modalities_applied"] = apps
    for existing in apps:
        if isinstance(existing, dict) and existing.get("modality") == modality:
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if value in (None, "", [], {}):
                        continue
                    if existing.get(key) in (None, "", [], {}):
                        existing[key] = value
            return
    payload: dict[str, Any] = {"modality": modality}
    if isinstance(meta, dict):
        for key, value in meta.items():
            if value in (None, "", [], {}):
                continue
            payload[key] = value
    apps.append(payload)


def _extract_cao_interventions_detail(
    note_text: str,
    *,
    collect_pct_candidates: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, set[int]]]]:
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return [], {}
    if not _CAO_HINT_RE.search(text):
        return [], {}

    sites: dict[str, dict[str, Any]] = {}
    pct_candidates: dict[str, dict[str, set[int]]] = {}
    current_location: str | None = None
    post_context_remaining = 0
    pre_context_remaining = 0
    fallback_location = None
    if re.search(
        r"(?i)\btrachea(?:l)?\b[^.\n]{0,80}\b(?:obstruct|stenos|narrow|lesion|mass|tumou?r|granulation|recanaliz|stent)\w*\b",
        text,
    ) or re.search(
        r"(?i)\b(?:obstruct|stenos|narrow|lesion|mass|tumou?r|granulation|recanaliz|stent)\w*\b[^.\n]{0,80}\btrachea(?:l)?\b",
        text,
    ):
        fallback_location = "Trachea"

    def _get_site(loc: str) -> dict[str, Any]:
        if loc not in sites:
            sites[loc] = {"location": loc}
        return sites[loc]

    def _assign_pct(loc: str, pct: int, *, is_post: bool) -> None:
        pct_int = max(0, min(100, int(pct)))
        site = _get_site(loc)

        if collect_pct_candidates:
            entry = pct_candidates.setdefault(
                loc,
                {"pre_obstruction_pct": set(), "post_obstruction_pct": set()},
            )
            if is_post:
                entry["post_obstruction_pct"].add(pct_int)
            else:
                entry["pre_obstruction_pct"].add(pct_int)

        if is_post:
            existing = site.get("post_obstruction_pct")
            if existing is None:
                site["post_obstruction_pct"] = pct_int
            else:
                site["post_obstruction_pct"] = min(int(existing), pct_int)
        else:
            existing = site.get("pre_obstruction_pct")
            if existing is None:
                site["pre_obstruction_pct"] = pct_int
            else:
                site["pre_obstruction_pct"] = max(int(existing), pct_int)

    for raw_sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = (raw_sentence or "").strip()
        if not sentence:
            continue

        is_post = post_context_remaining > 0
        is_pre = pre_context_remaining > 0

        if _POST_CUE_RE.search(sentence):
            post_context_remaining = max(post_context_remaining, 3)
            pre_context_remaining = 0
            is_post = True
            is_pre = False
        elif _PRE_CUE_RE.search(sentence):
            pre_context_remaining = max(pre_context_remaining, 3)
            post_context_remaining = 0
            is_pre = True
            is_post = False

        # Determine location context for sentences with no explicit location on the match.
        locations_in_sentence: list[str] = []
        for canonical, pattern in _LOCATION_PATTERNS:
            match = pattern.search(sentence)
            if match:
                # Measurement lines frequently use an anatomy term as a reference point
                # (e.g., "Distance ... to carina 90 mm") and should not set the CAO site.
                prefix = sentence[: match.start()]
                if _REFERENCE_MEASUREMENT_PREFIX_RE.search(prefix):
                    continue
                locations_in_sentence.append(canonical)
        if locations_in_sentence:
            heading_hint = bool(
                re.match(
                    r"(?i)^\s*(?:trachea|carina|rms|lms|bi|lingula|rul|rml|rll|lul|lll)\b\s*[:\-]",
                    sentence.strip(),
                )
            )
            if _CAO_LOCATION_CONTEXT_RE.search(sentence) or heading_hint:
                current_location = locations_in_sentence[0]

        # 1) Percent obstruction with explicit location group (loc before percent).
        for match in _OBSTRUCTION_PCT_AFTER_LOC_RE.finditer(sentence):
            loc_group = match.group("loc") or ""
            loc = _infer_location(loc_group) or current_location
            if not loc and fallback_location and re.search(r"(?i)\b(?:airway|lumen)\b", loc_group):
                loc = fallback_location
            if not loc:
                continue
            try:
                pct = int(match.group("pct"))
            except Exception:
                continue
            _assign_pct(loc, pct, is_post=is_post)

        # 2) Percent obstruction with explicit location group (percent before location).
        for match in _OBSTRUCTION_PCT_BEFORE_LOC_RE.finditer(sentence):
            loc_group = match.group("loc") or ""
            loc = _infer_location(loc_group) or current_location
            if not loc and fallback_location and re.search(r"(?i)\b(?:airway|lumen)\b", loc_group):
                loc = fallback_location
            if not loc:
                continue
            try:
                pct = int(match.group("pct"))
            except Exception:
                continue
            _assign_pct(loc, pct, is_post=is_post)

        # 2b) Obstruction described before percent (e.g., "stenosis is 80%") without explicit location groups.
        for match in _OBSTRUCTION_WORD_BEFORE_PCT_RE.finditer(sentence):
            prefix = sentence[max(0, match.start() - 160) : match.start()]
            loc = _infer_location_last(prefix) or current_location or fallback_location
            if not loc and len(locations_in_sentence) == 1:
                loc = locations_in_sentence[0]
            if not loc:
                continue
            try:
                pct = int(match.group("pct"))
            except Exception:
                continue
            _assign_pct(loc, pct, is_post=is_post)

        # 3) "Blocking 90% of the airway" (no explicit obstruction token after percent).
        for match in _BLOCKING_PCT_RE.finditer(sentence):
            loc = current_location or fallback_location
            if not loc and len(locations_in_sentence) == 1:
                loc = locations_in_sentence[0]
            if not loc:
                continue
            try:
                pct = int(match.group("pct"))
            except Exception:
                continue
            _assign_pct(loc, pct, is_post=is_post)

            # Preserve dynamic collapse context for clinician readability.
            if re.search(r"(?i)\b(?:exhalation|inhalation)\b", sentence):
                site = _get_site(loc)
                existing = (site.get("notes") or "").strip()
                snippet = sentence
                if len(snippet) > 320:
                    snippet = snippet[:320].rsplit(" ", 1)[0].strip()
                note_val = f"{existing}; {snippet}" if existing else snippet
                site["notes"] = note_val

        # 3b) "Occluding 80% of the tracheostomy tube lumen" patterns.
        for match in _OCCLUDING_PCT_RE.finditer(sentence):
            loc = _infer_location(match.group("loc") or "") or current_location
            if not loc and fallback_location and re.search(r"(?i)\b(?:airway|lumen)\b", match.group("loc") or ""):
                loc = fallback_location
            if not loc:
                continue
            try:
                pct = int(match.group("pct"))
            except Exception:
                continue
            _assign_pct(loc, pct, is_post=is_post)

        # 4) Percent open/patent -> obstruction = 100 - patency (usually post).
        for match in _PCT_OPEN_RE.finditer(sentence):
            loc = current_location or fallback_location
            if not loc and len(locations_in_sentence) == 1:
                loc = locations_in_sentence[0]
            if not loc:
                continue
            try:
                patency = int(match.group("pct"))
            except Exception:
                continue
            if not (0 <= patency <= 100):
                continue
            obstruction = 100 - patency
            # Default to post-procedure patency, but respect explicit "prior/before" cues.
            _assign_pct(loc, obstruction, is_post=not is_pre)

        # 4b) "patent/open to 80%" phrasing -> obstruction = 100 - patency (usually post).
        for match in _PATENCY_PCT_AFTER_WORD_RE.finditer(sentence):
            loc = current_location or fallback_location
            if not loc and len(locations_in_sentence) == 1:
                loc = locations_in_sentence[0]
            if not loc:
                continue
            try:
                patency = int(match.group("pct"))
            except Exception:
                continue
            if not (0 <= patency <= 100):
                continue
            obstruction = 100 - patency
            _assign_pct(loc, obstruction, is_post=not is_pre)

        # 4c) "residual 20% stenosis/obstruction" phrasing -> treat as obstruction percent.
        for match in _RESIDUAL_OBSTRUCTION_PCT_RE.finditer(sentence):
            loc = current_location or fallback_location
            if not loc and len(locations_in_sentence) == 1:
                loc = locations_in_sentence[0]
            if not loc:
                continue
            try:
                pct = int(match.group("pct"))
            except Exception:
                continue
            _assign_pct(loc, pct, is_post=not is_pre)

        # 4d) "20% residual stenosis/obstruction" phrasing (percent-first variant).
        for match in _PCT_RESIDUAL_OBSTRUCTION_RE.finditer(sentence):
            loc = current_location or fallback_location
            if not loc and len(locations_in_sentence) == 1:
                loc = locations_in_sentence[0]
            if not loc:
                continue
            try:
                pct = int(match.group("pct"))
            except Exception:
                continue
            _assign_pct(loc, pct, is_post=not is_pre)

        # 5) Complete obstruction/occlusion with explicit "of <location>" (multi-hit).
        for match in _COMPLETE_OBSTRUCTION_OF_RE.finditer(sentence):
            loc = _infer_location(match.group("loc") or "") or current_location
            if not loc:
                continue
            _assign_pct(loc, 100, is_post=is_post)

        # 6) "... remained completely obstructed" patterns (handles multi-location sentences).
        for match in _COMPLETELY_OBSTRUCTED_RE.finditer(sentence):
            prefix = sentence[max(0, match.start() - 140) : match.start()]
            loc = _infer_location_last(prefix) or current_location
            if not loc:
                continue
            _assign_pct(loc, 100, is_post=is_post)

        # Modalities + stent placement: attach to the best location context available.
        target_locations = locations_in_sentence or ([current_location] if current_location else [])
        target_locations = [loc for loc in target_locations if loc]
        lesion_locations = target_locations or ([fallback_location] if fallback_location else [])
        lesion_locations = [loc for loc in lesion_locations if loc]

        if lesion_locations:
            intrinsic = bool(_INTRINSIC_OBSTRUCTION_RE.search(sentence))
            extrinsic = bool(_EXTRINSIC_OBSTRUCTION_RE.search(sentence))
            mixed = bool(_MIXED_OBSTRUCTION_RE.search(sentence))
            inferred_type: str | None = None
            if mixed or (intrinsic and extrinsic):
                inferred_type = "Mixed"
            elif intrinsic:
                inferred_type = "Intraluminal"
            elif extrinsic:
                inferred_type = "Extrinsic"

            if inferred_type:
                for loc in lesion_locations:
                    site = _get_site(loc)
                    existing = str(site.get("obstruction_type") or "").strip()
                    if not existing:
                        site["obstruction_type"] = inferred_type
                    elif existing != inferred_type and existing in {"Intraluminal", "Extrinsic"}:
                        site["obstruction_type"] = "Mixed"

            classification_match = _CLASSIFICATION_RE.search(sentence)
            if classification_match:
                classification = classification_match.group(0).strip()
                if classification:
                    for loc in lesion_locations:
                        site = _get_site(loc)
                        if not str(site.get("classification") or "").strip():
                            site["classification"] = classification

            # Lesion morphology/count backstop (helps capture disease burden in templated notes).
            if _LESION_MORPHOLOGY_RE.search(sentence) and re.search(r"(?i)\b(?:lesion|lesions|tumou?r|mass)\b", sentence):
                m = _LESION_MORPHOLOGY_RE.search(sentence)
                morph = (m.group(0) if m else "").strip()
                if morph:
                    for loc in lesion_locations:
                        site = _get_site(loc)
                        if not site.get("lesion_morphology"):
                            site["lesion_morphology"] = morph.capitalize()

            count_match = _LESION_COUNT_RE.search(sentence)
            if count_match:
                raw_count = (count_match.group("count") or "").strip()
                if raw_count:
                    count_text = re.sub(r"\s+", "", raw_count)
                    for loc in lesion_locations:
                        site = _get_site(loc)
                        existing = str(site.get("lesion_count_text") or "").strip()
                        if not existing:
                            site["lesion_count_text"] = count_text
                        elif count_text not in existing:
                            site["lesion_count_text"] = f"{existing}; {count_text}"

        if target_locations:
            hemostasis_context = bool(
                re.search(r"(?i)\b(?:hemostas|ooz\w*|bleed\w*|hemorrhag\w*|control(?:led)?\s+bleed)\b", sentence)
            )
            freeze_seconds: int | None = None
            freeze_match = _FREEZE_TIME_SECONDS_RE.search(sentence)
            if freeze_match:
                try:
                    a = int(freeze_match.group("a"))
                except Exception:
                    a = None
                try:
                    b = int(freeze_match.group("b")) if freeze_match.group("b") else None
                except Exception:
                    b = None
                values = [v for v in (a, b) if isinstance(v, int)]
                if values:
                    freeze_seconds = max(values)

            for modality, pattern in _MODALITY_PATTERNS:
                if pattern.search(sentence):
                    for loc in target_locations:
                        site = _get_site(loc)

                        # Separate hemostasis from CAO "modalities applied" when supported by explicit language.
                        if modality in {
                            "Tranexamic acid instillation",
                            "Epinephrine instillation",
                            "Iced saline lavage",
                            "Balloon tamponade",
                        } and hemostasis_context:
                            site["hemostasis_required"] = True
                            methods = site.get("hemostasis_methods")
                            if not isinstance(methods, list):
                                methods = []
                                site["hemostasis_methods"] = methods
                            label = (
                                "Tranexamic acid"
                                if modality == "Tranexamic acid instillation"
                                else "Epinephrine"
                                if modality == "Epinephrine instillation"
                                else "Iced saline"
                                if modality == "Iced saline lavage"
                                else "Balloon tamponade"
                            )
                            if label not in methods:
                                methods.append(label)
                            continue

                        meta: dict[str, Any] = {}
                        if modality in {"Cryoextraction", "Cryotherapy - contact"} and freeze_seconds is not None:
                            meta["freeze_time_seconds"] = int(freeze_seconds)

                        _append_modality(site, modality, meta=meta or None)

            # Capture medication instillation notes when explicitly documented (schema-safe: notes field).
            if re.search(r"(?i)\bamphotericin\b", sentence):
                for loc in target_locations:
                    site = _get_site(loc)
                    existing = str(site.get("notes") or "").strip()
                    addition = "Amphotericin instilled."
                    if not existing:
                        site["notes"] = addition
                    elif addition.lower() not in existing.lower():
                        site["notes"] = (existing + " " + addition).strip()

        if _STENT_PLACED_RE.search(sentence):
            negated = bool(_STENT_NEGATION_CUES_RE.search(sentence))
            if negated and _STENT_DEPLOY_DECISION_RE.search(sentence):
                negated = False

            stent_locations = list(target_locations)
            if not stent_locations and not negated:
                if re.search(r"(?i)\by-?stent\b", sentence):
                    stent_locations = ["Carina"]
                elif fallback_location:
                    stent_locations = [fallback_location]
                else:
                    stent_locations = ["Trachea"]

            for loc in stent_locations:
                site = _get_site(loc)
                if negated:
                    if site.get("stent_placed_at_site") is None:
                        site["stent_placed_at_site"] = False
                else:
                    site["stent_placed_at_site"] = True

        if post_context_remaining > 0:
            post_context_remaining -= 1
        if pre_context_remaining > 0:
            pre_context_remaining -= 1

    # Return in stable order (proximal → distal-ish).
    priority = {"Tracheostomy tube lumen": 0, "Trachea": 1, "Carina": 2, "RMS": 3, "LMS": 4, "BI": 5}

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        loc = str(item.get("location") or "")
        return (priority.get(loc, 100), loc)

    details = list(sites.values())
    details.sort(key=sort_key)
    filtered: list[dict[str, Any]] = []
    for item in details:
        if item.get("pre_obstruction_pct") is not None:
            filtered.append(item)
            continue
        if item.get("post_obstruction_pct") is not None:
            filtered.append(item)
            continue
        if item.get("stent_placed_at_site") is True:
            filtered.append(item)
            continue
        if isinstance(item.get("modalities_applied"), list) and item.get("modalities_applied"):
            filtered.append(item)
            continue
    return filtered, pct_candidates


def extract_cao_interventions_detail(note_text: str) -> list[dict[str, Any]]:
    """Extract CAO site detail into v3 granular format.

    Returns a list of dicts compatible with granular_data.cao_interventions_detail.
    """
    details, _ = _extract_cao_interventions_detail(note_text, collect_pct_candidates=False)
    return details


def extract_cao_interventions_detail_with_candidates(
    note_text: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, set[int]]]]:
    """Extract CAO site detail plus per-site obstruction-% candidates."""
    return _extract_cao_interventions_detail(note_text, collect_pct_candidates=True)
