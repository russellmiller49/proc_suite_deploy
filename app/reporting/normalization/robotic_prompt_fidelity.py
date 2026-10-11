"""Conservative reporter seeding for short peripheral/EBUS dictations.

Only the supplied prompt is consulted. Counts are bound within the peripheral
or nodal clause; unknown counts, techniques, and results stay unknown. This
layer never changes registry fields or CPT derivation.
"""

from __future__ import annotations

import re

from proc_schemas.clinical import ProcedureBundle, ProcedureInput

_WORDS = dict(
    zip(
        ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"),
        map(str, range(1, 12)),
        strict=True,
    )
)
_LOBE = r"RUL|RML|RLL|LUL|LLL|lingula(?:r)?"
_SEGMENT = r"[RL]B(?:10|[1-9])(?:/[1-9])?"
_MODALITIES = {
    "transbronchial_needle_aspiration": r"TBNA|FNA|needles?|aspirat(?:es?|ions?)",
    "transbronchial_cryobiopsy": r"cryo(?:biops(?:y|ies))?|cryos|cryoprobe|TBCBx",
    "transbronchial_lung_biopsy": r"forceps",
    "bronchial_brushings": r"brush(?:ings?|ed)?",
}
_REPLACED = set(_MODALITIES) | {
    "transbronchial_biopsy",
    "robotic_navigation",
    "radial_ebus_survey",
    "radial_ebus_sampling",
    "bal",
    "bronchial_washing",
    "ebus_tbna",
}


def normalize_prompt(text: str) -> str:
    text = re.sub(r"\bone point five\b", "1.5", text, flags=re.I)
    for phrase, replacement in (
        ("eleven right inferior", "11RI"),
        ("eleven right superior", "11RS"),
        ("eleven left", "11L"),
        ("four right", "4R"),
        ("seven", "7"),
        ("upper left", "LUL"),
        ("right upper", "RUL"),
        ("right lower", "RLL"),
        ("left upper", "LUL"),
        ("left lower", "LLL"),
        ("right middle", "RML"),
        ("right upper lobe", "RUL"),
        ("right middle lobe", "RML"),
        ("right lower lobe", "RLL"),
        ("left upper lobe", "LUL"),
        ("left lower lobe", "LLL"),
    ):
        text = re.sub(r"\b" + phrase + r"\b", replacement, text, flags=re.I)
    text = re.sub(
        r"\b(" + "|".join(_WORDS) + r")\b", lambda m: _WORDS[m[0].lower()], text, flags=re.I
    )
    text = re.sub(r"(?i)\b11\s+(Rs|Ri)\b", r"11\1", text)
    text = re.sub(r"\b11\s+R\s*([is])\b", r"11R\1", text, flags=re.I)
    text = re.sub(r"\b11\s+left\b", "11L", text, flags=re.I)
    return text


def is_short_peripheral_prompt(text: str | None) -> bool:
    text = normalize_prompt(text or "")
    # This adapter handles procedural fragments. Existing report/section and
    # demographic adapters retain their richer information on complete notes.
    if "\n" in text.strip() or re.search(
        r"(?i)\b(?:bronchoscopy report|patient:|procedure:|"
        r"\d+[ -]*(?:yo|years?[ -]old)\s+(?:male|female|[MF])\b)",
        text,
    ):
        return False
    if re.search(r"(?i)\b(?:ILD|interstitial lung disease|UIP|NSIP)\b", text):
        return False
    if re.search(r"(?i)planned[^.;]*not performed", text):
        return False
    if re.search(r"(?i)\b(?:planned|scheduled|consider|history of)\b", text) and not re.search(
        r"(?i)performed|obtained|sampled", text
    ):
        return False
    if len(text) > 1600 or re.search(
        r"PROCEDURE IN DETAIL|DATE OF PROCEDURE|\[DESCRIPTION\]", text, re.I
    ):
        return False
    peripheral, _ = split_peripheral_nodal(text)
    if not re.search(r"(?i)\b(?:" + _LOBE + "|" + _SEGMENT + r")\b", peripheral):
        return False
    return bool(
        re.search(
            r"(?i)\b(?:" + _LOBE + "|" + _SEGMENT + r"|(?:right|left)\s+(?:upper|middle|lower))\b",
            text,
        )
        and re.search(
            r"(?i)\b(?:ion|robot\w*|rEBUS|CBCT|EBUS|cone[- ]beam|cryobiops\w*|cryos|radial|spin)\b",
            text,
        )
    )


def split_peripheral_nodal(text: str) -> tuple[str, str]:
    for match in re.finditer(r"(?i)\b(?:(?:linear\s+|c)?EBUS\b|Nodes:|Nodal evaluation:)", text):
        if re.search(r"(?i)\bradial\s*$", text[: match.start()]):
            continue
        start = match.start()
        preceding_passes = re.search(r"(?i)\b\d+\s+\d{2}G\s*$", text[:start])
        if preceding_passes:
            start = preceding_passes.start()
        preceding_nodal = re.search(
            r"(?i)\b(?:station\s+)?(?:11R[is]|11[RL]|10[RL]|[24][RL]|7)"
            r"(?:-node|\s+needle\s+staging)\s+(?:with\s+)?$",
            text[:start],
        )
        if preceding_nodal:
            start = preceding_nodal.start()
        # "11L EBUS x5" is a nodal clause, including its station label.
        station_before = re.search(
            r"(?i)\b(?:station\s+)?(?:11[RL][is]?|[24][RL]|"
            r"7)(?:[\s,/]+(?:and\s+)?(?:11[RL][is]?|[24][RL]|7))*(?:\s+\d+)?\s*$",
            text[:start],
        )
        if station_before:
            start = station_before.start()
        prior_action = re.search(r"(?i)(?:sampled|sampling|aspirat\w*)[^.;]{0,55}$", text[:start])
        if prior_action and re.search(r"(?i)\b(?:11[RL][is]?|[24][RL]|7)\b", prior_action[0]):
            start = prior_action.start()
        return text[:start], text[start:]
    # Conversational prompts sometimes omit EBUS but explicitly sample nodes.
    match = re.search(
        r"(?i)\b(?:Nodes:|(?:we (?:also|then) sampled|subsequent|then station)\s+(?:11[RL]|"
        r"[24][RL]|7)\b|we then sampled subcarinal|[24][RL] node sampled|"
        r"biopsied \d+ nodal stations)",
        text,
    )
    return (text[: match.start()], text[match.start() :]) if match else (text, "")


def _count(text: str, modality: str) -> int | None:
    # Device gauges/diameters are not specimen counts.
    cleaned = re.sub(r"\b\d{2}(?:\s*[/+]\s*\d{2})?\s*(?:-?gauge|g)\b", " ", text, flags=re.I)
    cleaned = re.sub(r"(?<!\d)\d+\.\d+\s*(?:mm)?", " ", cleaned)
    cleaned = re.sub(r"(?i)\bprotected\s+|\bneedle\s+(?=aspirat)|\btransbronchial\s+", "", cleaned)
    pattern = _MODALITIES[modality]
    suffix_style = (
        len(
            re.findall(r"(?i)\b(?:" + "|".join(_MODALITIES.values()) + r")\s*(?:x\s*)?\d+", cleaned)
        )
        >= 2
    )
    for match in re.finditer(r"(?i)\b(?:" + pattern + r")(?=\b|\d)", cleaned):
        before, after = cleaned[: match.start()], cleaned[match.end() :]
        suffix = re.match(
            r"\s*(?:x\s*)?(\d{1,2})(?![\d./])\b(?!\s*(?:mm|cm|g|seconds?))", after, re.I
        )
        prefix = re.search(r"(?<![\d.])(\d{1,2})\s+(?:transbronchial\s+)?$", before, re.I)
        if re.match(r"(?i)\s*(?:gauges|confirmed|location|position)", after):
            prefix = None
        if (
            suffix
            and not suffix_style
            and re.match(
                r"(?i)\s*(?:" + "|".join(_MODALITIES.values()) + r")\b", after[suffix.end() :]
            )
        ):
            suffix = None
        if (
            suffix_style
            and prefix
            and re.search(
                r"(?i)(?:" + "|".join(_MODALITIES.values()) + r")\s*$", before[: prefix.start()]
            )
        ):
            prefix = None
        candidate = (suffix or prefix) if suffix_style else (prefix or suffix)
        if candidate and 0 < int(candidate[1]) <= 99:
            return int(candidate[1])
    return None


def _targets(text: str) -> list[tuple[str, str]]:
    """Bind samples to target clauses, never to nearby nodal counts."""
    anchors = list(re.finditer(r"(?i)\b(?:" + _LOBE + "|" + _SEGMENT + r")\b", text))
    if not anchors:
        return []
    candidates = []
    for m in anchors:
        label = m[0].upper()
        if label.startswith("LINGUL"):
            label = "Lingula"
        # A segment immediately after its lobe refines the same target.
        if (
            candidates
            and re.fullmatch(_SEGMENT, label, re.I)
            and not re.fullmatch(_SEGMENT, candidates[-1][0], re.I)
            and m.start() - candidates[-1][2] < 40
        ):
            if not re.search(r"[.;]|\band\b", text[candidates[-1][2] : m.start()]):
                candidates[-1] = (label, candidates[-1][1], m.end())
                continue
        if not any(label == item[0] for item in candidates):
            candidates.append((label, m.start(), m.end()))
    multi = bool(
        re.search(
            r"(?i)\b(?:2[- ](?:site|target|RLL|robotic|separate|lung|spots)|both|"
            r"each target|second (?:site|spot|target)|separate (?:[RL]B|lung|target)|"
            r"repeated navigation|then navigated|sites:|targets:)",
            text,
        )
    )
    # Two segment-led sample clauses also delimit distinct targets.
    if len(candidates) > 1 and re.search(r"(?i)(?:;|\.)\s*(?:lingula|[RL]B\d)", text):
        multi = True
    if not multi:
        segment = next((c for c in candidates if re.fullmatch(_SEGMENT, c[0], re.I)), None)
        return [((segment or candidates[0])[0], text)]
    first_clause = re.search(r"(?i)\bFirst:", text)
    second_clause = re.search(
        r"(?i)\b(?:Second(?: site| spot| target)?:|Second (?:site|spot|target)\b|"
        r"Then navigated|For the anterior)",
        text,
    )
    if first_clause and second_clause and len(candidates) > 1:
        return [
            (candidates[0][0], text[first_clause.end() : second_clause.start()]),
            (candidates[1][0], text[second_clause.end() :]),
        ]
    if second_clause and (len(candidates) < 2 or "Second" in second_clause[0]):
        return [
            (candidates[0][0] + " (first target)", text[: second_clause.start()]),
            (candidates[0][0] + " (second target)", text[second_clause.end() :]),
        ]
    if len(candidates) > 1:
        first, second = candidates[:2]
        return [(first[0], text[: second[1]]), (second[0], text[second[1] :])]
    # Do not invent which target received which specimens in a sparse summary.
    return [(candidates[0][0] + " (2 targets; allocation unspecified)", text)]


def _new(kind: str, index: int, data: dict) -> ProcedureInput:
    return ProcedureInput(
        proc_type=kind, schema_id=kind + "_v1", proc_id=f"{kind}_prompt_{index}", data=data
    )


def _node_data(text: str) -> tuple[list[dict], bool]:
    text = re.sub(r"(?i)\b\d{2}\s*-?(?:G|gauge)\b", "", text)
    text = re.sub(r"(?i)\bneedle\s+", "", text)
    survey_only = bool(
        re.search(
            r"(?i)\b(?:survey (?:only|without)|inspection only|scanned only|"
            r"no (?:nodal (?:biopsies|specimens)|(?:node |nodal )?TBNA|bx)|"
            r"zero TBNA|without (?:lymph[- ]node |node )?(?:biopsy|puncture|sampling)|"
            r"none (?:were biopsied|aspirated)|did not sample|no nodes requiring biopsy|"
            r"no nodal specimens)",
            text,
        )
    )
    if re.search(
        r"(?i)did not warrant any node sampling|too small and soft to stick|elastography only", text
    ):
        survey_only = True
    stations = []
    matches = list(re.finditer(r"(?i)(?<![\w])(?:11R[is]|11[RL]|10[RL]|[24][RL]|7)(?=\b|\d)", text))
    common = re.search(r"(?i)\b(\d{1,2})\s+(?:(?:needle )?passes?\s+)?(?:each|ea)\b", text)
    # The compact final xN convention applies to a station list, not to a
    # preceding needle gauge or an unrelated peripheral sample count.
    if not common and len(matches) > 1:
        common = re.search(r"(?i)\bx\s*(\d{1,2})(?:\s+each)?\s*(?:[.;]|$)", text)
    for index, match in enumerate(matches):
        station = match[0].upper()
        if station in {s["station_name"] for s in stations}:
            continue
        tail = text[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(text)
        ]
        count = re.match(r"(?i)[\s,:]*(?:x\s*)?(\d{1,2})\b(?!\s*g)", tail)
        if not count:
            count = re.search(r"(?i)\b(?:sampled\s+)?(\d{1,2})\s+(?:times|passes|samples)\b", tail)
        prefix = text[matches[index - 1].end() if index else 0 : match.start()]
        before = re.search(r"(?i)\b(\d+)\s+(?:passes?\s+)?at\s+(?:station\s+)?$", prefix)
        if not count:
            count = re.search(r"(?i)\b(?:x\s*(\d+)|(\d+)\s+passes)", tail)
            if count:
                value = next(g for g in count.groups() if g is not None)
                tail = " " + value + " " + tail
                count = re.match(r"\s*(\d+)", tail)
        if not before and len(matches) == 1:
            before = re.search(r"(?i)\b(\d+)\s+EBUS passes", text)
        count = count or before or common
        result = re.search(
            r"(?i)\b(?:benign lymph\w*(?:\s+\w+)?|lymphocytes/benign|benign cytology|"
            r"benign lymphoid|lymphoid tissue)\b",
            tail,
        )
        stations.append(
            {
                "station_name": station,
                "passes": int(count[1]) if count else None,
                "biopsy_tools": ["TBNA"],
                "rose_result": result[0] if result else None,
            }
        )
    return stations, survey_only


# IASLC names already used by the EBUS template. Only unambiguous named
# stations are normalized; "right paratracheal" alone does not select 2R/4R.
# Reference: https://www.iaslc.org/file/10870/download?token=ZdsZFnAS (Ch. 11)
_NODE_ALIASES = {
    "subcarinal": "7",
    "left hilar": "10L",
    "right hilar": "10R",
    "right upper paratracheal": "2R",
    "right lower paratracheal": "4R",
    "left upper paratracheal": "2L",
    "left lower paratracheal": "4L",
}
_STATION_LABEL = r"(?:11R[is]|11[RL]|10[RL]|[24][RL]|7)"


def _has_adjacent_station(text: str, start: int, end: int) -> bool:
    return bool(
        re.search(r"(?i)\b" + _STATION_LABEL + r"\s*[(:-]?\s*$", text[:start])
        or re.match(r"(?i)\s*\(?\s*(?:station\s+)?" + _STATION_LABEL + r"\b", text[end:])
    )


def _assess_nodal(text: str) -> tuple[list[dict], bool, list[str], list[str]]:
    """Separate documented sampling from missing station identity/status."""
    alias_pattern = r"(?i)\b(?:" + "|".join(map(re.escape, _NODE_ALIASES)) + r")\b"
    canonical = re.sub(
        alias_pattern,
        lambda m: (
            m[0] if _has_adjacent_station(text, m.start(), m.end()) else _NODE_ALIASES[m[0].lower()]
        ),
        text,
    )
    stations, survey_only = _node_data(canonical)
    ambiguous = [
        m[0]
        for m in re.finditer(r"(?i)\b(?:right|left) paratracheal\b", text)
        if not _has_adjacent_station(text, m.start(), m.end())
    ]
    # Limit cytologic-result evidence to the nodal clause before subsequent
    # lung/lesion results. "Benign" alone can be an ultrasound assessment.
    sampling_clause = re.split(r"(?i)[.;]|\b(?:lung|lesion|nodule)\s+(?:ROSE|cytology)", canonical)[
        0
    ]
    sampled = bool(
        re.search(r"(?i)TBNA|biops|sampl|aspirat|passes|needle staging", sampling_clause)
        or any(st["passes"] is not None for st in stations)
        or re.search(r"(?i)benign lymphocytes|lymphoid tissue|lymphocytes/benign", sampling_clause)
    )
    gaps = []
    if not survey_only:
        if not stations or ambiguous:
            gaps.append("station_labels")
        if not sampled:
            gaps.append("sampling_status")
    return stations, survey_only, gaps, ambiguous


def ebus_detail_gaps(source: str) -> list[str]:
    """Structured reasons for the short-prompt EBUS review warning."""
    _, nodal = split_peripheral_nodal(normalize_prompt(source))
    return _assess_nodal(nodal)[2] if nodal else []


def has_source_supported_ebus_stations(bundle: ProcedureBundle, source: str) -> bool:
    """Whether short-prompt reconstruction resolved every sampled station label.

    Used only to retire stale upstream station warnings. Other missing details
    and partially ambiguous station lists must continue to request review.
    """
    if not is_short_peripheral_prompt(source):
        return False
    _, nodal = split_peripheral_nodal(normalize_prompt(source))
    stations, survey_only, gaps, _ = _assess_nodal(nodal)
    if not stations or survey_only or gaps:
        return False
    expected = {s["station_name"].upper() for s in stations}
    actual = set()
    for procedure in bundle.procedures:
        if procedure.proc_type != "ebus_tbna":
            continue
        data = procedure.data
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        for station in data.get("stations", []):
            actual.add(str(station.get("station_name") or "").upper())
    return expected == actual


def _lavages(text: str, location: str, index: int) -> list[ProcedureInput]:
    site = re.search(r"(?i)\b(" + _SEGMENT + r")\s+(?:BAL|lavage)", text)
    if site:
        location = site[1].upper()
    text = re.sub(r"(?i)(BAL|lavage)(?=\d)", r"\1 ", text)
    volumes = []
    for sentence in re.split(r"[.;](?!\d)", text):
        if not re.search(r"(?i)BAL|lavage|wash", sentence):
            continue
        pairs = list(
            re.finditer(r"(?i)(?<![\w./])(\d+)\s*(?:in\s*)?[:/]\s*(\d+)(?![\w\d])", sentence)
        )
        if not pairs:
            pairs = list(
                re.finditer(r"(?i)(\d+) instilled,? (\d+) (?:recovered|returned)", sentence)
            )
        volumes.extend((int(m[1]), int(m[2])) for m in pairs)
    if volumes and re.search(r"(?i)\bpooled\b", text):
        volumes = [(sum(v[0] for v in volumes), sum(v[1] for v in volumes))]
    if not volumes:
        return [_new("bal", index, {"lung_segment": location, "tests": []})]
    return [
        _new(
            "bal",
            index + i,
            {
                "lung_segment": location,
                "instilled_volume_cc": v[0],
                "returned_volume_cc": v[1],
                "tests": [],
            },
        )
        for i, v in enumerate(volumes)
    ]


def reconcile_robotic_prompt(bundle: ProcedureBundle, source: str) -> ProcedureBundle:
    if not is_short_peripheral_prompt(source):
        return bundle
    normalized = normalize_prompt(source)
    normalized = re.sub(r"(?i)(\d+)\s+(\d{2}G)\s+passes", r"\1 \2 needle passes", normalized)
    peripheral, nodal = split_peripheral_nodal(normalized)
    targets = _targets(peripheral)
    if not targets:
        return bundle
    bundle = bundle.model_copy(deep=True)
    bundle.procedures = [p for p in bundle.procedures if p.proc_type not in _REPLACED]
    new = []
    platform = next(
        (p for p in ("Ion", "Monarch", "Galaxy") if re.search(r"\b" + p + r"\b", source, re.I)),
        None,
    )
    nav_explicit = bool(platform or re.search(r"(?i)\brobot\w*|\bnavigat\w*", source))
    shared_counts = bool(re.search(r"(?i)\b(?:both|each)\b", peripheral)) and len(targets) > 1
    observations = []
    imaging = re.search(
        r"(?i)\b(?:used|acquired|obtained|performed)\s+"
        r"(?:(?:sequential|repeat|multiple)\s+)?cone[- ]beam\s+spins?\b[^.;]*",
        peripheral,
    )
    if imaging and not re.search(
        r"(?i)\b(?:not|never|planned to|did not)\s*$", peripheral[: imaging.start()]
    ):
        # Preserve the documented imaging/confirmation clause without inventing
        # a spin count, radiation dose, ventilation settings, or reconstruction.
        observations.append(f"CBCT guidance: {imaging[0].strip()}.")
    # Device deployment in these prompts usually concerns a fiducial. A stent
    # still requires its own explicitly documented placement action.
    if not re.search(
        r"(?i)stent.{0,50}(?:placed|deployed|inserted)|(?:placed|deployed|inserted).{0,50}stent",
        source,
    ):
        bundle.procedures = [
            p for p in bundle.procedures if p.proc_type != "airway_stent_placement"
        ]
    if not re.search(r"(?i)\bIndication:", source):
        bundle.indication_text = None
    # Preserve an explicitly named marker even when a short synonym was missed.
    if re.search(r"(?i)\bfiducial\b|\b(?:gold )?marker\b", peripheral) and not re.search(
        r"(?i)no fiducial|no marker|marker not", peripheral
    ):
        bundle.procedures = [
            p for p in bundle.procedures if p.proc_type != "fiducial_marker_placement"
        ]
        new.append(_new("fiducial_marker_placement", 1, {"airway_location": targets[0][0]}))
    for index, (location, local) in enumerate(targets, 1):
        # Explicit lesion dimensions survive even when the source has no
        # indication field. Reject measurements tied to a sampling probe.
        target_intro = re.split(
            r"(?i)\b(?:" + "|".join(_MODALITIES.values()) + r")(?=\b|\d)", local, maxsplit=1
        )[0]
        for size in re.finditer(
            r"(?i)(?<![\d.])(\d+(?:\.\d+)?)\s*(cm|mm|centimeters?|millimeters?)\b", target_intro
        ):
            if re.match(r"(?i)\s*(?:cryo|probe)", target_intro[size.end() :]):
                continue
            observations.append(f"Target {location}: {size[1]} {size[2]}.")
            break
        if nav_explicit:
            new.append(
                _new(
                    "robotic_navigation", index, {"platform": platform, "lesion_location": location}
                )
            )
        pattern = None
        if re.search(r"(?i)\b(?:radial|rEBUS)\b", local) or (
            len(targets) > 1
            and re.search(r"(?i)\b(?:radial|rEBUS)\b", peripheral)
            and re.search(r"(?i)\b(?:eccentric|concentric|ecc|conc)\b", local)
        ):
            if re.search(
                r"(?i)not seen|not diagnostic|nondiagnostic|could not identify|"
                r"white (?:specks|flecks)",
                local,
            ):
                pattern = "not identifiable"
            elif re.search(r"(?i)\b(?:eccentric|ecc)\b", local):
                pattern = "Eccentric"
            elif re.search(r"(?i)\b(?:concentric|conc)\b", local):
                pattern = "Concentric"
            new.append(
                _new("radial_ebus_survey", index, {"location": location, "rebus_features": pattern})
            )
        for kind, pattern in _MODALITIES.items():
            count = _count(local, kind)
            present = bool(re.search(r"(?i)\b(?:" + pattern + r")(?=\b|\d)", local))
            if shared_counts:
                count = _count(peripheral, kind)
                present = bool(re.search(r"(?i)\b(?:" + pattern + r")(?=\b|\d)", peripheral))
            if not present or re.search(
                r"(?i)\b(?:no|without)\s+(?:(?:TBNA|cryo(?:biopsy)?|needle|forceps)\s+(?:or|and)\s+)?(?:"
                + pattern
                + r")\b|\b(?:"
                + pattern
                + r")\s+(?:not performed|not obtained|available|on standby|not used)",
                local,
            ):
                continue
            if (
                kind == "transbronchial_needle_aspiration"
                and count is None
                and not re.search(
                    r"(?i)TBNA|FNA|aspirat|(?:sampled|samples?|sampling|obtained|took|"
                    r"collected).{0,60}needle|needle.{0,35}(?:samples?|aspirat)",
                    peripheral if shared_counts else local,
                )
            ):
                continue
            payload = {"lung_segment": location, "tests": []}
            if kind == "transbronchial_needle_aspiration":
                gauge = re.search(r"(?i)\b(\d{2}(?:[/+]\d{2})?)\s*-?(?:gauge|G)\b", local)
                if gauge:
                    payload["needle_gauge"] = gauge[1] + "G"
            if kind == "transbronchial_cryobiopsy":
                payload["num_samples"] = count
                lobe = re.search(
                    r"(?i)\b(" + _LOBE + r")\s+(?:\w+\s+){0,3}" + re.escape(location) + r"\b",
                    peripheral,
                )
                if lobe:
                    payload["notes"] = "Target lobe: " + lobe[1].upper() + "."

                probe = re.search(r"(?i)\b(\d\.\d)\s*(?:mm\s+)?(?:cryo|probe)", local)
                if probe:
                    payload["cryoprobe_size_mm"] = float(probe[1])
                freeze = re.search(
                    r"(?i)\b(\d+(?:\s*[-–]\s*\d+)?)\s*[- ]second(?:s| freezes| freeze)?", local
                )
                if freeze:
                    value = freeze[1]
                    payload["freeze_seconds"] = int(value) if value.isdigit() else None
                    payload["notes"] = (
                        f"Freeze time: {value} seconds" if not value.isdigit() else None
                    )
            elif kind == "transbronchial_lung_biopsy":
                # This schema requires a count; unknown forceps details remain
                # in findings rather than being replaced by a default of one.
                if count is None:
                    observations.append(
                        f"Forceps samples obtained from {location}; count not documented."
                    )
                    continue
                payload.update(samples_collected=count, forceps_tools="Forceps")
            else:
                payload["samples_collected"] = count
            new.append(_new(kind, index, payload))
        lavage_text = (
            peripheral
            if shared_counts and re.search(r"(?i)BAL.*both|(?:each|both).*BAL", peripheral)
            else local
        )
        if re.search(r"(?i)\b(?:mini[- ]?)?BAL(?=\b|\d)|lavag\w*|wash(?:ed|es)?\b", lavage_text):
            new.extend(_lavages(lavage_text, location, index * 10))
        if (
            not any(
                p.proc_type in (_MODALITIES.keys() - {"bronchial_brushings"})
                and p.data.get("lung_segment") == location
                for p in new
            )
            and re.search(r"(?i)\bbiops(?:y|ies|ied)\b", local)
            and not re.search(
                r"(?i)\bforceps\b|brush\s+biops|\b(?:no|without)\s+biops|"
                r"\bnot\s+biopsied|\bbiopsy\s+(?:not performed|not obtained)",
                local,
            )
        ):
            observations.append(
                f"Peripheral lung biopsy at {location}; sampling technique unspecified."
            )
        # Retain target-local, qualified ROSE wording without assigning it to a node.
        rose = re.search(r"(?i)\bROSE\s*[:=-]?\s*([^.;]+)", local)
        if rose:
            label = (
                ""
                if re.search(r"(?i)\b(?:" + _LOBE + "|" + _SEGMENT + r")\b", rose[1])
                else location + " "
            )
            observations.append(f"{label}ROSE: {rose[1].strip()}.")
        else:
            qualified_rose = re.search(
                r"(?i)\b((?:inflammatory|blood-only|lesional|atypical|negative malignant|"
                r"inflammatory-necrotic)(?: first-site)?)\s+ROSE\b",
                local,
            )
            if qualified_rose:
                observations.append(f"{location} ROSE: {qualified_rose[1]}.")
    # A separate lavage can occur after nodal EBUS; do not treat it as another
    # biopsy target or add the nodal pass count to the volume.
    for m in re.finditer(
        r"(?i)\b(RML)(?:\s+(RB4(?:/RB?5|/5)?))?\s+(?:BAL|lavage)(?:\s+\d+[:/]\d+)?", normalized
    ):
        loc = m[2] or m[1]
        if not any(
            p.proc_type == "bal" and p.data.get("lung_segment") in (loc, "RML") for p in new
        ):
            new.extend(_lavages(m[0], loc, 900))
    if re.search(
        r"(?i)(?:lavaged|lavage|BAL).{0,12}RUL (?:and|/)RML|RUL/RML BAL", normalized
    ) and not any(p.proc_type == "bal" and p.data.get("lung_segment") == "RML" for p in new):
        new.extend(_lavages("BAL", "RML", 910))
    stations, survey_only, nodal_gaps, ambiguous = _assess_nodal(nodal)
    if nodal:
        labels = ", ".join(st["station_name"] for st in stations)
        if survey_only:
            observations.append(
                "Linear EBUS survey without nodal sampling."
                + (" Stations inspected: " + labels + "." if stations else "")
            )
        else:
            if stations and "sampling_status" not in nodal_gaps:
                new.append(_new("ebus_tbna", 1, {"stations": stations}))
            if nodal_gaps:
                details = []
                if "station_labels" in nodal_gaps:
                    if ambiguous:
                        details.append(
                            "Station labels unspecified for "
                            + ", ".join(ambiguous)
                            + " (upper/lower not stated)."
                        )
                    else:
                        details.append("Individual station labels unspecified.")
                if "sampling_status" in nodal_gaps:
                    details.append("Nodal sampling versus inspection unspecified.")
                # Keep all supplied names/counts/actions in the report while
                # making the actual missing information explicit.
                observations.append(
                    "Linear EBUS evaluation: " + " ".join(details) + " Source: " + nodal.strip()
                )
    # Preserve qualified source results as observations, without upgrading
    # atypia/lesional cells to malignancy or assigning lesion ROSE to nodes.
    for sentence in re.split(r"(?<=[.;])\s+", source):
        if re.search(r"(?i)\bROSE\s*[:=-]?\s*([^.;]+)", sentence):
            continue  # already rendered above with its target-local qualification
        if re.search(
            r"(?i)cytology|purulent|\bpus\b|touch prep|rapid smear|white specks|no (?:PTX|"
            r"pneumothorax)|PTX neg|malignan|atypia|no CA",
            sentence,
        ):
            observations.append("Source detail: " + sentence.strip())
    bundle.procedures.extend(new)
    bundle.findings_text = "\n".join(dict.fromkeys(observations)) or None
    bundle.specimens_text = None  # templates enumerate only explicitly supplied samples
    bundle.impression_plan = None  # do not manufacture follow-up orders from sparse input
    bundle.estimated_blood_loss = (
        None if not re.search(r"(?i)blood loss|\bEBL\b", source) else bundle.estimated_blood_loss
    )
    if not re.search(r"(?i)complication|uncomplicated|no (?:issues|problems)|neg comps", source):
        bundle.complications_text = None
    elif re.search(
        r"(?i)no (?:immediate )?complications?|uncomplicated|no (?:issues|problems)|"
        r"neg comps|complications: none",
        source,
    ):
        bundle.complications_text = "No immediate complications documented."
    return bundle
