"""Bind report details to explicit operative target/site blocks.

This reporter-only reconciliation runs while seeding, never while editing a
bundle. It does not change registry flags, coding, or the shared schema. A
complete, recognizable narrative block outranks document-wide scalar guesses.
Unrecognized note formats retain the existing adapter path.
"""

from __future__ import annotations

import re

from proc_schemas.clinical import ProcedureBundle, ProcedureInput

_NAV_START = re.compile(r"(?im)^[ \t]*Robotic navigation bronchoscopy was performed with [^\n.]+\.")
_END = re.compile(
    r"(?im)^[ \t]*(?:EBUS[- ]Findings|SPECIMEN\(S\)|IMPRESSION/PLAN|"
    r"ESTIMATED BLOOD LOSS|COMPLICATIONS)\s*:?"
)
_SITE = re.compile(r"(?im)^[ \t]*Site\s+\d+\s*:\s*The\s+(11R[is]|[124]R|[124]L|7|10[RL]|11[RL])\b")
_SAMPLING = re.compile(
    r"(?im)^[ \t]*(Transbronchial needle aspiration|Transbronchial cryobiopsy|"
    r"Transbronchial biopsy|Transbronchial brushing|Bronchial alveolar lavage) was performed\b"
)
_TARGET_TYPES = {
    "robotic_navigation",
    "radial_ebus_survey",
    "radial_ebus_sampling",
    "transbronchial_needle_aspiration",
    "transbronchial_cryobiopsy",
    "transbronchial_biopsy",
    "transbronchial_lung_biopsy",
    "bronchial_brushings",
    "bal",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().rstrip(".")


def _match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.I)
    return _clean(match.group(1)) if match else None


def _proc(kind: str, index: str, data: dict) -> ProcedureInput:
    return ProcedureInput(
        proc_type=kind,
        schema_id=f"{kind}_v1",
        proc_id=f"{kind}_source_{index}",
        data=data,
    )


def _section(text: str, heading: str) -> str | None:
    match = re.search(r"(?im)^[ \t]*" + re.escape(heading) + r"[ \t]*:[ \t]*", text)
    if not match:
        return None
    body = text[match.end() :]
    end = re.search(r"(?m)^[ \t]*[A-Z][A-Z ()/]+[ \t]*:", body)
    return body[: end.start() if end else len(body)].strip() or None


def _target_procedures(text: str) -> list[ProcedureInput] | None:
    starts = list(_NAV_START.finditer(text))
    if not starts:
        return None
    procedures: list[ProcedureInput] = []
    for index, start in enumerate(starts, 1):
        block = text[start.start() : starts[index].start() if index < len(starts) else len(text)]
        end = _END.search(block)
        if end:
            block = block[: end.start()]
        location = _match(r"catheter was used to engage the\s+([^\n.]+)", block)
        if not location:
            # Do not partially replace a multi-target report.
            return None
        platform = _match(r"performed with\s+([^\n.]+?)\s+platform", block)
        registration = _match(r"\b(Partial|Full|Automatic) registration was used", block)
        size = _match(r"Target lesion is about\s+(\d+(?:\.\d+)?\s*(?:cm|mm))\b", block)
        distance = _match(r"advanced to\s+(\d+(?:\.\d+)?\s*cm)\s+away", block)
        rose = _match(r"ROSE from ION procedure was noted to be:[ \t]*\n?[ \t]*([^\n]+)", block)
        notes = []
        if size:
            notes.append(f"Target lesion diameter: {size}.")
        if distance:
            notes.append(f"Catheter-to-target distance: {distance}.")
        if rose:
            notes.append(f"Target ROSE: {rose}.")
        procedures.append(
            _proc(
                "robotic_navigation",
                str(index),
                {
                    "platform": platform,
                    "lesion_location": location,
                    "registration_method": registration,
                    "notes": " ".join(notes) or None,
                },
            )
        )
        pattern = _match(r"Radial EBUS was performed[^\n]*?nodule is\s+([^\n.]+)", block)
        if pattern:
            procedures.append(
                _proc(
                    "radial_ebus_survey",
                    str(index),
                    {
                        "location": location,
                        "rebus_features": pattern,
                        "notes": _match(r"following features were noted:\s*([^\n]+)", block),
                    },
                )
            )
        samples = list(_SAMPLING.finditer(block))
        for sample_index, sample in enumerate(samples):
            detail = block[
                sample.start() : samples[sample_index + 1].start()
                if sample_index + 1 < len(samples)
                else len(block)
            ]
            # Each step is a paragraph. Do not borrow a count/result from a later step.
            detail = re.split(r"\n[ \t]*\n", detail, maxsplit=1)[0]
            count = _match(r"Total\s+(\d{1,3})\s+sample(?:s|\(s\))?\s+were collected", detail)
            tests = _match(r"Samples sent for\s+([^\n.]+)", detail)
            data = {"lung_segment": location, "tests": [tests] if tests else []}
            name = sample.group(1).lower()
            if name == "transbronchial cryobiopsy":
                probe = _match(r"(\d+(?:\.\d+)?)\s*mm cryoprobe", detail)
                freeze = _match(r"Freeze time of\s+([\d]+(?:\s*[-–]\s*\d+)?)\s+seconds", detail)
                data.update(
                    num_samples=int(count) if count else None,
                    cryoprobe_size_mm=float(probe) if probe else None,
                    freeze_seconds=int(freeze) if freeze and freeze.isdigit() else None,
                    notes=f"Freeze time: {freeze} seconds"
                    if freeze and not freeze.isdigit()
                    else None,
                )
                kind = "transbronchial_cryobiopsy"
            elif name == "bronchial alveolar lavage":
                volumes = re.search(
                    r"Instilled\s+(\d+)\s*cc[^.]*?returned with\s+(\d+)\s*cc", detail, re.I
                )
                data.update(
                    instilled_volume_cc=int(volumes[1]) if volumes else None,
                    returned_volume_cc=int(volumes[2]) if volumes else None,
                )
                pooled = re.search(
                    r"instillation of\s+(\d+)\s*cc[^.]*?return of\s+(\d+)\s*cc[^.]*?pooled",
                    detail,
                    re.I,
                )
                if pooled and volumes:
                    data["instilled_volume_cc"] += int(pooled[1])
                    data["returned_volume_cc"] += int(pooled[2])
                kind = "bal"
            else:
                data["samples_collected"] = int(count) if count else None
                if name == "transbronchial needle aspiration":
                    gauges = re.findall(r"\b(\d{2})\s*G\s+Needle", detail, re.I)
                    data["needle_gauge"] = ", ".join(f"{g}G" for g in dict.fromkeys(gauges)) or None
                    kind = "transbronchial_needle_aspiration"
                elif name == "transbronchial brushing":
                    data["brush_tool"] = _match(
                        r"performed with\s+(.+?)\s+(?:through\s+)?the extended", detail
                    )
                    kind = "bronchial_brushings"
                else:
                    data["forceps_tools"] = _match(
                        r"performed with\s+(.+?)\s+(?:through\s+)?the extended", detail
                    )
                    kind = "transbronchial_lung_biopsy"
            procedures.append(_proc(kind, str(index), data))
    return procedures


def reconcile_operative_blocks(bundle: ProcedureBundle, text: str) -> ProcedureBundle:
    """Preserve target-local facts and distinguish sampled from inspected nodes."""
    target_procs = _target_procedures(text)
    ebus_start = re.search(r"(?im)^[ \t]*EBUS[- ]Findings\b", text)
    if target_procs is None and ebus_start is None:
        return bundle
    bundle = bundle.model_copy(deep=True)
    if target_procs is not None:
        bundle.procedures = [
            p for p in bundle.procedures if p.proc_type not in _TARGET_TYPES
        ] + target_procs
        ventilation = re.search(
            r"Mode\s+RR\s+TV\s+PEEP\s+FiO2\s+Flow Rate\s+Pmean\s*\n"
            r"[ \t]*(\w+)\s+(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+"
            r"(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)",
            text,
            re.I,
        )
        if ventilation:
            fields = dict(
                zip(
                    (
                        "vent_mode",
                        "vent_rr",
                        "vent_tv_ml",
                        "vent_peep_cm_h2o",
                        "vent_fio2_pct",
                        "vent_flow_rate",
                        "vent_pmean_cm_h2o",
                    ),
                    (
                        ventilation[1],
                        int(ventilation[2]),
                        int(ventilation[3]),
                        float(ventilation[4]),
                        int(ventilation[5]),
                        ventilation[6],
                        float(ventilation[7]),
                    ),
                    strict=True,
                )
            )
            for proc in bundle.procedures:
                if proc.proc_type == "robotic_ion_bronchoscopy":
                    payload = (
                        proc.data.model_dump()
                        if hasattr(proc.data, "model_dump")
                        else dict(proc.data)
                    )
                    proc.data = {**payload, **fields}
    # Keep explicit diagnoses and the submitted specimen inventory instead of
    # summaries synthesized earlier from document-wide counts.
    for heading, field in (
        ("PREOPERATIVE DIAGNOSIS", "preop_diagnosis_text"),
        ("POSTOPERATIVE DIAGNOSIS", "postop_diagnosis_text"),
        ("SPECIMEN(S)", "specimens_text"),
        ("ESTIMATED BLOOD LOSS", "estimated_blood_loss"),
        ("IMPRESSION/PLAN", "impression_plan"),
    ):
        value = _section(text, heading)
        if value:
            if heading == "POSTOPERATIVE DIAGNOSIS" and value.rstrip(".").lower() == "same":
                value = bundle.preop_diagnosis_text or value
            setattr(bundle, field, value)

    if ebus_start:
        ebus = text[ebus_start.end() :]
        end = _END.search(ebus)
        if end:
            ebus = ebus[: end.start()]
        sites = list(_SITE.finditer(ebus))
        stations, survey = [], []
        for index, site in enumerate(sites):
            detail = ebus[
                site.start() : sites[index + 1].start() if index + 1 < len(sites) else len(ebus)
            ]
            station = site[1].upper()
            sampled = bool(re.search(r"\bThe site was sampled\b", detail, re.I))
            not_sampled = bool(re.search(r"\bThe site was not sampled\b", detail, re.I))
            if not sampled and not not_sampled:
                # Only reconcile when every site states its sampling status.
                break
            elastography = _match(r"\b(Type\s+[123])\s+elastographic pattern", detail)
            if not_sampled:
                survey.append(
                    f"Station {station}: not sampled"
                    + (f"; elastography {elastography}" if elastography else "")
                    + "."
                )
                continue
            count = _match(
                r"\b(\d{1,3})\s+endobronchial ultrasound guided transbronchial biopsies", detail
            )
            rose = _match(r"Preliminary ROSE Cytology was reported as\s+([^\n.]+)", detail)
            stations.append(
                {
                    "station_name": station,
                    "passes": int(count) if count else None,
                    "echo_features": f"Elastography {elastography}" if elastography else None,
                    "biopsy_tools": ["TBNA"],
                    "rose_result": rose,
                    # CT size thresholds are not exact EBUS measurements.
                    "comments": _match(r"node was\s+(.+?\bon CT)\b", detail),
                }
            )
        else:
            if sites:
                bundle.procedures = [p for p in bundle.procedures if p.proc_type != "ebus_tbna"]
                if stations:
                    overall = _match(r"Overall EBUS TBNA ROSE Diagnosis:[ \t]*([^\n]+)", ebus)
                    gauges = re.findall(r"\b(\d{2})-gauge Needle", ebus, re.I)
                    bundle.procedures.append(
                        _proc(
                            "ebus_tbna",
                            "nodes",
                            {
                                "stations": stations,
                                "needle_gauge": ", ".join(f"{g}G" for g in dict.fromkeys(gauges))
                                or None,
                                "elastography_used": any(s["echo_features"] for s in stations),
                                "rose_available": False
                                if overall and re.fullmatch(r"not performed", overall, re.I)
                                else None,
                                "overall_rose_diagnosis": overall,
                            },
                        )
                    )
                if survey:
                    bundle.findings_text = "\n".join(
                        filter(
                            None, [bundle.findings_text, "Linear EBUS survey. " + " ".join(survey)]
                        )
                    )
    return bundle
