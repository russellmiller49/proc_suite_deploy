from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from proc_schemas.clinical import ProcedureBundle, ProcedureInput

FLAG_VERSION = "reporter_seed_flag.v1"

BLOCKER_CODES = {
    "UNRESOLVED_PLACEHOLDER",
    "COMPLICATIONS_NONE_CONTRADICTION",
    "OMITTED_MAJOR_PROCEDURE_FAMILY",
    "UNSUPPORTED_INVASIVE_PROCEDURE_ADDED",
    "ANATOMY_SITE_DRIFT",
    "STENT_DILATION_DIMENSION_MIXUP",
    "SPECIMEN_PLAN_MISMATCH",
    "PLEURAL_INSTILLATION_MISCLASSIFIED_AS_TUBE_INSERTION",
    "DIAGNOSIS_SOURCE_MISMATCH",
    "EXPLICIT_CPT_SHORTHAND_MISMATCH",
}

_LOCATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?i)\b(?:left\s+main\s*stem|left\s+mainstem|LMSB|LMS)\b", "LMSB"),
    (r"(?i)\b(?:right\s+main\s*stem|right\s+mainstem|RMSB|RMS)\b", "RMSB"),
    (r"(?i)\bbronchus\s+intermedius\b|\bBI\b", "BI"),
    (r"(?i)\btrachea\b", "TRACHEA"),
    (r"(?i)\bLINGULA\b", "LINGULA"),
    (r"(?i)\bRUL\b", "RUL"),
    (r"(?i)\bRML\b", "RML"),
    (r"(?i)\bRLL\b", "RLL"),
    (r"(?i)\bLUL\b", "LUL"),
    (r"(?i)\bLLL\b", "LLL"),
    (r"(?i)\b([RL]B\d{1,2}(?:\+\d{1,2})?)\b", ""),
)

_TISSUE_PROC_TYPES = {
    "transbronchial_biopsy",
    "transbronchial_lung_biopsy",
    "transbronchial_needle_aspiration",
    "transbronchial_cryobiopsy",
    "endobronchial_biopsy",
    "bronchial_brushings",
    "ebus_tbna",
    "ebus_ifb",
    "ebus_19g_fnb",
    "transthoracic_needle_biopsy",
}

_MAJOR_FAMILY_SPECS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("rigid bronchoscopy", (r"(?i)\brigid\s+bronch",), ("rigid_bronchoscopy",)),
    (
        "EBUS",
        (r"(?i)\bEBUS\b",),
        ("ebus_tbna", "ebus_ifb", "ebus_19g_fnb"),
    ),
    ("BAL", (r"(?i)\bBAL\b|\bbronchoalveolar\s+lavage\b",), ("bal", "bal_variant")),
    (
        "airway stent intervention",
        (r"(?i)\bstent\b",),
        ("airway_stent_placement", "airway_stent_removal_revision", "stent_surveillance"),
    ),
    (
        "therapeutic injection",
        (r"(?i)\b(?:voriconazole|amikacin|medication)\b", r"(?i)\binjection\b"),
        ("therapeutic_injection",),
    ),
    (
        "pleural fibrinolysis",
        (r"(?i)\b(?:tpa|alteplase|dnase|dornase|fibrinolytic)\b",),
        ("chest_tube", "pigtail_catheter"),
    ),
)


def build_reporter_quality_flags(
    *,
    source_text: str | None,
    bundle: ProcedureBundle,
    markdown: str | None = None,
    prior_flags: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    flags = list(prior_flags or [])
    source = str(source_text or bundle.free_text_hint or "")
    rendered = str(markdown or "")

    flags.extend(_placeholder_flags(rendered))
    flags.extend(_complication_contradiction_flags(source, bundle, rendered))
    flags.extend(_omitted_major_family_flags(source, bundle))
    flags.extend(_unsupported_invasive_add_flags(source, bundle, rendered))
    flags.extend(_anatomy_drift_flags(source, bundle, rendered))
    flags.extend(_stent_dilation_mixup_flags(source, bundle, rendered))
    flags.extend(_specimen_plan_mismatch_flags(bundle, rendered))
    flags.extend(_pleural_instillation_misclassification_flags(source, bundle, rendered))
    flags.extend(_diagnosis_source_mismatch_flags(source, bundle, rendered))
    flags.extend(_multi_target_collapse_flags(source, bundle, rendered))
    flags.extend(_complications_include_disposition_flags(bundle, rendered))
    flags.extend(_explicit_cpt_shorthand_flags(source, bundle, rendered))

    deduped = _dedupe_flags(flags)
    needs_manual_review = any(_is_blocking_flag(flag) for flag in deduped)
    return deduped, needs_manual_review


def has_blocking_quality_flags(flags: list[dict[str, Any]] | None) -> bool:
    return any(_is_blocking_flag(flag) for flag in list(flags or []))


def _make_flag(
    code: str,
    message: str,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
    blocking: bool = True,
) -> dict[str, Any]:
    payload = dict(metadata or {})
    payload.setdefault("blocking", bool(blocking))
    return {
        "version": FLAG_VERSION,
        "code": code,
        "severity": "blocker" if blocking else "warning",
        "source": source,
        "message": message,
        "legacy_warning": None,
        "metadata": payload,
    }


def _dedupe_flags(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for flag in flags:
        code = str(flag.get("code") or "").strip()
        source = str(flag.get("source") or "").strip()
        message = str(flag.get("message") or "").strip()
        key = (code, source, message)
        if not code or key in seen:
            continue
        seen.add(key)
        deduped.append(flag)
    return deduped


def _is_blocking_flag(flag: dict[str, Any]) -> bool:
    severity = str(flag.get("severity") or "").strip().lower()
    if severity == "blocker":
        return True
    metadata = flag.get("metadata")
    if isinstance(metadata, dict) and metadata.get("blocking") is True:
        return True
    return str(flag.get("code") or "").strip().upper() in BLOCKER_CODES


def _proc_data(proc: ProcedureInput) -> dict[str, Any]:
    if isinstance(proc.data, BaseModel):
        return proc.data.model_dump(exclude_none=False)
    if isinstance(proc.data, dict):
        return dict(proc.data)
    return {}


def _bundle_proc_types(bundle: ProcedureBundle) -> set[str]:
    return {str(proc.proc_type) for proc in list(bundle.procedures or [])}


def _placeholder_flags(markdown: str) -> list[dict[str, Any]]:
    if not markdown:
        return []
    placeholders = sorted(set(re.findall(r"\[[^\]\n]{2,}\]", markdown)))
    if not placeholders:
        return []
    return [
        _make_flag(
            "UNRESOLVED_PLACEHOLDER",
            "Rendered preview still contains unresolved placeholder text.",
            source="quality_gate.render",
            metadata={"placeholders": placeholders},
        )
    ]


def _complication_contradiction_flags(
    source_text: str,
    bundle: ProcedureBundle,
    markdown: str,
) -> list[dict[str, Any]]:
    if not source_text:
        return []
    adverse_patterns = (
        r"(?i)\b(?:complication|tear|bleeding|oozing|migration|migrated|failed|failure|removed|required|converted|switch\s+from)\b",
    )
    if not any(re.search(pattern, source_text) for pattern in adverse_patterns):
        return []
    existing = str(bundle.complications_text or "").strip()
    rendered_none = bool(re.search(r"(?im)^\s*COMPLICATIONS\s+None\s*$", markdown or ""))
    bundle_none = existing.lower() == "none"
    if not rendered_none and not bundle_none:
        return []
    return [
        _make_flag(
            "COMPLICATIONS_NONE_CONTRADICTION",
            "Source text describes an adverse event, but the report records complications as none.",
            source="quality_gate.complications",
        )
    ]


def _omitted_major_family_flags(source_text: str, bundle: ProcedureBundle) -> list[dict[str, Any]]:
    if not source_text:
        return []
    proc_types = _bundle_proc_types(bundle)
    flags: list[dict[str, Any]] = []
    if not proc_types and re.search(r"(?i)\bbronch", source_text):
        flags.append(
            _make_flag(
                "OMITTED_MAJOR_PROCEDURE_FAMILY",
                "Source text describes bronchoscopy, but the bundle has no rendered procedures.",
                source="quality_gate.bundle",
                metadata={"family": "bronchoscopy"},
            )
        )
    for family, patterns, expected_proc_types in _MAJOR_FAMILY_SPECS:
        if not all(re.search(pattern, source_text) for pattern in patterns):
            continue
        if family == "EBUS":
            has_sampling_detail = bool(re.search(r"(?i)\b(?:station|tbna|sampled|biops(?:y|ies))\b", source_text))
            has_procedure_context = bool(re.search(r"(?im)^\s*procedures?\s*:\s*[^\n]*\bEBUS\b", source_text))
            if not has_sampling_detail and not has_procedure_context and re.search(
                r"(?i)\bcomplication\b[^\n.]{0,80}\bEBUS\b",
                source_text,
            ):
                continue
        if proc_types.intersection(expected_proc_types):
            continue
        blocking = True
        if family == "EBUS" and not re.search(r"(?i)\b(?:station|tbna|sampled|biops(?:y|ies))\b", source_text):
            blocking = False
        flags.append(
            _make_flag(
                "OMITTED_MAJOR_PROCEDURE_FAMILY",
                f"Source text documents {family}, but the bundle does not preserve that major procedure family.",
                source="quality_gate.bundle",
                metadata={"family": family, "expected_proc_types": list(expected_proc_types)},
                blocking=blocking,
            )
        )
    return flags


def _unsupported_invasive_add_flags(
    source_text: str,
    bundle: ProcedureBundle,
    markdown: str,
) -> list[dict[str, Any]]:
    if not source_text:
        return []
    fibrinolytic_only = _is_fibrinolytic_only_source(source_text)
    insertion_added = bool(
        re.search(r"(?i)\b(?:thoracostomy\s+tube|pigtail\s+catheter)\b[^.\n]{0,30}\binsert", markdown or "")
    )
    if not fibrinolytic_only or not insertion_added:
        return []
    return [
        _make_flag(
            "UNSUPPORTED_INVASIVE_PROCEDURE_ADDED",
            "Rendered output adds a pleural tube insertion even though the source only documents ultrasound/fibrinolytic management via an existing drain.",
            source="quality_gate.render",
        )
    ]


def _anatomy_drift_flags(source_text: str, bundle: ProcedureBundle, markdown: str) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for label, source_pattern, render_pattern in (
        ("stent", r"(?i)\bstent\b", r"(?i)\bstent\b"),
        ("BAL", r"(?i)\bBAL\b|\bbronchoalveolar\s+lavage\b", r"(?i)\bBAL\b|\bbronchoalveolar\s+lavage\b"),
        ("cryo", r"(?i)\bcryo(?:therapy|biopsy|)\b", r"(?i)\bcryo(?:therapy|biopsy|)\b"),
    ):
        source_locations = _extract_context_locations(source_text, source_pattern)
        render_locations = _extract_context_locations(markdown or "", render_pattern)
        if len(source_locations) != 1 or len(render_locations) != 1:
            continue
        if source_locations.issubset(render_locations):
            continue
        flags.append(
            _make_flag(
                "ANATOMY_SITE_DRIFT",
                f"Rendered {label} anatomy does not match the single source-site documented in the prompt.",
                source="quality_gate.render",
                metadata={
                    "family": label,
                    "source_locations": sorted(source_locations),
                    "render_locations": sorted(render_locations),
                },
            )
        )
    return flags


def _stent_dilation_mixup_flags(source_text: str, bundle: ProcedureBundle, markdown: str) -> list[dict[str, Any]]:
    if not source_text or not re.search(r"(?i)\bstent\b", source_text):
        return []
    source_stent_numbers: set[int] = set()
    for match in re.finditer(r"(?i)\b(\d{1,2})(?:\s*mm)?\s*[x×]\s*(\d{1,3})\s*mm\b", source_text):
        try:
            source_stent_numbers.add(int(match.group(1)))
            source_stent_numbers.add(int(match.group(2)))
        except Exception:
            continue
    if not source_stent_numbers:
        return []
    explicit_dilation = {
        int(item)
        for item in re.findall(
            r"(?i)\b(?:balloon\s+dilat(?:ion|ed|e)?|dilat(?:ion|ed|e)?\s+to|dilation\s+to)\b[^.\n]{0,24}?(\d{1,2})\s*mm\b",
            source_text,
        )
    }
    bundle_dilation_sizes: set[int] = set()
    for proc in list(bundle.procedures or []):
        if proc.proc_type not in {"rigid_bronchoscopy", "airway_dilation"}:
            continue
        data = _proc_data(proc)
        values = data.get("dilation_sizes_mm")
        if isinstance(values, list):
            for value in values:
                try:
                    bundle_dilation_sizes.add(int(value))
                except Exception:
                    continue
        post_value = data.get("post_dilation_diameter_mm")
        if post_value not in (None, "", [], {}):
            try:
                bundle_dilation_sizes.add(int(post_value))
            except Exception:
                pass
    suspicious = sorted(bundle_dilation_sizes.intersection(source_stent_numbers) - explicit_dilation)
    if not suspicious:
        return []
    return [
        _make_flag(
            "STENT_DILATION_DIMENSION_MIXUP",
            "A rendered dilation size appears to be copied from documented stent dimensions rather than explicit dilation data.",
            source="quality_gate.bundle",
            metadata={"suspicious_sizes_mm": suspicious},
        )
    ]


def _specimen_plan_mismatch_flags(bundle: ProcedureBundle, markdown: str) -> list[dict[str, Any]]:
    rendered_plan = str(markdown or bundle.impression_plan or "")
    if not re.search(r"(?i)\bpathology\b|\bcytology\b", rendered_plan):
        return []
    has_tissue_proc = any(proc.proc_type in _TISSUE_PROC_TYPES for proc in list(bundle.procedures or []))
    specimens_text = str(bundle.specimens_text or "")
    if has_tissue_proc or re.search(r"(?i)\bbiops|tbna|cytology|pathology\b", specimens_text):
        return []
    return [
        _make_flag(
            "SPECIMEN_PLAN_MISMATCH",
            "The plan tells the reader to await pathology/cytology even though the bundle does not preserve tissue/cytology-producing specimens.",
            source="quality_gate.render",
        )
    ]


def _pleural_instillation_misclassification_flags(
    source_text: str,
    bundle: ProcedureBundle,
    markdown: str,
) -> list[dict[str, Any]]:
    if not _is_fibrinolytic_only_source(source_text):
        return []
    insertion_language = bool(
        re.search(
            r"(?i)\b(?:thoracostomy\s+tube|pigtail\s+catheter)\b[^.\n]{0,40}\b(?:inserted|placed)\b",
            markdown or "",
        )
    )
    if not insertion_language and not _bundle_proc_types(bundle).intersection({"chest_tube", "pigtail_catheter"}):
        return []
    if not insertion_language:
        return []
    return [
        _make_flag(
            "PLEURAL_INSTILLATION_MISCLASSIFIED_AS_TUBE_INSERTION",
            "The report rewrites pleural fibrinolytic instillation through an existing drain as a new tube insertion.",
            source="quality_gate.render",
        )
    ]


def _diagnosis_source_mismatch_flags(
    source_text: str,
    bundle: ProcedureBundle,
    markdown: str,
) -> list[dict[str, Any]]:
    if not source_text:
        return []
    diagnosis_text = "\n".join(
        [
            str(bundle.preop_diagnosis_text or ""),
            str(bundle.postop_diagnosis_text or ""),
            str(markdown or ""),
        ]
    )
    if not re.search(r"(?i)\bpulmonary\s+nodule\b", diagnosis_text):
        return []
    if re.search(r"(?i)\b(?:pulmonary\s+nodule|lung\s+nodule|lesion)\b", source_text):
        return []
    if not re.search(r"(?i)\b(?:thyroid\s+mass|stenosis\s+length|prolonged\s+intubation|tracheostomy)\b", source_text):
        return []
    return [
        _make_flag(
            "DIAGNOSIS_SOURCE_MISMATCH",
            "Rendered diagnosis appears to introduce a pulmonary nodule that is not supported by the source prompt.",
            source="quality_gate.render",
        )
    ]


def _multi_target_collapse_flags(
    source_text: str,
    bundle: ProcedureBundle,
    markdown: str,
) -> list[dict[str, Any]]:
    if not source_text:
        return []
    targets = [
        f"{match.group(1).upper()} ({str(match.group(2) or '').strip()})".strip()
        for match in re.finditer(r"(?i)\b(RUL|RML|RLL|LUL|LLL)\b(?:\s*\(([^)]+)\))?", source_text)
        if match.group(2)
    ]
    unique_targets = []
    seen: set[str] = set()
    for target in targets:
        key = target.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append(target)
    if len(unique_targets) < 2:
        return []
    nav_count = sum(1 for proc in list(bundle.procedures or []) if str(proc.proc_type) == "robotic_navigation")
    if nav_count >= len(unique_targets):
        return []
    return [
        _make_flag(
            "MULTI_TARGET_COLLAPSE",
            "Source text documents multiple explicit bronchoscopic targets, but the bundle preserves fewer targets than were dictated.",
            source="quality_gate.bundle",
            metadata={"targets": unique_targets},
            blocking=False,
        )
    ]


def _complications_include_disposition_flags(bundle: ProcedureBundle, markdown: str) -> list[dict[str, Any]]:
    payload = "\n".join([str(bundle.complications_text or ""), str(markdown or "")])
    if not re.search(r"(?i)\bdisposition\s*:", payload):
        return []
    return [
        _make_flag(
            "COMPLICATIONS_INCLUDE_DISPOSITION",
            "Disposition text appears inside the complications section and should be separated into the plan/disposition.",
            source="quality_gate.render",
            blocking=False,
        )
    ]


def _explicit_cpt_shorthand_flags(
    source_text: str,
    bundle: ProcedureBundle,
    markdown: str,
) -> list[dict[str, Any]]:
    if not source_text:
        return []
    explicit_cpts = {match.group(1) for match in re.finditer(r"(?<!\d)\b(31640|31641|31645)\b(?!\d)", source_text)}
    if not explicit_cpts:
        return []
    proc_types = _bundle_proc_types(bundle)
    rendered_lower = str(markdown or "").lower()
    missing: list[str] = []
    if "31645" in explicit_cpts and not (
        "therapeutic_aspiration" in proc_types or "therapeutic aspiration" in rendered_lower
    ):
        missing.append("31645 therapeutic aspiration")
    if {"31640", "31641"} & explicit_cpts and not (
        {"endobronchial_tumor_destruction", "microdebrider_debridement", "airway_dilation", "rigid_bronchoscopy"}
        & proc_types
        or "debulking" in rendered_lower
        or "stenosis relief" in rendered_lower
        or "endobronchial tumor destruction" in rendered_lower
    ):
        missing.append("31640/31641 therapeutic airway intervention")
    if not missing:
        return []
    return [
        _make_flag(
            "EXPLICIT_CPT_SHORTHAND_MISMATCH",
            "Source text includes explicit therapeutic bronchoscopy CPT shorthand, but the rendered bundle does not preserve the corresponding therapeutic intervention details.",
            source="quality_gate.bundle",
            metadata={"missing": missing, "codes": sorted(explicit_cpts)},
        )
    ]


def _extract_context_locations(text: str, keyword_pattern: str) -> set[str]:
    if not text:
        return set()
    locations: set[str] = set()
    for line in re.split(r"[\n\.]+", text):
        if not re.search(keyword_pattern, line):
            continue
        locations.update(_extract_locations(line))
    return locations


def _extract_locations(text: str) -> set[str]:
    locations: set[str] = set()
    for pattern, normalized in _LOCATION_PATTERNS:
        for match in re.finditer(pattern, text):
            if normalized:
                locations.add(normalized)
            else:
                locations.add(str(match.group(1) or "").upper().replace(" ", ""))
    return {item for item in locations if item}


def _is_fibrinolytic_only_source(source_text: str) -> bool:
    if not source_text:
        return False
    if not re.search(r"(?i)\b(?:tpa|alteplase|dnase|dornase|fibrinolytic|32561|32562)\b", source_text):
        return False
    insertion_language = re.search(
        r"(?i)\b(?:insert(?:ed|ion)|place(?:d|ment)|new\s+(?:tube|catheter)|seldinger|thoracostomy|pigtail)\b",
        source_text,
    )
    existing_drain_language = re.search(
        r"(?i)\b(?:existing|already\s+had|right-sided\s+tube|left-sided\s+tube|through\s+the\s+tube|via\s+(?:the\s+)?(?:tube|catheter)|subsequent\s+fibrinolytic|dose\s+number|dose\s*#)\b",
        source_text,
    )
    return bool(not insertion_language or existing_drain_language)


__all__ = [
    "BLOCKER_CODES",
    "build_reporter_quality_flags",
    "has_blocking_quality_flags",
]
