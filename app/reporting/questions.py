"""Question specification and prompt generation for interactive reporter flows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.reporting.metadata import MissingFieldIssue
from proc_schemas.clinical.common import ProcedureBundle, ProcedureInput


class QuestionInputType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"
    MULTISELECT = "multiselect"
    TEXTAREA = "textarea"


class QuestionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pointer: str
    label: str
    input_type: QuestionInputType
    required: bool
    options: list[str | int] = Field(default_factory=list)
    help: str | None = None
    group: str
    priority: int = 100


@dataclass(frozen=True)
class _PromptSpec:
    field_pattern: str
    label: str
    input_type: QuestionInputType
    required: bool
    options: tuple[str | int, ...] = ()
    help: str | None = None
    priority: int = 30


_EBUS_ECHO_FEATURE_OPTIONS: tuple[str, ...] = (
    "Round",
    "Oval",
    "Distinct margins",
    "Indistinct margins",
    "Hypoechoic",
    "Heterogeneous",
    "Necrosis",
    "Calcification",
)


_PROCEDURE_GROUP_LABELS: dict[str, str] = {
    "ebus_tbna": "EBUS-TBNA",
    "bal": "BAL",
    "bal_variant": "BAL",
    "endobronchial_biopsy": "Endobronchial Biopsy",
    "transbronchial_needle_aspiration": "Transbronchial Needle Aspiration",
    "transbronchial_lung_biopsy": "Transbronchial Lung Biopsy",
    "transbronchial_biopsy": "Transbronchial Lung Biopsy",
    "transbronchial_cryobiopsy": "Transbronchial Cryobiopsy",
    "thoracentesis": "Thoracentesis",
    "thoracentesis_detailed": "Thoracentesis",
}


PROMPT_SPEC_REGISTRY: dict[str, tuple[_PromptSpec, ...]] = {
    "ebus_tbna": (
        _PromptSpec(
            field_pattern="stations[].station_name",
            label="Sampled station",
            input_type=QuestionInputType.STRING,
            required=True,
            help="Document the nodal station sampled (for example: 7, 4R, 11L).",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="needle_gauge",
            label="Needle gauge",
            input_type=QuestionInputType.ENUM,
            required=True,
            options=("19", "21", "22", "25", "Unknown"),
            help="Select the EBUS TBNA needle gauge.",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="stations[].passes",
            label="Number of passes",
            input_type=QuestionInputType.INTEGER,
            required=True,
            help="Number of needle passes at this station.",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="stations[].size_mm",
            label="Nodal size (mm)",
            input_type=QuestionInputType.INTEGER,
            required=True,
            help="Largest short-axis nodal size in millimeters.",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="stations[].echo_features",
            label="Nodal echo features",
            input_type=QuestionInputType.MULTISELECT,
            required=False,
            options=_EBUS_ECHO_FEATURE_OPTIONS,
            help="Select any documented sonographic features.",
            priority=30,
        ),
        _PromptSpec(
            field_pattern="rose_available",
            label="ROSE performed",
            input_type=QuestionInputType.BOOLEAN,
            required=False,
            help="Was rapid on-site cytology evaluation performed?",
            priority=30,
        ),
        _PromptSpec(
            field_pattern="stations[].rose_result",
            label="ROSE adequacy/result",
            input_type=QuestionInputType.ENUM,
            required=False,
            options=("Adequate", "Inadequate", "Unknown"),
            help="If ROSE was performed, document adequacy/result.",
            priority=30,
        ),
        _PromptSpec(
            field_pattern="overall_rose_diagnosis",
            label="ROSE impression",
            input_type=QuestionInputType.TEXTAREA,
            required=False,
            help="Optional overall ROSE impression or preliminary diagnosis.",
            priority=35,
        ),
    ),
    "bal": (
        _PromptSpec(
            field_pattern="lung_segment",
            label="Lung segment/lobe",
            input_type=QuestionInputType.STRING,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="instilled_volume_cc",
            label="Instilled volume (cc)",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="returned_volume_cc",
            label="Returned volume (cc)",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
    ),
    "bal_variant": (
        _PromptSpec(
            field_pattern="lung_segment",
            label="Lung segment/lobe",
            input_type=QuestionInputType.STRING,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="instilled_volume_cc",
            label="Instilled volume (cc)",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="returned_volume_cc",
            label="Returned volume (cc)",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
    ),
    "endobronchial_biopsy": (
        _PromptSpec(
            field_pattern="airway_segment",
            label="Biopsy site (airway segment)",
            input_type=QuestionInputType.STRING,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="samples_collected",
            label="Biopsy count",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
    ),
    "transbronchial_lung_biopsy": (
        _PromptSpec(
            field_pattern="lung_segment",
            label="Biopsy segment",
            input_type=QuestionInputType.STRING,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="samples_collected",
            label="Biopsy count",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
    ),
    "transbronchial_biopsy": (
        _PromptSpec(
            field_pattern="segment",
            label="Biopsy segment",
            input_type=QuestionInputType.STRING,
            required=False,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="number_of_biopsies",
            label="Biopsy count",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
    ),
    "transbronchial_needle_aspiration": (
        _PromptSpec(
            field_pattern="lung_segment",
            label="Target lung segment/lobe",
            input_type=QuestionInputType.STRING,
            required=True,
            help="Document the bronchopulmonary segment/lobe targeted (example: RUL, RB10, LLL).",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="samples_collected",
            label="Number of samples",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="tests",
            label="Tests sent",
            input_type=QuestionInputType.TEXTAREA,
            required=True,
            help="Enter one per line or comma-separated (example: Cytology, Microbiology).",
            priority=20,
        ),
    ),
    "transbronchial_cryobiopsy": (
        _PromptSpec(
            field_pattern="lung_segment",
            label="Target lung segment/lobe",
            input_type=QuestionInputType.STRING,
            required=True,
            help="Document the bronchopulmonary segment/lobe targeted (example: RUL, RB10, LLL).",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="num_samples",
            label="Number of samples",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="cryoprobe_size_mm",
            label="Cryoprobe size (mm)",
            input_type=QuestionInputType.NUMBER,
            required=True,
            options=("1.1", "1.7", "1.9", "2.4", "Unknown"),
            help="Select the cryoprobe size if documented.",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="blocker_type",
            label="Blocker type",
            input_type=QuestionInputType.ENUM,
            required=True,
            options=("Fogarty", "Arndt", "Balloon blocker", "Other", "Unknown"),
            help="If an endobronchial blocker was used for tamponade, document the type.",
            priority=20,
        ),
        _PromptSpec(
            field_pattern="tests",
            label="Tests sent",
            input_type=QuestionInputType.TEXTAREA,
            required=False,
            help="Optional tests requested (one per line or comma-separated).",
            priority=30,
        ),
    ),
    "thoracentesis": (
        _PromptSpec(
            field_pattern="side",
            label="Procedure side",
            input_type=QuestionInputType.ENUM,
            required=True,
            options=("Left", "Right", "Bilateral", "Unknown"),
            priority=20,
        ),
        _PromptSpec(
            field_pattern="volume_removed_ml",
            label="Volume removed (mL)",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="ultrasound_findings",
            label="Ultrasound findings",
            input_type=QuestionInputType.TEXTAREA,
            required=False,
            help="Optional ultrasound findings (echogenicity, loculations, septations).",
            priority=30,
        ),
    ),
    "thoracentesis_detailed": (
        _PromptSpec(
            field_pattern="side",
            label="Procedure side",
            input_type=QuestionInputType.ENUM,
            required=True,
            options=("Left", "Right", "Bilateral", "Unknown"),
            priority=20,
        ),
        _PromptSpec(
            field_pattern="volume_removed_ml",
            label="Volume removed (mL)",
            input_type=QuestionInputType.INTEGER,
            required=True,
            priority=20,
        ),
        _PromptSpec(
            field_pattern="pleural_guidance",
            label="Ultrasound guidance/findings",
            input_type=QuestionInputType.TEXTAREA,
            required=False,
            help="Optional ultrasound findings or guidance details.",
            priority=30,
        ),
    ),
}


_INDEX_RE = re.compile(r"\[\d+\]")
_TOKEN_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def questions_from_missing_issues(
    bundle: ProcedureBundle,
    issues: list[MissingFieldIssue],
) -> list[QuestionSpec]:
    return _build_questions(bundle=bundle, issues=issues, include_prompt_registry=False)


def build_questions(
    bundle: ProcedureBundle,
    issues: list[MissingFieldIssue],
) -> list[QuestionSpec]:
    return _build_questions(bundle=bundle, issues=issues, include_prompt_registry=True)


def _build_questions(
    bundle: ProcedureBundle,
    issues: list[MissingFieldIssue],
    *,
    include_prompt_registry: bool,
) -> list[QuestionSpec]:
    proc_id_to_index = {
        (proc.proc_id or f"{proc.proc_type}_{idx}"): idx for idx, proc in enumerate(bundle.procedures)
    }
    questions: list[QuestionSpec] = []
    seen_pointers: set[str] = set()

    for issue in issues:
        proc_index = _resolve_proc_index(bundle, proc_id_to_index, issue)
        if proc_index is None:
            continue
        proc = bundle.procedures[proc_index]
        proc_data = _coerce_proc_data(proc)
        normalized_path = _normalize_field_path(issue.field_path)
        prompt_spec = _find_prompt_spec(proc.proc_type, normalized_path)
        pointer = _field_path_to_pointer(proc_index, issue.field_path)
        if pointer in seen_pointers:
            continue
        seen_pointers.add(pointer)
        question = _question_from_issue(
            issue=issue,
            pointer=pointer,
            proc=proc,
            proc_data=proc_data,
            prompt_spec=prompt_spec,
        )
        questions.append(question)

    if include_prompt_registry:
        for proc_index, proc in enumerate(bundle.procedures):
            proc_data = _coerce_proc_data(proc)
            for prompt_spec in PROMPT_SPEC_REGISTRY.get(proc.proc_type, ()):
                if not _should_include_prompt_spec(prompt_spec, proc_data):
                    continue
                expanded_paths = _expand_field_pattern(proc_data, prompt_spec.field_pattern)
                if not expanded_paths and "[]" not in prompt_spec.field_pattern:
                    expanded_paths = [prompt_spec.field_pattern]
                for expanded_path in expanded_paths:
                    value = _get_field_value(proc_data, expanded_path)
                    if not _is_missing_value(value):
                        continue
                    pointer = _field_path_to_pointer(proc_index, expanded_path)
                    if pointer in seen_pointers:
                        continue
                    seen_pointers.add(pointer)
                    questions.append(
                        _question_from_prompt_spec(
                            proc=proc,
                            proc_data=proc_data,
                            pointer=pointer,
                            field_path=expanded_path,
                            prompt_spec=prompt_spec,
                        )
                    )

    questions.sort(key=lambda q: (q.priority, q.group, q.label))
    return questions


def _resolve_proc_index(
    bundle: ProcedureBundle,
    proc_id_to_index: dict[str, int],
    issue: MissingFieldIssue,
) -> int | None:
    if issue.proc_id in proc_id_to_index:
        return proc_id_to_index[issue.proc_id]
    for idx, proc in enumerate(bundle.procedures):
        if proc.proc_type == issue.proc_type:
            return idx
    return None


def _coerce_proc_data(proc: ProcedureInput) -> dict[str, Any]:
    data = proc.data
    if isinstance(data, BaseModel):
        return data.model_dump(exclude_none=False)
    if isinstance(data, dict):
        return data
    return {}


def _question_from_issue(
    *,
    issue: MissingFieldIssue,
    pointer: str,
    proc: ProcedureInput,
    proc_data: dict[str, Any],
    prompt_spec: _PromptSpec | None,
) -> QuestionSpec:
    default_label = _label_from_field_path(issue.field_path)
    default_priority = _priority_from_field_path(issue.field_path)
    input_type = prompt_spec.input_type if prompt_spec else QuestionInputType.STRING
    required = prompt_spec.required if prompt_spec else issue.severity != "recommended"
    options = list(prompt_spec.options) if prompt_spec else []
    help_text = prompt_spec.help if prompt_spec else issue.message
    priority = prompt_spec.priority if prompt_spec else default_priority
    group = _group_label(proc=proc, proc_data=proc_data, field_path=issue.field_path)
    return QuestionSpec(
        id=_question_id(proc=proc, pointer=pointer),
        pointer=pointer,
        label=prompt_spec.label if prompt_spec else default_label,
        input_type=input_type,
        required=required,
        options=options,
        help=help_text,
        group=group,
        priority=priority,
    )


def _question_from_prompt_spec(
    *,
    proc: ProcedureInput,
    proc_data: dict[str, Any],
    pointer: str,
    field_path: str,
    prompt_spec: _PromptSpec,
) -> QuestionSpec:
    return QuestionSpec(
        id=_question_id(proc=proc, pointer=pointer),
        pointer=pointer,
        label=prompt_spec.label,
        input_type=prompt_spec.input_type,
        required=prompt_spec.required,
        options=list(prompt_spec.options),
        help=prompt_spec.help,
        group=_group_label(proc=proc, proc_data=proc_data, field_path=field_path),
        priority=prompt_spec.priority,
    )


def _find_prompt_spec(proc_type: str, normalized_field_path: str) -> _PromptSpec | None:
    for prompt_spec in PROMPT_SPEC_REGISTRY.get(proc_type, ()):
        if prompt_spec.field_pattern == normalized_field_path:
            return prompt_spec
    return None


def _normalize_field_path(field_path: str) -> str:
    return _INDEX_RE.sub("[]", field_path)


def _expand_field_pattern(proc_data: dict[str, Any], field_pattern: str) -> list[str]:
    parts = field_pattern.split(".")
    active: list[tuple[str, Any]] = [("", proc_data)]
    for part in parts:
        is_list = part.endswith("[]")
        key = part[:-2] if is_list else part
        next_active: list[tuple[str, Any]] = []
        for current_path, current_value in active:
            if not isinstance(current_value, dict):
                continue
            child = current_value.get(key)
            if is_list:
                if not isinstance(child, list) or not child:
                    continue
                for idx, item in enumerate(child):
                    path = f"{current_path}.{key}[{idx}]" if current_path else f"{key}[{idx}]"
                    next_active.append((path, item))
            else:
                path = f"{current_path}.{key}" if current_path else key
                next_active.append((path, child))
        active = next_active
        if not active:
            return []
    return [path for path, _ in active]


def _field_path_to_pointer(proc_index: int, field_path: str) -> str:
    tokens = _field_path_tokens(field_path)
    pointer_tokens = ["procedures", str(proc_index), "data", *tokens]
    escaped = [_escape_pointer_token(token) for token in pointer_tokens]
    return "/" + "/".join(escaped)


def _field_path_tokens(field_path: str) -> list[str]:
    tokens: list[str] = []
    for name, idx in _TOKEN_RE.findall(field_path):
        if name:
            tokens.append(name)
        elif idx:
            tokens.append(idx)
    return tokens


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _question_id(proc: ProcedureInput, pointer: str) -> str:
    prefix = proc.proc_id or proc.proc_type
    normalized_pointer = pointer.strip("/").replace("/", "_")
    return f"{prefix}:{normalized_pointer}"


def _group_label(proc: ProcedureInput, proc_data: dict[str, Any], field_path: str) -> str:
    station_idx = _extract_station_index(field_path)
    if proc.proc_type == "ebus_tbna" and station_idx is not None:
        station_name = _station_name(proc_data, station_idx)
        return f"EBUS Station {station_name}" if station_name else f"EBUS Station {station_idx + 1}"
    return _PROCEDURE_GROUP_LABELS.get(proc.proc_type, proc.proc_type.replace("_", " ").title())


def _extract_station_index(field_path: str) -> int | None:
    match = re.match(r"^stations\[(\d+)\]", field_path)
    if not match:
        return None
    return int(match.group(1))


def _station_name(proc_data: dict[str, Any], station_idx: int) -> str | None:
    stations = proc_data.get("stations")
    if not isinstance(stations, list):
        return None
    if station_idx < 0 or station_idx >= len(stations):
        return None
    station = stations[station_idx]
    if not isinstance(station, dict):
        return None
    station_name = station.get("station_name")
    if station_name in (None, ""):
        return None
    return str(station_name)


def _label_from_field_path(field_path: str) -> str:
    tokens = _field_path_tokens(field_path)
    if not tokens:
        return "Missing field"
    leaf = tokens[-1]
    if leaf.isdigit() and len(tokens) >= 2:
        leaf = tokens[-2]
    return leaf.replace("_", " ").strip().title()


def _priority_from_field_path(field_path: str) -> int:
    lowered = field_path.lower()
    if any(term in lowered for term in ("complication", "bleed", "hemostasis", "oxygen", "anaphylaxis")):
        return 10
    if any(
        term in lowered
        for term in (
            "needle",
            "passes",
            "size",
            "volume",
            "samples",
            "biopsy",
            "segment",
            "station",
            "side",
        )
    ):
        return 20
    return 30


def _is_missing_value(value: Any) -> bool:
    return value in (None, "", [], {})


def _get_field_value(payload: Any, field_path: str) -> Any:
    current = payload
    for token in _field_path_tokens(field_path):
        if token.isdigit():
            idx = int(token)
            if not isinstance(current, list) or idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        if isinstance(current, dict):
            current = current.get(token)
        else:
            return None
    return current


def _should_include_prompt_spec(prompt_spec: _PromptSpec, proc_data: dict[str, Any]) -> bool:
    if prompt_spec.field_pattern == "stations[].rose_result":
        rose_available = proc_data.get("rose_available")
        return rose_available is not False
    return True


__all__ = [
    "QuestionInputType",
    "QuestionSpec",
    "PROMPT_SPEC_REGISTRY",
    "questions_from_missing_issues",
    "build_questions",
]
