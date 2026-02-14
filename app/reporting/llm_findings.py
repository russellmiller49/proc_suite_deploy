"""Reporter Structured-First (v2) seeding via LLM findings.

This module implements a safer alternative to "prompt -> full ProcedureBundle JSON":

Scrubbed + menu-masked prompt text
  -> GPT emits atomic, evidence-cited findings (JSON)
  -> findings are converted to synthetic NER entities
  -> existing deterministic NERToRegistryMapper flips canonical registry flags
  -> ClinicalGuardrails suppress common false positives
  -> deterministic registry->CPT derivation
  -> build_procedure_bundle_from_extraction feeds existing reporter templates

Design goals:
- Evidence quotes must be verbatim substrings of the exact masked input text.
- procedure_key must be allowlisted (registry NER procedure universe).
- Prefer omission over hallucination; drop unsupported/unevidenced findings.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.common.llm import OpenAILLM, _resolve_openai_model
from app.ner.inference import NEREntity, NERExtractionResult
from app.ner.entity_types import normalize_lobe, normalize_station
from app.registry.ner_mapping.entity_to_registry import NERToRegistryMapper
from app.registry.ner_mapping.procedure_extractor import PROCEDURE_MAPPINGS
from app.registry.schema import RegistryRecord

from app.extraction.postprocessing.clinical_guardrails import ClinicalGuardrails
from app.coder.domain_rules.registry_to_cpt.engine import apply as derive_registry_to_cpt

from ml.lib.reporter_prompt_masking import mask_prompt_cpt_noise

_CPT_CODE_RE = re.compile(r"\b\d{5}\b")


class FindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    procedure_key: str = Field(..., description="Procedure key allowlisted by the registry NER mapper.")
    finding_text: str = Field(..., description="Short normalized finding. No CPT codes.")
    evidence_quote: str = Field(..., description="Verbatim substring copied from the input prompt text.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class ReporterFindingsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["reporter_findings_v1"] = "reporter_findings_v1"
    findings: list[FindingV1] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


ALLOWED_PROCEDURE_KEYS: set[str] = set(PROCEDURE_MAPPINGS.keys()) | {"peripheral_tbna"}


class ReporterFindingsError(RuntimeError):
    pass


class ReporterFindingsUnavailable(ReporterFindingsError):
    pass


class ReporterFindingsParseError(ReporterFindingsError):
    pass


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _strip_markdown_code_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3].strip()
    return cleaned.strip()


def _keyword_hit(text_lower: str, needle: str) -> bool:
    needle_lower = (needle or "").lower()
    if not needle_lower:
        return False
    if " " in needle_lower or len(needle_lower) >= 5:
        return needle_lower in text_lower
    return re.search(rf"\b{re.escape(needle_lower)}\b", text_lower) is not None


def _procedure_keywords(proc_key: str) -> list[str]:
    mapping = PROCEDURE_MAPPINGS.get(proc_key)
    if not mapping:
        if proc_key == "peripheral_tbna":
            return ["tbna", "transbronchial needle aspiration"]
        return []
    keywords, _field_path = mapping
    return sorted({str(k).strip() for k in keywords if str(k).strip()}, key=len, reverse=True)


def _contains_any_keyword(text: str, proc_key: str) -> bool:
    text_lower = (text or "").lower()
    for keyword in _procedure_keywords(proc_key):
        if _keyword_hit(text_lower, keyword):
            return True
    return False


def _build_findings_prompt(masked_prompt_text: str) -> str:
    keys = ", ".join(sorted(ALLOWED_PROCEDURE_KEYS))
    schema_hint = {
        "version": "reporter_findings_v1",
        "findings": [
            {
                "procedure_key": "bal",
                "finding_text": "BAL performed in RUL",
                "evidence_quote": "Bilateral BAL performed.",
                "confidence": 0.9,
            }
        ],
        "notes": [],
    }
    return (
        "You are a clinical information extraction engine.\n"
        "Task: Read the clinical prompt text and output atomic performed-procedure findings.\n\n"
        "Output MUST be exactly one JSON object matching this shape:\n"
        f"{json.dumps(schema_hint, indent=2)}\n\n"
        "Rules:\n"
        f"- procedure_key MUST be one of: {keys}\n"
        "- finding_text MUST be short, normalized, and MUST NOT include CPT codes.\n"
        "- finding_text MUST include a canonical procedure keyword/abbreviation for the procedure_key.\n"
        "  (Examples: BAL, EBUS, TBNA, TBBx, Cryotherapy, APC, Airway dilation, Stent, Chest ultrasound, Balloon occlusion.)\n"
        "- evidence_quote MUST be a verbatim substring copied from the prompt text below.\n"
        "- evidence_quote MUST be >= 10 characters and include enough local context to uniquely support the finding.\n"
        "- evidence_quote does NOT need to contain the canonical keyword if the prompt uses shorthand.\n"
        "  If the prompt says 'cryo', write finding_text as 'Cryotherapy ...' and quote the exact 'cryo ...' evidence.\n"
        "  If the prompt says 'balloon' for stenosis, write finding_text as 'Airway dilation ...' and quote the balloon evidence.\n"
        "- Only include procedures that are explicitly documented as performed.\n"
        "- If something is planned/considered/denied/inspection-only, OMIT it.\n"
        "- If uncertain, OMIT it.\n\n"
        "*** CRITICAL CLINICAL GUARDRAILS (ANTI-HALLUCINATION) ***\n"
        "1. TOOLS DO NOT EQUAL INTENT: The mere mention of a tool (cryoprobe, snare, forceps) does NOT mean a therapeutic intervention was performed.\n"
        "2. ACTION-ON-TISSUE REQUIRED: DO NOT output tags for ablation, debulking, or therapeutic aspiration unless there is explicit 'action-on-tissue' language (e.g., 'tissue was destroyed', 'secretions were aspirated' to clear obstruction).\n"
        "3. INSPECTION IS NOT INTERVENTION: Visualizing a stent or patent airway is NOT a stent placement or mechanical dilation.\n\n"
        "PROMPT TEXT (use evidence_quote from here):\n"
        f"{masked_prompt_text.strip()}\n"
    )


def _resolve_openai_llm() -> OpenAILLM:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider != "openai_compat":
        raise ReporterFindingsUnavailable(f"LLM_PROVIDER must be openai_compat (got {provider!r})")

    if _truthy_env("OPENAI_OFFLINE") or not os.getenv("OPENAI_API_KEY"):
        raise ReporterFindingsUnavailable("OpenAI unavailable (OPENAI_OFFLINE=1 or OPENAI_API_KEY not set)")

    model = _resolve_openai_model("structurer") or "gpt-5-mini"
    return OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model=model, task="structurer")


def extract_reporter_findings_v1(masked_prompt_text: str, *, llm: OpenAILLM | None = None) -> ReporterFindingsV1:
    llm = llm or _resolve_openai_llm()
    prompt = _build_findings_prompt(masked_prompt_text)
    raw = llm.generate(prompt, task="structurer")
    cleaned = _strip_markdown_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except Exception as exc:  # noqa: BLE001
        raise ReporterFindingsParseError(f"Failed parsing LLM JSON: {type(exc).__name__}") from exc
    try:
        return ReporterFindingsV1.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ReporterFindingsParseError(f"LLM response did not match ReporterFindingsV1: {type(exc).__name__}") from exc


def validate_findings_against_text(
    findings: ReporterFindingsV1,
    *,
    masked_prompt_text: str,
    min_evidence_len: int = 10,
) -> tuple[list[FindingV1], list[str]]:
    accepted: list[FindingV1] = []
    warnings: list[str] = []

    for idx, item in enumerate(findings.findings or []):
        proc_key = str(item.procedure_key or "").strip()
        if proc_key not in ALLOWED_PROCEDURE_KEYS:
            warnings.append(f"LLM_FINDINGS_DROPPED: invalid_procedure_key index={idx} key={proc_key!r}")
            continue

        finding_text = str(item.finding_text or "").strip()
        if _CPT_CODE_RE.search(finding_text):
            warnings.append(f"LLM_FINDINGS_DROPPED: contains_cpt_code index={idx} key={proc_key!r}")
            continue

        evidence = str(item.evidence_quote or "").strip()
        if len(evidence) < int(min_evidence_len):
            warnings.append(f"LLM_FINDINGS_DROPPED: evidence_too_short index={idx} key={proc_key!r}")
            continue

        if _find_evidence_span(masked_prompt_text or "", evidence) is None:
            warnings.append(f"LLM_FINDINGS_DROPPED: missing_evidence_quote index={idx} key={proc_key!r}")
            continue

        evidence_lower = evidence.lower()
        if proc_key == "peripheral_ablation" and not any(
            token in evidence_lower for token in ("ablat", "destroy", "freez")
        ):
            warnings.append(f"LLM_FINDINGS_DROPPED: missing_action_intent index={idx} key={proc_key!r}")
            continue
        if proc_key == "therapeutic_aspiration" and not any(
            token in evidence_lower for token in ("plug", "thick", "obstruct", "clear")
        ):
            warnings.append(f"LLM_FINDINGS_DROPPED: missing_action_intent index={idx} key={proc_key!r}")
            continue

        keyword_text = f"{finding_text}\n{evidence}".strip()
        flexible_keys = {"pharmacological_instillation", "complication", "other_intervention"}
        if proc_key not in flexible_keys and not _contains_any_keyword(keyword_text, proc_key):
            warnings.append(f"LLM_FINDINGS_DROPPED: keyword_missing index={idx} key={proc_key!r}")
            continue

        accepted.append(item)

    return accepted, warnings


def _find_evidence_span(text: str, evidence_quote: str) -> tuple[int, int] | None:
    raw = text or ""
    needle = (evidence_quote or "").strip()
    if not raw or not needle:
        return None

    direct = raw.find(needle)
    if direct >= 0:
        return direct, direct + len(needle)

    parts = [p for p in re.split(r"\s+", needle) if p]
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(p) for p in parts)
    match = re.search(pattern, raw, flags=re.MULTILINE)
    if not match:
        return None
    return match.start(), match.end()


_STATION_WITH_PREFIX_RE = re.compile(r"(?i)\b(?:station|level)\s*(?P<num>5|6|7|8|9)\b")
_STATION_TOKEN_RE = re.compile(r"\b(?P<tok>2[RL]|4[RL]|10[RL]|11[RL](?:s|i|si)?)\b", re.IGNORECASE)
_LOBAR_TOKEN_RE = re.compile(r"\b(?P<tok>RUL|RML|RLL|LUL|LLL|LINGULA)\b", re.IGNORECASE)
_LOBAR_LONG_RE = re.compile(
    r"(?i)\b(?P<side>right|left)\s+(?P<lobe>upper|middle|lower)\s+lobe\b"
)

_PROC_ACTION_CUE_OVERRIDES: dict[str, str] = {
    # Avoid accidental endobronchial_biopsy triggers from "transbronchial biopsy" substrings.
    "transbronchial_biopsy": "tbbx",
    "endobronchial_biopsy": "ebbx",
}

_PROC_ACTION_CANONICAL_CUES: dict[str, str] = {
    "linear_ebus": "ebus",
    "radial_ebus": "radial ebus",
    "navigational_bronchoscopy": "navigation",
    "transbronchial_biopsy": "tbbx",
    "transbronchial_cryobiopsy": "cryobiopsy",
    "endobronchial_biopsy": "ebbx",
    "bal": "bal",
    "brushings": "brushings",
    "tbna_conventional": "tbna",
    "therapeutic_aspiration": "therapeutic aspiration",
    "airway_dilation": "airway dilation",
    "airway_stent": "stent",
    "balloon_occlusion": "balloon occlusion",
    "thermal_ablation": "apc",
    "cryotherapy": "cryotherapy",
    "tumor_debulking": "debulking",
    "blvr": "blvr",
    "thoracentesis": "thoracentesis",
    "chest_tube": "chest tube",
    "ipc": "ipc",
    "medical_thoracoscopy": "medical thoracoscopy",
    "pleurodesis": "pleurodesis",
    "chest_ultrasound": "chest ultrasound",
    "percutaneous_tracheostomy": "tracheostomy",
    "rigid_bronchoscopy": "rigid bronchoscopy",
    "foreign_body_removal": "foreign body removal",
}


def _entities_for_helper_spans(
    *,
    label: str,
    evidence_quote: str,
    evidence_abs_start: int,
    matches: list[tuple[int, int, str]],
    confidence: float,
) -> list[NEREntity]:
    out: list[NEREntity] = []
    for rel_start, rel_end, text in matches:
        abs_start = evidence_abs_start + rel_start
        abs_end = evidence_abs_start + rel_end
        out.append(
            NEREntity(
                text=text,
                label=label,
                start_char=abs_start,
                end_char=abs_end,
                confidence=confidence,
                evidence_quote=evidence_quote,
            )
        )
    return out


def _extract_station_spans(evidence_quote: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    for match in _STATION_TOKEN_RE.finditer(evidence_quote or ""):
        token = match.group("tok")
        if token:
            out.append((match.start("tok"), match.end("tok"), token))

    for match in _STATION_WITH_PREFIX_RE.finditer(evidence_quote or ""):
        token = match.group("num")
        if token:
            out.append((match.start("num"), match.end("num"), token))

    return out


def _extract_lobe_spans(evidence_quote: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    for match in _LOBAR_TOKEN_RE.finditer(evidence_quote or ""):
        token = match.group("tok")
        if token:
            norm = token.upper()
            out.append((match.start("tok"), match.end("tok"), norm))

    for match in _LOBAR_LONG_RE.finditer(evidence_quote or ""):
        side = (match.group("side") or "").strip().lower()
        lobe = (match.group("lobe") or "").strip().lower()
        code = None
        if side == "right" and lobe == "upper":
            code = "RUL"
        elif side == "right" and lobe == "middle":
            code = "RML"
        elif side == "right" and lobe == "lower":
            code = "RLL"
        elif side == "left" and lobe == "upper":
            code = "LUL"
        elif side == "left" and lobe == "lower":
            code = "LLL"
        if code:
            out.append((match.start(), match.end(), code))

    return out


def build_synthetic_ner_result(
    *,
    masked_prompt_text: str,
    accepted_findings: list[FindingV1],
) -> NERExtractionResult:
    entities: list[NEREntity] = []

    for item in accepted_findings or []:
        proc_key = str(item.procedure_key or "").strip()
        evidence = str(item.evidence_quote or "").strip()
        if not evidence:
            continue
        evidence_span = _find_evidence_span(masked_prompt_text or "", evidence)
        if not evidence_span:
            continue
        abs_start, abs_end = evidence_span

        confidence = float(item.confidence) if item.confidence is not None else 1.0
        finding_text = str(item.finding_text or "").strip()
        proc_action_text = _PROC_ACTION_CANONICAL_CUES.get(proc_key) or _PROC_ACTION_CUE_OVERRIDES.get(proc_key)
        if not proc_action_text:
            if finding_text and _contains_any_keyword(finding_text, proc_key):
                proc_action_text = finding_text
            elif evidence and _contains_any_keyword(evidence, proc_key):
                proc_action_text = evidence
            else:
                proc_action_text = finding_text or evidence

        entities.append(
            NEREntity(
                text=proc_action_text,
                label="PROC_ACTION",
                start_char=abs_start,
                end_char=abs_end,
                confidence=confidence,
                evidence_quote=evidence,
            )
        )

        if proc_key in {"linear_ebus", "tbna_conventional", "ebus_tbna"}:
            station_spans = _extract_station_spans(evidence)
            if station_spans:
                entities.extend(
                    _entities_for_helper_spans(
                        label="ANAT_LN_STATION",
                        evidence_quote=evidence,
                        evidence_abs_start=abs_start,
                        matches=station_spans,
                        confidence=confidence,
                    )
                )

        lobar_procedures = {
            "bal",
            "brushings",
            "transbronchial_biopsy",
            "transbronchial_cryobiopsy",
            "peripheral_tbna",
            "radial_ebus",
            "navigational_bronchoscopy",
            "airway_stent",
            "airway_dilation",
            "cryotherapy",
            "thermal_ablation",
            "mechanical_debulking",
            "tumor_debulking",
        }

        if proc_key in lobar_procedures:
            lobe_spans = _extract_lobe_spans(evidence)
            if lobe_spans:
                entities.extend(
                    _entities_for_helper_spans(
                        label="ANAT_LUNG_LOC",
                        evidence_quote=evidence,
                        evidence_abs_start=abs_start,
                        matches=lobe_spans,
                        confidence=confidence,
                    )
                )

    entities_by_type: dict[str, list[NEREntity]] = {}
    for entity in entities:
        entities_by_type.setdefault(entity.label, []).append(entity)

    return NERExtractionResult(
        entities=entities,
        entities_by_type=entities_by_type,
        raw_text=masked_prompt_text or "",
    )


def _set_nested_field(d: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in str(path or "").split(".") if p]
    if not parts:
        return
    cur: dict[str, Any] = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _performed_field_path_for(proc_key: str) -> str:
    mapping = PROCEDURE_MAPPINGS.get(proc_key)
    if mapping:
        _keywords, field_path = mapping
        return str(field_path)
    if proc_key == "peripheral_tbna":
        return "procedures_performed.peripheral_tbna.performed"
    return f"procedures_performed.{proc_key}.performed"


def _object_path_for(proc_key: str) -> str:
    performed_path = _performed_field_path_for(proc_key)
    if performed_path.endswith(".performed"):
        return performed_path[: -len(".performed")]
    return performed_path


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = str(item or "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


_SEGMENT_TOKEN_RE = re.compile(r"\b(?P<tok>[RL]B(?:1[01]|[1-9]))\b", re.IGNORECASE)
_VALVE_SIZE_RE = re.compile(r"(?i)\bsize\s*(?P<num>\d{1,2})\b")
_FRENCH_RE = re.compile(r"(?i)\b(?P<num>\d+(?:\.\d+)?)\s*(?:fr|french)\b")


def _extract_lobes(text: str) -> list[str]:
    lobes: list[str] = []
    for _start, _end, tok in _extract_lobe_spans(text or ""):
        norm = normalize_lobe(tok)
        if norm:
            lobes.append(norm)
    return _dedupe_keep_order(lobes)


def _extract_stations(text: str) -> list[str]:
    stations: list[str] = []
    for _start, _end, tok in _extract_station_spans(text or ""):
        norm = normalize_station(tok)
        if norm:
            stations.append(norm)
    return _dedupe_keep_order(stations)


def _extract_segments(text: str) -> list[str]:
    out: list[str] = []
    for match in _SEGMENT_TOKEN_RE.finditer(text or ""):
        tok = match.group("tok")
        if tok:
            out.append(tok.upper())
    return _dedupe_keep_order(out)


def _extract_stent_location(text: str) -> str | None:
    lower = (text or "").lower()
    if re.search(r"\btrachea\b", lower):
        return "Trachea"
    if re.search(r"\b(left|l)\s*mainstem\b|\blms\b", lower):
        return "Left mainstem"
    if re.search(r"\b(right|r)\s*mainstem\b|\brms\b", lower):
        return "Right mainstem"
    if re.search(r"\bbronchus\s+intermedius\b", lower):
        return "Bronchus intermedius"
    if "carina" in lower or "y-stent" in lower or "y stent" in lower:
        return "Carina (Y)"
    return None


def _apply_findings_backfill(
    record: RegistryRecord,
    *,
    accepted_findings: list[FindingV1],
    masked_prompt_text: str,
) -> tuple[RegistryRecord, list[str]]:
    payload = record.model_dump()
    warnings: list[str] = []

    procedures_payload = payload.get("procedures_performed")
    if not isinstance(procedures_payload, dict):
        procedures_payload = {}
        payload["procedures_performed"] = procedures_payload

    pleural_payload = payload.get("pleural_procedures")
    if not isinstance(pleural_payload, dict):
        pleural_payload = {}
        payload["pleural_procedures"] = pleural_payload

    accepted_keys: set[str] = set()
    for item in accepted_findings or []:
        proc_key = str(item.procedure_key or "").strip()
        if not proc_key:
            continue
        accepted_keys.add(proc_key)

        performed_path = _performed_field_path_for(proc_key)
        _set_nested_field(payload, performed_path, True)

    # Prevent common overlaps that inflate CPTs.
    if "linear_ebus" in accepted_keys and "tbna_conventional" not in accepted_keys:
        _set_nested_field(payload, "procedures_performed.tbna_conventional.performed", False)
    if "peripheral_tbna" in accepted_keys and "tbna_conventional" not in accepted_keys:
        _set_nested_field(payload, "procedures_performed.tbna_conventional.performed", False)

    blvr_segments: list[str] = []
    blvr_sizes: list[str] = []
    blvr_type: str | None = None
    blvr_action: str | None = None
    blvr_lobes: list[str] = []
    blvr_finding_count = 0

    for item in accepted_findings or []:
        proc_key = str(item.procedure_key or "").strip()
        evidence = str(item.evidence_quote or "").strip()
        if not proc_key or not evidence:
            continue

        evidence_span = _find_evidence_span(masked_prompt_text or "", evidence)
        if not evidence_span:
            continue
        abs_start, abs_end = evidence_span

        line_start = (masked_prompt_text or "").rfind("\n", 0, abs_start)
        line_start = 0 if line_start < 0 else line_start + 1
        line_end = (masked_prompt_text or "").find("\n", abs_end)
        line_end = len(masked_prompt_text or "") if line_end < 0 else line_end
        line_text = (masked_prompt_text or "")[line_start:line_end]
        context_text = f"{evidence}\n{line_text}".strip()

        lobes = _extract_lobes(context_text)
        stations = _extract_stations(context_text)

        if proc_key == "bal" and lobes:
            existing = procedures_payload.get("bal") if isinstance(procedures_payload.get("bal"), dict) else None
            if not existing or not existing.get("location"):
                _set_nested_field(payload, "procedures_performed.bal.location", ", ".join(lobes))

        if proc_key == "brushings" and lobes:
            existing = procedures_payload.get("brushings") if isinstance(procedures_payload.get("brushings"), dict) else None
            current = list((existing or {}).get("locations") or [])
            merged = _dedupe_keep_order(current + lobes)
            _set_nested_field(payload, "procedures_performed.brushings.locations", merged)

        if proc_key == "transbronchial_biopsy" and lobes:
            existing = procedures_payload.get("transbronchial_biopsy") if isinstance(
                procedures_payload.get("transbronchial_biopsy"), dict
            ) else None
            current = list((existing or {}).get("locations") or [])
            merged = _dedupe_keep_order(current + lobes)
            _set_nested_field(payload, "procedures_performed.transbronchial_biopsy.locations", merged)

        if proc_key == "transbronchial_cryobiopsy" and lobes:
            existing = procedures_payload.get("transbronchial_cryobiopsy") if isinstance(
                procedures_payload.get("transbronchial_cryobiopsy"), dict
            ) else None
            current = list((existing or {}).get("locations_biopsied") or [])
            merged = _dedupe_keep_order(current + lobes)
            _set_nested_field(payload, "procedures_performed.transbronchial_cryobiopsy.locations_biopsied", merged)

        if proc_key == "tbna_conventional" and stations:
            existing = procedures_payload.get("tbna_conventional") if isinstance(
                procedures_payload.get("tbna_conventional"), dict
            ) else None
            current = list((existing or {}).get("stations_sampled") or [])
            merged = _dedupe_keep_order(current + stations)
            _set_nested_field(payload, "procedures_performed.tbna_conventional.stations_sampled", merged)
            # Guardrails may infer linear EBUS from station sampling language. Keep linear_ebus
            # station arrays populated so CPT derivation can run when that inference happens.
            existing_linear = procedures_payload.get("linear_ebus") if isinstance(
                procedures_payload.get("linear_ebus"), dict
            ) else None
            current_linear = list((existing_linear or {}).get("stations_sampled") or [])
            merged_linear = _dedupe_keep_order(current_linear + stations)
            _set_nested_field(payload, "procedures_performed.linear_ebus.stations_sampled", merged_linear)

        if proc_key == "airway_stent":
            loc = _extract_stent_location(context_text)
            if loc:
                existing = procedures_payload.get("airway_stent") if isinstance(procedures_payload.get("airway_stent"), dict) else None
                if not existing or not existing.get("location"):
                    _set_nested_field(payload, "procedures_performed.airway_stent.location", loc)

        if proc_key in {"airway_dilation", "cryotherapy", "thermal_ablation", "mechanical_debulking", "therapeutic_aspiration"}:
            loc = _extract_stent_location(context_text)
            if not loc and lobes:
                loc = ", ".join(lobes)
            if loc:
                obj_path = _object_path_for(proc_key)
                existing = procedures_payload.get(proc_key) if isinstance(procedures_payload.get(proc_key), dict) else None
                if not existing or not existing.get("location"):
                    _set_nested_field(payload, f"{obj_path}.location", loc)

        if proc_key == "tumor_debulking":
            loc = _extract_stent_location(context_text)
            if not loc and lobes:
                loc = ", ".join(lobes)
            if loc:
                existing = procedures_payload.get("mechanical_debulking") if isinstance(
                    procedures_payload.get("mechanical_debulking"), dict
                ) else None
                if not existing or not existing.get("location"):
                    _set_nested_field(payload, "procedures_performed.mechanical_debulking.location", loc)

        if proc_key == "balloon_occlusion":
            size_match = _FRENCH_RE.search(context_text)
            if size_match:
                size_num = size_match.group("num")
                if size_num:
                    existing = procedures_payload.get("balloon_occlusion") if isinstance(
                        procedures_payload.get("balloon_occlusion"), dict
                    ) else None
                    if not existing or not existing.get("device_size"):
                        _set_nested_field(payload, "procedures_performed.balloon_occlusion.device_size", f"{size_num} Fr")
            if lobes:
                existing = procedures_payload.get("balloon_occlusion") if isinstance(
                    procedures_payload.get("balloon_occlusion"), dict
                ) else None
                if not existing or not existing.get("occlusion_location"):
                    _set_nested_field(payload, "procedures_performed.balloon_occlusion.occlusion_location", ", ".join(lobes))

        if proc_key == "blvr":
            blvr_finding_count += 1
            segs = _extract_segments(context_text)
            if segs:
                blvr_segments.extend(segs)
            if lobes:
                blvr_lobes.extend(lobes)

            for match in _VALVE_SIZE_RE.finditer(context_text):
                num = match.group("num")
                if num:
                    blvr_sizes.append(str(num))

            lower = context_text.lower()
            if "zephyr" in lower:
                blvr_type = blvr_type or "Zephyr (Pulmonx)"
            if "spiration" in lower:
                blvr_type = blvr_type or "Spiration (Olympus)"

            if re.search(r"\b(remove(?:d|al)?|retriev(?:e|ed|al)|extract(?:ed|ion))\b", lower):
                blvr_action = blvr_action or "Valve removal"
            if re.search(r"\b(place(?:d|ment)?|deploy(?:ed|ment)?|insert(?:ed|ion)?|implant(?:ed|ation)?)\b", lower):
                blvr_action = "Valve placement"

    if blvr_finding_count:
        blvr_segments = _dedupe_keep_order(blvr_segments)
        blvr_sizes = _dedupe_keep_order(blvr_sizes)
        blvr_lobes = _dedupe_keep_order(blvr_lobes)

        if blvr_action:
            _set_nested_field(payload, "procedures_performed.blvr.procedure_type", blvr_action)
        if blvr_type:
            _set_nested_field(payload, "procedures_performed.blvr.valve_type", blvr_type)
        if blvr_sizes:
            _set_nested_field(payload, "procedures_performed.blvr.valve_sizes", blvr_sizes)
        if blvr_segments:
            _set_nested_field(payload, "procedures_performed.blvr.segments_treated", blvr_segments)
            _set_nested_field(payload, "procedures_performed.blvr.number_of_valves", len(blvr_segments))
        else:
            existing_blvr = procedures_payload.get("blvr") if isinstance(procedures_payload.get("blvr"), dict) else None
            if not existing_blvr or existing_blvr.get("number_of_valves") is None:
                _set_nested_field(payload, "procedures_performed.blvr.number_of_valves", int(blvr_finding_count))

        # Target lobe is a strict literal; only set when unambiguous.
        if len(blvr_lobes) == 1:
            _set_nested_field(payload, "procedures_performed.blvr.target_lobe", blvr_lobes[0])

    try:
        return RegistryRecord.model_validate(payload), warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"LLM_FINDINGS_BACKFILL_FAILED: {type(exc).__name__}")
        return record, warnings


@dataclass(frozen=True)
class LLMFindingsSeedResult:
    record: RegistryRecord
    masked_prompt_text: str
    cpt_codes: list[str]
    warnings: list[str]
    needs_review: bool
    accepted_findings: int
    dropped_findings: int


def seed_registry_record_from_llm_findings(prompt_text: str, *, llm: OpenAILLM | None = None) -> LLMFindingsSeedResult:
    masked_prompt_text = mask_prompt_cpt_noise(prompt_text or "")

    findings = extract_reporter_findings_v1(masked_prompt_text, llm=llm)
    accepted, dropped_warnings = validate_findings_against_text(
        findings,
        masked_prompt_text=masked_prompt_text,
    )

    ner_result = build_synthetic_ner_result(
        masked_prompt_text=masked_prompt_text,
        accepted_findings=accepted,
    )

    warnings: list[str] = []
    warnings.extend(dropped_warnings)

    mapping = NERToRegistryMapper().map_entities(ner_result)
    warnings.extend(mapping.warnings or [])
    record = mapping.record

    record, backfill_warnings = _apply_findings_backfill(
        record,
        accepted_findings=accepted,
        masked_prompt_text=masked_prompt_text,
    )
    warnings.extend(backfill_warnings)

    record = _maybe_infer_diagnostic_bronchoscopy(record)

    guardrails = ClinicalGuardrails()
    guardrail_outcome = guardrails.apply_record_guardrails(masked_prompt_text, record)
    if guardrail_outcome.warnings:
        warnings.extend(guardrail_outcome.warnings)
    record = guardrail_outcome.record or record

    derivation = derive_registry_to_cpt(record)
    derived_codes = [str(c.code).strip() for c in (derivation.codes or []) if str(c.code).strip()]
    if derivation.warnings:
        warnings.extend([str(w) for w in derivation.warnings if str(w).strip()])

    return LLMFindingsSeedResult(
        record=record,
        masked_prompt_text=masked_prompt_text,
        cpt_codes=derived_codes,
        warnings=warnings,
        needs_review=bool(guardrail_outcome.needs_review),
        accepted_findings=len(accepted),
        dropped_findings=len(dropped_warnings),
    )


def _maybe_infer_diagnostic_bronchoscopy(record: RegistryRecord) -> RegistryRecord:
    procedures = record.procedures_performed
    if procedures is None:
        return record

    diagnostic = getattr(procedures, "diagnostic_bronchoscopy", None)
    if diagnostic is not None and getattr(diagnostic, "performed", None) is True:
        return record

    bronch_signal_keys = (
        "bal",
        "brushings",
        "tbna_conventional",
        "peripheral_tbna",
        "linear_ebus",
        "radial_ebus",
        "navigational_bronchoscopy",
        "transbronchial_biopsy",
        "transbronchial_cryobiopsy",
        "endobronchial_biopsy",
        "therapeutic_aspiration",
        "airway_dilation",
        "airway_stent",
        "balloon_occlusion",
        "thermal_ablation",
        "cryotherapy",
        "mechanical_debulking",
        "rigid_bronchoscopy",
        "foreign_body_removal",
    )

    any_bronch_proc = False
    for key in bronch_signal_keys:
        proc = getattr(procedures, key, None)
        if proc is not None and getattr(proc, "performed", None) is True:
            any_bronch_proc = True
            break

    if not any_bronch_proc:
        return record

    payload = record.model_dump()
    procedures_payload = payload.get("procedures_performed")
    if not isinstance(procedures_payload, dict):
        procedures_payload = {}
        payload["procedures_performed"] = procedures_payload

    diagnostic_payload = procedures_payload.get("diagnostic_bronchoscopy")
    if not isinstance(diagnostic_payload, dict):
        diagnostic_payload = {}
        procedures_payload["diagnostic_bronchoscopy"] = diagnostic_payload

    diagnostic_payload["performed"] = True
    return RegistryRecord.model_validate(payload)


def build_record_payload_for_reporting(seed: LLMFindingsSeedResult) -> dict[str, Any]:
    payload = seed.record.model_dump(exclude_none=True)
    if seed.cpt_codes:
        payload["cpt_codes"] = list(seed.cpt_codes)
    return payload


__all__ = [
    "ALLOWED_PROCEDURE_KEYS",
    "FindingV1",
    "ReporterFindingsV1",
    "ReporterFindingsError",
    "ReporterFindingsParseError",
    "ReporterFindingsUnavailable",
    "LLMFindingsSeedResult",
    "build_record_payload_for_reporting",
    "extract_reporter_findings_v1",
    "seed_registry_record_from_llm_findings",
    "validate_findings_against_text",
]
