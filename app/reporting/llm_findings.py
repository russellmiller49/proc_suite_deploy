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

from app.common.spans import Span, dedupe_spans
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


class ClinicalContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indication: str | None = Field(
        None,
        description="Primary reason for procedure, extracted exactly as stated (no paraphrasing).",
    )
    preop_diagnosis: str | None = Field(None, description="Pre-op diagnosis as stated.")
    postop_diagnosis: str | None = Field(None, description="Post-op diagnosis as stated.")
    specimens: str | None = Field(None, description="Specimen list as stated (free text).")
    plan: str | None = Field(None, description="Impression/plan as stated (free text).")
    modifier_22_rationale: str | None = Field(
        None,
        description="Modifier 22 justification as stated (e.g., '>40% increased work due to hair removal').",
    )


class FindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    procedure_key: str = Field(..., description="Procedure key allowlisted by the registry NER mapper.")
    action: Literal[
        "placement",
        "removal",
        "revision",
        "ablation",
        "aspiration",
        "inspection",
        "diagnostic",
        "other",
    ] = Field(..., description="Explicit action for this finding (do not infer).")
    anatomy: list[str] = Field(
        default_factory=list,
        description="Specific anatomic locations for THIS finding only (e.g., ['Trachea'], ['RB4'], ['station 7']).",
    )
    finding_text: str = Field(..., description="Short normalized finding. No CPT codes.")
    evidence_quote: str = Field(..., description="Verbatim substring copied from the input prompt text.")
    clinical_details: str | None = Field(
        None,
        description=(
            "1-3 sentences expanding the shorthand into a detailed operative-report narrative. "
            "MUST include specific tools, sizes, settings (e.g., APC pulse 20 effect 2), sealants, and techniques used "
            "when they are explicitly documented. If not documented, return null."
        ),
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class ReporterFindingsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["reporter_findings_v1"] = "reporter_findings_v1"
    context: ClinicalContextV1 | None = None
    findings: list[FindingV1] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


ALLOWED_PROCEDURE_KEYS: set[str] = set(PROCEDURE_MAPPINGS.keys()) | {
    "peripheral_tbna",
    "bpf_sealant",
    "tracheal_puncture",
}


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
        if proc_key == "bpf_sealant":
            return ["sealant", "glue", "fibrin", "tisseel", "tissel"]
        if proc_key == "tracheal_puncture":
            return ["tracheal puncture", "transtracheal", "angiocath", "angiocatheter", "punctur"]
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
        "context": {
            "indication": "Persistent air leak",
            "preop_diagnosis": "Persistent air leak",
            "postop_diagnosis": "Persistent air leak",
            "specimens": "BAL ×2",
            "plan": "Follow up BAL results; plan valve removal in 6 weeks.",
            "modifier_22_rationale": "BAL performed in multiple lobes (increased work).",
        },
        "findings": [
            {
                "procedure_key": "bal",
                "action": "diagnostic",
                "anatomy": ["Lingula", "LB4", "LB5"],
                "finding_text": "BAL performed in Lingula (LB4/LB5)",
                "evidence_quote": "BAL lingula LB4/LB5",
                "clinical_details": "BAL was performed with saline instilled and returned; specimens were sent as documented.",
                "confidence": 0.9,
            },
            {
                "procedure_key": "bpf_sealant",
                "action": "other",
                "anatomy": ["RLL posterior subsegment"],
                "finding_text": "Endobronchial sealant applied for air leak",
                "evidence_quote": "tisseel 2cc RLL posterior subsegment.",
                "clinical_details": "2 cc Tisseel sealant was applied to the documented segment for air leak management.",
                "confidence": 0.85,
            },
            {
                "procedure_key": "blvr",
                "action": "removal",
                "anatomy": ["RB10"],
                "finding_text": "Endobronchial valve removed from RB10",
                "evidence_quote": "tried size 7 spiration valve RB10 too big removed.",
                "clinical_details": "A previously attempted endobronchial valve was removed from RB10 as documented.",
                "confidence": 0.85,
            },
        ],
        "notes": [],
    }
    return (
        "You are a clinical information extraction engine.\n"
        "Task: Read the clinical prompt text and output global clinical context + atomic performed-procedure findings.\n\n"
        "Output MUST be exactly one JSON object matching this shape:\n"
        f"{json.dumps(schema_hint, indent=2)}\n\n"
        "Rules:\n"
        "- context.* fields MUST be extracted from the prompt text below. If not explicitly documented, use null.\n"
        f"- procedure_key MUST be one of: {keys}\n"
        "- action MUST be one of: placement, removal, revision, ablation, aspiration, inspection, diagnostic, other\n"
        "- anatomy MUST list only the specific anatomic targets for THIS finding (do not leak anatomy across findings).\n"
        "- finding_text MUST be short, normalized, and MUST NOT include CPT codes.\n"
        "- finding_text MUST include a canonical procedure keyword/abbreviation for the procedure_key.\n"
        "  (Examples: BAL, EBUS, TBNA, TBBx, Cryotherapy, APC, Airway dilation, Stent, Chest ultrasound, Balloon occlusion.)\n"
        "- evidence_quote MUST be a verbatim substring copied from the prompt text below.\n"
        "- evidence_quote MUST be >= 10 characters and include enough local context to uniquely support the finding.\n"
        "- evidence_quote SHOULD include the anatomy tokens (RB4, station 7, Trachea, etc.) that you list in anatomy.\n"
        "- evidence_quote does NOT need to contain the canonical keyword if the prompt uses shorthand.\n"
        "  If the prompt says 'cryo', write finding_text as 'Cryotherapy ...' and quote the exact 'cryo ...' evidence.\n"
        "  If the prompt says 'balloon' for stenosis, write finding_text as 'Airway dilation ...' and quote the balloon evidence.\n"
        "- clinical_details MUST be 1-3 sentences summarizing the procedure details for this finding using ONLY what is explicitly documented.\n"
        "  If tools/sizes/settings are documented, include them. If not documented, set clinical_details to null.\n"
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
        anatomy_tokens = [str(tok).strip() for tok in (item.anatomy or []) if str(tok).strip()]
        if anatomy_tokens:
            missing_anatomy = [tok for tok in anatomy_tokens if tok.lower() not in evidence_lower]
            if missing_anatomy:
                warnings.append(
                    "LLM_FINDINGS_DROPPED: anatomy_not_in_evidence "
                    f"index={idx} key={proc_key!r} tokens={missing_anatomy!r}"
                )
                continue
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


def _find_case_insensitive_span(text: str, needle: str) -> tuple[int, int] | None:
    raw = text or ""
    tok = (needle or "").strip()
    if not raw or not tok:
        return None
    raw_lower = raw.lower()
    tok_lower = tok.lower()
    idx = raw_lower.find(tok_lower)
    if idx < 0:
        return None
    return idx, idx + len(tok)


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
            station_matches: list[tuple[int, int, str]] = []
            for tok in _dedupe_keep_order([str(t).strip() for t in (item.anatomy or []) if str(t).strip()]):
                station_norm = normalize_station(tok)
                if not station_norm:
                    continue
                span_norm = _find_case_insensitive_span(evidence, station_norm)
                if span_norm:
                    rel_start, rel_end = span_norm
                    station_matches.append((rel_start, rel_end, station_norm))
                    continue
                span_raw = _find_case_insensitive_span(evidence, tok)
                if span_raw:
                    rel_start, rel_end = span_raw
                    station_matches.append((rel_start, rel_end, evidence[rel_start:rel_end]))
            station_spans = station_matches or _extract_station_spans(evidence)
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
            lobe_matches: list[tuple[int, int, str]] = []
            for tok in _dedupe_keep_order([str(t).strip() for t in (item.anatomy or []) if str(t).strip()]):
                lobe_norm = normalize_lobe(tok)
                if not lobe_norm:
                    continue
                span_norm = _find_case_insensitive_span(evidence, lobe_norm)
                if span_norm:
                    rel_start, rel_end = span_norm
                    lobe_matches.append((rel_start, rel_end, lobe_norm))
                    continue
                span_raw = _find_case_insensitive_span(evidence, tok)
                if span_raw:
                    rel_start, rel_end = span_raw
                    lobe_matches.append((rel_start, rel_end, evidence[rel_start:rel_end]))
            lobe_spans = lobe_matches or _extract_lobe_spans(evidence)
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
    context: ClinicalContextV1 | None,
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

    if context is not None:
        clinical_payload = payload.get("clinical_context")
        if not isinstance(clinical_payload, dict):
            clinical_payload = {}
            payload["clinical_context"] = clinical_payload

        if context.indication and not clinical_payload.get("primary_indication"):
            clinical_payload["primary_indication"] = str(context.indication).strip()

        if context.plan and not payload.get("follow_up_plan"):
            payload["follow_up_plan"] = str(context.plan).strip()

    accepted_keys: set[str] = set()
    for item in accepted_findings or []:
        proc_key = str(item.procedure_key or "").strip()
        if not proc_key:
            continue
        accepted_keys.add(proc_key)

        evidence = str(item.evidence_quote or "").strip()
        confidence = float(item.confidence) if item.confidence is not None else 1.0
        if proc_key == "tracheal_puncture":
            evidence_span = _find_evidence_span(masked_prompt_text or "", evidence)
            if evidence_span:
                start, end = evidence_span
                evidence_dict = payload.get("evidence")
                if not isinstance(evidence_dict, dict):
                    evidence_dict = {}
                    payload["evidence"] = evidence_dict
                key = "procedures_performed.tracheal_puncture.performed"
                spans = evidence_dict.get(key)
                if not isinstance(spans, list):
                    spans = []
                spans.append(Span(text=evidence, start=int(start), end=int(end), confidence=confidence))
                evidence_dict[key] = dedupe_spans(spans)
            continue

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
    blvr_lobes: list[str] = []
    blvr_finding_count = 0
    blvr_any_placement = False
    blvr_any_removal = False
    blvr_any_assessment = False

    def _as_anatomy_tokens(item: FindingV1) -> list[str]:
        return _dedupe_keep_order([str(tok).strip() for tok in (item.anatomy or []) if str(tok).strip()])

    def _join_anatomy(tokens: list[str]) -> str | None:
        return ", ".join(tokens) if tokens else None

    def _normalize_airway_stent_location(tokens: list[str]) -> str | None:
        allowed = {"Trachea", "Right mainstem", "Left mainstem", "Bronchus intermedius", "Carina (Y)", "Other"}
        for tok in tokens:
            if tok in allowed:
                return tok
        return None

    def _coerce_stent_action(value: str) -> str | None:
        mapping = {
            "placement": "Placement",
            "removal": "Removal",
            "revision": "Revision/Repositioning",
            "inspection": "Assessment only",
        }
        return mapping.get((value or "").strip().lower())

    def _proc_object_key(proc_key: str) -> str:
        path = _object_path_for(proc_key)
        return path.split(".")[-1]

    for item in accepted_findings or []:
        proc_key = str(item.procedure_key or "").strip()
        evidence = str(item.evidence_quote or "").strip()
        if not proc_key:
            continue

        anatomy_tokens = _as_anatomy_tokens(item)
        location_text = _join_anatomy(anatomy_tokens)

        if proc_key == "bal" and location_text:
            existing = procedures_payload.get("bal") if isinstance(procedures_payload.get("bal"), dict) else None
            if not existing or not existing.get("location"):
                _set_nested_field(payload, "procedures_performed.bal.location", location_text)

        if proc_key == "brushings" and anatomy_tokens:
            existing = procedures_payload.get("brushings") if isinstance(procedures_payload.get("brushings"), dict) else None
            current = list((existing or {}).get("locations") or [])
            merged = _dedupe_keep_order(current + anatomy_tokens)
            _set_nested_field(payload, "procedures_performed.brushings.locations", merged)

        if proc_key == "transbronchial_biopsy" and anatomy_tokens:
            existing = procedures_payload.get("transbronchial_biopsy") if isinstance(
                procedures_payload.get("transbronchial_biopsy"), dict
            ) else None
            current = list((existing or {}).get("locations") or [])
            merged = _dedupe_keep_order(current + anatomy_tokens)
            _set_nested_field(payload, "procedures_performed.transbronchial_biopsy.locations", merged)

        if proc_key == "transbronchial_cryobiopsy" and anatomy_tokens:
            existing = procedures_payload.get("transbronchial_cryobiopsy") if isinstance(
                procedures_payload.get("transbronchial_cryobiopsy"), dict
            ) else None
            current = list((existing or {}).get("locations_biopsied") or [])
            merged = _dedupe_keep_order(current + anatomy_tokens)
            _set_nested_field(payload, "procedures_performed.transbronchial_cryobiopsy.locations_biopsied", merged)

        if proc_key == "tbna_conventional":
            stations = _dedupe_keep_order(
                [normalize_station(tok) for tok in anatomy_tokens if normalize_station(tok)]
            )
            if stations:
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
            stent_action = _coerce_stent_action(str(item.action or ""))
            if stent_action:
                _set_nested_field(payload, "procedures_performed.airway_stent.action", stent_action)
                _set_nested_field(payload, "procedures_performed.airway_stent.airway_stent_removal", stent_action == "Removal")

            loc = _normalize_airway_stent_location(anatomy_tokens)
            if loc:
                existing = procedures_payload.get("airway_stent") if isinstance(procedures_payload.get("airway_stent"), dict) else None
                if not existing or not existing.get("location"):
                    _set_nested_field(payload, "procedures_performed.airway_stent.location", loc)

        if proc_key in {"airway_dilation", "cryotherapy", "thermal_ablation", "mechanical_debulking", "therapeutic_aspiration", "foreign_body_removal", "tumor_debulking"}:
            if location_text:
                obj_path = _object_path_for(proc_key)
                obj_key = _proc_object_key(proc_key)
                existing = procedures_payload.get(obj_key) if isinstance(procedures_payload.get(obj_key), dict) else None
                if not existing or not existing.get("location"):
                    _set_nested_field(payload, f"{obj_path}.location", location_text)

        if proc_key == "bpf_sealant":
            if location_text:
                existing = procedures_payload.get("bpf_sealant") if isinstance(procedures_payload.get("bpf_sealant"), dict) else None
                if not existing or not existing.get("location"):
                    _set_nested_field(payload, "procedures_performed.bpf_sealant.location", location_text)
            if item.clinical_details:
                existing = procedures_payload.get("bpf_sealant") if isinstance(procedures_payload.get("bpf_sealant"), dict) else None
                if not existing or not existing.get("notes"):
                    _set_nested_field(payload, "procedures_performed.bpf_sealant.notes", str(item.clinical_details).strip())
            if evidence:
                existing = procedures_payload.get("bpf_sealant") if isinstance(procedures_payload.get("bpf_sealant"), dict) else None
                if not existing or not existing.get("sealant_type"):
                    lower = evidence.lower()
                    if "tisseel" in lower or "tissel" in lower:
                        _set_nested_field(payload, "procedures_performed.bpf_sealant.sealant_type", "Tisseel")

        if proc_key == "balloon_occlusion":
            if evidence:
                size_match = _FRENCH_RE.search(evidence)
                if size_match:
                    size_num = size_match.group("num")
                    if size_num:
                        existing = procedures_payload.get("balloon_occlusion") if isinstance(
                            procedures_payload.get("balloon_occlusion"), dict
                        ) else None
                        if not existing or not existing.get("device_size"):
                            _set_nested_field(payload, "procedures_performed.balloon_occlusion.device_size", f"{size_num} Fr")
            if location_text:
                existing = procedures_payload.get("balloon_occlusion") if isinstance(
                    procedures_payload.get("balloon_occlusion"), dict
                ) else None
                if not existing or not existing.get("occlusion_location"):
                    _set_nested_field(payload, "procedures_performed.balloon_occlusion.occlusion_location", location_text)

        if proc_key == "blvr":
            blvr_finding_count += 1
            blvr_segments.extend(_extract_segments(" ".join(anatomy_tokens)))

            for tok in anatomy_tokens:
                lobe = normalize_lobe(tok)
                if lobe:
                    blvr_lobes.append(lobe)

            if evidence:
                for match in _VALVE_SIZE_RE.finditer(evidence):
                    num = match.group("num")
                    if num:
                        blvr_sizes.append(str(num))

                lower = evidence.lower()
                if "zephyr" in lower:
                    blvr_type = blvr_type or "Zephyr (Pulmonx)"
                if "spiration" in lower:
                    blvr_type = blvr_type or "Spiration (Olympus)"

            action_lower = str(item.action or "").strip().lower()
            if action_lower == "placement":
                blvr_any_placement = True
            elif action_lower == "removal":
                blvr_any_removal = True
            elif action_lower in {"inspection", "diagnostic"}:
                blvr_any_assessment = True

    if blvr_finding_count:
        blvr_segments = _dedupe_keep_order(blvr_segments)
        blvr_sizes = _dedupe_keep_order(blvr_sizes)
        blvr_lobes = _dedupe_keep_order(blvr_lobes)

        blvr_action: str | None = None
        if blvr_any_placement:
            blvr_action = "Valve placement"
        elif blvr_any_removal:
            blvr_action = "Valve removal"
        elif blvr_any_assessment:
            blvr_action = "Valve assessment"

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
    context: ClinicalContextV1 | None
    accepted_items: list[FindingV1]
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
        context=findings.context,
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
        context=findings.context,
        accepted_items=list(accepted),
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

    context = seed.context
    if context is not None:
        indication = str(context.indication or "").strip()
        mod_22 = str(context.modifier_22_rationale or "").strip()
        if indication:
            ind = indication.strip().rstrip(".")
            if mod_22:
                mod = mod_22.strip().rstrip(".")
                if not re.search(r"(?i)modifier\\s*22", mod):
                    mod = f"Modifier 22 Declaration: {mod}"
                payload["primary_indication"] = f"{ind}. {mod}".strip()
            else:
                payload["primary_indication"] = ind
        elif mod_22:
            mod = mod_22.strip().rstrip(".")
            if not re.search(r"(?i)modifier\\s*22", mod):
                mod = f"Modifier 22 Declaration: {mod}"
            payload["primary_indication"] = mod

        preop = str(context.preop_diagnosis or "").strip()
        if preop:
            payload["preop_diagnosis_text"] = preop

        postop = str(context.postop_diagnosis or "").strip()
        if postop:
            payload["postop_diagnosis_text"] = postop

        specimens = str(context.specimens or "").strip()
        if specimens:
            payload["specimens_text"] = specimens

        plan = str(context.plan or "").strip()
        if plan and not payload.get("follow_up_plan"):
            payload["follow_up_plan"] = plan

    def _dedupe_tokens(tokens: list[str]) -> list[str]:
        return _dedupe_keep_order([str(t).strip() for t in (tokens or []) if str(t).strip()])

    def _join_tokens(tokens: list[str]) -> str | None:
        tokens = _dedupe_tokens(tokens)
        return ", ".join(tokens) if tokens else None

    def _merge_notes(existing: Any, new: str) -> str:
        old = str(existing or "").strip()
        new_clean = str(new or "").strip()
        if not new_clean:
            return old
        if not old:
            return new_clean
        if new_clean in old:
            return old
        return f"{old}\n\n{new_clean}"

    def _infer_tools(text: str) -> list[str]:
        lower = (text or "").lower()
        tools: list[str] = []
        if "forceps" in lower:
            tools.append("Forceps")
        if "snare" in lower:
            tools.append("Snare")
        if "basket" in lower:
            tools.append("Basket")
        if "cryo" in lower:
            tools.append("Cryoprobe")
        return tools or ["Forceps"]

    tracheal_puncture_details: list[str] = []
    blvr_items = [item for item in (seed.accepted_items or []) if str(item.procedure_key or "").strip() == "blvr"]
    prebuilt_procedures: list[dict[str, Any]] = []

    for item in seed.accepted_items or []:
        proc_key = str(item.procedure_key or "").strip()
        if not proc_key:
            continue

        notes = str(item.clinical_details or "").strip()
        anatomy_tokens = _dedupe_tokens(list(item.anatomy or []))
        location_text = _join_tokens(anatomy_tokens)
        evidence = str(item.evidence_quote or "").strip()

        if proc_key == "tracheal_puncture":
            if notes:
                tracheal_puncture_details.append(notes)
            continue

        if proc_key == "airway_dilation":
            proc = payload.get("airway_dilation")
            if not isinstance(proc, dict):
                proc = {}
            if location_text and not proc.get("airway_segment"):
                proc["airway_segment"] = location_text
            if notes:
                proc["notes"] = _merge_notes(proc.get("notes"), notes)
            payload["airway_dilation"] = proc
            continue

        if proc_key == "thermal_ablation":
            proc = payload.get("endobronchial_tumor_destruction")
            if not isinstance(proc, dict):
                proc = {}
            if location_text and not proc.get("airway_segment"):
                proc["airway_segment"] = location_text
            if not proc.get("modality"):
                lower = f"{evidence}\n{item.finding_text}".lower()
                if "apc" in lower or "argon" in lower:
                    proc["modality"] = "APC"
                elif "laser" in lower:
                    proc["modality"] = "Laser"
                elif "cautery" in lower or "electrocaut" in lower:
                    proc["modality"] = "Electrocautery"
            if notes:
                proc["notes"] = _merge_notes(proc.get("notes"), notes)
            payload["endobronchial_tumor_destruction"] = proc
            continue

        if proc_key == "bpf_sealant":
            proc = payload.get("bpf_sealant_application")
            if not isinstance(proc, dict):
                proc = {}
            if not proc.get("sealant_type"):
                lower = f"{evidence}\n{item.finding_text}".lower()
                if "tisseel" in lower or "tissel" in lower:
                    proc["sealant_type"] = "Tisseel"
                else:
                    proc["sealant_type"] = "Sealant"
            if proc.get("volume_ml") in (None, "", [], {}) and evidence:
                match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*(?:cc|ml)\b", evidence)
                if match:
                    try:
                        proc["volume_ml"] = float(match.group(1))
                    except Exception:
                        proc["volume_ml"] = None
            if notes:
                proc["notes"] = _merge_notes(proc.get("notes"), notes)
            payload["bpf_sealant_application"] = proc
            continue

        if proc_key == "airway_stent":
            action = str(item.action or "").strip().lower()
            if action == "placement":
                proc = payload.get("airway_stent_placement")
                if not isinstance(proc, dict):
                    proc = {}
                if location_text and not proc.get("airway_segment"):
                    proc["airway_segment"] = location_text
                if notes:
                    proc["notes"] = _merge_notes(proc.get("notes"), notes)
                payload["airway_stent_placement"] = proc
            elif action == "removal":
                proc = payload.get("foreign_body_removal")
                if not isinstance(proc, dict):
                    proc = {}
                if location_text and not proc.get("airway_segment"):
                    proc["airway_segment"] = location_text
                if not proc.get("tools_used"):
                    proc["tools_used"] = _infer_tools(f"{evidence}\n{notes}")
                if notes:
                    proc["notes"] = _merge_notes(proc.get("notes"), notes)
                payload["foreign_body_removal"] = proc
            else:
                proc = payload.get("airway_stent_placement")
                if not isinstance(proc, dict):
                    proc = {}
                if location_text and not proc.get("airway_segment"):
                    proc["airway_segment"] = location_text
                if notes:
                    proc["notes"] = _merge_notes(proc.get("notes"), notes)
                payload["airway_stent_placement"] = proc
            continue

    if tracheal_puncture_details:
        extra = "\n\n".join([t for t in tracheal_puncture_details if t.strip()])
        if isinstance(payload.get("airway_stent_placement"), dict):
            payload["airway_stent_placement"]["notes"] = _merge_notes(
                payload["airway_stent_placement"].get("notes"), extra
            )
        elif isinstance(payload.get("airway_dilation"), dict):
            payload["airway_dilation"]["notes"] = _merge_notes(payload["airway_dilation"].get("notes"), extra)

    if blvr_items:
        def _segment_to_lobe(seg: str) -> str | None:
            tok = (seg or "").upper()
            match = re.match(r"^([RL])B(\d{1,2})$", tok)
            if not match:
                return None
            side = match.group(1)
            num = int(match.group(2))
            if side == "R":
                if num in {1, 2, 3}:
                    return "RUL"
                if num in {4, 5}:
                    return "RML"
                if 6 <= num <= 10:
                    return "RLL"
            if side == "L":
                if 1 <= num <= 5:
                    # LB4/LB5 are Lingula; keep LUL as a safe umbrella unless caller documents Lingula explicitly.
                    return "LUL"
                if 6 <= num <= 10:
                    return "LLL"
            return None

        placement_valves: list[dict[str, Any]] = []
        placement_notes: list[str] = []
        removal_locations: list[str] = []
        removal_notes: list[str] = []
        device_brand: str | None = None

        for item in blvr_items:
            action = str(item.action or "").strip().lower()
            evidence = str(item.evidence_quote or "").strip()
            notes = str(item.clinical_details or "").strip()
            anatomy_tokens = _dedupe_tokens(list(item.anatomy or []))
            segs = _extract_segments(" ".join(anatomy_tokens) + " " + evidence)
            lower = (evidence + "\n" + notes).lower()
            if "zephyr" in lower:
                device_brand = device_brand or "Zephyr (Pulmonx)"
            if "spiration" in lower:
                device_brand = device_brand or "Spiration (Olympus)"
            size = None
            match = _VALVE_SIZE_RE.search(evidence) or _VALVE_SIZE_RE.search(notes)
            if match:
                size = match.group("num")

            if action == "placement":
                if notes:
                    placement_notes.append(notes)
                for seg in segs or []:
                    lobe = _segment_to_lobe(seg) or "target lobe"
                    placement_valves.append(
                        {
                            "valve_type": device_brand or "Valve",
                            "valve_size": str(size) if size else None,
                            "lobe": lobe,
                            "segment": seg,
                        }
                    )
            elif action == "removal":
                if notes:
                    removal_notes.append(notes)
                removal_locations.extend(segs or anatomy_tokens)

        if placement_valves:
            lobes_treated = _dedupe_keep_order(
                [
                    str(v.get("lobe") or "").strip()
                    for v in placement_valves
                    if str(v.get("lobe") or "").strip()
                ]
            )
            prebuilt_procedures.append(
                {
                    "proc_type": "blvr_valve_placement",
                    "schema_id": "blvr_valve_placement_v1",
                    "data": {
                        "lobes_treated": lobes_treated or ["target lobe"],
                        "valves": placement_valves,
                        "notes": "\n\n".join([n for n in placement_notes if n.strip()]) or None,
                    },
                }
            )

        if removal_locations:
            ind = (
                str(context.indication).strip()
                if context and context.indication
                else "Endobronchial valve removal/exchange"
            )
            prebuilt_procedures.append(
                {
                    "proc_type": "blvr_valve_removal_exchange",
                    "schema_id": "blvr_valve_removal_exchange_v1",
                    "data": {
                        "indication": ind,
                        "device_brand": device_brand,
                        "locations": _dedupe_keep_order(
                            [str(l).strip() for l in removal_locations if str(l).strip()]
                        ),
                        "valves_removed": max(
                            1,
                            len(
                                _dedupe_keep_order(
                                    [str(l).strip() for l in removal_locations if str(l).strip()]
                                )
                            ),
                        ),
                        "tolerance_notes": "\n\n".join([n for n in removal_notes if n.strip()]) or None,
                    },
                }
            )

    if prebuilt_procedures:
        existing = payload.get("procedures")
        procedures: list[dict[str, Any]] = list(existing) if isinstance(existing, list) else []
        procedures.extend(prebuilt_procedures)
        payload["procedures"] = procedures

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
