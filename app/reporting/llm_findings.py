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

        keyword_text = f"{finding_text}\n{evidence}".strip()
        if not _contains_any_keyword(keyword_text, proc_key):
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
        proc_action_text = _PROC_ACTION_CUE_OVERRIDES.get(proc_key)
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

        if proc_key == "linear_ebus":
            station_spans = _extract_station_spans(evidence)
            entities.extend(
                _entities_for_helper_spans(
                    label="ANAT_LN_STATION",
                    evidence_quote=evidence,
                    evidence_abs_start=abs_start,
                    matches=station_spans,
                    confidence=confidence,
                )
            )

        if proc_key == "transbronchial_biopsy":
            lobe_spans = _extract_lobe_spans(evidence)
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


@dataclass(frozen=True)
class LLMFindingsSeedResult:
    record: RegistryRecord
    masked_prompt_text: str
    cpt_codes: list[str]
    warnings: list[str]
    needs_review: bool
    accepted_findings: int
    dropped_findings: int


def seed_registry_record_from_llm_findings(prompt_text: str) -> LLMFindingsSeedResult:
    masked_prompt_text = mask_prompt_cpt_noise(prompt_text or "")

    findings = extract_reporter_findings_v1(masked_prompt_text)
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
