from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import CoderSettings
from app.common.knowledge import get_knowledge
from app.common.spans import Span
from app.registry.schema import RegistryRecord


_HYPHEN_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)

_PATHOLOGY_CONTEXT_RE = re.compile(r"\b(?:patholog|cytolog|histolog|cytopatholog)\w*\b", re.IGNORECASE)
_PATHOLOGY_RESULT_CUE_RE = re.compile(
    r"\b(?:final|diagnos(?:is|es)|consistent\s+with|positive\s+for|reve(?:al|aled)|show(?:s|ed)|confirm(?:ed|s))\b",
    re.IGNORECASE,
)
_PATHOLOGY_SENT_FOR_CUE_RE = re.compile(
    r"\b(?:sent|submit(?:ted)?|submitted|await|pending)\b[^.\n]{0,80}\b(?:patholog|cytolog|histolog|cytopatholog)\w*\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Extraction:
    value: Any
    start: int
    end: int
    text: str
    confidence: float = 0.9


@lru_cache(maxsize=1)
def _get_pathology_knowledge() -> dict[str, Any]:
    kb_path = Path(CoderSettings().kb_path)
    kb = get_knowledge(kb_path)
    payload = kb.get("pathology_knowledge")
    return payload if isinstance(payload, dict) else {}


def _first_match(patterns: list[str], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            continue
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match
    return None


def _normalize_hyphens(text: str) -> str:
    """Normalize unicode hyphen/dash characters to '-' without changing string length."""
    return (text or "").translate(_HYPHEN_TRANSLATION)


def _histology_is_result_context(text: str, start: int, end: int) -> bool:
    """Return True if histology appears in a pathology-results context (not just staging/indication)."""
    if not text or start < 0 or end <= start:
        return False

    window_start = max(0, start - 260)
    window_end = min(len(text), end + 260)
    window = text[window_start:window_end]

    if _PATHOLOGY_CONTEXT_RE.search(window) is None:
        return False

    # If the only context is that samples were *sent for* pathology/cytology (no result yet), skip.
    if _PATHOLOGY_SENT_FOR_CUE_RE.search(window) is not None and _PATHOLOGY_RESULT_CUE_RE.search(window) is None:
        return False

    # Common result formats: "FINAL PATHOLOGY:" or "Cytology: <result>"
    if re.search(r"(?i)\b(?:patholog|cytolog|histolog|cytopatholog)\w*\s*[:\-]", window) is not None:
        return True
    if _PATHOLOGY_RESULT_CUE_RE.search(window) is not None:
        return True

    return False


def _extract_histology(text: str, knowledge: dict[str, Any]) -> _Extraction | None:
    histology = knowledge.get("histology")
    if not isinstance(histology, dict) or not histology:
        return None

    normalized = _normalize_hyphens(text)

    # Prefer specific histology types by iterating in KB order.
    for _key, cfg in histology.items():
        if not isinstance(cfg, dict):
            continue
        label = cfg.get("label")
        patterns = cfg.get("patterns")
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(patterns, list) or not patterns:
            continue

        match = _first_match([str(p) for p in patterns if p], normalized)
        if not match:
            continue

        start, end = match.span(0)

        # Avoid matching "small cell ..." inside "non-small cell ..." (common staging/indication phrase).
        if str(_key).strip().lower() == "small_cell":
            prefix = normalized[max(0, start - 8) : start].lower()
            if re.search(r"non\s*[- ]\s*$", prefix):
                continue

        if not _histology_is_result_context(normalized, start, end):
            continue

        return _Extraction(value=label.strip(), start=start, end=end, text=text[start:end])

    return None


def _extract_biomarkers(text: str, knowledge: dict[str, Any]) -> dict[str, _Extraction]:
    biomarkers = knowledge.get("biomarkers")
    if not isinstance(biomarkers, dict) or not biomarkers:
        return {}

    extracted: dict[str, _Extraction] = {}
    for marker, cfg in biomarkers.items():
        if not isinstance(marker, str) or not marker.strip():
            continue
        if not isinstance(cfg, dict):
            continue

        patterns = cfg.get("patterns")
        result_patterns = cfg.get("result_patterns")
        if not isinstance(patterns, list) or not patterns:
            continue
        if not isinstance(result_patterns, list) or not result_patterns:
            continue

        marker_match = _first_match([str(p) for p in patterns if p], text)
        if not marker_match:
            continue

        result_match: re.Match[str] | None = None
        result_global_start: int | None = None
        result_global_end: int | None = None

        # Prefer results that appear after the marker (common pathology format: "EGFR: Negative").
        tail_start = marker_match.end()
        tail_end = min(len(text), marker_match.end() + 240)
        tail = text[tail_start:tail_end]
        tail_match = _first_match([str(p) for p in result_patterns if p], tail)
        if tail_match:
            result_match = tail_match
            result_global_start = tail_start + tail_match.start()
            result_global_end = tail_start + tail_match.end()
        else:
            # Fallback: search in a local window around the marker mention.
            window_start = max(0, marker_match.start() - 60)
            window_end = min(len(text), marker_match.end() + 240)
            window = text[window_start:window_end]
            window_match = _first_match([str(p) for p in result_patterns if p], window)
            if window_match:
                result_match = window_match
                result_global_start = window_start + window_match.start()
                result_global_end = window_start + window_match.end()

        if not result_match or result_global_start is None or result_global_end is None:
            continue

        start = min(marker_match.start(), result_global_start)
        end = max(marker_match.end(), result_global_end)
        snippet = text[start:end].strip()
        value = text[result_global_start:result_global_end].strip()
        if not value:
            continue

        extracted[marker.strip().upper()] = _Extraction(
            value=value,
            start=start,
            end=end,
            text=snippet,
        )

    return extracted


def _extract_pdl1(text: str, knowledge: dict[str, Any]) -> _Extraction | None:
    pdl1 = knowledge.get("pdl1")
    if not isinstance(pdl1, dict) or not pdl1:
        return None

    patterns = pdl1.get("patterns")
    if isinstance(patterns, list) and patterns:
        if _first_match([str(p) for p in patterns if p], text) is None:
            return None

    extractors = pdl1.get("tps_extractors")
    if not isinstance(extractors, list) or not extractors:
        return None

    for extractor in extractors:
        if not isinstance(extractor, dict):
            continue
        kind = extractor.get("kind")
        regex = extractor.get("regex")
        if kind not in {"percent", "range"}:
            continue
        if not isinstance(regex, str) or not regex.strip():
            continue

        match = re.search(regex, text, flags=re.IGNORECASE)
        if not match:
            continue

        start, end = match.span(0)
        snippet = text[start:end].strip()

        if kind == "percent":
            raw = match.group(1).strip() if match.lastindex and match.lastindex >= 1 else ""
            try:
                value_int = int(raw)
            except (TypeError, ValueError):
                continue
            if value_int < 0 or value_int > 100:
                continue
            return _Extraction(value=value_int, start=start, end=end, text=snippet)

        # kind == "range"
        raw = match.group(1).strip() if match.lastindex and match.lastindex >= 1 else ""
        if not raw:
            continue
        raw = re.sub(r"\s+", "", raw)
        if "%" in snippet and "%" not in raw:
            raw = raw + "%"
        return _Extraction(value=raw, start=start, end=end, text=snippet)

    return None


def apply_pathology_extraction(
    record: RegistryRecord,
    note_text: str,
) -> tuple[RegistryRecord, list[str]]:
    """Deterministically populate registry.pathology_results from note text.

    This never calls an LLM. It is designed to be safe and conservative:
    - Only fills missing fields.
    - Attaches evidence spans for UI highlighting and auditability.
    """
    if record is None or not (note_text or "").strip():
        return record, []

    knowledge = _get_pathology_knowledge()
    if not knowledge:
        return record, []

    text = note_text
    warnings: list[str] = []

    histology = _extract_histology(text, knowledge)
    biomarkers = _extract_biomarkers(text, knowledge)
    pdl1 = _extract_pdl1(text, knowledge)

    if histology is None and not biomarkers and pdl1 is None:
        return record, []

    record_data = record.model_dump()
    pathology = record_data.get("pathology_results")
    if not isinstance(pathology, dict):
        pathology = {}

    evidence = getattr(record, "evidence", None)
    if not isinstance(evidence, dict):
        evidence = {}

    def _add_evidence(key: str, extraction: _Extraction) -> None:
        evidence.setdefault(key, []).append(
            Span(
                text=extraction.text,
                start=int(extraction.start),
                end=int(extraction.end),
                confidence=float(extraction.confidence),
            )
        )

    if histology is not None and not pathology.get("histology"):
        pathology["histology"] = histology.value
        _add_evidence("pathology_results.histology", histology)

    if biomarkers:
        existing = pathology.get("molecular_markers")
        if not isinstance(existing, dict):
            existing = {}
        for marker, extraction in biomarkers.items():
            if marker not in existing or existing.get(marker) in (None, ""):
                existing[marker] = extraction.value
                _add_evidence(f"pathology_results.molecular_markers.{marker}", extraction)
        if existing:
            pathology["molecular_markers"] = existing

    if pdl1 is not None:
        if isinstance(pdl1.value, int):
            if pathology.get("pdl1_tps_percent") is None:
                pathology["pdl1_tps_percent"] = pdl1.value
                _add_evidence("pathology_results.pdl1_tps_percent", pdl1)
        else:
            if not pathology.get("pdl1_tps_text"):
                pathology["pdl1_tps_text"] = str(pdl1.value)
                _add_evidence("pathology_results.pdl1_tps_text", pdl1)

    record_data["pathology_results"] = pathology
    record_data["evidence"] = evidence
    record_out = RegistryRecord(**record_data)

    extracted_parts: list[str] = []
    if histology is not None:
        extracted_parts.append(f"histology={histology.value}")
    if biomarkers:
        extracted_parts.append(f"biomarkers={sorted(biomarkers.keys())}")
    if pdl1 is not None:
        extracted_parts.append(f"pdl1={pdl1.value}")
    if extracted_parts:
        warnings.append("PATHOLOGY_EXTRACTED: " + "; ".join(extracted_parts))

    return record_out, warnings


__all__ = ["apply_pathology_extraction"]
