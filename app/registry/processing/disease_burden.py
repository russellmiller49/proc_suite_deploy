"""Deterministic extraction + safe overrides for disease burden numeric fields.

Goal: prevent LLM numeric hallucinations from entering the production registry
record when the note contains an unambiguous deterministic value.

Current targets (v3 schema):
- clinical_context.lesion_size_mm
- clinical_context.suv_max
- granular_data.cao_interventions_detail[].pre_obstruction_pct / post_obstruction_pct
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.common.sectionizer import SectionizerService
from app.common.spans import Span
from app.registry.processing.cao_interventions_detail import (
    extract_cao_interventions_detail_with_candidates,
)
from app.registry.schema import RegistryRecord


@dataclass(frozen=True)
class ExtractedNumeric:
    value: float
    span: Span


@dataclass(frozen=True)
class ExtractedLesionAxes:
    long_axis_mm: float
    short_axis_mm: float
    craniocaudal_mm: float | None
    size_text: str
    span: Span


_MULTI_DIM_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:[x×]|by)\s*\d+(?:\.\d+)?"
    r"(?:\s*(?:[x×]|by)\s*\d+(?:\.\d+)?)?\s*(?:cm|mm)\b"
)

_LESION_SIZE_TERM_BEFORE_RE = re.compile(
    r"(?i)\b(?:target\s+lesion|lesion|nodule|mass|tumou?r)\b"
    r"(?:\s+(?:is|was))?"
    r"(?:\s+(?:measuring|measures|measure|measured|sized|size|diameter(?:\s+of)?))?"
    r"[^0-9]{0,20}"
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm)\b"
)
_LESION_SIZE_TERM_AFTER_RE = re.compile(
    r"(?i)\b(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>cm|mm)\b"
    r"(?:\s+(?:spiculated|solid|part-?solid|ground-?glass|cavitary|calcified|fdg-?avid|pet-?avid))?"
    r"\s+(?:target\s+lesion|lesion|nodule|mass|tumou?r)\b"
)

_LESION_AXES_TERM_BEFORE_RE = re.compile(
    r"(?i)\b(?:target\s+lesion|lesion|nodule|mass|tumou?r)\b"
    r"(?:\s+(?:is|was))?"
    r"(?:\s+(?:measuring|measures|measure|measured|sized|size|dimensions?(?:\s+of)?))?"
    r"[^0-9]{0,20}"
    r"(?P<a>\d+(?:\.\d+)?)\s*(?:[x×]|by)\s*(?P<b>\d+(?:\.\d+)?)"
    r"(?:\s*(?:[x×]|by)\s*(?P<c>\d+(?:\.\d+)?))?\s*(?P<unit>cm|mm)\b"
)
_LESION_AXES_TERM_AFTER_RE = re.compile(
    r"(?i)\b(?P<a>\d+(?:\.\d+)?)\s*(?:[x×]|by)\s*(?P<b>\d+(?:\.\d+)?)"
    r"(?:\s*(?:[x×]|by)\s*(?P<c>\d+(?:\.\d+)?))?\s*(?P<unit>cm|mm)\b"
    r"(?:\s+(?:spiculated|solid|part-?solid|ground-?glass|cavitary|calcified|fdg-?avid|pet-?avid))?"
    r"\s+(?:target\s+lesion|lesion|nodule|mass|tumou?r)\b"
)

_SUV_RE = re.compile(
    r"(?i)\bSUV(?:\s*max(?:imum)?|\s*max)?\b[^0-9]{0,12}(?P<num>\d+(?:\.\d+)?)"
)

_LESION_CONTEXT_RE = re.compile(r"(?i)\b(?:target\s+lesion|lesion|nodule|mass|tumou?r)\b")

_MORPHOLOGY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Spiculated", re.compile(r"(?i)\bspiculat(?:ed|ion)\b")),
    ("Ground Glass", re.compile(r"(?i)\b(?:ground[-\s]?glass|ggo)\b")),
    ("Part-solid", re.compile(r"(?i)\bpart[-\s]?solid\b")),
    ("Solid", re.compile(r"(?i)\bsolid\b")),
    ("Cavitary", re.compile(r"(?i)\bcavit(?:ary|at(?:e|ed)|ation)\b")),
    ("Calcified", re.compile(r"(?i)\bcalcif(?:ied|ication)\b")),
)

_MORPHOLOGY_SECTIONIZER: SectionizerService | None = None


def _get_morphology_sectionizer() -> SectionizerService:
    global _MORPHOLOGY_SECTIONIZER
    if _MORPHOLOGY_SECTIONIZER is None:
        # Keep this lightweight: only scan the most reliable radiology/context sections.
        _MORPHOLOGY_SECTIONIZER = SectionizerService(headings=("INDICATION", "FINDINGS"))
    return _MORPHOLOGY_SECTIONIZER


def _extract_target_lesion_morphology(note_text: str) -> tuple[list[str], list[Span]]:
    """Extract morphology terms for the target lesion from INDICATION/FINDINGS sections."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return [], []

    sectionizer = _get_morphology_sectionizer()
    sections = sectionizer.sectionize(text)

    terms: list[str] = []
    seen: set[str] = set()
    spans: list[Span] = []

    for section in sections:
        if section.title.upper() not in {"INDICATION", "FINDINGS"}:
            continue

        # Important: `Section.text` is stripped, so indices can drift. Use the original slice.
        slice_text = text[section.start : section.end]
        if not slice_text.strip():
            continue

        for canonical, pattern in _MORPHOLOGY_PATTERNS:
            for match in pattern.finditer(slice_text):
                window = slice_text[max(0, match.start() - 80) : min(len(slice_text), match.end() + 80)]
                if not _LESION_CONTEXT_RE.search(window):
                    continue

                if canonical not in seen:
                    terms.append(canonical)
                    seen.add(canonical)

                spans.append(
                    Span(
                        text=match.group(0).strip(),
                        start=section.start + match.start(),
                        end=section.start + match.end(),
                    )
                )

    return terms, spans


def _maybe_unescape_newlines(raw: str) -> str:
    if not raw:
        return ""
    if ("\n" not in raw and "\r" not in raw) and ("\\n" in raw or "\\r" in raw):
        return raw.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    return raw


def _coerce_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mm_from(value: float, unit: str) -> float:
    return value * 10.0 if unit.lower() == "cm" else value


def _unique_value(values: list[float], *, places: int = 1) -> float | None:
    if not values:
        return None
    normalized = [round(float(v), places) for v in values]
    uniques = sorted({v for v in normalized})
    if len(uniques) == 1:
        return uniques[0]
    return None


def _extract_lesion_size_candidates(text: str) -> tuple[list[ExtractedNumeric], list[ExtractedLesionAxes]]:
    single_candidates: list[ExtractedNumeric] = []
    axes_candidates: list[ExtractedLesionAxes] = []

    multi_dim_spans = [(m.start(), m.end()) for m in _MULTI_DIM_RE.finditer(text)]

    def _overlaps_multi_dim(start: int, end: int) -> bool:
        for md_start, md_end in multi_dim_spans:
            if start < md_end and end > md_start:
                return True
        return False

    for pattern in (_LESION_SIZE_TERM_AFTER_RE, _LESION_SIZE_TERM_BEFORE_RE):
        for match in pattern.finditer(text):
            snippet = match.group(0)
            # Exclude matches that are part of a multi-axis dimension string like "2.5 x 1.7 cm".
            if _overlaps_multi_dim(match.start(), match.end()):
                continue
            raw_num = match.group("num")
            raw_unit = match.group("unit")
            num = _coerce_float(raw_num)
            if num is None:
                continue
            value_mm = _mm_from(num, raw_unit)
            if value_mm <= 0 or value_mm > 500:
                continue
            single_candidates.append(
                ExtractedNumeric(
                    value=value_mm,
                    span=Span(text=snippet.strip(), start=match.start(), end=match.end()),
                )
            )

    for pattern in (_LESION_AXES_TERM_AFTER_RE, _LESION_AXES_TERM_BEFORE_RE):
        for match in pattern.finditer(text):
            snippet = match.group(0)

            a = _coerce_float(match.group("a"))
            b = _coerce_float(match.group("b"))
            c = _coerce_float(match.group("c")) if match.groupdict().get("c") is not None else None
            if a is None or b is None:
                continue

            unit = match.group("unit")
            a_mm = _mm_from(a, unit)
            b_mm = _mm_from(b, unit)
            c_mm = _mm_from(c, unit) if c is not None else None

            dims = [d for d in (a_mm, b_mm, c_mm) if d is not None]
            if any(d <= 0 or d > 500 for d in dims):
                continue

            long_axis = max(dims)
            short_axis = min(dims)

            axes_candidates.append(
                ExtractedLesionAxes(
                    long_axis_mm=float(long_axis),
                    short_axis_mm=float(short_axis),
                    craniocaudal_mm=float(c_mm) if c_mm is not None else None,
                    size_text=snippet.strip(),
                    span=Span(text=snippet.strip(), start=match.start(), end=match.end()),
                )
            )

    return single_candidates, axes_candidates


def extract_unambiguous_target_lesion_size_and_axes_mm(
    note_text: str,
) -> tuple[ExtractedNumeric | None, ExtractedLesionAxes | None, list[str]]:
    """Return deterministic target-lesion size evidence when a single lesion is supported.

    - `lesion_size_mm` reflects the unambiguous long-axis dimension (single value).
    - `axes` is populated only when an unambiguous multi-axis size string is present
      and the overall long-axis size is not ambiguous across candidates.
    """
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return None, None, []

    single_candidates, axes_candidates = _extract_lesion_size_candidates(text)
    long_axis_values = [c.value for c in single_candidates] + [c.long_axis_mm for c in axes_candidates]

    if not long_axis_values:
        return None, None, []

    warnings: list[str] = []
    unique_long = _unique_value(long_axis_values, places=1)
    if unique_long is None:
        unique_values = sorted({round(v, 1) for v in long_axis_values})
        return None, None, [f"AMBIGUOUS_DISEASE_BURDEN: lesion_size_mm candidates={unique_values}"]

    # Best evidence span: prefer a multi-axis span if available, else single-axis.
    lesion_span = None
    for cand in axes_candidates:
        if round(cand.long_axis_mm, 1) == round(unique_long, 1):
            lesion_span = cand.span
            break
    if lesion_span is None:
        for cand in single_candidates:
            if round(cand.value, 1) == round(unique_long, 1):
                lesion_span = cand.span
                break

    lesion = ExtractedNumeric(value=unique_long, span=lesion_span or single_candidates[0].span)

    axes: ExtractedLesionAxes | None = None
    if axes_candidates:
        matching = [c for c in axes_candidates if round(c.long_axis_mm, 1) == round(unique_long, 1)]
        if matching:
            tuples = {
                (
                    round(c.long_axis_mm, 1),
                    round(c.short_axis_mm, 1),
                    round(c.craniocaudal_mm, 1) if c.craniocaudal_mm is not None else None,
                )
                for c in matching
            }
            if len(tuples) == 1:
                axes = matching[0]
            else:
                warnings.append(
                    "AMBIGUOUS_DISEASE_BURDEN: "
                    f"target_lesion_axes_mm candidates={sorted(tuples)}"
                )

    return lesion, axes, warnings


def extract_unambiguous_target_lesion_axes_mm(note_text: str) -> tuple[ExtractedLesionAxes | None, list[str]]:
    lesion, axes, warnings = extract_unambiguous_target_lesion_size_and_axes_mm(note_text)
    if lesion is None:
        return None, warnings
    return axes, warnings


def extract_unambiguous_lesion_size_mm(note_text: str) -> tuple[ExtractedNumeric | None, list[str]]:
    """Return a deterministic lesion long-axis size in mm when a single value is supported."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return None, []

    lesion, _axes, warnings = extract_unambiguous_target_lesion_size_and_axes_mm(note_text)
    # Keep this function focused on the single numeric size warning surface.
    filtered = [w for w in warnings if "target_lesion_axes_mm" not in w]
    return lesion, filtered


def extract_unambiguous_suv_max(note_text: str) -> tuple[ExtractedNumeric | None, list[str]]:
    """Return a deterministic SUV max when a single value is supported."""
    text = _maybe_unescape_newlines(note_text or "")
    if not text.strip():
        return None, []

    candidates: list[ExtractedNumeric] = []
    for match in _SUV_RE.finditer(text):
        raw_num = match.group("num")
        num = _coerce_float(raw_num)
        if num is None:
            continue
        if num < 0 or num > 100:
            continue
        snippet = match.group(0)
        candidates.append(
            ExtractedNumeric(
                value=float(num),
                span=Span(text=snippet.strip(), start=match.start(), end=match.end()),
            )
        )

    if not candidates:
        return None, []

    unique = _unique_value([c.value for c in candidates], places=1)
    if unique is None:
        unique_values = sorted({round(c.value, 1) for c in candidates})
        return None, [f"AMBIGUOUS_DISEASE_BURDEN: suv_max candidates={unique_values}"]

    best = next((c for c in candidates if round(c.value, 1) == unique), candidates[0])
    return ExtractedNumeric(value=unique, span=best.span), []


def apply_disease_burden_overrides(
    record_in: RegistryRecord,
    *,
    note_text: str,
) -> tuple[RegistryRecord, list[str]]:
    """Override disease-burden numeric fields when deterministic extraction is unambiguous."""
    warnings: list[str] = []
    if record_in is None:
        return RegistryRecord(), warnings

    record_data = record_in.model_dump()

    # ----------------------------
    # Clinical context (lesion size, SUV max)
    # ----------------------------
    clinical = record_data.get("clinical_context") or {}
    if not isinstance(clinical, dict):
        clinical = {}

    evidence = record_data.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}

    lesion, axes, lesion_warnings = extract_unambiguous_target_lesion_size_and_axes_mm(note_text)
    warnings.extend(lesion_warnings)
    if lesion is not None:
        old = clinical.get("lesion_size_mm")
        old_val: float | None
        try:
            old_val = None if old is None else float(old)
        except (TypeError, ValueError):
            old_val = None

        if old_val is None or round(old_val, 1) != round(lesion.value, 1):
            if old_val is not None:
                warnings.append(
                    f"OVERRIDE_LLM_NUMERIC: clinical_context.lesion_size_mm {round(old_val, 1)} -> {round(lesion.value, 1)}"
                )
            clinical["lesion_size_mm"] = lesion.value
            evidence.setdefault("clinical_context.lesion_size_mm", []).append(lesion.span)

    if axes is not None:
        target = clinical.get("target_lesion")
        if target is None or not isinstance(target, dict):
            target = {}

        def _override_axis(field: str, new_val: float | None) -> None:
            if new_val is None:
                return
            old_raw = target.get(field)
            old_val: float | None
            try:
                old_val = None if old_raw is None else float(old_raw)
            except (TypeError, ValueError):
                old_val = None

            if old_val is None or round(old_val, 1) != round(float(new_val), 1):
                if old_val is not None:
                    warnings.append(
                        f"OVERRIDE_LLM_NUMERIC: clinical_context.target_lesion.{field} {round(old_val, 1)} -> {round(float(new_val), 1)}"
                    )
                target[field] = float(new_val)
                evidence.setdefault(f"clinical_context.target_lesion.{field}", []).append(axes.span)

        _override_axis("long_axis_mm", axes.long_axis_mm)
        _override_axis("short_axis_mm", axes.short_axis_mm)
        _override_axis("craniocaudal_mm", axes.craniocaudal_mm)

        old_text = target.get("size_text")
        if not isinstance(old_text, str) or old_text.strip() != axes.size_text.strip():
            target["size_text"] = axes.size_text
            evidence.setdefault("clinical_context.target_lesion.size_text", []).append(axes.span)

        clinical["target_lesion"] = target

    suv, suv_warnings = extract_unambiguous_suv_max(note_text)
    warnings.extend(suv_warnings)
    if suv is not None:
        old = clinical.get("suv_max")
        old_val: float | None
        try:
            old_val = None if old is None else float(old)
        except (TypeError, ValueError):
            old_val = None

        if old_val is None or round(old_val, 1) != round(suv.value, 1):
            if old_val is not None:
                warnings.append(
                    f"OVERRIDE_LLM_NUMERIC: clinical_context.suv_max {round(old_val, 1)} -> {round(suv.value, 1)}"
                )
            clinical["suv_max"] = suv.value
            evidence.setdefault("clinical_context.suv_max", []).append(suv.span)

        target = clinical.get("target_lesion")
        if target is None or not isinstance(target, dict):
            target = {}
        old_raw = target.get("suv_max")
        old_val: float | None
        try:
            old_val = None if old_raw is None else float(old_raw)
        except (TypeError, ValueError):
            old_val = None
        if old_val is None or round(old_val, 1) != round(suv.value, 1):
            if old_val is not None:
                warnings.append(
                    f"OVERRIDE_LLM_NUMERIC: clinical_context.target_lesion.suv_max {round(old_val, 1)} -> {round(suv.value, 1)}"
                )
            target["suv_max"] = suv.value
            evidence.setdefault("clinical_context.target_lesion.suv_max", []).append(suv.span)
        clinical["target_lesion"] = target

    morph_terms, morph_spans = _extract_target_lesion_morphology(note_text)
    if morph_terms:
        target = clinical.get("target_lesion")
        if target is None or not isinstance(target, dict):
            target = {}

        existing = target.get("morphology")
        if not isinstance(existing, str) or not existing.strip():
            target["morphology"] = "; ".join(morph_terms) if len(morph_terms) > 1 else morph_terms[0]
            for span in morph_spans:
                evidence.setdefault("clinical_context.target_lesion.morphology", []).append(span)
            clinical["target_lesion"] = target

    if clinical:
        record_data["clinical_context"] = clinical
    if evidence:
        record_data["evidence"] = evidence

    # ----------------------------
    # CAO detail backstop (obstruction %)
    # ----------------------------
    parsed_cao, cao_candidates = extract_cao_interventions_detail_with_candidates(note_text)
    if parsed_cao:
        # Surface a conservative aggregate view for easy downstream UX: only when there is
        # exactly one unambiguous pre/post obstruction value across all CAO sites.
        pre_union: set[int] = set()
        post_union: set[int] = set()
        for loc_fields in cao_candidates.values():
            if not isinstance(loc_fields, dict):
                continue
            pre_set = loc_fields.get("pre_obstruction_pct")
            post_set = loc_fields.get("post_obstruction_pct")
            if isinstance(pre_set, set):
                pre_union |= {int(v) for v in pre_set}
            if isinstance(post_set, set):
                post_union |= {int(v) for v in post_set}

        procedures = record_data.get("procedures_performed")
        if procedures is None or not isinstance(procedures, dict):
            procedures = {}
        therapeutic = procedures.get("therapeutic_outcomes")
        if therapeutic is None or not isinstance(therapeutic, dict):
            therapeutic = {}

        def _find_pct_span(pct: int, *, allow_open: bool) -> Span | None:
            # Prefer explicit obstruction language; fall back to "% open" for derived post values.
            obstruction_re = re.compile(
                rf"(?i)\b{pct}\s*%\s*(?:obstruct(?:ed|ion)?|occlud(?:ed|ing|e)?|stenos(?:is|ed)|narrow(?:ed|ing)?|block(?:ed|ing)?|obstruction|occlusion)\b"
            )
            m = obstruction_re.search(note_text)
            if m:
                return Span(text=m.group(0).strip(), start=m.start(), end=m.end())

            if allow_open:
                open_pct = 100 - pct
                open_re = re.compile(rf"(?i)\b{open_pct}\s*%\s*(?:open|patent|recanaliz(?:ed|ation))\b")
                m2 = open_re.search(note_text)
                if m2:
                    return Span(text=m2.group(0).strip(), start=m2.start(), end=m2.end())

                # Support "patent/open to 80%" phrasing (percent after the patency keyword).
                open_after_re = re.compile(
                    rf"(?i)\b(?:open|patent|recanaliz(?:ed|ation))\w*(?:\s+(?:to|of))?\s*{open_pct}\s*%"
                )
                m3 = open_after_re.search(note_text)
                if m3:
                    return Span(text=m3.group(0).strip(), start=m3.start(), end=m3.end())
            return None

        if len(pre_union) == 1:
            pre_pct = next(iter(pre_union))
            old_raw = therapeutic.get("pre_obstruction_pct")
            old_val = old_raw if isinstance(old_raw, int) else None
            if old_val is None or old_val != pre_pct:
                if old_val is not None:
                    warnings.append(
                        f"OVERRIDE_LLM_NUMERIC: procedures_performed.therapeutic_outcomes.pre_obstruction_pct {old_val} -> {pre_pct}"
                    )
                therapeutic["pre_obstruction_pct"] = pre_pct
                span = _find_pct_span(pre_pct, allow_open=False)
                if span is not None:
                    evidence.setdefault(
                        "procedures_performed.therapeutic_outcomes.pre_obstruction_pct",
                        [],
                    ).append(span)
        elif len(pre_union) > 1:
            warnings.append(
                "AMBIGUOUS_DISEASE_BURDEN: "
                f"procedures_performed.therapeutic_outcomes.pre_obstruction_pct candidates={sorted(pre_union)}"
            )

        if len(post_union) == 1:
            post_pct = next(iter(post_union))
            old_raw = therapeutic.get("post_obstruction_pct")
            old_val = old_raw if isinstance(old_raw, int) else None
            if old_val is None or old_val != post_pct:
                if old_val is not None:
                    warnings.append(
                        f"OVERRIDE_LLM_NUMERIC: procedures_performed.therapeutic_outcomes.post_obstruction_pct {old_val} -> {post_pct}"
                    )
                therapeutic["post_obstruction_pct"] = post_pct
                span = _find_pct_span(post_pct, allow_open=True)
                if span is not None:
                    evidence.setdefault(
                        "procedures_performed.therapeutic_outcomes.post_obstruction_pct",
                        [],
                    ).append(span)
        elif len(post_union) > 1:
            warnings.append(
                "AMBIGUOUS_DISEASE_BURDEN: "
                f"procedures_performed.therapeutic_outcomes.post_obstruction_pct candidates={sorted(post_union)}"
            )

        if therapeutic:
            procedures["therapeutic_outcomes"] = therapeutic
            record_data["procedures_performed"] = procedures

        granular = record_data.get("granular_data")
        if granular is None or not isinstance(granular, dict):
            granular = {}

        existing_raw = granular.get("cao_interventions_detail")
        existing: list[dict[str, Any]] = []
        if isinstance(existing_raw, list):
            existing = [dict(item) for item in existing_raw if isinstance(item, dict)]

        def _key(item: dict[str, Any]) -> str:
            return str(item.get("location") or "").strip()

        by_loc: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for item in existing:
            loc = _key(item)
            if not loc:
                continue
            if loc not in by_loc:
                order.append(loc)
            by_loc[loc] = item

        modified = False
        for item in parsed_cao:
            if not isinstance(item, dict):
                continue
            loc = _key(item)
            if not loc:
                continue

            existing_item = by_loc.get(loc)
            created = False
            if existing_item is None:
                existing_item = {"location": loc}
                created = True

            changed_any = False
            for field in ("pre_obstruction_pct", "post_obstruction_pct"):
                value = item.get(field)
                if value is None:
                    continue
                try:
                    pct_int = max(0, min(100, int(value)))
                except (TypeError, ValueError):
                    continue

                candidate_set = cao_candidates.get(loc, {}).get(field)
                if not isinstance(candidate_set, set) or len(candidate_set) != 1 or pct_int not in candidate_set:
                    if isinstance(candidate_set, set) and len(candidate_set) > 1:
                        warnings.append(
                            "AMBIGUOUS_DISEASE_BURDEN: "
                            f"granular_data.cao_interventions_detail[{loc}].{field} candidates={sorted(candidate_set)}"
                        )
                    elif isinstance(candidate_set, set):
                        warnings.append(
                            "AMBIGUOUS_DISEASE_BURDEN: "
                            f"granular_data.cao_interventions_detail[{loc}].{field} candidates=[]"
                        )
                    else:
                        warnings.append(
                            "AMBIGUOUS_DISEASE_BURDEN: "
                            f"granular_data.cao_interventions_detail[{loc}].{field} candidates=<unavailable>"
                        )
                    continue

                existing_val = existing_item.get(field)
                if existing_val is None:
                    existing_item[field] = pct_int
                    changed_any = True
                    continue
                try:
                    existing_pct = int(existing_val)
                except (TypeError, ValueError):
                    existing_pct = None

                if existing_pct is None or existing_pct != pct_int:
                    if existing_pct is not None:
                        warnings.append(
                            f"OVERRIDE_LLM_NUMERIC: granular_data.cao_interventions_detail[{loc}].{field} {existing_pct} -> {pct_int}"
                        )
                    existing_item[field] = pct_int
                    changed_any = True

            if created:
                if changed_any:
                    by_loc[loc] = existing_item
                    order.append(loc)
                    modified = True
            else:
                if changed_any:
                    by_loc[loc] = existing_item
                    modified = True

        if modified:
            merged = [by_loc[loc] for loc in order if loc in by_loc]
            granular["cao_interventions_detail"] = merged
            record_data["granular_data"] = granular

    try:
        if evidence:
            record_data["evidence"] = evidence
        return RegistryRecord(**record_data), warnings
    except ValidationError as exc:
        warnings.append(f"DISEASE_BURDEN_OVERRIDE_FAILED: {type(exc).__name__}")
        return record_in, warnings


__all__ = [
    "ExtractedNumeric",
    "ExtractedLesionAxes",
    "extract_unambiguous_lesion_size_mm",
    "extract_unambiguous_target_lesion_axes_mm",
    "extract_unambiguous_target_lesion_size_and_axes_mm",
    "extract_unambiguous_suv_max",
    "apply_disease_burden_overrides",
]
