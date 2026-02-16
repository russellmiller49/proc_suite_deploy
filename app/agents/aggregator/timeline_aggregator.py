"""Timeline Aggregator v2: build an entity ledger with explicit cross-doc links.

Design goals:
- Stateless: operates only on per-doc structured outputs (no persistence).
- Zero-knowledge friendly: does not require raw PHI; avoids echoing full note text.
- Explicit linking: emits `linked_to_id`, `confidence`, and `reasoning_short` link proposals.

This module intentionally starts with conservative, deterministic heuristics. As the system
evolves, it can be extended to incorporate model-based relation extraction in shadow mode.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

_LOBE_RE = re.compile(r"\b(RUL|RML|RLL|LUL|LLL|LINGULA)\b", flags=re.IGNORECASE)
_SEGMENT_RE = re.compile(r"\b([RL]B\d{1,2})\b", flags=re.IGNORECASE)


class LedgerDocRef(BaseModel):
    timepoint_role: str
    seq: int
    doc_t_offset_days: int | None = None


class LedgerEntity(BaseModel):
    entity_id: str = Field(..., description="Stable entity identifier within the bundle response.")
    kind: str = Field(..., description="Entity kind (e.g. canonical_lesion, nav_target, specimen).")
    label: str = Field(..., description="Short human-readable label.")
    doc_ref: LedgerDocRef | None = Field(
        default=None,
        description="Source document reference; null for canonical (cross-doc) entities.",
    )
    attributes: dict[str, Any] = Field(default_factory=dict)


class LinkProposal(BaseModel):
    entity_id: str = Field(..., description="Source entity id.")
    linked_to_id: str = Field(..., description="Target entity id.")
    relation: str = Field(..., description="Relationship type (e.g. same_as, specimen_from).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0-1 confidence score.")
    reasoning_short: str = Field(..., description="Short rationale for reviewers/UI.")


class EntityLedger(BaseModel):
    schema_version: str = Field(default="entity_ledger_v1")
    entities: list[LedgerEntity] = Field(default_factory=list)
    link_proposals: list[LinkProposal] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BundleDocInput(BaseModel):
    """Minimal doc input contract for aggregation.

    The aggregator intentionally consumes only structured, scrubbed outputs.
    """

    timepoint_role: str
    seq: int
    doc_t_offset_days: int | None = None
    registry: dict[str, Any] = Field(default_factory=dict)


def _stable_id(prefix: str, parts: Sequence[str]) -> str:
    joined = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(joined).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _as_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _as_list(value: Any) -> list[Any] | None:
    return value if isinstance(value, list) else None


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_granular(registry: dict[str, Any]) -> dict[str, Any] | None:
    return _as_dict(registry.get("granular_data"))


def _iter_nav_targets(registry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    granular = _extract_granular(registry) or {}
    targets = _as_list(granular.get("navigation_targets")) or []
    for item in targets:
        if isinstance(item, dict):
            yield item


def _iter_specimens(registry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    granular = _extract_granular(registry) or {}
    specimens = _as_list(granular.get("specimens_collected")) or []
    for item in specimens:
        if isinstance(item, dict):
            yield item


@dataclass(frozen=True)
class _NavTarget:
    entity: LedgerEntity
    lobe: str
    segment: str
    size_mm: float | None
    location_text: str


def _norm_lobe(value: str) -> str:
    m = _LOBE_RE.search(value or "")
    return (m.group(1).upper() if m else "").replace("LINGULA", "Lingula")


def _norm_segment(value: str) -> str:
    m = _SEGMENT_RE.search(value or "")
    return m.group(1).upper() if m else ""


def _nav_target_label(lobe: str, segment: str, size_mm: float | None, location_text: str) -> str:
    loc = " ".join([p for p in (lobe, segment) if p]).strip()
    if not loc:
        loc = location_text.strip() or "Unknown target"
    if size_mm is not None:
        return f"Nav target: {loc} ({int(round(size_mm))}mm)"
    return f"Nav target: {loc}"


def _specimen_label(source_proc: str, source_location: str, specimen_number: int | None) -> str:
    base = f"Specimen {specimen_number}" if specimen_number else "Specimen"
    proc = source_proc.strip() or "Unknown procedure"
    loc = source_location.strip() or "Unknown location"
    return f"{base}: {proc} — {loc}"


def _canonical_lesion_label(lobe: str, segment: str, size_mm: float | None) -> str:
    loc = " ".join([p for p in (lobe, segment) if p]).strip() or "Unknown location"
    if size_mm is not None:
        return f"Lesion: {loc} (~{int(round(size_mm))}mm)"
    return f"Lesion: {loc}"


def _size_bucket(size_mm: float | None) -> str:
    if size_mm is None:
        return ""
    # Bucket to reduce accidental splits on minor measurement differences.
    # Use floor bucketing to avoid boundary flips (e.g., 22mm vs 23mm).
    return str(int(size_mm // 5) * 5)


def _location_key(lobe: str, segment: str, location_text: str) -> str:
    if lobe or segment:
        return f"{lobe}:{segment}".strip(":")
    # Fall back to minimal normalized tokens from free-text.
    lobe_from_text = _norm_lobe(location_text)
    seg_from_text = _norm_segment(location_text)
    if lobe_from_text or seg_from_text:
        return f"{lobe_from_text}:{seg_from_text}".strip(":")
    return (location_text or "").strip().lower()[:40]


def aggregate_entity_ledger(docs: Sequence[BundleDocInput]) -> EntityLedger:
    """Build an entity ledger + explicit link proposals from bundle docs."""

    entities: list[LedgerEntity] = []
    links: list[LinkProposal] = []
    warnings: list[str] = []

    nav_targets: list[_NavTarget] = []
    specimen_entities: list[LedgerEntity] = []

    for doc in docs:
        doc_ref = LedgerDocRef(
            timepoint_role=str(doc.timepoint_role),
            seq=int(doc.seq),
            doc_t_offset_days=doc.doc_t_offset_days,
        )

        for t in _iter_nav_targets(doc.registry):
            lobe_raw = _as_text(t.get("target_lobe")) or _as_text(t.get("target_location_text"))
            seg_raw = _as_text(t.get("target_segment")) or _as_text(t.get("target_location_text"))
            lobe = _norm_lobe(lobe_raw)
            segment = _norm_segment(seg_raw)
            location_text = _as_text(t.get("target_location_text"))
            size_mm = _as_float(t.get("lesion_size_mm"))
            target_number = str(t.get("target_number") or "")

            entity_id = _stable_id(
                "nav",
                [
                    doc_ref.timepoint_role,
                    str(doc_ref.seq),
                    target_number,
                    lobe,
                    segment,
                    _size_bucket(size_mm),
                    location_text.lower(),
                ],
            )
            label = _nav_target_label(lobe, segment, size_mm, location_text)
            entity = LedgerEntity(
                entity_id=entity_id,
                kind="nav_target",
                label=label,
                doc_ref=doc_ref,
                attributes={
                    "target_number": t.get("target_number"),
                    "target_location_text": location_text or None,
                    "target_lobe": lobe or None,
                    "target_segment": segment or None,
                    "lesion_size_mm": size_mm,
                    "tool_in_lesion_confirmed": t.get("tool_in_lesion_confirmed"),
                },
            )
            entities.append(entity)
            nav_targets.append(
                _NavTarget(
                    entity=entity,
                    lobe=lobe,
                    segment=segment,
                    size_mm=size_mm,
                    location_text=location_text,
                )
            )

        for s in _iter_specimens(doc.registry):
            specimen_number = None
            try:
                raw_number = s.get("specimen_number")
                specimen_number = int(raw_number) if raw_number else None
            except (TypeError, ValueError):
                specimen_number = None

            source_proc = _as_text(s.get("source_procedure"))
            source_location = _as_text(s.get("source_location"))
            final_dx = _as_text(s.get("final_pathology_diagnosis")) or None
            entity_id = _stable_id(
                "spec",
                [
                    doc_ref.timepoint_role,
                    str(doc_ref.seq),
                    str(specimen_number or ""),
                    source_proc.lower(),
                    source_location.lower(),
                ],
            )
            entity = LedgerEntity(
                entity_id=entity_id,
                kind="specimen",
                label=_specimen_label(source_proc, source_location, specimen_number),
                doc_ref=doc_ref,
                attributes={
                    "specimen_number": specimen_number,
                    "source_procedure": source_proc or None,
                    "source_location": source_location or None,
                    "final_pathology_diagnosis": final_dx,
                },
            )
            entities.append(entity)
            specimen_entities.append(entity)

    # 1) Create canonical lesions by clustering nav targets using conservative keys.
    clusters: dict[str, list[_NavTarget]] = {}
    for t in nav_targets:
        key = _location_key(t.lobe, t.segment, t.location_text)
        size_bucket = _size_bucket(t.size_mm)
        cluster_key = f"{key}|{size_bucket}"
        clusters.setdefault(cluster_key, []).append(t)

    canonical_by_cluster: dict[str, LedgerEntity] = {}
    for cluster_key in sorted(clusters.keys()):
        members = clusters[cluster_key]
        member_ids = sorted(m.entity.entity_id for m in members)
        lesion_id = _stable_id("lesion", [cluster_key, ",".join(member_ids)])
        # Use the first member's normalized location as a summary; keep it conservative.
        first = members[0]
        label = _canonical_lesion_label(first.lobe, first.segment, first.size_mm)
        lesion_entity = LedgerEntity(
            entity_id=lesion_id,
            kind="canonical_lesion",
            label=label,
            doc_ref=None,
            attributes={
                "location_key": cluster_key.split("|", maxsplit=1)[0],
                "member_count": len(members),
                "member_entity_ids": member_ids,
            },
        )
        entities.append(lesion_entity)
        canonical_by_cluster[cluster_key] = lesion_entity

        for m in members:
            confidence, reasoning = _confidence_nav_to_lesion(m, first)
            links.append(
                LinkProposal(
                    entity_id=m.entity.entity_id,
                    linked_to_id=lesion_id,
                    relation="linked_to_lesion",
                    confidence=confidence,
                    reasoning_short=reasoning,
                )
            )

    # 2) Link specimens to canonical lesions using same-doc string heuristics.
    nav_by_doc: dict[tuple[str, int], list[_NavTarget]] = {}
    for t in nav_targets:
        doc_ref = t.entity.doc_ref
        if doc_ref is None:
            continue
        nav_by_doc.setdefault((doc_ref.timepoint_role, doc_ref.seq), []).append(t)

    for specimen in specimen_entities:
        if specimen.doc_ref is None:
            continue
        candidates = nav_by_doc.get((specimen.doc_ref.timepoint_role, specimen.doc_ref.seq), [])
        if not candidates:
            continue
        best = _best_nav_match_for_specimen(specimen, candidates)
        if best is None:
            continue
        location = _location_key(best.lobe, best.segment, best.location_text)
        cluster_key = f"{location}|{_size_bucket(best.size_mm)}"
        lesion = canonical_by_cluster.get(cluster_key)
        if lesion is None:
            continue
        conf, reasoning = _confidence_specimen_to_lesion(specimen, best)
        if conf <= 0.0:
            continue
        links.append(
            LinkProposal(
                entity_id=specimen.entity_id,
                linked_to_id=lesion.entity_id,
                relation="specimen_from_lesion",
                confidence=conf,
                reasoning_short=reasoning,
            )
        )

    if not nav_targets and not specimen_entities:
        warnings.append(
            "ENTITY_LEDGER_EMPTY: no granular navigation_targets/specimens_collected found."
        )

    return EntityLedger(entities=entities, link_proposals=links, warnings=warnings)


def _confidence_nav_to_lesion(member: _NavTarget, representative: _NavTarget) -> tuple[float, str]:
    conf = 0.6
    reasons: list[str] = []
    if member.lobe and member.lobe == representative.lobe:
        conf += 0.15
        reasons.append("same lobe")
    if member.segment and representative.segment and member.segment == representative.segment:
        conf += 0.15
        reasons.append("same segment")
    if member.size_mm is not None and representative.size_mm is not None:
        delta = abs(member.size_mm - representative.size_mm)
        if delta <= 2.0:
            conf += 0.10
            reasons.append("size within 2mm")
        elif delta <= 5.0:
            conf += 0.05
            reasons.append("size within 5mm")
    conf = min(conf, 0.98)
    return conf, ", ".join(reasons) or "clustered by location/size bucket"


def _best_nav_match_for_specimen(
    specimen: LedgerEntity,
    candidates: Sequence[_NavTarget],
) -> _NavTarget | None:
    source_location = _as_text(specimen.attributes.get("source_location"))
    if not source_location:
        return None

    src_lobe = _norm_lobe(source_location)
    src_segment = _norm_segment(source_location)

    def score(t: _NavTarget) -> int:
        s = 0
        if src_lobe and t.lobe and src_lobe == t.lobe:
            s += 3
        if src_segment and t.segment and src_segment == t.segment:
            s += 3
        # Bonus for exact substring match of normalized tokens.
        if t.segment and t.segment.lower() in source_location.lower():
            s += 1
        if t.lobe and t.lobe.lower() in source_location.lower():
            s += 1
        return s

    ranked = sorted(candidates, key=lambda t: (score(t), t.entity.entity_id), reverse=True)
    best = ranked[0] if ranked else None
    if best is None:
        return None
    if score(best) <= 0:
        return None
    return best


def _confidence_specimen_to_lesion(specimen: LedgerEntity, nav: _NavTarget) -> tuple[float, str]:
    source_location = _as_text(specimen.attributes.get("source_location"))
    src_lobe = _norm_lobe(source_location)
    src_segment = _norm_segment(source_location)

    conf = 0.5
    reasons: list[str] = []
    if src_lobe and nav.lobe and src_lobe == nav.lobe:
        conf += 0.2
        reasons.append("source_location matches lobe")
    if src_segment and nav.segment and src_segment == nav.segment:
        conf += 0.2
        reasons.append("source_location matches segment")
    if conf < 0.6:
        return 0.0, ""
    return min(conf, 0.95), ", ".join(reasons) or "matched by location tokens"


__all__ = [
    "BundleDocInput",
    "EntityLedger",
    "LedgerEntity",
    "LedgerDocRef",
    "LinkProposal",
    "aggregate_entity_ledger",
]
