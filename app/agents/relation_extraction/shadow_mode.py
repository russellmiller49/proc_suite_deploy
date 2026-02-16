"""Shadow-mode relation extraction: heuristics + optional ML proposals.

Phase 8 goal:
- Emit `relations_ml` and `relations_heuristic`
- Merge: prefer ML relations when confidence is high, otherwise fall back to heuristics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

from app.agents.aggregator.timeline_aggregator import LinkProposal


class ShadowRelationsResult(BaseModel):
    relations_heuristic: list[LinkProposal] = Field(default_factory=list)
    relations_ml: list[LinkProposal] = Field(default_factory=list)
    relations: list[LinkProposal] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class _Key:
    entity_id: str
    relation: str


def _key(edge: LinkProposal) -> _Key:
    return _Key(entity_id=str(edge.entity_id), relation=str(edge.relation))


def merge_relations_shadow_mode(
    *,
    relations_heuristic: Sequence[LinkProposal],
    relations_ml: Sequence[LinkProposal],
    confidence_threshold: float = 0.85,
) -> ShadowRelationsResult:
    """Merge heuristic + ML edges with a conservative preference for high-confidence ML.

    For each (entity_id, relation) group:
    - If ML has any edge >= threshold: keep only the top-confidence ML edge.
    - Else: keep all heuristic edges for the group.
    """

    warnings: list[str] = []
    override_keys = 0
    ml_only_keys = 0
    below_threshold_keys = 0
    used_ml_keys = 0

    heur_by_key: dict[_Key, list[LinkProposal]] = {}
    for edge in relations_heuristic:
        heur_by_key.setdefault(_key(edge), []).append(edge)

    ml_by_key: dict[_Key, list[LinkProposal]] = {}
    for edge in relations_ml:
        ml_by_key.setdefault(_key(edge), []).append(edge)

    merged: list[LinkProposal] = []

    keys = set(heur_by_key.keys()) | set(ml_by_key.keys())
    for k in sorted(keys, key=lambda kk: (kk.entity_id, kk.relation)):
        ml_edges = ml_by_key.get(k, [])
        if ml_edges:
            best = max(ml_edges, key=lambda e: float(e.confidence))
            if float(best.confidence) >= float(confidence_threshold):
                merged.append(best)
                used_ml_keys += 1
                if heur_by_key.get(k):
                    override_keys += 1
                else:
                    ml_only_keys += 1
                continue
            below_threshold_keys += 1
            warnings.append(
                f"RELATIONS_ML_BELOW_THRESHOLD: entity_id={k.entity_id} relation={k.relation}"
            )
        merged.extend(heur_by_key.get(k, []))

    merged_sorted = sorted(
        merged,
        key=lambda e: (str(e.entity_id), str(e.relation), str(e.linked_to_id)),
    )

    merged_from_ml_edges = int(used_ml_keys)
    metrics = {
        "confidence_threshold": float(confidence_threshold),
        "heuristic_edges": len(relations_heuristic),
        "ml_edges": len(relations_ml),
        "merged_edges": len(merged_sorted),
        "heuristic_keys": len(heur_by_key),
        "ml_keys": len(ml_by_key),
        "keys_total": len(keys),
        "used_ml_keys": used_ml_keys,
        "override_keys": override_keys,
        "ml_only_keys": ml_only_keys,
        "below_threshold_keys": below_threshold_keys,
        "merged_from_ml_edges": merged_from_ml_edges,
        "merged_from_heuristic_edges": max(0, int(len(merged_sorted) - merged_from_ml_edges)),
    }

    return ShadowRelationsResult(
        relations_heuristic=list(relations_heuristic),
        relations_ml=list(relations_ml),
        relations=merged_sorted,
        warnings=warnings,
        metrics=metrics,
    )


def iter_relation_tasks(
    *,
    case_id: str,
    entities_by_id: dict[str, dict],
    relations: Iterable[LinkProposal],
    source: str,
) -> Iterable[dict]:
    """Utility for silver-data generation: produce one JSON-serializable task per edge."""

    for edge in relations:
        src = entities_by_id.get(str(edge.entity_id)) or {}
        dst = entities_by_id.get(str(edge.linked_to_id)) or {}
        yield {
            "case_id": case_id,
            "source": source,
            "edge": edge.model_dump(),
            "source_entity": src,
            "target_entity": dst,
            "text": f"{src.get('label','?')}  ->({edge.relation})->  {dst.get('label','?')}",
        }


__all__ = [
    "ShadowRelationsResult",
    "iter_relation_tasks",
    "merge_relations_shadow_mode",
]
