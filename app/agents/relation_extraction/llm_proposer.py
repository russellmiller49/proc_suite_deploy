"""LLM-based relation proposer for Phase 8 (shadow mode).

Safety constraints:
- Zero-knowledge friendly: consumes ONLY the structured entity ledger (labels/attributes),
  never raw note text.
- Disabled by default; enabled only via env flag.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field

from app.agents.aggregator.timeline_aggregator import EntityLedger, LinkProposal

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    raw = raw.strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _llm_config_allows_runtime_calls() -> tuple[bool, str]:
    if not _truthy_env("RELATIONS_ML_ENABLED"):
        return False, "RELATIONS_ML_DISABLED"

    if _truthy_env("REGISTRY_USE_STUB_LLM"):
        return False, "RELATIONS_ML_SKIPPED: REGISTRY_USE_STUB_LLM enabled"

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "openai_compat":
        if _truthy_env("OPENAI_OFFLINE") or not os.getenv("OPENAI_API_KEY"):
            return False, "RELATIONS_ML_SKIPPED: OPENAI offline/unconfigured"
        if not (os.getenv("OPENAI_MODEL_JUDGE") or os.getenv("OPENAI_MODEL")):
            return False, "RELATIONS_ML_SKIPPED: OPENAI model not set"
        return True, "RELATIONS_ML_ENABLED"

    if provider == "gemini":
        if _truthy_env("GEMINI_OFFLINE") or not os.getenv("GEMINI_API_KEY"):
            return False, "RELATIONS_ML_SKIPPED: GEMINI offline/unconfigured"
        return True, "RELATIONS_ML_ENABLED"

    return False, f"RELATIONS_ML_SKIPPED: unknown LLM_PROVIDER='{provider}'"


TModel = TypeVar("TModel", bound=BaseModel)


class _LLMServiceLike(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
        temperature: float = 0.0,
    ) -> TModel: ...


class RelationsMLProposerResult(BaseModel):
    relations_ml: list[LinkProposal] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class _LLMProposalBundle(BaseModel):
    proposals: list[LinkProposal] = Field(default_factory=list)


def propose_relations_ml(
    *,
    ledger: EntityLedger | None,
    relations_heuristic: list[LinkProposal] | None = None,
    llm: _LLMServiceLike | None = None,
) -> RelationsMLProposerResult:
    """Optionally propose relations from the entity ledger using an LLM.

    The proposer is intentionally conservative and only emits edges that link
    `specimen`/`nav_target` entities to `canonical_lesion` entities.
    """

    relations_heuristic = relations_heuristic or []
    enabled, reason = _llm_config_allows_runtime_calls()
    metrics: dict[str, Any] = {
        "enabled": bool(enabled),
        "reason": reason,
    }

    if ledger is None:
        return RelationsMLProposerResult(relations_ml=[], warnings=["RELATIONS_ML_NO_LEDGER"], metrics=metrics)

    entities = list(ledger.entities or [])
    if not entities:
        metrics["entity_count"] = 0
        return RelationsMLProposerResult(relations_ml=[], warnings=["RELATIONS_ML_EMPTY_LEDGER"], metrics=metrics)

    entities_by_id = {e.entity_id: e for e in entities if e.entity_id}
    canonical_ids = {e.entity_id for e in entities if e.kind == "canonical_lesion" and e.entity_id}
    specimen_ids = {e.entity_id for e in entities if e.kind == "specimen" and e.entity_id}
    nav_target_ids = {e.entity_id for e in entities if e.kind == "nav_target" and e.entity_id}

    metrics.update(
        {
            "entity_count": len(entities),
            "canonical_lesion_count": len(canonical_ids),
            "specimen_count": len(specimen_ids),
            "nav_target_count": len(nav_target_ids),
        }
    )

    if not enabled:
        # Treat "disabled" as informational (metrics will still explain why).
        warnings = [] if reason == "RELATIONS_ML_DISABLED" else [reason]
        return RelationsMLProposerResult(relations_ml=[], warnings=warnings, metrics=metrics)

    if not canonical_ids:
        return RelationsMLProposerResult(
            relations_ml=[],
            warnings=["RELATIONS_ML_NO_CANONICAL_LESIONS"],
            metrics=metrics,
        )

    only_missing = _truthy_env("RELATIONS_ML_ONLY_MISSING")
    propose_nav_targets = _truthy_env("RELATIONS_ML_PROPOSE_NAV_TARGETS")

    existing_keys = {(e.entity_id, e.relation) for e in relations_heuristic}

    candidate_entities: list[dict[str, Any]] = []
    candidate_source_ids: set[str] = set()
    for e in entities:
        if e.kind == "specimen":
            if only_missing and (e.entity_id, "specimen_from_lesion") in existing_keys:
                continue
            candidate_source_ids.add(str(e.entity_id))
            candidate_entities.append(
                {
                    "entity_id": e.entity_id,
                    "kind": e.kind,
                    "label": e.label,
                    "attributes": e.attributes,
                    "doc_ref": e.doc_ref.model_dump() if e.doc_ref else None,
                }
            )
        elif propose_nav_targets and e.kind == "nav_target":
            if only_missing and (e.entity_id, "linked_to_lesion") in existing_keys:
                continue
            candidate_source_ids.add(str(e.entity_id))
            candidate_entities.append(
                {
                    "entity_id": e.entity_id,
                    "kind": e.kind,
                    "label": e.label,
                    "attributes": e.attributes,
                    "doc_ref": e.doc_ref.model_dump() if e.doc_ref else None,
                }
            )

    metrics["candidate_entity_count"] = len(candidate_entities)
    if not candidate_entities:
        return RelationsMLProposerResult(relations_ml=[], warnings=["RELATIONS_ML_NO_CANDIDATES"], metrics=metrics)

    max_candidates = _get_int_env("RELATIONS_ML_MAX_CANDIDATES", 40)
    if max_candidates > 0 and len(candidate_entities) > max_candidates:
        metrics["candidate_entity_count_before_truncation"] = len(candidate_entities)
        candidate_entities = sorted(
            candidate_entities,
            key=lambda x: (
                str(x.get("kind") or ""),
                str(x.get("entity_id") or ""),
            ),
        )[:max_candidates]
        candidate_source_ids = {
            str(e.get("entity_id") or "")
            for e in candidate_entities
            if str(e.get("entity_id") or "")
        }
        metrics["candidate_entity_count"] = len(candidate_entities)
        metrics["candidates_truncated"] = True
    else:
        metrics["candidates_truncated"] = False

    canonical_lesions = [
        {
            "entity_id": entities_by_id[eid].entity_id,
            "label": entities_by_id[eid].label,
            "attributes": entities_by_id[eid].attributes,
        }
        for eid in sorted(canonical_ids)
        if eid in entities_by_id
    ]
    max_canonical = _get_int_env("RELATIONS_ML_MAX_CANONICAL_LESIONS", 20)
    if max_canonical > 0 and len(canonical_lesions) > max_canonical:
        metrics["canonical_lesion_count_before_truncation"] = len(canonical_lesions)
        canonical_lesions = canonical_lesions[:max_canonical]
        metrics["canonical_lesion_count"] = len(canonical_lesions)
        metrics["canonical_lesions_truncated"] = True
    else:
        metrics["canonical_lesions_truncated"] = False

    system_prompt = (
        "You propose cross-document clinical relations from entity labels/attributes.\n"
        "Return ONLY JSON.\n"
        "Rules:\n"
        "- Use ONLY entity_ids provided.\n"
        "- Only propose relations:\n"
        "  - linked_to_lesion: nav_target -> canonical_lesion\n"
        "  - specimen_from_lesion: specimen -> canonical_lesion\n"
        "- Propose at most ONE target per (entity_id, relation).\n"
        "- confidence must be a number 0.0-1.0.\n"
        "- reasoning_short must be <= 12 words.\n"
        "- If you are not confident, omit the proposal.\n"
    )
    user_prompt = json.dumps(
        {
            "candidate_entities": candidate_entities,
            "canonical_lesions": canonical_lesions,
        },
        ensure_ascii=False,
    )
    metrics["prompt_chars"] = int(len(system_prompt) + len(user_prompt))

    if llm is None:
        # Lazy import to avoid side effects when ML is disabled.
        from app.common.llm import LLMService

        llm = LLMService(task="judge")

    warnings: list[str] = []
    try:
        t0 = time.time()
        bundle = llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=_LLMProposalBundle,
            temperature=0.0,
        )
        metrics["llm_duration_ms"] = float((time.time() - t0) * 1000.0)
    except Exception as e:
        metrics["llm_duration_ms"] = None
        warnings.append(f"RELATIONS_ML_ERROR: {type(e).__name__}")
        return RelationsMLProposerResult(relations_ml=[], warnings=warnings, metrics=metrics)

    # Defensive filtering + dedupe.
    out_by_key: dict[tuple[str, str], LinkProposal] = {}
    for p in bundle.proposals:
        if str(p.entity_id) not in candidate_source_ids:
            continue
        rel = str(p.relation)
        if rel == "specimen_from_lesion":
            if p.entity_id not in specimen_ids or p.linked_to_id not in canonical_ids:
                continue
        elif rel == "linked_to_lesion":
            if p.entity_id not in nav_target_ids or p.linked_to_id not in canonical_ids:
                continue
        else:
            continue

        key = (str(p.entity_id), rel)
        prev = out_by_key.get(key)
        if prev is None or float(p.confidence) > float(prev.confidence):
            out_by_key[key] = p

    out = list(out_by_key.values())
    metrics["proposal_count"] = len(out)
    return RelationsMLProposerResult(relations_ml=out, warnings=warnings, metrics=metrics)


__all__ = ["RelationsMLProposerResult", "propose_relations_ml"]
