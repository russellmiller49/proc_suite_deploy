"""UMLS linking helpers.

This module preserves the public API (`UmlsConcept`, `umls_link`) while adding a
lightweight deterministic backend powered by a distilled IP-UMLS map.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Set
import os
import warnings

from config.settings import UmlsSettings

_ALLOWED_SEMTYPES: Set[str] = {
    "T061",  # Therapeutic or Preventive Procedure
    "T060",  # Diagnostic Procedure
    "T074",  # Medical Device
    "T168",  # Food / Substance Device catch-all
    "T017",  # Anatomical Structure
    "T029",  # Body Location or Region
}


@dataclass
class UmlsConcept:
    cui: str
    score: float
    semtypes: Sequence[str]
    preferred_name: str
    text: str
    start_char: int
    end_char: int


def _import_scispacy():  # pragma: no cover - exercised by integration
    try:
        import spacy  # type: ignore
        from scispacy.linking import EntityLinker  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("spaCy/scispaCy not available - install per README") from exc
    return spacy, EntityLinker


def _require_spacy():  # pragma: no cover - exercised by integration
    spacy, _ = _import_scispacy()
    model_name = os.getenv("PROCSUITE_SPACY_MODEL", "en_core_sci_sm")
    try:
        return _load_model(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' missing. Download with python -m spacy download {model_name}."
        ) from exc


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    spacy, _ = _import_scispacy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r'Field "model_.*" has conflict with protected namespace "model_"',
            category=UserWarning,
            module=r"pydantic\._internal\._fields",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Possible set union at position \d+",
            category=FutureWarning,
            module=r"spacy\.language",
        )
        nlp = spacy.load(model_name)  # type: ignore[union-attr]
    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "max_entities_per_mention": 3,
                "k": 5,
                "linker_name": "umls",
            },
        )
    return nlp


def _umls_link_scispacy(text: str, allowed_semtypes: Iterable[str] | None = None) -> List[UmlsConcept]:
    _, EntityLinker = _import_scispacy()
    clean_text = (text or "").strip()
    if not clean_text:
        return []

    semtypes = set(allowed_semtypes or _ALLOWED_SEMTYPES)
    nlp = _require_spacy()
    doc = nlp(clean_text)
    concepts: List[UmlsConcept] = []
    linker: EntityLinker = nlp.get_pipe("scispacy_linker")  # type: ignore[assignment]

    for ent in doc.ents:
        for cui, score in ent._.kb_ents:  # type: ignore[attr-defined]
            ent_info = linker.kb.cui_to_entity[cui]
            if not semtypes.intersection(ent_info.types):
                continue
            concepts.append(
                UmlsConcept(
                    cui=cui,
                    score=score,
                    semtypes=tuple(ent_info.types),
                    preferred_name=ent_info.canonical_name,
                    text=ent.text,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )
    return concepts


def umls_link_terms(terms: Iterable[str], category: str | None = None) -> List[UmlsConcept]:
    settings = UmlsSettings()
    if not settings.enable_linker:
        return []

    backend = (settings.linker_backend or "distilled").strip().lower()
    if backend == "scispacy":
        results: List[UmlsConcept] = []
        for term in terms:
            results.extend(_umls_link_scispacy(term))
        return results

    from app.umls.ip_umls_store import get_ip_umls_store

    store = get_ip_umls_store()
    concepts: List[UmlsConcept] = []
    for term in terms:
        match = store.match(term, category=category)
        if not match:
            continue
        score = 1.0 if match.get("match_type") == "exact" else 0.95
        concepts.append(
            UmlsConcept(
                cui=str(match["chosen_cui"]),
                score=score,
                semtypes=tuple(match.get("semtypes", []) or ()),
                preferred_name=str(match.get("preferred_name") or ""),
                text=str(term),
                start_char=0,
                end_char=len(str(term)),
            )
        )
    return concepts


def umls_link(text: str, allowed_semtypes: Iterable[str] | None = None) -> List[UmlsConcept]:
    settings = UmlsSettings()
    if not settings.enable_linker:
        return []

    backend = (settings.linker_backend or "distilled").strip().lower()
    if backend == "scispacy":
        return _umls_link_scispacy(text, allowed_semtypes=allowed_semtypes)

    return umls_link_terms([text])


__all__ = ["UmlsConcept", "umls_link", "umls_link_terms"]
