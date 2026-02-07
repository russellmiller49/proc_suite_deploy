"""Thin wrapper around scispaCy's UMLS linker with semantic-type filtering."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Set
import os
import warnings

try:
    import spacy  # type: ignore
    from scispacy.linking import EntityLinker  # type: ignore
except ImportError as exc:  # pragma: no cover - surfaced by preflight/tests
    spacy = None  # type: ignore
    EntityLinker = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

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


def _require_spacy() -> "spacy.Language":  # type: ignore[name-defined]
    if spacy is None or EntityLinker is None:  # pragma: no cover - exercised by integration
        raise RuntimeError(
            "spaCy/scispaCy not available - install per README"
        ) from _IMPORT_ERROR
    model_name = os.getenv("PROCSUITE_SPACY_MODEL", "en_core_sci_sm")
    try:
        return _load_model(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' missing. Download with python -m spacy download {model_name}."
        ) from exc


@lru_cache(maxsize=2)
def _load_model(model_name: str):
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


def umls_link(text: str, allowed_semtypes: Iterable[str] | None = None) -> List[UmlsConcept]:
    """Return filtered UMLS concepts for a free-text snippet."""
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


__all__ = ["UmlsConcept", "umls_link"]
