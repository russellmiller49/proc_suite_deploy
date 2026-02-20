"""Lightweight UMLS lookup using the pre-extracted ip_umls_map.json.

Drop-in replacement for the heavy scispaCy-based umls_linker when running
on memory-constrained environments (e.g., Railway). Uses exact and fuzzy
term matching against the pre-built concept map instead of loading the
full UMLS Metathesaurus into memory.

Usage
-----
    from proc_nlp.umls_lite import umls_link_lite

    concepts = umls_link_lite("bronchoscopy with EBUS-TBNA")
    for c in concepts:
        print(c.cui, c.preferred_name, c.score)

The map file is loaded lazily on first call and cached for the process
lifetime (~2-5 MB memory vs ~1 GB for scispaCy UMLS linker).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Re-export the same semantic type set used by the extraction script
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
    """Mirrors proc_nlp.umls_linker.UmlsConcept for compatibility."""

    cui: str
    score: float
    semtypes: Sequence[str]
    preferred_name: str
    text: str
    start_char: int
    end_char: int


@lru_cache(maxsize=1)
def _load_map() -> Dict[str, Any]:
    """Load the pre-built IP UMLS map from disk (cached)."""
    map_path = os.getenv(
        "IP_UMLS_MAP_PATH",
        str(Path(__file__).resolve().parents[1] / "data" / "knowledge" / "ip_umls_map.json"),
    )
    path = Path(map_path)
    if not path.exists():
        logger.warning("IP UMLS map not found at %s — umls_link_lite will return empty results", path)
        return {"concepts": {}, "term_index": {}}

    logger.info("Loading IP UMLS map from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept_count = len(data.get("concepts", {}))
    term_count = len(data.get("term_index", {}))
    logger.info("Loaded %d concepts, %d index terms", concept_count, term_count)
    return data


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"[a-z][a-z0-9\-]{1,}", text.lower())


def _ngrams(tokens: List[str], max_n: int = 4) -> List[Tuple[str, int, int]]:
    """Generate n-grams from tokens with approximate character spans.

    Returns (ngram_text, approx_start_char, approx_end_char).
    """
    results = []
    for n in range(min(max_n, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            results.append((phrase, i, i + n))
    return results


def umls_link_lite(
    text: str,
    allowed_semtypes: Optional[Set[str]] = None,
    max_concepts: int = 20,
) -> List[UmlsConcept]:
    """Return UMLS concepts matched from the pre-built IP concept map.

    This is a lightweight alternative to proc_nlp.umls_linker.umls_link().
    It performs exact term-index lookup using n-grams extracted from the input.

    Parameters
    ----------
    text : str
        Free-text clinical snippet to link.
    allowed_semtypes : set[str] | None
        Semantic types to accept. Defaults to the standard IP set.
    max_concepts : int
        Maximum number of concepts to return.

    Returns
    -------
    list[UmlsConcept]
        Matched concepts sorted by descending score (longer matches score higher).
    """
    clean = (text or "").strip()
    if not clean:
        return []

    semtype_filter = allowed_semtypes or _ALLOWED_SEMTYPES
    data = _load_map()
    term_index: Dict[str, List[str]] = data.get("term_index", {})
    concepts_db: Dict[str, Dict[str, Any]] = data.get("concepts", {})

    if not term_index:
        return []

    tokens = _tokenize(clean)
    ngrams = _ngrams(tokens)

    seen_cuis: Set[str] = set()
    results: List[UmlsConcept] = []

    # Try to find character positions for matched terms
    text_lower = clean.lower()

    for phrase, tok_start, tok_end in ngrams:
        if len(results) >= max_concepts:
            break

        cuis = term_index.get(phrase)
        if not cuis:
            continue

        # Find actual character span in the original text
        char_start = text_lower.find(phrase)
        char_end = char_start + len(phrase) if char_start >= 0 else -1

        for cui in cuis:
            if cui in seen_cuis:
                continue
            concept_data = concepts_db.get(cui)
            if not concept_data:
                continue

            concept_semtypes = set(concept_data.get("semtypes", []))
            if not concept_semtypes.intersection(semtype_filter):
                continue

            seen_cuis.add(cui)
            # Score: longer phrase matches get higher scores
            score = min(1.0, len(phrase.split()) / 4.0 + 0.5)

            results.append(
                UmlsConcept(
                    cui=cui,
                    score=round(score, 3),
                    semtypes=tuple(concept_data.get("semtypes", [])),
                    preferred_name=concept_data.get("name", ""),
                    text=phrase,
                    start_char=max(char_start, 0),
                    end_char=max(char_end, 0),
                )
            )

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:max_concepts]


def lookup_cui(cui: str) -> Optional[Dict[str, Any]]:
    """Look up a single CUI in the pre-built map.

    Returns the concept dict or None if not found.
    """
    data = _load_map()
    return data.get("concepts", {}).get(cui)


def search_terms(query: str, limit: int = 10) -> List[Tuple[str, List[str]]]:
    """Search the term index for entries containing the query string.

    Returns list of (term, [CUI, ...]) tuples.
    """
    data = _load_map()
    term_index = data.get("term_index", {})
    query_lower = query.strip().lower()
    matches = []
    for term, cuis in term_index.items():
        if query_lower in term:
            matches.append((term, cuis))
            if len(matches) >= limit:
                break
    return matches


__all__ = ["UmlsConcept", "umls_link_lite", "lookup_cui", "search_terms"]
