"""Wrapper for spaCy/scispaCy UMLS linking with caching and guards."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import spacy
    from spacy.tokens import Doc
except ImportError:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore
    Doc = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from scispacy.linking import EntityLinker
except ImportError:  # pragma: no cover - optional dependency
    EntityLinker = None  # type: ignore


class UmlsLinker:
    """Convenience wrapper around the scispaCy EntityLinker component."""

    def __init__(self, model: str = "en_core_sci_lg") -> None:
        self.model_name = model
        self._nlp = None
        self._available = False
        self._cache: Dict[Tuple[int, Tuple[int, int]], list[str]] = {}
        self._doc_cache: Dict[int, Doc] = {}

        if spacy is None or EntityLinker is None:  # pragma: no cover - optional dependency
            return

        try:
            self._nlp = spacy.load(model)
        except Exception:  # pragma: no cover - loading failure guard
            self._nlp = None
            return

        if "scispacy_linker" not in self._nlp.pipe_names:
            try:
                self._nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                )
            except Exception:  # pragma: no cover - optional dependency guard
                self._nlp = None
                return

        self._available = True

    @property
    def available(self) -> bool:
        """Return True when the spaCy/scispaCy pipeline is ready."""

        return self._available and self._nlp is not None

    def link_spans(self, text: str, spans: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], list[str]]:
        """Return UMLS CUI candidates for each span in *spans*."""

        doc = self._ensure_doc(text)
        doc_hash = hash(text)
        results: Dict[Tuple[int, int], list[str]] = {}
        for start, end in spans:
            key = (doc_hash, (start, end))
            if key not in self._cache:
                self._cache[key] = self._link_single_span(doc, start, end)
            results[(start, end)] = list(self._cache[key])
        return results

    def _ensure_doc(self, text: str) -> Doc | None:
        if not self.available:  # pragma: no cover - guard path
            return None
        doc_hash = hash(text)
        if doc_hash not in self._doc_cache:
            self._doc_cache[doc_hash] = self._nlp(text)  # type: ignore[arg-type]
        return self._doc_cache[doc_hash]

    def _link_single_span(self, doc: Doc | None, start: int, end: int) -> list[str]:
        if doc is None:
            return []
        span = doc.char_span(start, end)
        if span is None or not getattr(span._, "kb_ents", None):
            return []
        return [cui for cui, score in span._.kb_ents]


__all__ = ["UmlsLinker"]
