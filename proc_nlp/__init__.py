"""NLP utilities shared across Procedure Suite."""

from __future__ import annotations

from typing import Any

from .normalize_proc import normalize_dictation

__all__ = ["normalize_dictation", "UmlsConcept", "umls_link"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    # Avoid importing spaCy/scispaCy at module import time (startup performance).
    if name == "UmlsConcept":
        from .umls_linker import UmlsConcept as _UmlsConcept

        return _UmlsConcept
    if name == "umls_link":
        from .umls_linker import umls_link as _umls_link

        return _umls_link
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(globals().keys()) | set(__all__))
