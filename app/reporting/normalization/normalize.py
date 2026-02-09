from __future__ import annotations

from proc_schemas.clinical import ProcedureBundle

from .compat_enricher import add_compat_flat_fields
from .text_enricher import enrich_from_text
from .types import NormalizationResult


def normalize_bundle(bundle: ProcedureBundle, *, source_text: str | None = None) -> NormalizationResult:
    """Normalize a ProcedureBundle prior to validation/rendering.

    Order:
    1) add compatibility fields
    2) optionally enrich from source text
    3) ensure idempotency (fill-missing only; no repeated appends)
    """
    notes = []

    bundle, compat_notes = add_compat_flat_fields(bundle)
    notes.extend(compat_notes)

    text = source_text if source_text is not None else bundle.free_text_hint
    if text:
        bundle, text_notes = enrich_from_text(bundle, text)
        notes.extend(text_notes)

    return NormalizationResult(bundle=bundle, notes=notes)

