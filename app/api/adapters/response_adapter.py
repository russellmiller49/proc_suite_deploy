from __future__ import annotations

from typing import Any


def _source_for_field(field: str | None) -> str:
    if not field:
        return "registry_span"
    if field == "ner_spans" or field.startswith("ner"):
        return "ner_span"
    return "registry_span"


def _span_item_from_obj(span: Any, source: str) -> dict[str, Any] | None:
    if isinstance(span, dict):
        if "source" in span and "text" in span and "span" in span:
            return {
                "source": span.get("source"),
                "text": span.get("text") or "",
                "span": list(span.get("span") or []),
                "confidence": span.get("confidence"),
            }
        text = span.get("text") or span.get("quote")
        start = span.get("start")
        end = span.get("end")
        if start is None:
            start = span.get("start_char")
        if end is None:
            end = span.get("end_char")
        confidence = span.get("confidence")
    else:
        text = getattr(span, "text", None) or getattr(span, "quote", None)
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        if start is None:
            start = getattr(span, "start_char", None)
        if end is None:
            end = getattr(span, "end_char", None)
        confidence = getattr(span, "confidence", None)

    if text is None or start is None or end is None:
        return None

    try:
        start_val = int(start)
        end_val = int(end)
    except (TypeError, ValueError):
        return None

    return {
        "source": source,
        "text": str(text),
        "span": [start_val, end_val],
        "confidence": confidence if confidence is not None else 1.0,
    }


def _merge_evidence(
    target: dict[str, list[dict[str, Any]]],
    evidence: dict[str, list[Any]],
) -> None:
    for field, spans in evidence.items():
        if not spans:
            continue
        source = _source_for_field(field)
        items: list[dict[str, Any]] = []
        for span in spans:
            item = _span_item_from_obj(span, source)
            if item:
                items.append(item)
        if items:
            target.setdefault(field, []).extend(items)


def build_v3_evidence_payload(
    *,
    record: Any | None = None,
    evidence: dict[str, list[Any]] | None = None,
    codes: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}

    if record is not None:
        record_evidence = getattr(record, "evidence", None)
        if isinstance(record_evidence, dict):
            _merge_evidence(merged, record_evidence)

    if isinstance(evidence, dict):
        _merge_evidence(merged, evidence)

    if not merged and codes:
        placeholders = [
            {"source": "derived_code", "text": str(code), "span": [0, 0], "confidence": 0.0}
            for code in codes
            if code
        ]
        if placeholders:
            merged["code_evidence"] = placeholders

    return merged


__all__ = ["build_v3_evidence_payload"]
