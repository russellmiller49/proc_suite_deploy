"""Note focusing helpers for deterministic RegistryEngine extraction.

This module is used by RegistryService.extract_record() when
REGISTRY_EXTRACTION_ENGINE=agents_focus_then_engine.

Guardrail: RAW-ML auditing must always run on the full raw note text. Focusing
is for deterministic registry extraction only.
"""

from __future__ import annotations

from typing import Any

from app.registry.processing.focus import get_procedure_focus

PREFERRED_SEGMENT_TYPES: tuple[str, ...] = (
    "Procedure",
    "Technique",
    "Findings",
    "Complications",
    "Sedation",
    "Disposition",
)


def focus_note_for_extraction(note_text: str) -> tuple[str, dict[str, Any]]:
    """Return (focused_text, meta) for deterministic extraction.

    The focused text should be a best-effort subset of the note that preserves
    procedural evidence while reducing noise from indications/history.
    """
    meta: dict[str, Any] = {"method": "unknown", "fallback": False}

    normalized = note_text or ""
    if not normalized.strip():
        meta["method"] = "empty"
        meta["fallback"] = True
        return note_text, meta

    # Prefer the deterministic parser agent if present.
    try:
        from app.agents.contracts import ParserIn
        from app.agents.parser.parser_agent import ParserAgent

        parser_out = ParserAgent().run(ParserIn(note_id="", raw_text=normalized))
        segments = getattr(parser_out, "segments", []) or []

        selected = [s for s in segments if getattr(s, "type", None) in PREFERRED_SEGMENT_TYPES]
        if selected:
            meta["method"] = "agents.parser"
            meta["selected_sections"] = [s.type for s in selected]
            focused_parts: list[str] = []
            for seg in selected:
                text = (seg.text or "").strip()
                if not text:
                    continue
                focused_parts.append(f"{seg.type.upper()}:\n{text}")
            focused = "\n\n".join(focused_parts).strip()
            if focused:
                return focused, meta

        # Parser ran but didn't find recognized headings; fall back to heuristics.
        meta["parser_sections"] = [getattr(s, "type", None) for s in segments]
        meta["parser_status"] = getattr(parser_out, "status", None)
    except Exception as exc:
        meta["agent_error"] = str(exc)

    focused = _heuristic_focus_sections(normalized)
    if focused and focused.strip() and focused.strip() != normalized.strip():
        meta["method"] = "heuristic.sections_v1"
        meta["fallback"] = False
        return focused, meta

    meta["method"] = "noop"
    meta["fallback"] = True
    return note_text, meta


def _heuristic_focus_sections(note_text: str) -> str:
    """Best-effort section extractor using common heading patterns."""
    import re

    text = note_text or ""
    if not text.strip():
        return text

    # Match headings like "PROCEDURE:", "FINDINGS:", etc at start of line.
    heading_re = re.compile(r"^(?P<header>[A-Za-z][A-Za-z /_-]{0,40}):\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return text

    # Slice into (header, body) segments.
    segments: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        header = match.group("header").strip()
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            segments.append((header, body))

    wanted = {h.lower() for h in PREFERRED_SEGMENT_TYPES} | {"specimens", "impression"}
    selected = [(h, b) for (h, b) in segments if h.strip().lower() in wanted]
    if not selected:
        return text

    return "\n\n".join(f"{h.upper()}:\n{b}" for h, b in selected).strip()


__all__ = ["focus_note_for_extraction", "get_procedure_focus"]
