from __future__ import annotations

import re
from typing import Any

from app.common.knowledge import lobe_aliases


_SEGMENT_TOKEN_RE = re.compile(r"\b(?P<side>[RL])B(?P<num>10|[1-9])\b", re.IGNORECASE)
_SEGMENT_NAME_RE = re.compile(
    r"\b(?P<name>(?:apical|posterior|anterior|superior|medial|lateral|basal|inferior)\s+segment)\b",
    re.IGNORECASE,
)

_INSTILLED_VOL_RE = re.compile(r"(?i)\b(?:instilled|infused)\s+(?P<num>\d{1,4})\s*(?:cc|ml)\b")
_RETURNED_VOL_RE = re.compile(
    r"(?i)\b(?:return(?:ed)?|recovered|suction\s*returned|suction(?:ed)?|aspirat(?:ed|ion))\s+(?:with\s+)?(?P<num>\d{1,4})\s*(?:cc|ml)\b"
)


def _segment_token_to_lobe(side: str, num: int) -> str | None:
    side_norm = (side or "").strip().upper()
    if side_norm not in {"R", "L"}:
        return None

    if side_norm == "R":
        if 1 <= num <= 3:
            return "RUL"
        if 4 <= num <= 5:
            return "RML"
        if 6 <= num <= 10:
            return "RLL"
        return None

    # Left
    if 1 <= num <= 3:
        return "LUL"
    if 4 <= num <= 5:
        # Prefer the KB key for consistency; downstream tokenizers can map to "Lingula".
        return "LINGULA"
    if 6 <= num <= 10:
        return "LLL"
    return None


def _add_anchor(result: dict[str, list[str]], seen: dict[str, set[str]], *, lobe: str, value: str) -> None:
    lobe_clean = (lobe or "").strip().upper()
    val_clean = (value or "").strip()
    if not lobe_clean or not val_clean:
        return

    bucket = result.setdefault(lobe_clean, [])
    seen_set = seen.setdefault(lobe_clean, set())
    if val_clean in seen_set:
        return
    bucket.append(val_clean)
    seen_set.add(val_clean)


def _infer_lobe_near(text: str, *, start: int, end: int) -> str | None:
    if not text:
        return None
    window = text[max(0, start - 120) : min(len(text), end + 120)]

    seg = _SEGMENT_TOKEN_RE.search(window)
    if seg:
        try:
            num = int(seg.group("num"))
        except Exception:
            num = -1
        return _segment_token_to_lobe(seg.group("side"), num)

    # Try direct lobe tokens.
    for token in ("RUL", "RML", "RLL", "LUL", "LLL", "LINGULA"):
        if re.search(rf"(?i)\b{token}\b", window):
            return token

    # Try KB aliases (e.g., "right upper lobe").
    aliases = lobe_aliases()
    for lobe, names in aliases.items():
        candidates = [lobe] + list(names or [])
        for candidate in candidates:
            if not candidate:
                continue
            pat = re.compile(rf"(?i)\b{re.escape(candidate)}\b")
            if pat.search(window):
                return str(lobe).strip().upper()

    return None


def extract_deterministic_anatomy(text: str) -> dict[str, list[str]]:
    """Extract anatomic anchors (lobes + segment tokens/names) from text.

    Returns a map like:
      {"RML": ["RB4", "Lateral Segment"], "RLL": ["RB10"]}
    """
    raw = text or ""
    if not raw.strip():
        return {}

    anchors: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}

    # 1) Segment tokens (RB4/LB4/etc) → inferred lobe.
    for match in _SEGMENT_TOKEN_RE.finditer(raw):
        side = match.group("side")
        try:
            num = int(match.group("num"))
        except Exception:
            continue
        lobe = _segment_token_to_lobe(side, num)
        if not lobe:
            continue
        token = f"{side.upper()}B{num}"
        _add_anchor(anchors, seen, lobe=lobe, value=token)

    # 2) Explicit lobe mentions (RML / "right middle lobe") from KB synonyms.
    for lobe, names in (lobe_aliases() or {}).items():
        lobe_key = str(lobe).strip().upper()
        if not lobe_key:
            continue
        candidates = [lobe_key] + list(names or [])
        for candidate in candidates:
            if not candidate:
                continue
            pat = re.compile(rf"(?i)\b{re.escape(candidate)}\b")
            for match in pat.finditer(raw):
                _add_anchor(anchors, seen, lobe=lobe_key, value=match.group(0))

    # 3) Segment names (e.g., "Lateral Segment") anchored to a nearby lobe token.
    for match in _SEGMENT_NAME_RE.finditer(raw):
        name = match.group("name").strip()
        lobe = _infer_lobe_near(raw, start=match.start(), end=match.end())
        if not lobe:
            continue
        # Preserve title casing for readability in prompts.
        pretty = " ".join(w.capitalize() for w in name.split())
        _add_anchor(anchors, seen, lobe=lobe, value=pretty)

    return anchors


def extract_volume_anchors(text: str) -> dict[str, int]:
    """Extract simple volume anchors (instilled + returned/recovered) in mL."""
    raw = text or ""
    if not raw.strip():
        return {}

    instilled = None
    returned = None

    m = _INSTILLED_VOL_RE.search(raw)
    if m:
        try:
            instilled = int(m.group("num"))
        except Exception:
            instilled = None

    m = _RETURNED_VOL_RE.search(raw)
    if m:
        try:
            returned = int(m.group("num"))
        except Exception:
            returned = None

    out: dict[str, int] = {}
    if isinstance(instilled, int):
        out["volume_instilled_ml"] = instilled
    if isinstance(returned, int):
        out["volume_returned_ml"] = returned
    return out


def to_prompt_payload(*, anatomy: dict[str, list[str]] | None, volumes: dict[str, int] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if anatomy:
        payload["anatomy"] = anatomy
    if volumes:
        payload["volumes"] = volumes
    return payload


__all__ = [
    "extract_deterministic_anatomy",
    "extract_volume_anchors",
    "to_prompt_payload",
]

