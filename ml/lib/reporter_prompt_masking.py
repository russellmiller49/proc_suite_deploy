"""Utilities for making reporter prompts narrative-first.

Goal: reduce model over-reliance on CPT/menu blocks by masking CPT-heavy lines
and "IP ... CODE MOD DETAILS" sections while preserving the clinical narrative.

We intentionally do NOT mask non-procedural sections (INDICATION/CONSENT/etc.)
here; that logic exists in `app.registry.processing.masking.mask_extraction_noise`
and is aimed at extraction quality, not model input quality.
"""

from __future__ import annotations

import re

from app.registry.processing.masking import mask_offset_preserving

_CPT_CODE_RE = re.compile(r"\b\d{5}\b")
_IP_CODE_MOD_DETAILS_RE = re.compile(r"(?im)^\s*IP\b[^\n]{0,80}CODE\s+MOD\s+DETAILS\b")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def prompt_contains_cpt_codes(text: str) -> bool:
    return bool(_CPT_CODE_RE.search(text or ""))


def prompt_contains_ip_code_mod_details(text: str) -> bool:
    return bool(_IP_CODE_MOD_DETAILS_RE.search(text or ""))


def mask_prompt_cpt_noise(text: str) -> str:
    """Return a compacted prompt with CPT/menu noise masked out.

    This uses the same masking primitives as the extraction-first pipeline
    (but not the non-procedural section masking).
    """
    masked = mask_offset_preserving(text or "")

    out_lines: list[str] = []
    prev_blank = True
    for raw_line in masked.splitlines():
        # Collapse long runs of spaces introduced by offset-preserving masking.
        line = _MULTI_SPACE_RE.sub(" ", raw_line).rstrip()
        if not line.strip():
            if prev_blank:
                continue
            prev_blank = True
            out_lines.append("")
            continue
        prev_blank = False
        out_lines.append(line)

    return "\n".join(out_lines).strip()


__all__ = [
    "mask_prompt_cpt_noise",
    "prompt_contains_cpt_codes",
    "prompt_contains_ip_code_mod_details",
]

