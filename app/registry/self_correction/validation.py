"""Strict validator for registry self-correction patches."""

from __future__ import annotations

import difflib
import os
import re
from typing import Any

try:
    from rapidfuzz.fuzz import partial_ratio
except Exception:  # pragma: no cover
    partial_ratio = None  # type: ignore[assignment]

ALLOWED_PATHS: set[str] = {
    # Performed flags
    "/procedures_performed/bal/performed",
    "/procedures_performed/bronchial_wash/performed",
    "/procedures_performed/brushings/performed",
    "/procedures_performed/diagnostic_bronchoscopy/performed",
    "/procedures_performed/endobronchial_biopsy/performed",
    "/procedures_performed/mechanical_debulking/performed",
    "/procedures_performed/therapeutic_aspiration/performed",
    "/procedures_performed/transbronchial_biopsy/performed",
    "/procedures_performed/transbronchial_cryobiopsy/performed",
    "/procedures_performed/tbna_conventional/performed",
    "/procedures_performed/peripheral_tbna/performed",
    "/procedures_performed/linear_ebus/performed",
    "/procedures_performed/radial_ebus/performed",
    "/procedures_performed/navigational_bronchoscopy/performed",
    "/procedures_performed/airway_dilation/performed",
    "/procedures_performed/airway_stent/performed",
    "/procedures_performed/foreign_body_removal/performed",
    "/procedures_performed/percutaneous_tracheostomy/performed",
    "/procedures_performed/eus_b/performed",
    "/procedures_performed/blvr/performed",
    "/procedures_performed/rigid_bronchoscopy/performed",
    "/procedures_performed/intubation/performed",
    "/procedures_performed/linear_ebus/elastography_used",
    "/procedures_performed/linear_ebus/elastography_pattern",
    "/pleural_procedures/ipc/performed",
    "/pleural_procedures/thoracentesis/performed",
    "/pleural_procedures/chest_tube/performed",
    "/pleural_procedures/fibrinolytic_therapy/performed",
    "/pleural_procedures/pleurodesis/performed",
    "/established_tracheostomy_route",
    # Add other safe fields as needed
}

ALLOWED_PATH_PREFIXES: set[str] = {
    "/procedures_performed/navigational_bronchoscopy",
    "/procedures_performed/tbna_conventional",
    "/procedures_performed/peripheral_tbna",
    "/procedures_performed/brushings",
    "/procedures_performed/mechanical_debulking",
    "/procedures_performed/therapeutic_aspiration",
    "/procedures_performed/transbronchial_cryobiopsy",
    "/procedures_performed/thermal_ablation",
    "/procedures_performed/peripheral_ablation",
    "/procedures_performed/airway_dilation",
    "/procedures_performed/airway_stent",
    "/procedures_performed/foreign_body_removal",
    "/procedures_performed/eus_b",
    "/procedures_performed/blvr",
    "/procedures_performed/rigid_bronchoscopy",
    "/procedures_performed/intubation",
    # Allow self-correction to seed the nested pleural_procedures object when missing.
    # Individual pleural child paths remain explicitly allowlisted below.
    "/pleural_procedures",
    "/pleural_procedures/ipc",
    "/pleural_procedures/thoracentesis",
    "/pleural_procedures/chest_tube",
    "/pleural_procedures/fibrinolytic_therapy",
    "/pleural_procedures/pleurodesis",
    "/granular_data",
}

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_PATCH_PATH_ALIASES: dict[str, str] = {
    "/procedures_performed/bronchial_valve_insertion": "/procedures_performed/blvr",
    "/procedures_performed/endobronchial_excision": "/procedures_performed/mechanical_debulking",
    "/procedures_performed/balloon_dilation": "/procedures_performed/airway_dilation",
    "/procedures_performed/diagnostic_bronchoscopy/bronchial_wash": "/procedures_performed/bronchial_wash/performed",
    "/procedures_performed/diagnostic_bronchoscopy/bronchial_washing": "/procedures_performed/bronchial_wash/performed",
    "/procedures_performed/foreign_body_removal/tool_used": "/procedures_performed/foreign_body_removal/retrieval_tool",
    "/procedures_performed/foreign_body_removal/tool": "/procedures_performed/foreign_body_removal/retrieval_tool",
    "/procedures_performed/rigid_bronchoscopy/mechanical_debulking": "/procedures_performed/mechanical_debulking",
    "/pleural_procedures/fibrinolysis_instillation": "/pleural_procedures/fibrinolytic_therapy",
    "/pleural_procedures/tunneled_pleural_catheter": "/pleural_procedures/ipc",
    "/procedures_performed/linear_ebus/elastography_performed": "/procedures_performed/linear_ebus/elastography_used",
    "/procedures_performed/ebus_elastography/performed": "/procedures_performed/linear_ebus/elastography_used",
    "/procedures_performed/ebus_elastography/method": "/procedures_performed/linear_ebus/elastography_pattern",
    "/procedures_performed/ebus_elastography/pattern": "/procedures_performed/linear_ebus/elastography_pattern",
}

_EBUS_ELASTOGRAPHY_ROOT = "/procedures_performed/ebus_elastography"
_EBUS_ELASTOGRAPHY_ROOT_CANONICAL = "/procedures_performed/linear_ebus"


def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def _normalize_alnum(text: str) -> str:
    """Lowercase + strip punctuation for resilient quote containment checks."""
    lowered = (text or "").lower()
    no_punct = _NON_ALNUM_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", no_punct).strip()


def _difflib_fuzzy_contains(needle: str, haystack: str, *, threshold: int) -> bool:
    """Best-effort fuzzy quote containment when rapidfuzz is unavailable.

    Safety: restrict to anchored windows so we don't accept unrelated matches.
    """
    if not needle or not haystack:
        return False
    tokens = needle.split()
    if len(tokens) < 6:
        return False

    anchors = [" ".join(tokens[:3]), " ".join(tokens[-3:])]
    target_len = len(needle)
    threshold_ratio = max(0.0, min(1.0, float(threshold) / 100.0))

    for anchor in anchors:
        start = 0
        while True:
            idx = haystack.find(anchor, start)
            if idx == -1:
                break
            # Window around the anchor with a modest margin; keep compute bounded.
            w_start = max(0, idx - 240)
            w_end = min(len(haystack), idx + target_len + 240)
            window = haystack[w_start:w_end]
            if not window:
                start = idx + 1
                continue
            ratio = difflib.SequenceMatcher(None, needle, window).ratio()
            if ratio >= threshold_ratio:
                return True
            start = idx + 1
    return False


def validate_proposal(
    proposal: Any,
    raw_note_text: str,
    *,
    extraction_text: str | None = None,
    max_patch_ops: int | None = None,
) -> tuple[bool, str]:
    """Return (is_valid, reason)."""

    quote = getattr(proposal, "evidence_quote", "")
    if not isinstance(quote, str) or not quote.strip():
        return False, "Missing evidence quote"
    quote = quote.strip()

    if extraction_text is not None and extraction_text.strip():
        text = extraction_text
        text_label = "focused procedure text"
    else:
        text = raw_note_text or ""
        text_label = "raw note text"
    if quote not in text:
        normalized_quote = _normalize_whitespace(quote)
        normalized_text = _normalize_whitespace(text)
        if normalized_quote and normalized_quote in normalized_text:
            pass
        else:
            alnum_quote = _normalize_alnum(quote)
            alnum_text = _normalize_alnum(text)
            if alnum_quote and alnum_quote in alnum_text:
                pass
            else:
                threshold = _env_int("REGISTRY_SELF_CORRECT_QUOTE_FUZZY_THRESHOLD", 90)
                min_len = _env_int("REGISTRY_SELF_CORRECT_QUOTE_FUZZY_MIN_LEN", 24)
                if (
                    callable(partial_ratio)
                    and alnum_quote
                    and len(alnum_quote) >= min_len
                    and partial_ratio(alnum_quote, alnum_text) >= threshold
                ):
                    pass
                elif alnum_quote and len(alnum_quote) >= min_len and _difflib_fuzzy_contains(
                    alnum_quote, alnum_text, threshold=threshold
                ):
                    pass
                else:
                    return False, f"Quote not found verbatim in {text_label}: '{quote[:50]}...'"

    patches = getattr(proposal, "json_patch", [])
    if not isinstance(patches, list) or not patches:
        return False, "Empty patch"

    _canonicalize_elastography_root_patch_ops(patches)

    if max_patch_ops is None:
        max_patch_ops = _env_int("REGISTRY_SELF_CORRECT_MAX_PATCH_OPS", 5)

    if len(patches) > max_patch_ops:
        return False, f"Patch too large: {len(patches)} ops (max {max_patch_ops})"

    allowed_paths, allowed_prefixes = _allowed_paths_from_env(
        default_paths=ALLOWED_PATHS,
        default_prefixes=ALLOWED_PATH_PREFIXES,
    )

    for op in patches:
        if not isinstance(op, dict):
            return False, "Patch operation must be an object"

        path = op.get("path")
        if isinstance(path, str):
            canonical = _canonicalize_patch_path(path)
            if canonical != path:
                op["path"] = canonical
                path = canonical
        if not _path_allowed(path, allowed_paths, allowed_prefixes):
            return False, f"Path not allowed: {path}"

        verb = op.get("op")
        if verb not in ("add", "replace"):
            return False, f"Op not allowed: {verb}"

    return True, "Valid"


def _allowed_paths_from_env(
    *,
    default_paths: set[str],
    default_prefixes: set[str],
) -> tuple[set[str], set[str]]:
    raw = os.getenv("REGISTRY_SELF_CORRECT_ALLOWLIST", "")
    if not raw.strip():
        return set(default_paths), set(default_prefixes)

    parsed_paths: set[str] = set()
    parsed_prefixes: set[str] = set()
    for entry in raw.split(","):
        cleaned = entry.strip()
        if not cleaned:
            continue
        if cleaned.endswith("/*"):
            prefix = cleaned[:-2].rstrip("/")
            if prefix:
                parsed_prefixes.add(prefix)
            continue
        parsed_paths.add(cleaned)

    if not parsed_paths and not parsed_prefixes:
        return set(default_paths), set(default_prefixes)
    return parsed_paths, parsed_prefixes


def _path_allowed(path: object, allowed_paths: set[str], allowed_prefixes: set[str]) -> bool:
    if not isinstance(path, str):
        return False
    if path in allowed_paths:
        return True
    for prefix in allowed_prefixes:
        if path == prefix or path.startswith(f"{prefix}/"):
            return True
    return False


def _canonicalize_patch_path(path: str) -> str:
    if not path.startswith("/"):
        return path
    for alias_prefix, canonical_prefix in _PATCH_PATH_ALIASES.items():
        if path == alias_prefix or path.startswith(f"{alias_prefix}/"):
            suffix = path[len(alias_prefix):]
            return f"{canonical_prefix}{suffix}"
    return path


def _canonicalize_elastography_root_patch_ops(patches: list[dict[str, Any]]) -> None:
    """Rewrite object-level /procedures_performed/ebus_elastography patches into canonical fields.

    Some proposals treat ebus_elastography as its own object; the registry schema stores
    elastography under procedures_performed.linear_ebus.*.
    """
    expanded: list[dict[str, Any]] = []

    for op in patches:
        path = op.get("path")
        if path != _EBUS_ELASTOGRAPHY_ROOT:
            expanded.append(op)
            continue

        value = op.get("value")
        verb = op.get("op")
        verb_out = verb if verb in ("add", "replace") else "add"

        rewritten: list[dict[str, Any]] = []
        performed: bool | None = None
        pattern: str | None = None

        if isinstance(value, bool):
            performed = value
        elif isinstance(value, dict):
            raw_performed = value.get("performed")
            if isinstance(raw_performed, bool):
                performed = raw_performed
            raw_pattern = value.get("pattern")
            if isinstance(raw_pattern, str) and raw_pattern.strip():
                pattern = raw_pattern.strip()
            raw_method = value.get("method")
            if pattern is None and isinstance(raw_method, str) and raw_method.strip():
                pattern = raw_method.strip()
        elif isinstance(value, str) and value.strip():
            performed = True
            pattern = value.strip()

        if performed is not None:
            rewritten.append(
                {
                    "op": verb_out,
                    "path": f"{_EBUS_ELASTOGRAPHY_ROOT_CANONICAL}/elastography_used",
                    "value": performed,
                }
            )

        if pattern is not None:
            # If a pattern/method is provided without performed, treat as performed.
            if performed is None:
                rewritten.append(
                    {
                        "op": verb_out,
                        "path": f"{_EBUS_ELASTOGRAPHY_ROOT_CANONICAL}/elastography_used",
                        "value": True,
                    }
                )
            rewritten.append(
                {
                    "op": verb_out,
                    "path": f"{_EBUS_ELASTOGRAPHY_ROOT_CANONICAL}/elastography_pattern",
                    "value": pattern,
                }
            )

        expanded.extend(rewritten or [op])

    patches[:] = expanded


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


__all__ = ["ALLOWED_PATHS", "ALLOWED_PATH_PREFIXES", "validate_proposal"]
