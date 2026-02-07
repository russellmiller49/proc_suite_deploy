import re
from typing import Iterable, Tuple

from app.phi.adapters.phi_redactor_hybrid import (
    ANATOMICAL_TERMS,
    CLINICAL_ALLOW_LIST,
    DEVICE_MANUFACTURERS,
    PROTECTED_DEVICE_NAMES,
)

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_LN_STATION_RE = re.compile(r"^\d{1,2}[lr](?:[is])?$")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[a-z0-9]+\s+(street|st|road|rd|ave|avenue|blvd|boulevard|ln|lane|dr|drive)\b"
)

LN_CONTEXT_WORDS = {"station", "stations", "nodes", "node", "sampled", "ln", "ebus", "tbna"}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _NORMALIZE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_set(values: Iterable[str]) -> set[str]:
    return {normalize(value) for value in values if value}


ANATOMICAL_TERMS_NORM = _normalize_set(ANATOMICAL_TERMS)
DEVICE_MANUFACTURERS_NORM = _normalize_set(DEVICE_MANUFACTURERS)
PROTECTED_DEVICE_NAMES_NORM = _normalize_set(PROTECTED_DEVICE_NAMES)
CLINICAL_ALLOW_LIST_NORM = _normalize_set(CLINICAL_ALLOW_LIST)

PROTECTED_DEVICE_TERMS = DEVICE_MANUFACTURERS_NORM | PROTECTED_DEVICE_NAMES_NORM
PROTECTED_GEO_TERMS = (
    ANATOMICAL_TERMS_NORM
    | DEVICE_MANUFACTURERS_NORM
    | PROTECTED_DEVICE_NAMES_NORM
    | CLINICAL_ALLOW_LIST_NORM
)
PROTECTED_PERSON_TERMS = ANATOMICAL_TERMS_NORM | DEVICE_MANUFACTURERS_NORM | PROTECTED_DEVICE_NAMES_NORM


def reconstruct_wordpiece(tokens: list[str], start_idx: int) -> Tuple[str, int]:
    if start_idx >= len(tokens):
        return "", start_idx
    word = tokens[start_idx]
    end_idx = start_idx
    while end_idx + 1 < len(tokens) and tokens[end_idx + 1].startswith("##"):
        word += tokens[end_idx + 1][2:]
        end_idx += 1
    return word, end_idx


def is_protected_anatomy_phrase(text: str) -> bool:
    norm = normalize(text)
    if not norm:
        return False
    if norm in ANATOMICAL_TERMS_NORM:
        return True
    parts = norm.split()
    return "lobe" in parts or "segment" in parts or "bronchus" in parts or "bronchi" in parts


def is_ln_station(text: str, context_tokens: list[str] | None = None) -> bool:
    norm = normalize(text).replace(" ", "")
    if _LN_STATION_RE.match(norm):
        return True
    if norm == "7" and context_tokens:
        context = {normalize(tok) for tok in context_tokens}
        return any(word in context for word in LN_CONTEXT_WORDS)
    return False


def is_protected_device(text: str) -> bool:
    return normalize(text) in PROTECTED_DEVICE_TERMS


def looks_like_real_address(line_text: str) -> bool:
    return bool(_ADDRESS_RE.search(normalize(line_text)))
