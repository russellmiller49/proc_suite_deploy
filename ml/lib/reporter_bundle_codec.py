"""Bundle key codec utilities for reporter prompt-to-bundle training.

The codec is intentionally narrow and path-aware so we only compress known
ProcedureBundle keys. Unknown keys are preserved verbatim.
"""

from __future__ import annotations

import json
from math import ceil
from typing import Any

CODEC_MARKER_KEY = "__codec__"
CODEC_VERSION_V1 = "reporter_bundle_v1"

# Root-level ProcedureBundle keys.
_ROOT_ENCODE_MAP = {
    "patient": "pt",
    "encounter": "enc",
    "procedures": "pr",
    "sedation": "sd",
    "anesthesia": "an",
    "pre_anesthesia": "pa",
    "indication_text": "it",
    "preop_diagnosis_text": "pre",
    "postop_diagnosis_text": "post",
    "impression_plan": "ip",
    "estimated_blood_loss": "ebl",
    "complications_text": "comp",
    "specimens_text": "spec",
    "free_text_hint": "fth",
    "acknowledged_omissions": "ao",
    "addons": "ad",
}

_PATIENT_ENCODE_MAP = {
    "name": "n",
    "age": "a",
    "sex": "s",
    "patient_id": "pid",
    "mrn": "m",
}

_ENCOUNTER_ENCODE_MAP = {
    "date": "d",
    "encounter_id": "eid",
    "location": "loc",
    "referred_physician": "ref",
    "attending": "att",
    "assistant": "asst",
}

_SEDATION_ENCODE_MAP = {
    "type": "t",
    "description": "desc",
}

_PROCEDURE_ENCODE_MAP = {
    "proc_type": "t",
    "schema_id": "sid",
    "proc_id": "id",
    "data": "d",
    "cpt_candidates": "cpt",
    "sequence": "seq",
}


def _reverse_map(mapping: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in mapping.items()}


_ROOT_DECODE_MAP = _reverse_map(_ROOT_ENCODE_MAP)
_PATIENT_DECODE_MAP = _reverse_map(_PATIENT_ENCODE_MAP)
_ENCOUNTER_DECODE_MAP = _reverse_map(_ENCOUNTER_ENCODE_MAP)
_SEDATION_DECODE_MAP = _reverse_map(_SEDATION_ENCODE_MAP)
_PROCEDURE_DECODE_MAP = _reverse_map(_PROCEDURE_ENCODE_MAP)


def _map_for_path(path: tuple[Any, ...], *, encode: bool) -> dict[str, str]:
    """Return path-specific key map for encode/decode."""
    if not path:
        return _ROOT_ENCODE_MAP if encode else _ROOT_DECODE_MAP

    head = path[0]

    if head in ("patient", "pt") and len(path) == 1:
        return _PATIENT_ENCODE_MAP if encode else _PATIENT_DECODE_MAP

    if head in ("encounter", "enc") and len(path) == 1:
        return _ENCOUNTER_ENCODE_MAP if encode else _ENCOUNTER_DECODE_MAP

    if head in ("sedation", "sd") and len(path) == 1:
        return _SEDATION_ENCODE_MAP if encode else _SEDATION_DECODE_MAP

    if head in ("anesthesia", "an") and len(path) == 1:
        return _SEDATION_ENCODE_MAP if encode else _SEDATION_DECODE_MAP

    if head in ("procedures", "pr") and len(path) == 2 and isinstance(path[1], int):
        return _PROCEDURE_ENCODE_MAP if encode else _PROCEDURE_DECODE_MAP

    return {}


def _transform_keys(value: Any, *, path: tuple[Any, ...], encode: bool) -> Any:
    if isinstance(value, dict):
        key_map = _map_for_path(path, encode=encode)
        out: dict[str, Any] = {}
        for key, child in value.items():
            mapped_key = key_map.get(key, key)
            if mapped_key in out:
                raise ValueError(f"Codec collision at path={path!r}, key={mapped_key!r}")
            out[mapped_key] = _transform_keys(
                child,
                path=path + (mapped_key,),
                encode=encode,
            )
        return out

    if isinstance(value, list):
        return [
            _transform_keys(item, path=path + (idx,), encode=encode)
            for idx, item in enumerate(value)
        ]

    return value


def is_encoded_bundle_v1(payload: dict[str, Any]) -> bool:
    return payload.get(CODEC_MARKER_KEY) == CODEC_VERSION_V1


def encode_bundle_keys_v1(payload: dict[str, Any]) -> dict[str, Any]:
    """Encode known ProcedureBundle keys into compact aliases.

    Adds a top-level codec marker so decoding can be safely gated.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    transformed = _transform_keys(payload, path=(), encode=True)
    if CODEC_MARKER_KEY in transformed:
        raise ValueError(f"payload already contains reserved marker key {CODEC_MARKER_KEY!r}")
    transformed[CODEC_MARKER_KEY] = CODEC_VERSION_V1
    return transformed


def decode_bundle_keys_v1(payload: dict[str, Any]) -> dict[str, Any]:
    """Decode compact keys if payload carries v1 marker.

    If marker is absent, payload is returned unchanged.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    if not is_encoded_bundle_v1(payload):
        return payload

    raw = dict(payload)
    raw.pop(CODEC_MARKER_KEY, None)
    return _transform_keys(raw, path=(), encode=False)


def rough_token_len(text: str) -> int:
    """Approximate token count using 4 chars/token heuristic."""
    return max(1, ceil(len(text) / 4))


def rough_token_len_for_payload(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return rough_token_len(encoded)


__all__ = [
    "CODEC_MARKER_KEY",
    "CODEC_VERSION_V1",
    "decode_bundle_keys_v1",
    "encode_bundle_keys_v1",
    "is_encoded_bundle_v1",
    "rough_token_len",
    "rough_token_len_for_payload",
]
