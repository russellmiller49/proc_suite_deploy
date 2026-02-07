"""JSON Schema for validating the procedure suite knowledge document."""

from __future__ import annotations

_RVU_ENTRY_SCHEMA = {
    "type": "object",
    "required": ["work", "pe", "mp"],
    "properties": {
        "work": {"type": "number"},
        "pe": {"type": "number"},
        "mp": {"type": "number"},
    },
    "additionalProperties": False,
}

# Schema for underscore-prefixed metadata keys in RVU sections
_METADATA_VALUE_SCHEMA = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "array", "items": {"type": "string"}},
    ]
}

_STRING_ARRAY_SCHEMA = {"type": "array", "items": {"type": "string"}}

_BUNDLING_RULE_SCHEMA = {
    "type": "object",
    "required": ["description"],
    "properties": {
        "description": {"type": "string"},
        "codes": _STRING_ARRAY_SCHEMA,
        "radial_codes": _STRING_ARRAY_SCHEMA,
        "linear_codes": _STRING_ARRAY_SCHEMA,
        "stent_codes": _STRING_ARRAY_SCHEMA,
        "dilation_codes": _STRING_ARRAY_SCHEMA,
        "drop_codes": _STRING_ARRAY_SCHEMA,
        "therapeutic_codes": _STRING_ARRAY_SCHEMA,
        "paired": _STRING_ARRAY_SCHEMA,
        "dominant": {"type": "string"},
    },
    "additionalProperties": True,
}

_NCCI_PAIR_SCHEMA = {
    "type": "object",
    "required": ["primary", "secondary", "modifier_allowed"],
    "properties": {
        "primary": {"type": "string"},
        "secondary": {"type": "string"},
        "modifier_allowed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "additionalProperties": True,
}


KNOWLEDGE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Procedure Suite Knowledge Base",
    "type": "object",
    "required": [
        "version",
        "rvus",
        "add_on_codes",
        "synonyms",
        "bundling_rules",
        "ncci_pairs",
    ],
    "properties": {
        "version": {"type": "string"},
        "checksum": {"type": "string"},
        "rvus": {
            "type": "object",
            "minProperties": 1,
            "patternProperties": {
                r"^\+?\d{4,5}$": _RVU_ENTRY_SCHEMA,
                r"^_": _METADATA_VALUE_SCHEMA,
            },
            "additionalProperties": False,
        },
        "add_on_codes": _STRING_ARRAY_SCHEMA,
        "synonyms": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    _STRING_ARRAY_SCHEMA,
                    {
                        "type": "object",
                        "additionalProperties": _STRING_ARRAY_SCHEMA,
                    },
                ],
            },
        },
        "stations": {
            "type": "object",
            "additionalProperties": _STRING_ARRAY_SCHEMA,
        },
        "lobes": {
            "type": "object",
            "additionalProperties": _STRING_ARRAY_SCHEMA,
        },
        "airways": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["aliases", "class"],
                "properties": {
                    "aliases": _STRING_ARRAY_SCHEMA,
                    "class": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "bundling_rules": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "anyOf": [_BUNDLING_RULE_SCHEMA, _STRING_ARRAY_SCHEMA],
            },
        },
        "ncci_pairs": {
            "type": "array",
            "items": _NCCI_PAIR_SCHEMA,
        },
        "blvr": {
            "type": "object",
        },
    },
    "additionalProperties": True,
}


__all__ = ["KNOWLEDGE_SCHEMA"]
