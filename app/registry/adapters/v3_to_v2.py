"""Compatibility shim for V3 (event-log) → V2 RegistryRecord projection."""

from __future__ import annotations

from app.registry.schema.adapters.v3_to_v2 import project_v3_to_v2

__all__ = ["project_v3_to_v2"]

