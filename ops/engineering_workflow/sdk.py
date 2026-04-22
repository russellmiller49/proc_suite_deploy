"""Lazy Agents SDK import helpers."""

from __future__ import annotations

import importlib


def load_agents_sdk():
    try:
        return importlib.import_module("agents")
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime path
        raise RuntimeError(
            "The Agents SDK is not installed. Install the optional 'agents' extra to enable planning and optional classifier/reporter agents."
        ) from exc

