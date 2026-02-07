"""Loader for static code family hierarchy configuration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from config.settings import KnowledgeSettings


@lru_cache()
def load_code_families(path: str | Path | None = None) -> Dict[str, Any]:
    """Load code family configuration."""
    config_path = Path(path) if path is not None else KnowledgeSettings().families_path
    with config_path.open(encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


__all__ = ["load_code_families"]
