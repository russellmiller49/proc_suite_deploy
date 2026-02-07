"""EBUS Configuration Loader.

This module provides utilities for loading and normalizing EBUS coding
configuration. The configuration defines:
- Valid EBUS station codes (allow-list)
- Station name aliases for normalization
- Target action for counting (e.g., "Sampling")

Configuration can be stored in YAML or JSON format.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # YAML is optional; JSON configs still work


# Default path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "proc_kb" / "ebus_config.yaml"


def load_ebus_config(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load and normalize EBUS coding configuration.

    Args:
        path: Path to config file. Defaults to proc_kb/ebus_config.yaml

    Returns:
        Dict containing:
            - target_action: str - Action to trigger counting (default "Sampling")
            - valid_stations: Set[str] - Uppercase station codes
            - aliases: Dict[str, str] - Uppercase key->value alias mappings

    Raises:
        FileNotFoundError: If config file doesn't exist
        RuntimeError: If YAML file specified but PyYAML not installed
    """
    p = Path(path) if path else DEFAULT_CONFIG_PATH

    if not p.exists():
        raise FileNotFoundError(f"EBUS config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is required to load YAML EBUS config; "
                    "either install it or use JSON for ebus_config."
                )
            raw_config = yaml.safe_load(f)
        else:
            raw_config = json.load(f)

    settings = raw_config.get("settings", {}) or {}

    # Normalize valid_stations to uppercase set
    valid_stations: Set[str] = {
        s.strip().upper() for s in settings.get("valid_stations", []) if s
    }

    # Normalize aliases keys and values to uppercase
    raw_aliases: Dict[str, str] = settings.get("aliases", {}) or {}
    aliases: Dict[str, str] = {
        k.strip().upper(): v.strip().upper()
        for k, v in raw_aliases.items()
        if k and v
    }

    return {
        "target_action": settings.get("target_action", "Sampling"),
        "valid_stations": valid_stations,
        "aliases": aliases,
    }


@lru_cache(maxsize=1)
def get_default_ebus_config() -> Dict[str, Any]:
    """Get the default EBUS config (cached).

    This function caches the config so it's only loaded once per process.

    Returns:
        The normalized EBUS configuration dict.

    Raises:
        FileNotFoundError: If default config file doesn't exist
    """
    return load_ebus_config()


def resolve_station_alias(
    station: str,
    aliases: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve a station name to its canonical code using aliases.

    Args:
        station: Raw station name/code
        aliases: Optional alias mapping. Uses default config if not provided.

    Returns:
        Canonical station code (uppercase)
    """
    normalized = station.strip().upper()
    if aliases is None:
        config = get_default_ebus_config()
        aliases = config["aliases"]
    return aliases.get(normalized, normalized)


def is_valid_station(
    station: str,
    valid_stations: Optional[Set[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> bool:
    """Check if a station (after alias resolution) is in the valid set.

    Args:
        station: Raw station name/code
        valid_stations: Optional valid station set. Uses default config if not provided.
        aliases: Optional alias mapping. Uses default config if not provided.

    Returns:
        True if station is valid, False otherwise
    """
    if valid_stations is None or aliases is None:
        config = get_default_ebus_config()
        if valid_stations is None:
            valid_stations = config["valid_stations"]
        if aliases is None:
            aliases = config["aliases"]

    resolved = resolve_station_alias(station, aliases)
    return resolved in valid_stations


__all__ = [
    "load_ebus_config",
    "get_default_ebus_config",
    "resolve_station_alias",
    "is_valid_station",
    "DEFAULT_CONFIG_PATH",
]
