"""Mapping from EBUS node evidence to CPT code candidates.

This module provides config-driven EBUS station counting for CPT code selection.
Station counting uses an allow-list and alias resolution from proc_kb/ebus_config.yaml.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Set

from app.coder.types import CodeCandidate, EBUSNodeEvidence
from app.registry.ebus_config import get_default_ebus_config


# Load config once at module import for performance
_EBUS_CONFIG: Dict[str, Any] = {}


def _get_ebus_config() -> Dict[str, Any]:
    """Lazy-load EBUS config on first use."""
    global _EBUS_CONFIG
    if not _EBUS_CONFIG:
        try:
            _EBUS_CONFIG = get_default_ebus_config()
        except FileNotFoundError:
            # Fallback to permissive defaults if config not found
            _EBUS_CONFIG = {
                "target_action": "Sampling",
                "valid_stations": set(),  # Empty = accept all
                "aliases": {},
            }
    return _EBUS_CONFIG


def _normalize_station(station: str) -> str:
    """Normalize station name to uppercase."""
    return station.strip().upper()


def _normalize_method(method: str | None) -> str:
    """Normalize method name to lowercase."""
    return (method or "").strip().lower()


def _count_sampled_nodes(
    evidence: Iterable[EBUSNodeEvidence],
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Count unique sampled EBUS stations using config-driven allow list.

    IMPORTANT: This counts UNIQUE stations only, not unique (station, method) pairs.
    Per CPT guidelines:
    - 31652: 1-2 nodal stations sampled
    - 31653: 3+ nodal stations sampled

    Multiple passes or different needle types at the SAME station still count as ONE station.

    Station counting is config-driven:
    - Only entries with action matching config['target_action'] are considered
    - Station labels are normalized (uppercase, stripped)
    - Aliases are resolved via config['aliases'] (e.g., "SUBCARINAL" -> "7")
    - Only stations in config['valid_stations'] are counted (if set is non-empty)
    - Unknown/invalid stations are ignored

    Args:
        evidence: Iterable of EBUSNodeEvidence entries
        config: Optional config dict. Uses default EBUS config if not provided.
            Expected keys: target_action, valid_stations, aliases

    Returns:
        Count of unique valid sampled stations
    """
    cfg = config if config is not None else _get_ebus_config()

    target_action: str = cfg.get("target_action", "Sampling")
    valid_stations: Set[str] = cfg.get("valid_stations", set())
    aliases: Dict[str, str] = cfg.get("aliases", {})

    sampled_stations: Set[str] = set()

    for entry in evidence:
        # Only count sampling actions
        if getattr(entry, "action", None) != target_action:
            continue

        raw_station = str(getattr(entry, "station", "")).strip().upper()
        if not raw_station:
            continue

        # Resolve alias (e.g., "SUBCARINAL" -> "7")
        station_code = aliases.get(raw_station, raw_station)

        # Only count stations in the allow list (if allow list is non-empty)
        if valid_stations and station_code not in valid_stations:
            # Station not in allow list - skip it
            continue

        sampled_stations.add(station_code)

    return len(sampled_stations)


def ebus_nodes_to_candidates(
    evidence: Sequence[EBUSNodeEvidence],
    config: Optional[Dict[str, Any]] = None,
) -> list[CodeCandidate]:
    """Map EBUS node sampling counts to CPT code candidates.

    Args:
        evidence: Sequence of EBUSNodeEvidence entries
        config: Optional config dict for station counting. Uses default if not provided.

    Returns:
        List containing a single CodeCandidate (31652 or 31653) based on station count,
        or empty list if no stations were sampled.
    """
    sampled_count = _count_sampled_nodes(evidence, config)
    if sampled_count == 0:
        return []

    if 1 <= sampled_count <= 2:
        code = "31652"
    else:
        code = "31653"

    reason = f"ebus_nodes:sampled_count={sampled_count}"
    return [CodeCandidate(code=code, confidence=0.9, reason=reason)]


def get_sampled_station_list(
    evidence: Iterable[EBUSNodeEvidence],
    config: Optional[Dict[str, Any]] = None,
) -> list[str]:
    """Get list of unique sampled station codes (after alias resolution).

    Useful for populating registry fields like ebus_stations_sampled.

    Args:
        evidence: Iterable of EBUSNodeEvidence entries
        config: Optional config dict. Uses default EBUS config if not provided.

    Returns:
        Sorted list of unique station codes that were sampled
    """
    cfg = config if config is not None else _get_ebus_config()

    target_action: str = cfg.get("target_action", "Sampling")
    valid_stations: Set[str] = cfg.get("valid_stations", set())
    aliases: Dict[str, str] = cfg.get("aliases", {})

    sampled_stations: Set[str] = set()

    for entry in evidence:
        if getattr(entry, "action", None) != target_action:
            continue

        raw_station = str(getattr(entry, "station", "")).strip().upper()
        if not raw_station:
            continue

        station_code = aliases.get(raw_station, raw_station)

        if valid_stations and station_code not in valid_stations:
            continue

        sampled_stations.add(station_code)

    return sorted(sampled_stations)


__all__ = [
    "ebus_nodes_to_candidates",
    "_count_sampled_nodes",
    "get_sampled_station_list",
]
