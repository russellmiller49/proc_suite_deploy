"""Shared warning message helpers for extraction heuristics."""

from __future__ import annotations


def nav_target_parsed(count: int) -> str:
    return f"NAV_TARGET_HEURISTIC: parsed {count} navigation target(s) from target text"


def nav_target_added(count: int) -> str:
    return f"NAV_TARGET_HEURISTIC: added {count} navigation target(s) from text"


def cryobiopsy_site_added(count: int) -> str:
    return f"CRYOBIOPSY_SITE_HEURISTIC: added {count} cryobiopsy site(s) from text"


def ebus_station_detail_parsed(count: int) -> str:
    suffix = "y" if count == 1 else "ies"
    return f"EBUS_STATION_DETAIL_HEURISTIC: parsed {count} station detail entr{suffix} from text"


def cao_detail_parsed(count: int) -> str:
    suffix = "y" if count == 1 else "ies"
    return f"CAO_DETAIL_HEURISTIC: parsed {count} CAO site entr{suffix} from text"


__all__ = [
    "nav_target_parsed",
    "nav_target_added",
    "cryobiopsy_site_added",
    "ebus_station_detail_parsed",
    "cao_detail_parsed",
]
