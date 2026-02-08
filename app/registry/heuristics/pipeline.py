"""Composable heuristic pipeline primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from app.registry.schema import RegistryRecord


class RecordHeuristic(Protocol):
    def apply(self, note_text: str, record: RegistryRecord) -> tuple[RegistryRecord, list[str]]:
        ...


@dataclass(frozen=True)
class FunctionHeuristic:
    """Adapt a plain callable to the RecordHeuristic protocol."""

    fn: Callable[[str, RegistryRecord], tuple[RegistryRecord, list[str]]]

    def apply(self, note_text: str, record: RegistryRecord) -> tuple[RegistryRecord, list[str]]:
        return self.fn(note_text, record)


def apply_heuristics(
    *,
    note_text: str,
    record: RegistryRecord,
    heuristics: Iterable[RecordHeuristic],
) -> tuple[RegistryRecord, list[str]]:
    """Apply heuristics in-order and accumulate warnings."""
    current = record
    warnings: list[str] = []
    for heuristic in heuristics:
        current, emitted = heuristic.apply(note_text, current)
        if emitted:
            warnings.extend(emitted)
    return current, warnings


__all__ = ["RecordHeuristic", "FunctionHeuristic", "apply_heuristics"]
