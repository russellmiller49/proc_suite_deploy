"""Threshold configuration for ML coder case difficulty classification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CaseDifficulty(str, Enum):
    """Ternary classification of case difficulty based on ML confidence."""

    HIGH_CONF = "high_confidence"
    GRAY_ZONE = "gray_zone"
    LOW_CONF = "low_confidence"


@dataclass
class Thresholds:
    """
    Threshold configuration for case difficulty classification.

    Attributes:
        upper: Global upper threshold for HIGH_CONF classification
        lower: Global lower threshold (below this = LOW_CONF)
        per_code: Per-code upper threshold overrides for specific CPT codes
    """

    upper: float = 0.7
    lower: float = 0.4
    per_code: dict[str, float] = field(default_factory=dict)

    def upper_for(self, code: str) -> float:
        """Get upper threshold for a specific code (uses per-code override if available)."""
        return self.per_code.get(code, self.upper)

    def lower_for(self, code: str) -> float:
        """Get lower threshold for a specific code (currently global)."""
        return self.lower

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Thresholds:
        """Create Thresholds from a dictionary."""
        return cls(
            upper=data.get("upper", 0.7),
            lower=data.get("lower", 0.4),
            per_code=data.get("per_code", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Thresholds:
        """Load thresholds from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "upper": self.upper,
            "lower": self.lower,
            "per_code": self.per_code,
        }

    def to_json(self, path: str | Path) -> None:
        """Save thresholds to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Default thresholds path
THRESHOLDS_PATH = Path("data/models/ml_coder_thresholds_v1.json")


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """
    Load thresholds from file or return defaults.

    Args:
        path: Path to thresholds JSON. If None, uses THRESHOLDS_PATH.
              If file doesn't exist, returns default thresholds.
    """
    path = Path(path) if path else THRESHOLDS_PATH
    if path.exists():
        return Thresholds.from_json(path)
    return Thresholds()


__all__ = [
    "CaseDifficulty",
    "Thresholds",
    "load_thresholds",
    "THRESHOLDS_PATH",
]
