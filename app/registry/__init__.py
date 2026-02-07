"""Registry extraction pipeline.

Keep this package import-light.

Many scripts import `app.registry.*` utilities (e.g., boolean mappers) without
needing the full LLM extractor stack. Importing `RegistryEngine` eagerly pulls in
LLM/config dependencies, which breaks running leaf scripts directly (e.g.
`python -m ml.lib.ml_coder.data_prep`) in environments where `config` isn't on
`PYTHONPATH`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import RegistryRecord

if TYPE_CHECKING:  # pragma: no cover
    from .engine import RegistryEngine as RegistryEngine


def __getattr__(name: str):
    # Lazy import to avoid side effects during package import.
    if name == "RegistryEngine":
        from .engine import RegistryEngine as _RegistryEngine

        return _RegistryEngine
    raise AttributeError(name)


__all__ = ["RegistryRecord", "RegistryEngine"]
