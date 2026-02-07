#!/usr/bin/env python3
"""Patch validator / status script.

The repository previously had a large, non-executable draft `ops/tools/patch.py` that mixed
code snippets and patch instructions. That draft could not run (SyntaxError).

This script is now a *validator* you can run to confirm the intended updates exist
in the current checkout.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure repo root is on sys.path (so `import app.*` works when running as a script).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _check_import(module: str) -> None:
    importlib.import_module(module)


def main() -> int:
    failures: list[str] = []

    # New/centralized registry modules
    for mod in ("app.registry.tags", "app.registry.schema_filter"):
        try:
            _check_import(mod)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"FAILED import {mod}: {exc}")

    # Ensure auditors support warm/is_loaded
    try:
        from app.registry.audit.raw_ml_auditor import RawMLAuditor, RegistryFlagAuditor

        for cls in (RawMLAuditor, RegistryFlagAuditor):
            for meth in ("warm", "is_loaded"):
                if not hasattr(cls, meth):
                    failures.append(f"{cls.__name__} missing method {meth}()")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"FAILED import auditors: {exc}")

    if failures:
        print("Patch validation FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("Patch validation OK: registry tag/schema_filter modules and auditor helpers are present.")
    print("Optional prompt gating is available via: REGISTRY_PROMPT_FILTER_BY_FAMILY=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


