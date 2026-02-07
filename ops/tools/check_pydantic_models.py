#!/usr/bin/env python3
"""
Smoke-test imports for core Pydantic models and key services.

This is intentionally simple: if any import fails, the script exits non-zero
so `make validate-schemas` / CI will catch it.
"""

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODULES = [
    # Clinical schemas
    "proc_schemas.clinical.airway",
    "proc_schemas.clinical.pleural",

    # Reasoning + coding lifecycle models
    "proc_schemas.reasoning",
    "proc_schemas.coding",

    # Registry models
    "proc_schemas.registry.ip_v2",
    "proc_schemas.registry.ip_v3",

    # Reporter agents contracts
    "app.agents.contracts",

    # Coder application (pulls in smart_hybrid policy etc.)
    "app.coder.application.coding_service",
    "app.coder.application.smart_hybrid_policy",
]


def main() -> int:
    failed = False

    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"Imported OK: {mod}")
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"FAILED to import {mod}: {exc}", file=sys.stderr)

    if failed:
        print("One or more modules failed to import.", file=sys.stderr)
        return 1

    print("All Pydantic model imports OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
