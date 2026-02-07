"""REST API surface for the Procedure Suite package.

Keep this module import-light: core pipeline code imports submodules under
`app.api.*` (e.g., normalization helpers) in non-API contexts such as
CLI smoke tests. Importing the FastAPI app pulls in optional infrastructure
dependencies (DB drivers, observability, etc.) that are not required for
pure extraction/coding.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "app":
        from .fastapi_app import app

        return app
    raise AttributeError(name)


__all__ = ["app"]
