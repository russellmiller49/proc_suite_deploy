"""Deprecated shim forwarding to app.reporting.engine."""

from __future__ import annotations

import warnings

from app.reporting.engine import *  # noqa: F401,F403

warnings.warn(
    "app.reporter.engine is deprecated; please import from app.reporting.engine instead.",
    DeprecationWarning,
    stacklevel=2,
)
