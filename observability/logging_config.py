"""Structured logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, MutableMapping


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs JSON-structured log records."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields if present
        if hasattr(record, "extra_fields"):
            log_dict.update(record.extra_fields)

        # Include exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict)


class StructuredLogger(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter that supports structured fields."""

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


_configured = False


def configure_logging(
    level: int = logging.INFO,
    structured: bool = True,
) -> None:
    """Configure logging for the application."""
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root_logger.addHandler(handler)
    _configured = True


def get_logger(name: str, **extra: Any) -> StructuredLogger:
    """Get a structured logger with optional default extra fields."""
    configure_logging()
    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger, extra)
