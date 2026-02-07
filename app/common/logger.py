"""Standardized logging configuration for the Procedure Suite."""

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configures and returns a logger with Rich formatting."""
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(level.upper())
    
    console = Console(stderr=True)
    handler = RichHandler(console=console, show_time=False, show_path=False)
    
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one."""
    return setup_logger(name)
