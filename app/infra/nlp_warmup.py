"""NLP model warmup and resource loading.

This module handles loading heavy NLP resources (spaCy models, UMLS linker,
sectionizer) at application startup to avoid cold-start latency on first request.

Usage:
    from app.infra.nlp_warmup import warm_heavy_resources, should_skip_warmup

    @app.on_event("startup")
    async def startup():
        if not should_skip_warmup():
            try:
                await warm_heavy_resources()
            except Exception:
                logger.exception("Heavy NLP warmup failed, starting in degraded mode")

Environment Variables:
    PROCSUITE_SKIP_WARMUP / SKIP_WARMUP: Set to "1", "true", or "yes" to skip warmup
    PROCSUITE_SPACY_MODEL: SpaCy model name (default: en_core_sci_sm)
    ENABLE_UMLS_LINKER: Set to "false", "0", or "no" to skip UMLS linker (saves memory)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any
import warnings

_logger = logging.getLogger(__name__)

# Global flag to track NLP warmup status (useful for degraded mode)
_nlp_warmup_successful: bool = False


def should_skip_warmup() -> bool:
    """Check if warmup should be skipped based on environment variables.

    Returns:
        True if warmup should be skipped, False otherwise.
    """
    skip_warmup = os.getenv("SKIP_WARMUP") or os.getenv("PROCSUITE_SKIP_WARMUP") or ""
    return skip_warmup.lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def get_spacy_model() -> Any:
    """Return the spaCy/scispaCy model used for UMLS linking and NER.

    The model is loaded once and cached. The model name is configurable via
    the PROCSUITE_SPACY_MODEL environment variable (default: en_core_sci_sm).

    Returns:
        The loaded spaCy model, or None if spaCy is not available.
    """
    try:
        import spacy
    except ImportError:
        _logger.warning("spaCy not available - NLP features will be disabled")
        return None

    model_name = os.getenv("PROCSUITE_SPACY_MODEL", "en_core_sci_sm")
    try:
        _logger.info("Loading spaCy model: %s", model_name)
        with warnings.catch_warnings():
            # spaCy/scispaCy may define Pydantic models with `model_*` fields which
            # trigger warnings under Pydantic v2 protected namespaces.
            warnings.filterwarnings(
                "ignore",
                message=r'Field "model_.*" has conflict with protected namespace "model_"',
                category=UserWarning,
                module=r"pydantic\._internal\._fields",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Possible set union at position \d+",
                category=FutureWarning,
                module=r"spacy\.language",
            )
            nlp = spacy.load(model_name)
        _logger.info("spaCy model %s loaded successfully", model_name)
        return nlp
    except OSError:
        _logger.warning(
            "spaCy model '%s' not found. Install with: pip install %s",
            model_name,
            model_name.replace("_", "-"),
        )
        return None


@lru_cache(maxsize=1)
def get_sectionizer() -> Any:
    """Return a cached SectionizerService instance.

    The sectionizer uses medspaCy under the hood and is initialized once.

    Returns:
        SectionizerService instance, or None if initialization fails.
    """
    try:
        from app.common.sectionizer import SectionizerService

        _logger.info("Initializing SectionizerService")
        sectionizer = SectionizerService()
        _logger.info("SectionizerService initialized successfully")
        return sectionizer
    except Exception as exc:
        _logger.warning("Failed to initialize SectionizerService: %s", exc)
        return None


def _do_heavy_warmup() -> None:
    """Perform the actual heavy NLP warmup (synchronous helper).

    This loads spaCy models, sectionizer, and UMLS linker to ensure
    they're ready for first requests.
    """
    global _nlp_warmup_successful

    _logger.info("Warming up heavy NLP resources...")

    # Load spaCy model (used by proc_nlp and app.common.umls_linking)
    nlp = get_spacy_model()
    if nlp:
        # Warm up the pipeline with a small text to ensure all components are ready
        _ = nlp("Warmup text for pipeline initialization.")

    # Initialize sectionizer (uses medspaCy)
    _ = get_sectionizer()

    # Also warm up the UMLS linker from proc_nlp if available
    # Check env var first to save memory on Railway
    enable_umls = os.getenv("ENABLE_UMLS_LINKER", "true").lower() in ("true", "1", "yes")

    if enable_umls:
        try:
            from proc_nlp.umls_linker import _load_model

            model_name = os.getenv("PROCSUITE_SPACY_MODEL", "en_core_sci_sm")
            _logger.info("Warming up UMLS linker with model: %s", model_name)
            _load_model(model_name)
            _logger.info("UMLS linker warmed up successfully")
        except Exception as exc:
            _logger.warning("UMLS linker warmup skipped: %s", exc)
    else:
        _logger.info("UMLS linker warmup skipped (ENABLE_UMLS_LINKER=false)")

    _logger.info("Heavy NLP resources warmed up successfully")
    _nlp_warmup_successful = True


async def warm_heavy_resources() -> None:
    """Preload heavy NLP models (async wrapper for startup hook).

    This is the main entry point for the FastAPI startup hook. It wraps
    the synchronous warmup logic and handles errors gracefully.

    Raises:
        Exception: If warmup fails (app should handle gracefully).
    """
    _do_heavy_warmup()


def warm_heavy_resources_sync() -> None:
    """Synchronous warmup entry point (for background threads/executors)."""
    _do_heavy_warmup()


def is_nlp_warmed() -> bool:
    """Check if NLP warmup completed successfully.

    Useful for endpoints that want to return 503 if NLP is unavailable.

    Returns:
        True if warmup completed successfully, False otherwise.
    """
    return _nlp_warmup_successful


def reset_warmup_state() -> None:
    """Reset the warmup state (useful for testing).

    This clears the cached models and warmup flag so they can be
    reloaded on the next call.
    """
    global _nlp_warmup_successful
    _nlp_warmup_successful = False
    get_spacy_model.cache_clear()
    get_sectionizer.cache_clear()


__all__ = [
    "should_skip_warmup",
    "warm_heavy_resources",
    "warm_heavy_resources_sync",
    "is_nlp_warmed",
    "get_spacy_model",
    "get_sectionizer",
    "reset_warmup_state",
]
