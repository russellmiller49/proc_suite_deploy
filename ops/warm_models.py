#!/usr/bin/env python3
"""Warm up NLP models before starting the FastAPI server.

This script pre-loads heavy NLP models (spaCy, scispaCy, medspaCy) to ensure
they are ready before the first HTTP request. This is critical for Railway
deployments where cold-start latency can cause timeouts.

Usage:
    python ops/warm_models.py

Environment Variables:
    PROCSUITE_SPACY_MODEL - spaCy model to use (default: en_core_sci_sm)
    ENABLE_UMLS_LINKER - Set to "false" to skip UMLS linker (saves ~1GB memory)

Exit codes:
    0 - All models loaded successfully
    1 - One or more models failed to load
"""

from __future__ import annotations

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Load and warm up all heavy NLP models."""
    model_name = os.getenv("PROCSUITE_SPACY_MODEL", "en_core_sci_sm")
    errors: list[str] = []

    # 1. Load spaCy model
    logger.info("Loading spaCy model: %s", model_name)
    try:
        import spacy

        nlp = spacy.load(model_name)
        # Warm up the pipeline with a test document
        doc = nlp("Patient underwent bronchoscopy for evaluation of lung nodule.")
        logger.info(
            "spaCy model loaded successfully (%d entities, %d tokens)",
            len(doc.ents),
            len(doc),
        )
    except ImportError:
        errors.append("spaCy not installed")
        logger.warning("spaCy not installed - skipping")
    except OSError as exc:
        errors.append(f"spaCy model '{model_name}' not found: {exc}")
        logger.error("spaCy model '%s' not found: %s", model_name, exc)

    # 2. Load medspaCy sectionizer
    logger.info("Initializing medspaCy sectionizer...")
    try:
        from app.common.sectionizer import SectionizerService

        sectionizer = SectionizerService()
        # Test sectionization
        sections = sectionizer.sectionize(
            "INDICATION: Lung nodule\n\nPROCEDURE: Bronchoscopy\n\nFINDINGS: Normal airways"
        )
        logger.info("medspaCy sectionizer initialized (%d sections)", len(sections))
    except ImportError as exc:
        errors.append(f"medspaCy not installed: {exc}")
        logger.warning("medspaCy not available: %s", exc)
    except Exception as exc:
        errors.append(f"Sectionizer initialization failed: {exc}")
        logger.error("Sectionizer initialization failed: %s", exc)

    # 3. Load UMLS linker (if available and enabled)
    enable_umls = os.getenv("ENABLE_UMLS_LINKER", "true").lower() in ("true", "1", "yes")
    if enable_umls:
        logger.info("Initializing UMLS linker...")
        try:
            from proc_nlp.umls_linker import _load_model, umls_link

            _load_model(model_name)
            # Test UMLS linking with a simple term
            concepts = umls_link("bronchoscopy")
            logger.info("UMLS linker initialized (%d concepts found)", len(concepts))
        except ImportError as exc:
            logger.warning("UMLS linker not available: %s", exc)
        except RuntimeError as exc:
            logger.warning("UMLS linker not available: %s", exc)
        except Exception as exc:
            # UMLS linker is optional, so we just warn
            logger.warning("UMLS linker initialization skipped: %s", exc)
    else:
        logger.info("UMLS linker skipped (ENABLE_UMLS_LINKER=false)")

    # 4. Import FastAPI app to trigger any module-level initialization
    logger.info("Importing FastAPI app...")
    try:
        from app.api.fastapi_app import app

        logger.info("FastAPI app imported successfully (version: %s)", app.version)
    except ImportError as exc:
        errors.append(f"FastAPI app import failed: {exc}")
        logger.error("FastAPI app import failed: %s", exc)
    except Exception as exc:
        errors.append(f"FastAPI app initialization failed: {exc}")
        logger.error("FastAPI app initialization failed: %s", exc)

    # Summary
    if errors:
        logger.error("Model warmup completed with %d error(s):", len(errors))
        for err in errors:
            logger.error("  - %s", err)
        return 1

    logger.info("All models warmed up successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
