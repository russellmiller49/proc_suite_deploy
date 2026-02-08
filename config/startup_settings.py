"""Startup environment validation settings."""

from __future__ import annotations

import logging

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class StartupSettings(BaseSettings):
    """Env-backed startup invariants for API boot."""

    pipeline_mode: str = Field(
        default="",
        validation_alias=AliasChoices("PROCSUITE_PIPELINE_MODE"),
    )
    coder_require_phi_review: bool = Field(
        default=False,
        validation_alias=AliasChoices("CODER_REQUIRE_PHI_REVIEW"),
    )
    procsuite_env: str = Field(default="", validation_alias=AliasChoices("PROCSUITE_ENV"))
    extraction_engine: str = Field(
        default="",
        validation_alias=AliasChoices("REGISTRY_EXTRACTION_ENGINE"),
    )
    allow_registry_engine_override: bool = Field(
        default=False,
        validation_alias=AliasChoices("PROCSUITE_ALLOW_REGISTRY_ENGINE_OVERRIDE"),
    )
    schema_version: str = Field(
        default="",
        validation_alias=AliasChoices("REGISTRY_SCHEMA_VERSION"),
    )
    auditor_source: str = Field(
        default="",
        validation_alias=AliasChoices("REGISTRY_AUDITOR_SOURCE"),
    )

    model_config = {"extra": "ignore"}

    def validate_runtime_contract(self) -> None:
        pipeline_mode = (self.pipeline_mode or "").strip().lower()
        if pipeline_mode != "extraction_first":
            if pipeline_mode == "parallel_ner":
                raise RuntimeError(
                    "PROCSUITE_PIPELINE_MODE=parallel_ner is invalid; use extraction_first."
                )
            raise RuntimeError(
                f"PROCSUITE_PIPELINE_MODE must be 'extraction_first', got '{pipeline_mode or 'unset'}'."
            )

        is_production = bool(self.coder_require_phi_review) or (
            (self.procsuite_env or "").strip().lower() == "production"
        )
        if not is_production:
            return

        extraction_engine = (self.extraction_engine or "").strip().lower()
        if extraction_engine != "parallel_ner" and not bool(self.allow_registry_engine_override):
            raise RuntimeError(
                "REGISTRY_EXTRACTION_ENGINE must be 'parallel_ner' in production "
                "(or set PROCSUITE_ALLOW_REGISTRY_ENGINE_OVERRIDE=true for an explicit override)."
            )
        if extraction_engine != "parallel_ner" and bool(self.allow_registry_engine_override):
            logging.getLogger(__name__).warning(
                "Production override enabled: REGISTRY_EXTRACTION_ENGINE=%s (expected parallel_ner).",
                extraction_engine,
            )

        schema_version = (self.schema_version or "").strip().lower()
        if schema_version != "v3":
            raise RuntimeError("REGISTRY_SCHEMA_VERSION must be 'v3' in production.")

        auditor_source = (self.auditor_source or "").strip().lower()
        if auditor_source != "raw_ml":
            raise RuntimeError("REGISTRY_AUDITOR_SOURCE must be 'raw_ml' in production.")


def validate_startup_env() -> None:
    """Validate startup env invariants."""
    StartupSettings().validate_runtime_contract()


__all__ = ["StartupSettings", "validate_startup_env"]
