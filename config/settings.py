"""Configuration settings using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


class KnowledgeSettings(BaseSettings):
    """Centralized configuration for local knowledge/config files.

    This is the single source of truth for file paths used across the app. It
    preserves backwards compatibility with legacy environment variables during
    the transition period.
    """

    kb_path: Path = Field(
        default=Path("data/knowledge/ip_coding_billing_v3_0.json"),
        validation_alias=AliasChoices("CODER_KB_PATH", "PSUITE_KNOWLEDGE_FILE"),
    )
    ncci_path: Path = Field(
        default=Path("data/knowledge/ncci_ptp.v1.json"),
        validation_alias="PSUITE_NCCI_FILE",
    )
    families_path: Path = Field(
        default=Path("data/knowledge/code_families.v1.json"),
        validation_alias="PSUITE_FAMILIES_FILE",
    )
    registry_schema_path: Path = Field(
        default=Path("data/knowledge/IP_Registry.json"),
        validation_alias="PSUITE_REGISTRY_SCHEMA_FILE",
    )
    addon_templates_path: Path = Field(
        default=Path("data/knowledge/ip_addon_templates_parsed.json"),
        validation_alias="PSUITE_ADDON_TEMPLATES_FILE",
    )

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _resolve_paths(self) -> "KnowledgeSettings":
        self.kb_path = _resolve_repo_path(self.kb_path)
        self.ncci_path = _resolve_repo_path(self.ncci_path)
        self.families_path = _resolve_repo_path(self.families_path)
        self.registry_schema_path = _resolve_repo_path(self.registry_schema_path)
        self.addon_templates_path = _resolve_repo_path(self.addon_templates_path)
        return self


class CoderSettings(BaseSettings):
    """Settings for the CPT coding pipeline."""

    model_version: str = "gemini-2.5-flash"
    kb_path: Path = Field(default_factory=lambda: KnowledgeSettings().kb_path)
    kb_version: str = "v3_0"
    keyword_mapping_dir: Path = Path("data/keyword_mappings")
    keyword_mapping_version: str = "v1"

    # Smart hybrid thresholds
    advisor_confidence_auto_accept: float = 0.85
    rule_confidence_low_threshold: float = 0.6
    context_window_chars: int = 200

    # CMS conversion factor (updated annually by CMS)
    # CY 2026 Medicare Physician Fee Schedule Conversion Factor (non-QP)
    # Override via CODER_CMS_CONVERSION_FACTOR environment variable
    cms_conversion_factor: float = 33.4009

    model_config = {"env_prefix": "CODER_", "protected_namespaces": ()}

    @model_validator(mode="after")
    def _resolve_paths(self) -> "CoderSettings":
        self.kb_path = _resolve_repo_path(self.kb_path)
        self.keyword_mapping_dir = _resolve_repo_path(self.keyword_mapping_dir)
        return self


class ReporterSettings(BaseSettings):
    """Settings for the reporter pipeline."""

    llm_model: str = "gemini-2.5-flash"
    max_retries: int = 3
    cache_strategy: str = "by_note_hash"
    fast_path_confidence_threshold: float = 0.95
    timeout_per_attempt_ms: int = 5000

    model_config = {"env_prefix": "REPORTER_"}


class RegistrySettings(BaseSettings):
    """Settings for registry export."""

    default_registry_version: str = "v2"
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    model_config = {"env_prefix": "REGISTRY_"}


class LLMExtractionConfig(BaseSettings):
    """Settings for LLM extraction with self-correction."""

    max_retries: int = 3
    cache_strategy: str = "by_note_hash"
    fast_path_confidence_threshold: float = 0.95
    timeout_per_attempt_ms: int = 5000

    model_config = {"env_prefix": "LLM_"}
