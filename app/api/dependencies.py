"""Dependency injection factories for API endpoints.

Provides fully wired service instances for FastAPI dependency injection.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends

from config.settings import CoderSettings, KnowledgeSettings

# QA Pipeline imports
from app.api.services.qa_pipeline import (
    QAPipelineService,
    ReportingStrategy,
    SimpleReporterStrategy,
)
from app.coder.adapters.llm.gemini_advisor import (
    GeminiAdvisorAdapter,
    LLMAdvisorPort,
)
from app.coder.adapters.nlp.keyword_mapping_loader import YamlKeywordMappingRepository
from app.coder.adapters.nlp.simple_negation_detector import SimpleNegationDetector
from app.coder.adapters.persistence.csv_kb_adapter import JsonKnowledgeBaseAdapter
from app.coder.adapters.persistence.inmemory_procedure_store import (
    InMemoryProcedureStore,
)
from app.coder.application.coding_service import CodingService
from app.coder.application.smart_hybrid_policy import (
    build_hybrid_orchestrator,
)
from app.domain.coding_rules.rule_engine import RuleEngine
from app.domain.knowledge_base.repository import KnowledgeBaseRepository
from app.domain.procedure_store.repository import ProcedureStore
from app.registry.adapters.schema_registry import get_schema_registry
from app.registry.application.registry_service import RegistryService
from app.registry.engine import RegistryEngine
from app.reporting.engine import (
    ReporterEngine,
    _load_procedure_order,
    default_schema_registry,
    default_template_registry,
)
from app.reporting.inference import InferenceEngine
from app.reporting.validation import ValidationEngine
from observability.logging_config import get_logger

logger = get_logger("api_dependencies")

# Global singleton for the procedure store (supports both memory and supabase backends)
_procedure_store: ProcedureStore | None = None


@lru_cache(maxsize=1)
def get_coder_settings() -> CoderSettings:
    """Get cached CoderSettings from environment."""
    return CoderSettings()


@lru_cache(maxsize=1)
def get_knowledge_settings() -> KnowledgeSettings:
    """Get cached KnowledgeSettings from environment."""
    return KnowledgeSettings()


@lru_cache(maxsize=1)
def get_kb_document() -> dict[str, object]:
    """Load and cache the raw KB JSON document once per process."""
    from app.common.knowledge import get_knowledge

    settings = get_knowledge_settings()
    data = get_knowledge(settings.kb_path)
    if not isinstance(data, dict):
        return {}
    return dict(data)


@lru_cache(maxsize=1)
def get_kb_repo() -> KnowledgeBaseRepository:
    """Create a cached KB repository instance from the centralized KB path."""
    settings = get_knowledge_settings()
    try:
        return JsonKnowledgeBaseAdapter(settings.kb_path, raw_data=get_kb_document())
    except TypeError:
        # Backwards compatibility: older adapter signature.
        return JsonKnowledgeBaseAdapter(settings.kb_path)


@lru_cache(maxsize=1)
def get_code_families_config() -> dict[str, object]:
    """Load and cache the static code family hierarchy config."""
    import json

    settings = get_knowledge_settings()
    with settings.families_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return dict(data) if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def get_ncci_ptp_config() -> dict[str, object]:
    """Load and cache merged NCCI PTP bundling rules (KB + external).

    Precedence: external file overrides internal KB for conflicting pairs.
    """
    import json

    settings = get_knowledge_settings()
    with settings.ncci_path.open(encoding="utf-8") as handle:
        external = json.load(handle)
    if not isinstance(external, dict):
        external = {}

    from app.coder.ncci import merge_ncci_sources

    merged = merge_ncci_sources(
        kb_document=dict(get_kb_document()),
        external_cfg=external,
    )
    return dict(merged) if isinstance(merged, dict) else {}


@lru_cache(maxsize=1)
def get_addon_templates_document() -> dict[str, object]:
    """Load and cache add-on templates JSON."""
    import json

    settings = get_knowledge_settings()
    with settings.addon_templates_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return dict(data) if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def get_coding_service() -> CodingService:
    """Create a fully wired CodingService instance.

    This factory:
    - Loads CoderSettings from environment
    - Instantiates JsonKnowledgeBaseAdapter (KnowledgeBaseRepository)
    - Instantiates YamlKeywordMappingRepository
    - Instantiates SimpleNegationDetector
    - Instantiates RuleEngine using the KB
    - Instantiates GeminiAdvisorAdapter (or MockLLMAdvisor in rules_only mode)
    - Returns a CodingService with all dependencies wired

    The instance is cached for reuse across requests.
    """
    config = get_coder_settings()
    logger.info(
        "Initializing CodingService",
        extra={
            "kb_path": config.kb_path,
            "kb_version": config.kb_version,
            "keyword_mapping_dir": config.keyword_mapping_dir,
        },
    )

    # 1. Knowledge base repository
    kb_repo = get_kb_repo()
    logger.info(f"Loaded KB version: {kb_repo.version}")

    # 2. Keyword mapping repository
    keyword_repo = YamlKeywordMappingRepository(config.keyword_mapping_dir)
    logger.info(f"Loaded keyword mappings version: {keyword_repo.version}")

    # 3. Negation detector
    negation_detector = SimpleNegationDetector()
    logger.info(f"Negation detector version: {negation_detector.version}")

    # 4. Rule engine (with CodingRulesEngine mode from environment)
    rules_mode = os.getenv("CODING_RULES_MODE", "python")
    families_cfg = get_code_families_config()
    ncci_cfg = get_ncci_ptp_config()

    ip_kb = None
    try:
        from app.autocode.ip_kb.ip_kb import IPCodingKnowledgeBase

        ip_kb = IPCodingKnowledgeBase(get_kb_document())
    except Exception as exc:
        logger.warning(
            "IPCodingKnowledgeBase unavailable (%s); using fallback keyword matching",
            exc,
        )
        ip_kb = None

    rule_engine = RuleEngine(
        kb_repo,
        rules_mode=rules_mode,
        code_families_config=families_cfg,
        ncci_data=ncci_cfg,
        ip_kb=ip_kb,
    )
    logger.info(
        f"Rule engine version: {rule_engine.version}",
        extra={"rules_mode": rules_mode},
    )

    # 5. LLM advisor (conditionally enabled)
    llm_advisor: LLMAdvisorPort | None = None
    use_llm = os.getenv("CODER_USE_LLM_ADVISOR", "").lower() in ("true", "1", "yes")

    if use_llm:
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider == "openai_compat":
            offline = os.getenv("OPENAI_OFFLINE", "").strip().lower() in ("1", "true", "yes")
            api_key = os.getenv("OPENAI_API_KEY", "")
            model_name = os.getenv("OPENAI_MODEL", "")
            if not offline and api_key and not model_name:
                logger.warning("OPENAI_MODEL not set, LLM advisor disabled")
            else:
                from app.coder.adapters.llm.openai_compat_advisor import (
                    OpenAICompatAdvisorAdapter,
                )

                llm_advisor = OpenAICompatAdvisorAdapter(
                    model_name=model_name,
                    allowed_codes=list(kb_repo.get_all_codes()),
                    api_key=api_key,
                )
                logger.info("LLM advisor enabled: openai_compat")
        else:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
            if api_key:
                llm_advisor = GeminiAdvisorAdapter(
                    model_name=config.model_version,
                    allowed_codes=list(kb_repo.get_all_codes()),
                    api_key=api_key,
                )
                logger.info(f"LLM advisor enabled: {config.model_version}")
            else:
                logger.warning("GOOGLE_API_KEY not set, LLM advisor disabled")
    else:
        logger.info("LLM advisor disabled (CODER_USE_LLM_ADVISOR not set)")

    # 6. PHI scrubbing is now handled at route level (app/api/phi_redaction.py).
    # The phi_scrubber parameter on CodingService is deprecated.

    # 7. Build CodingService
    registry_service = get_registry_service()
    service = CodingService(
        kb_repo=kb_repo,
        keyword_repo=keyword_repo,
        negation_detector=negation_detector,
        rule_engine=rule_engine,
        llm_advisor=llm_advisor,
        config=config,
        # phi_scrubber omitted - now handled at route level
        registry_service=registry_service,
    )

    logger.info("CodingService initialized successfully")
    return service


def reset_coding_service_cache() -> None:
    """Reset the cached CodingService instance.

    Useful for testing or when settings change.
    """
    get_coding_service.cache_clear()
    get_coder_settings.cache_clear()
    get_knowledge_settings.cache_clear()
    get_kb_document.cache_clear()
    get_kb_repo.cache_clear()
    get_code_families_config.cache_clear()
    get_ncci_ptp_config.cache_clear()
    get_addon_templates_document.cache_clear()


@lru_cache(maxsize=1)
def get_registry_service() -> RegistryService:
    """Create a RegistryService instance for registry export operations.

    This factory:
    - Uses the default RegistrySchemaRegistry for schema access
    - Configures default version from environment (REGISTRY_DEFAULT_VERSION)
    - Builds SmartHybridOrchestrator for ML-first coding
    - Returns a cached instance for reuse across requests
    """
    import os

    default_version = (
        os.getenv("REGISTRY_DEFAULT_VERSION")
        or os.getenv("REGISTRY_SCHEMA_VERSION")
        or "v3"
    ).strip()
    if default_version not in {"v2", "v3"}:
        raise ValueError(
            f"Invalid REGISTRY_DEFAULT_VERSION={default_version!r}. Expected 'v2' or 'v3'."
        )
    schema_registry = get_schema_registry()

    # Build hybrid orchestrator for ML-first coding
    from app.coder.ncci import NCCIEngine
    from app.coder.rules_engine import CodingRulesEngine

    rules_engine = CodingRulesEngine(
        families_cfg=get_code_families_config(),
        ncci_engine=NCCIEngine(ptp_cfg=get_ncci_ptp_config()),
    )
    hybrid_orchestrator = build_hybrid_orchestrator(rules_engine=rules_engine)
    logger.info("SmartHybridOrchestrator built for RegistryService")

    service = RegistryService(
        schema_registry=schema_registry,
        default_version=default_version,
        hybrid_orchestrator=hybrid_orchestrator,
    )

    logger.info(
        "RegistryService initialized",
        extra={"default_version": default_version},
    )

    return service


def reset_registry_service_cache() -> None:
    """Reset the cached RegistryService instance.

    Useful for testing or when settings change.
    """
    get_registry_service.cache_clear()
    get_code_families_config.cache_clear()
    get_ncci_ptp_config.cache_clear()


def get_procedure_store() -> ProcedureStore:
    """Get the procedure store instance.

    This factory:
    - Reads PROCEDURE_STORE_BACKEND environment variable (default: "memory")
    - If "memory" → returns InMemoryProcedureStore singleton
    - If "supabase" → returns SupabaseProcedureStore singleton (requires SUPABASE_URL and
      SUPABASE_SERVICE_ROLE_KEY)

    The instance is cached as a singleton for reuse across requests.

    Returns:
        ProcedureStore implementation based on configuration
    """
    global _procedure_store

    if _procedure_store is not None:
        return _procedure_store

    backend = os.getenv("PROCEDURE_STORE_BACKEND", "memory").lower()

    if backend == "supabase":
        # Try to use Supabase backend
        from app.coder.adapters.persistence.supabase_procedure_store import (
            SupabaseProcedureStore,
            is_supabase_available,
        )

        if is_supabase_available():
            try:
                _procedure_store = SupabaseProcedureStore()
                logger.info("ProcedureStore initialized with Supabase backend")
                return _procedure_store
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Supabase store, falling back to memory: {e}"
                )
        else:
            logger.warning(
                "Supabase credentials not configured, falling back to memory backend"
            )

    # Default: in-memory backend
    _procedure_store = InMemoryProcedureStore()
    logger.info("ProcedureStore initialized with in-memory backend")
    return _procedure_store


def reset_procedure_store() -> None:
    """Reset the procedure store singleton.

    This clears all data and resets the singleton to None,
    forcing a fresh instance on the next get_procedure_store() call.

    Useful for testing to ensure clean state between tests.
    """
    global _procedure_store

    if _procedure_store is not None:
        _procedure_store.clear_all()
        _procedure_store = None
        logger.debug("ProcedureStore reset")


_coding_service_dep = Depends(get_coding_service)


def get_qa_pipeline_service(
    coding_service: CodingService = _coding_service_dep,
) -> QAPipelineService:
    """Create a fully wired QAPipelineService instance.

    This factory:
    - Instantiates RegistryEngine for procedure extraction
    - Builds ReporterEngine with templates, schemas, and procedure order
    - Creates InferenceEngine and ValidationEngine for report enrichment
    - Creates SimpleReporterStrategy as fallback
    - Builds ReportingStrategy with all engines
    - Gets CodingService from get_coding_service()
    - Returns a QAPipelineService with all dependencies wired

    The instance is cached for reuse across requests.
    """
    logger.info("Initializing QAPipelineService")

    # 1. Registry engine
    registry_engine = RegistryEngine()
    logger.debug("RegistryEngine initialized")

    # 2. Reporter infrastructure
    templates = default_template_registry()
    schemas = default_schema_registry()
    procedure_order = _load_procedure_order()

    reporter_engine = ReporterEngine(
        templates,
        schemas,
        procedure_order=procedure_order,
    )
    logger.debug("ReporterEngine initialized")

    # 3. Inference and validation engines
    inference_engine = InferenceEngine()
    validation_engine = ValidationEngine(templates, schemas)
    logger.debug("InferenceEngine and ValidationEngine initialized")

    # 4. Simple reporter fallback
    simple_strategy = SimpleReporterStrategy()

    # 5. Reporting strategy
    reporting_strategy = ReportingStrategy(
        reporter_engine=reporter_engine,
        inference_engine=inference_engine,
        validation_engine=validation_engine,
        registry_engine=registry_engine,
        simple_strategy=simple_strategy,
    )
    logger.debug("ReportingStrategy initialized")

    # 7. Build QAPipelineService
    service = QAPipelineService(
        registry_engine=registry_engine,
        reporting_strategy=reporting_strategy,
        coding_service=coding_service,
    )

    logger.info("QAPipelineService initialized successfully")
    return service


def reset_qa_pipeline_service_cache() -> None:
    """Reset the cached QAPipelineService instance.

    Useful for testing or when settings change.
    """
    # No-op: get_qa_pipeline_service no longer caches instances.
    return
