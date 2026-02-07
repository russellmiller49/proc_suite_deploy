"""
ML Advisor Module for Procedure Suite

This module provides ML/LLM-based advisory capabilities for the entire
Procedure Suite pipeline: Reporter, Coder, and Registry app.

Key Design Principles:
- Rules remain authoritative (v1): Advisor only suggests, never decides
- Transparent: All disagreements clearly marked
- Auditable: Complete trace logging for evaluation
- Pluggable: Multiple backends (stub, gemini, future: openai, local)
- Integrated: Cross-module error attribution and feedback loops

Trace Models (for ML feedback loops):
- CodingTrace: Tracks coder module runs (Phase 1)
- ReporterTrace: Tracks reporter extraction quality (Phase 2)
- RegistryTrace: Tracks registry export validation (Phase 3)
- UnifiedTrace: Links all three for error attribution (Phase 4)

Usage:
    from app.proc_ml_advisor import (
        MLAdvisorInput,
        MLAdvisorSuggestion,
        HybridCodingResult,
        CodingTrace,
        ReporterTrace,
        RegistryTrace,
        UnifiedTrace,
    )
"""

from app.proc_ml_advisor.schemas import (
    # Enums
    AdvisorBackend,
    CodingPolicy,
    CodeType,
    ConfidenceLevel,
    ProcedureCategory,
    # Code-level models
    CodeWithConfidence,
    CodeModifier,
    NCCIWarning,
    # Structured report models
    SamplingStation,
    PleuralProcedureDetails,
    BronchoscopyProcedureDetails,
    SedationDetails,
    StructuredProcedureReport,
    # ML Advisor models
    MLAdvisorInput,
    MLAdvisorSuggestion,
    # Hybrid result models
    RuleEngineResult,
    HybridCodingResult,
    # Trace models (all modules)
    CodingTrace,
    ReporterTrace,
    RegistryTrace,
    UnifiedTrace,
    # API models
    CodeRequest,
    CodeResponse,
    EvaluationMetrics,
)

__all__ = [
    # Enums
    "AdvisorBackend",
    "CodingPolicy",
    "CodeType",
    "ConfidenceLevel",
    "ProcedureCategory",
    # Code-level models
    "CodeWithConfidence",
    "CodeModifier",
    "NCCIWarning",
    # Structured report models
    "SamplingStation",
    "PleuralProcedureDetails",
    "BronchoscopyProcedureDetails",
    "SedationDetails",
    "StructuredProcedureReport",
    # ML Advisor models
    "MLAdvisorInput",
    "MLAdvisorSuggestion",
    # Hybrid result models
    "RuleEngineResult",
    "HybridCodingResult",
    # Trace models (all modules)
    "CodingTrace",
    "ReporterTrace",
    "RegistryTrace",
    "UnifiedTrace",
    # API models
    "CodeRequest",
    "CodeResponse",
    "EvaluationMetrics",
]
