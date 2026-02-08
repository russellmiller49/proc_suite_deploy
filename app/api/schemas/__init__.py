"""API schemas package.

This package contains all Pydantic schemas for the FastAPI integration layer.
"""

# Base schemas (legacy compatibility)
from app.api.schemas.base import (
    CoderRequest,
    CoderResponse,
    CodeSuggestionSummary,
    HybridPipelineMetadata,
    KnowledgeMeta,
    MissingFieldPrompt,
    QARunRequest,
    QuestionsRequest,
    QuestionsResponse,
    RegistryRequest,
    RegistryResponse,
    RenderRequest,
    RenderResponse,
    SeedFromTextRequest,
    SeedFromTextResponse,
    JsonPatchOperation,
    UnifiedProcessRequest,
    UnifiedProcessResponse,
    VerifyRequest,
    VerifyResponse,
)

# QA pipeline schemas (new structured response)
from app.api.schemas.qa import (
    CodeEntry,
    CoderData,
    ModuleResult,
    ModuleStatus,
    QARunResponse,
    RegistryData,
    ReporterData,
)

__all__ = [
    # Base schemas
    "CoderRequest",
    "CoderResponse",
    "CodeSuggestionSummary",
    "HybridPipelineMetadata",
    "KnowledgeMeta",
    "MissingFieldPrompt",
    "QARunRequest",
    "QuestionsRequest",
    "QuestionsResponse",
    "RegistryRequest",
    "RegistryResponse",
    "RenderRequest",
    "RenderResponse",
    "SeedFromTextRequest",
    "SeedFromTextResponse",
    "JsonPatchOperation",
    "UnifiedProcessRequest",
    "UnifiedProcessResponse",
    "VerifyRequest",
    "VerifyResponse",
    # QA pipeline schemas
    "CodeEntry",
    "CoderData",
    "ModuleResult",
    "ModuleStatus",
    "QARunResponse",
    "RegistryData",
    "ReporterData",
]
