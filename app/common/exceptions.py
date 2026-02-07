"""Exception hierarchy for the Procedure Suite system."""

from __future__ import annotations


class CodingError(Exception):
    """Base error for coding pipeline."""

    pass


class ValidationError(CodingError):
    """Schema or business rule validation failed."""

    pass


class LLMError(CodingError):
    """LLM call failed (timeout, invalid response, etc.)."""

    pass


class KnowledgeBaseError(CodingError):
    """Knowledge base loading or lookup error."""

    pass


class RegistryError(Exception):
    """Registry export or validation error."""

    pass


class ReporterError(Exception):
    """Reporter pipeline error."""

    pass


class PersistenceError(Exception):
    """Database or storage persistence error."""

    def __init__(self, message: str, operation: str | None = None, proc_id: str | None = None):
        self.operation = operation
        self.proc_id = proc_id
        super().__init__(message)


class AgentError(Exception):
    """Agent execution error."""

    def __init__(self, code: str, message: str, section: str | None = None):
        self.code = code
        self.message = message
        self.section = section
        super().__init__(f"[{code}] {message}")
