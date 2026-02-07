"""Port interfaces for procedure-related persistence.

These are the domain-layer interfaces (ports) that define how the application
accesses and persists procedure-related data. Adapters implement these interfaces
for specific storage backends (in-memory, Supabase, PostgreSQL, etc.).

The interfaces follow the Repository pattern from DDD.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from proc_schemas.coding import CodeSuggestion, FinalCode, ReviewAction, CodingResult


class ProcedureCodeSuggestionRepository(ABC):
    """Repository interface for procedure code suggestions."""

    @abstractmethod
    def get_suggestions(self, proc_id: str) -> list[CodeSuggestion]:
        """Get all code suggestions for a procedure.

        Args:
            proc_id: Procedure identifier

        Returns:
            List of CodeSuggestion objects (empty if none found)
        """
        ...

    @abstractmethod
    def save_suggestions(self, proc_id: str, suggestions: list[CodeSuggestion]) -> None:
        """Save code suggestions for a procedure.

        Overwrites any existing suggestions for this procedure.

        Args:
            proc_id: Procedure identifier
            suggestions: List of CodeSuggestion objects to save
        """
        ...

    @abstractmethod
    def delete_suggestions(self, proc_id: str) -> None:
        """Delete all suggestions for a procedure.

        Args:
            proc_id: Procedure identifier
        """
        ...

    @abstractmethod
    def exists(self, proc_id: str) -> bool:
        """Check if suggestions exist for a procedure.

        Args:
            proc_id: Procedure identifier

        Returns:
            True if suggestions exist
        """
        ...


class ProcedureCodingResultRepository(ABC):
    """Repository interface for complete coding results."""

    @abstractmethod
    def get_result(self, proc_id: str) -> CodingResult | None:
        """Get the coding result for a procedure.

        Args:
            proc_id: Procedure identifier

        Returns:
            CodingResult if found, None otherwise
        """
        ...

    @abstractmethod
    def save_result(self, proc_id: str, result: CodingResult) -> None:
        """Save the coding result for a procedure.

        Overwrites any existing result for this procedure.

        Args:
            proc_id: Procedure identifier
            result: CodingResult to save
        """
        ...

    @abstractmethod
    def delete_result(self, proc_id: str) -> None:
        """Delete the coding result for a procedure.

        Args:
            proc_id: Procedure identifier
        """
        ...


class ProcedureCodeReviewRepository(ABC):
    """Repository interface for code review actions."""

    @abstractmethod
    def get_reviews(self, proc_id: str) -> list[ReviewAction]:
        """Get all review actions for a procedure.

        Args:
            proc_id: Procedure identifier

        Returns:
            List of ReviewAction objects (empty if none found)
        """
        ...

    @abstractmethod
    def add_review(self, proc_id: str, review: ReviewAction) -> None:
        """Add a review action for a procedure.

        Args:
            proc_id: Procedure identifier
            review: ReviewAction to add
        """
        ...

    @abstractmethod
    def delete_reviews(self, proc_id: str) -> None:
        """Delete all reviews for a procedure.

        Args:
            proc_id: Procedure identifier
        """
        ...


class ProcedureFinalCodeRepository(ABC):
    """Repository interface for final approved codes."""

    @abstractmethod
    def get_final_codes(self, proc_id: str) -> list[FinalCode]:
        """Get all final approved codes for a procedure.

        Args:
            proc_id: Procedure identifier

        Returns:
            List of FinalCode objects (empty if none found)
        """
        ...

    @abstractmethod
    def add_final_code(self, proc_id: str, final_code: FinalCode) -> None:
        """Add a final approved code for a procedure.

        Args:
            proc_id: Procedure identifier
            final_code: FinalCode to add
        """
        ...

    @abstractmethod
    def delete_final_codes(self, proc_id: str) -> None:
        """Delete all final codes for a procedure.

        Args:
            proc_id: Procedure identifier
        """
        ...


class ProcedureRegistryExportRepository(ABC):
    """Repository interface for registry exports."""

    @abstractmethod
    def get_export(self, proc_id: str) -> dict[str, Any] | None:
        """Get the registry export for a procedure.

        Args:
            proc_id: Procedure identifier

        Returns:
            Export record dict if found, None otherwise
        """
        ...

    @abstractmethod
    def save_export(self, proc_id: str, export: dict[str, Any]) -> None:
        """Save a registry export for a procedure.

        Overwrites any existing export for this procedure.

        Args:
            proc_id: Procedure identifier
            export: Export record dict to save
        """
        ...

    @abstractmethod
    def delete_export(self, proc_id: str) -> None:
        """Delete the registry export for a procedure.

        Args:
            proc_id: Procedure identifier
        """
        ...

    @abstractmethod
    def exists(self, proc_id: str) -> bool:
        """Check if an export exists for a procedure.

        Args:
            proc_id: Procedure identifier

        Returns:
            True if export exists
        """
        ...


class ProcedureStore(
    ProcedureCodeSuggestionRepository,
    ProcedureCodingResultRepository,
    ProcedureCodeReviewRepository,
    ProcedureFinalCodeRepository,
    ProcedureRegistryExportRepository,
):
    """Composite interface combining all procedure-related repositories.

    Implementations can either implement this single interface or implement
    each repository interface separately. This composite interface is useful
    when all data should be stored in the same backend.
    """

    @abstractmethod
    def clear_all(self, proc_id: str | None = None) -> None:
        """Clear all data for a procedure or all procedures.

        Args:
            proc_id: If provided, clear only that procedure's data.
                     If None, clear all procedure data.
        """
        ...
