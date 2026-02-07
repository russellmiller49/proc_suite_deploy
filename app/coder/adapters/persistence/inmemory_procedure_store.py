"""In-memory implementation of the ProcedureStore interface.

This adapter stores all procedure-related data in memory using dictionaries.
It's suitable for development, testing, and single-instance deployments.

For production with multiple instances, use a database-backed adapter
(e.g., SupabaseProcedureStore or PostgresProcedureStore).
"""

from __future__ import annotations

from typing import Any

from app.domain.procedure_store.repository import ProcedureStore
from proc_schemas.coding import CodeSuggestion, FinalCode, ReviewAction, CodingResult


class InMemoryProcedureStore(ProcedureStore):
    """In-memory implementation of the ProcedureStore interface.

    Thread-safety: This implementation is NOT thread-safe. For production use
    with concurrent access, wrap with a lock or use a thread-safe backend.
    """

    def __init__(self):
        """Initialize empty stores."""
        self._suggestions: dict[str, list[CodeSuggestion]] = {}
        self._coding_results: dict[str, CodingResult] = {}
        self._reviews: dict[str, list[ReviewAction]] = {}
        self._final_codes: dict[str, list[FinalCode]] = {}
        self._registry_exports: dict[str, dict[str, Any]] = {}

    # =========================================================================
    # ProcedureCodeSuggestionRepository
    # =========================================================================

    def get_suggestions(self, proc_id: str) -> list[CodeSuggestion]:
        """Get all code suggestions for a procedure."""
        return list(self._suggestions.get(proc_id, []))

    def save_suggestions(self, proc_id: str, suggestions: list[CodeSuggestion]) -> None:
        """Save code suggestions for a procedure."""
        self._suggestions[proc_id] = list(suggestions)

    def delete_suggestions(self, proc_id: str) -> None:
        """Delete all suggestions for a procedure."""
        self._suggestions.pop(proc_id, None)

    def exists(self, proc_id: str) -> bool:
        """Check if suggestions exist for a procedure."""
        return proc_id in self._suggestions

    # =========================================================================
    # ProcedureCodingResultRepository
    # =========================================================================

    def get_result(self, proc_id: str) -> CodingResult | None:
        """Get the coding result for a procedure."""
        return self._coding_results.get(proc_id)

    def save_result(self, proc_id: str, result: CodingResult) -> None:
        """Save the coding result for a procedure."""
        self._coding_results[proc_id] = result

    def delete_result(self, proc_id: str) -> None:
        """Delete the coding result for a procedure."""
        self._coding_results.pop(proc_id, None)

    # =========================================================================
    # ProcedureCodeReviewRepository
    # =========================================================================

    def get_reviews(self, proc_id: str) -> list[ReviewAction]:
        """Get all review actions for a procedure."""
        return list(self._reviews.get(proc_id, []))

    def add_review(self, proc_id: str, review: ReviewAction) -> None:
        """Add a review action for a procedure."""
        if proc_id not in self._reviews:
            self._reviews[proc_id] = []
        self._reviews[proc_id].append(review)

    def delete_reviews(self, proc_id: str) -> None:
        """Delete all reviews for a procedure."""
        self._reviews.pop(proc_id, None)

    # =========================================================================
    # ProcedureFinalCodeRepository
    # =========================================================================

    def get_final_codes(self, proc_id: str) -> list[FinalCode]:
        """Get all final approved codes for a procedure."""
        return list(self._final_codes.get(proc_id, []))

    def add_final_code(self, proc_id: str, final_code: FinalCode) -> None:
        """Add a final approved code for a procedure."""
        if proc_id not in self._final_codes:
            self._final_codes[proc_id] = []
        self._final_codes[proc_id].append(final_code)

    def delete_final_codes(self, proc_id: str) -> None:
        """Delete all final codes for a procedure."""
        self._final_codes.pop(proc_id, None)

    # =========================================================================
    # ProcedureRegistryExportRepository
    # =========================================================================

    def get_export(self, proc_id: str) -> dict[str, Any] | None:
        """Get the registry export for a procedure."""
        return self._registry_exports.get(proc_id)

    def save_export(self, proc_id: str, export: dict[str, Any]) -> None:
        """Save a registry export for a procedure."""
        self._registry_exports[proc_id] = export

    def delete_export(self, proc_id: str) -> None:
        """Delete the registry export for a procedure."""
        self._registry_exports.pop(proc_id, None)

    def export_exists(self, proc_id: str) -> bool:
        """Check if an export exists for a procedure."""
        return proc_id in self._registry_exports

    # =========================================================================
    # ProcedureStore composite methods
    # =========================================================================

    def clear_all(self, proc_id: str | None = None) -> None:
        """Clear all data for a procedure or all procedures."""
        if proc_id:
            self.delete_suggestions(proc_id)
            self.delete_result(proc_id)
            self.delete_reviews(proc_id)
            self.delete_final_codes(proc_id)
            self.delete_export(proc_id)
        else:
            self._suggestions.clear()
            self._coding_results.clear()
            self._reviews.clear()
            self._final_codes.clear()
            self._registry_exports.clear()

    # =========================================================================
    # Debug/inspection helpers
    # =========================================================================

    def get_all_procedure_ids(self) -> set[str]:
        """Get all procedure IDs that have any data stored.

        Useful for debugging and testing.
        """
        ids = set()
        ids.update(self._suggestions.keys())
        ids.update(self._coding_results.keys())
        ids.update(self._reviews.keys())
        ids.update(self._final_codes.keys())
        ids.update(self._registry_exports.keys())
        return ids

    def get_stats(self) -> dict[str, int]:
        """Get statistics about stored data.

        Returns:
            Dict with counts for each data type
        """
        return {
            "procedures_with_suggestions": len(self._suggestions),
            "procedures_with_results": len(self._coding_results),
            "procedures_with_reviews": len(self._reviews),
            "procedures_with_final_codes": len(self._final_codes),
            "procedures_with_exports": len(self._registry_exports),
            "total_suggestions": sum(len(s) for s in self._suggestions.values()),
            "total_reviews": sum(len(r) for r in self._reviews.values()),
            "total_final_codes": sum(len(f) for f in self._final_codes.values()),
        }


# Singleton instance for backward compatibility with current module-level dicts
_default_store: InMemoryProcedureStore | None = None


def get_default_procedure_store() -> InMemoryProcedureStore:
    """Get the default in-memory procedure store instance.

    This is provided for backward compatibility during migration from
    module-level dicts to the repository pattern.
    """
    global _default_store
    if _default_store is None:
        _default_store = InMemoryProcedureStore()
    return _default_store


def reset_default_procedure_store() -> None:
    """Reset the default procedure store instance.

    Useful for testing to ensure clean state.
    """
    global _default_store
    if _default_store:
        _default_store.clear_all()
    _default_store = None
