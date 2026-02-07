"""Supabase/PostgreSQL implementation of the ProcedureStore interface.

This adapter stores all procedure-related data in Supabase/PostgreSQL.
It's suitable for production deployments with multiple instances.

Required Tables:
-----------------

1. procedure_code_suggestions
   - id: UUID (primary key)
   - proc_id: TEXT (indexed)
   - suggestion_json: JSONB (the CodeSuggestion as JSON)
   - created_at: TIMESTAMPTZ
   - updated_at: TIMESTAMPTZ

2. procedure_coding_results
   - id: UUID (primary key)
   - proc_id: TEXT (unique, indexed)
   - result_json: JSONB (the CodingResult as JSON)
   - created_at: TIMESTAMPTZ
   - updated_at: TIMESTAMPTZ

3. procedure_code_reviews
   - id: UUID (primary key)
   - proc_id: TEXT (indexed)
   - review_json: JSONB (the ReviewAction as JSON)
   - created_at: TIMESTAMPTZ

4. procedure_final_codes
   - id: UUID (primary key)
   - proc_id: TEXT (indexed)
   - code: TEXT
   - final_code_json: JSONB (the FinalCode as JSON)
   - created_at: TIMESTAMPTZ

5. procedure_registry_exports
   - id: UUID (primary key)
   - proc_id: TEXT (unique, indexed)
   - registry_id: TEXT
   - schema_version: TEXT
   - export_id: TEXT (unique)
   - export_json: JSONB (the full export record)
   - created_at: TIMESTAMPTZ


Environment Variables:
----------------------
- SUPABASE_URL: The Supabase project URL
- SUPABASE_SERVICE_ROLE_KEY: The service role key for server-side access

Usage:
------
    store = SupabaseProcedureStore()  # Uses env vars
    # or
    store = SupabaseProcedureStore(url="...", key="...")
"""

from __future__ import annotations

import os
from typing import Any

from app.domain.procedure_store.repository import ProcedureStore
from app.common.exceptions import PersistenceError
from proc_schemas.coding import CodeSuggestion, FinalCode, ReviewAction, CodingResult
from observability.logging_config import get_logger

logger = get_logger("supabase_procedure_store")


# Table names - centralized for consistency
TABLE_SUGGESTIONS = "procedure_code_suggestions"
TABLE_CODING_RESULTS = "procedure_coding_results"
TABLE_REVIEWS = "procedure_code_reviews"
TABLE_FINAL_CODES = "procedure_final_codes"
TABLE_REGISTRY_EXPORTS = "procedure_registry_exports"


class SupabaseClient:
    """Thin wrapper around supabase-py client.

    Provides connection management and error handling.
    """

    def __init__(self, url: str | None = None, key: str | None = None):
        """Initialize Supabase client.

        Args:
            url: Supabase project URL. If not provided, reads from SUPABASE_URL env var.
            key: Supabase service role key. If not provided, reads from SUPABASE_SERVICE_ROLE_KEY.

        Raises:
            PersistenceError: If required credentials are missing.
        """
        self._url = url or os.getenv("SUPABASE_URL", "")
        self._key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

        if not self._url or not self._key:
            raise PersistenceError(
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.",
                operation="init",
            )

        self._client = None

    @property
    def client(self):
        """Lazy-initialize and return the Supabase client."""
        if self._client is None:
            try:
                from supabase import create_client
                self._client = create_client(self._url, self._key)
                logger.info("Supabase client initialized")
            except ImportError:
                raise PersistenceError(
                    "supabase-py is not installed. Run: pip install supabase",
                    operation="init",
                )
            except Exception as e:
                raise PersistenceError(
                    f"Failed to create Supabase client: {e}",
                    operation="init",
                )
        return self._client

    def table(self, name: str):
        """Get a table reference."""
        return self.client.table(name)


class SupabaseProcedureStore(ProcedureStore):
    """Supabase/PostgreSQL implementation of ProcedureStore.

    Thread-safety: This implementation is thread-safe as it uses
    Supabase's connection pooling.
    """

    def __init__(self, url: str | None = None, key: str | None = None):
        """Initialize the Supabase procedure store.

        Args:
            url: Supabase project URL (optional, uses env var if not provided)
            key: Supabase service role key (optional, uses env var if not provided)
        """
        self._supabase = SupabaseClient(url, key)

    def _handle_error(self, operation: str, proc_id: str | None, error: Exception) -> None:
        """Convert Supabase errors to PersistenceError."""
        logger.error(
            f"Supabase error during {operation}",
            extra={"proc_id": proc_id, "error": str(error)},
        )
        raise PersistenceError(
            f"Database error during {operation}: {error}",
            operation=operation,
            proc_id=proc_id,
        )

    # =========================================================================
    # ProcedureCodeSuggestionRepository
    # =========================================================================

    def get_suggestions(self, proc_id: str) -> list[CodeSuggestion]:
        """Get all code suggestions for a procedure."""
        try:
            response = (
                self._supabase.table(TABLE_SUGGESTIONS)
                .select("suggestion_json")
                .eq("proc_id", proc_id)
                .order("created_at")
                .execute()
            )
            return [
                CodeSuggestion.model_validate(row["suggestion_json"])
                for row in response.data
            ]
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("get_suggestions", proc_id, e)
            return []  # unreachable

    def save_suggestions(self, proc_id: str, suggestions: list[CodeSuggestion]) -> None:
        """Save code suggestions for a procedure (replaces existing)."""
        try:
            # Delete existing suggestions for this procedure
            self._supabase.table(TABLE_SUGGESTIONS).delete().eq("proc_id", proc_id).execute()

            # Insert new suggestions
            if suggestions:
                rows = [
                    {"proc_id": proc_id, "suggestion_json": s.model_dump(mode="json")}
                    for s in suggestions
                ]
                self._supabase.table(TABLE_SUGGESTIONS).insert(rows).execute()

            logger.debug(
                f"Saved {len(suggestions)} suggestions",
                extra={"proc_id": proc_id},
            )
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("save_suggestions", proc_id, e)

    def delete_suggestions(self, proc_id: str) -> None:
        """Delete all suggestions for a procedure."""
        try:
            self._supabase.table(TABLE_SUGGESTIONS).delete().eq("proc_id", proc_id).execute()
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("delete_suggestions", proc_id, e)

    def exists(self, proc_id: str) -> bool:
        """Check if suggestions exist for a procedure."""
        try:
            response = (
                self._supabase.table(TABLE_SUGGESTIONS)
                .select("id", count="exact")
                .eq("proc_id", proc_id)
                .limit(1)
                .execute()
            )
            return response.count > 0 if response.count is not None else len(response.data) > 0
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("exists", proc_id, e)
            return False  # unreachable

    # =========================================================================
    # ProcedureCodingResultRepository
    # =========================================================================

    def get_result(self, proc_id: str) -> CodingResult | None:
        """Get the coding result for a procedure."""
        try:
            response = (
                self._supabase.table(TABLE_CODING_RESULTS)
                .select("result_json")
                .eq("proc_id", proc_id)
                .limit(1)
                .execute()
            )
            if response.data:
                return CodingResult.model_validate(response.data[0]["result_json"])
            return None
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("get_result", proc_id, e)
            return None  # unreachable

    def save_result(self, proc_id: str, result: CodingResult) -> None:
        """Save the coding result for a procedure (upsert)."""
        try:
            row = {
                "proc_id": proc_id,
                "result_json": result.model_dump(mode="json"),
            }
            # Upsert: insert or update on conflict
            self._supabase.table(TABLE_CODING_RESULTS).upsert(
                row, on_conflict="proc_id"
            ).execute()

            logger.debug("Saved coding result", extra={"proc_id": proc_id})
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("save_result", proc_id, e)

    def delete_result(self, proc_id: str) -> None:
        """Delete the coding result for a procedure."""
        try:
            self._supabase.table(TABLE_CODING_RESULTS).delete().eq("proc_id", proc_id).execute()
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("delete_result", proc_id, e)

    # =========================================================================
    # ProcedureCodeReviewRepository
    # =========================================================================

    def get_reviews(self, proc_id: str) -> list[ReviewAction]:
        """Get all review actions for a procedure."""
        try:
            response = (
                self._supabase.table(TABLE_REVIEWS)
                .select("review_json")
                .eq("proc_id", proc_id)
                .order("created_at")
                .execute()
            )
            return [
                ReviewAction.model_validate(row["review_json"])
                for row in response.data
            ]
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("get_reviews", proc_id, e)
            return []  # unreachable

    def add_review(self, proc_id: str, review: ReviewAction) -> None:
        """Add a review action for a procedure."""
        try:
            row = {
                "proc_id": proc_id,
                "review_json": review.model_dump(mode="json"),
            }
            self._supabase.table(TABLE_REVIEWS).insert(row).execute()

            logger.debug(
                "Added review",
                extra={"proc_id": proc_id, "action": review.action},
            )
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("add_review", proc_id, e)

    def delete_reviews(self, proc_id: str) -> None:
        """Delete all reviews for a procedure."""
        try:
            self._supabase.table(TABLE_REVIEWS).delete().eq("proc_id", proc_id).execute()
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("delete_reviews", proc_id, e)

    # =========================================================================
    # ProcedureFinalCodeRepository
    # =========================================================================

    def get_final_codes(self, proc_id: str) -> list[FinalCode]:
        """Get all final approved codes for a procedure."""
        try:
            response = (
                self._supabase.table(TABLE_FINAL_CODES)
                .select("final_code_json")
                .eq("proc_id", proc_id)
                .order("created_at")
                .execute()
            )
            return [
                FinalCode.model_validate(row["final_code_json"])
                for row in response.data
            ]
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("get_final_codes", proc_id, e)
            return []  # unreachable

    def add_final_code(self, proc_id: str, final_code: FinalCode) -> None:
        """Add a final approved code for a procedure."""
        try:
            row = {
                "proc_id": proc_id,
                "code": final_code.code,
                "final_code_json": final_code.model_dump(mode="json"),
            }
            self._supabase.table(TABLE_FINAL_CODES).insert(row).execute()

            logger.debug(
                "Added final code",
                extra={"proc_id": proc_id, "code": final_code.code},
            )
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("add_final_code", proc_id, e)

    def delete_final_codes(self, proc_id: str) -> None:
        """Delete all final codes for a procedure."""
        try:
            self._supabase.table(TABLE_FINAL_CODES).delete().eq("proc_id", proc_id).execute()
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("delete_final_codes", proc_id, e)

    # =========================================================================
    # ProcedureRegistryExportRepository
    # =========================================================================

    def get_export(self, proc_id: str) -> dict[str, Any] | None:
        """Get the registry export for a procedure."""
        try:
            response = (
                self._supabase.table(TABLE_REGISTRY_EXPORTS)
                .select("export_json")
                .eq("proc_id", proc_id)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0]["export_json"]
            return None
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("get_export", proc_id, e)
            return None  # unreachable

    def save_export(self, proc_id: str, export: dict[str, Any]) -> None:
        """Save a registry export for a procedure (upsert)."""
        try:
            row = {
                "proc_id": proc_id,
                "registry_id": export.get("registry_id", "ip_registry"),
                "schema_version": export.get("schema_version", "v2"),
                "export_id": export.get("export_id", ""),
                "export_json": export,
            }
            # Upsert: insert or update on conflict
            self._supabase.table(TABLE_REGISTRY_EXPORTS).upsert(
                row, on_conflict="proc_id"
            ).execute()

            logger.debug(
                "Saved registry export",
                extra={"proc_id": proc_id, "export_id": export.get("export_id")},
            )
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("save_export", proc_id, e)

    def delete_export(self, proc_id: str) -> None:
        """Delete the registry export for a procedure."""
        try:
            self._supabase.table(TABLE_REGISTRY_EXPORTS).delete().eq("proc_id", proc_id).execute()
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("delete_export", proc_id, e)

    def export_exists(self, proc_id: str) -> bool:
        """Check if an export exists for a procedure."""
        try:
            response = (
                self._supabase.table(TABLE_REGISTRY_EXPORTS)
                .select("id", count="exact")
                .eq("proc_id", proc_id)
                .limit(1)
                .execute()
            )
            return response.count > 0 if response.count is not None else len(response.data) > 0
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("export_exists", proc_id, e)
            return False  # unreachable

    # =========================================================================
    # ProcedureStore composite methods
    # =========================================================================

    def clear_all(self, proc_id: str | None = None) -> None:
        """Clear all data for a procedure or all procedures."""
        try:
            if proc_id:
                # Clear specific procedure
                self.delete_suggestions(proc_id)
                self.delete_result(proc_id)
                self.delete_reviews(proc_id)
                self.delete_final_codes(proc_id)
                self.delete_export(proc_id)
                logger.info("Cleared all data for procedure", extra={"proc_id": proc_id})
            else:
                # Clear all data (use with caution!)
                # Note: In production, you might want to add a confirmation or
                # restrict this operation
                for table in [
                    TABLE_SUGGESTIONS,
                    TABLE_CODING_RESULTS,
                    TABLE_REVIEWS,
                    TABLE_FINAL_CODES,
                    TABLE_REGISTRY_EXPORTS,
                ]:
                    # Delete all rows by selecting all
                    self._supabase.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
                logger.warning("Cleared ALL procedure data from database")
        except PersistenceError:
            raise
        except Exception as e:
            self._handle_error("clear_all", proc_id, e)

    # =========================================================================
    # Debug/inspection helpers
    # =========================================================================

    def get_all_procedure_ids(self) -> set[str]:
        """Get all procedure IDs that have any data stored.

        Useful for debugging and testing.
        """
        try:
            ids = set()

            for table in [
                TABLE_SUGGESTIONS,
                TABLE_CODING_RESULTS,
                TABLE_REVIEWS,
                TABLE_FINAL_CODES,
                TABLE_REGISTRY_EXPORTS,
            ]:
                response = self._supabase.table(table).select("proc_id").execute()
                ids.update(row["proc_id"] for row in response.data)

            return ids
        except Exception as e:
            logger.error(f"Error getting all procedure IDs: {e}")
            return set()

    def get_stats(self) -> dict[str, int]:
        """Get statistics about stored data.

        Returns:
            Dict with counts for each data type
        """
        try:
            stats = {}

            # Count procedures with suggestions
            response = self._supabase.table(TABLE_SUGGESTIONS).select("proc_id", count="exact").execute()
            stats["procedures_with_suggestions"] = len(set(row["proc_id"] for row in response.data))
            stats["total_suggestions"] = response.count if response.count else len(response.data)

            # Count procedures with results
            response = self._supabase.table(TABLE_CODING_RESULTS).select("id", count="exact").execute()
            stats["procedures_with_results"] = response.count if response.count else len(response.data)

            # Count reviews
            response = self._supabase.table(TABLE_REVIEWS).select("proc_id", count="exact").execute()
            stats["procedures_with_reviews"] = len(set(row["proc_id"] for row in response.data))
            stats["total_reviews"] = response.count if response.count else len(response.data)

            # Count final codes
            response = self._supabase.table(TABLE_FINAL_CODES).select("proc_id", count="exact").execute()
            stats["procedures_with_final_codes"] = len(set(row["proc_id"] for row in response.data))
            stats["total_final_codes"] = response.count if response.count else len(response.data)

            # Count exports
            response = self._supabase.table(TABLE_REGISTRY_EXPORTS).select("id", count="exact").execute()
            stats["procedures_with_exports"] = response.count if response.count else len(response.data)

            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


def is_supabase_available() -> bool:
    """Check if Supabase credentials are configured.

    Returns:
        True if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set.
    """
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def get_supabase_procedure_store() -> SupabaseProcedureStore | None:
    """Get a SupabaseProcedureStore if credentials are available.

    Returns:
        SupabaseProcedureStore instance, or None if not configured.
    """
    if is_supabase_available():
        try:
            return SupabaseProcedureStore()
        except PersistenceError as e:
            logger.warning(f"Failed to create Supabase store: {e}")
    return None
