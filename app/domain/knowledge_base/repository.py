"""Knowledge Base Repository port (interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set

from .models import ProcedureInfo, NCCIPair


class KnowledgeBaseRepository(ABC):
    """Port for accessing the Knowledge Base.

    This is the domain interface that infrastructure adapters implement.
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version identifier for this knowledge base."""
        ...

    @abstractmethod
    def get_procedure_info(self, code: str) -> Optional[ProcedureInfo]:
        """Get procedure information for a CPT code."""
        ...

    @abstractmethod
    def get_mer_group(self, code: str) -> Optional[str]:
        """Get the MER group ID for a code, if any."""
        ...

    @abstractmethod
    def get_ncci_pairs(self, code: str) -> list[NCCIPair]:
        """Get all NCCI pairs where this code is involved."""
        ...

    @abstractmethod
    def is_addon_code(self, code: str) -> bool:
        """Check if a code is an add-on code."""
        ...

    @abstractmethod
    def get_all_codes(self) -> Set[str]:
        """Get all valid CPT codes in the knowledge base."""
        ...

    @abstractmethod
    def get_parent_codes(self, addon_code: str) -> list[str]:
        """Get valid parent codes for an add-on code."""
        ...

    @abstractmethod
    def get_bundled_codes(self, code: str) -> list[str]:
        """Get codes that are bundled with the given code."""
        ...
