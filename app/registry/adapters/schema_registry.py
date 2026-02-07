"""Registry Schema Registry for versioned schema management.

Provides access to versioned Pydantic models for registry data.
"""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from proc_schemas.registry.ip_v2 import IPRegistryV2
from proc_schemas.registry.ip_v3 import IPRegistryV3


class RegistrySchemaRegistry:
    """Registry for accessing versioned schema models.

    Supports dual-schema mode for migration between versions.
    """

    def __init__(self):
        self._schemas: dict[tuple[str, str], Type[BaseModel]] = {
            ("ip_registry", "v2"): IPRegistryV2,
            ("ip_registry", "v3"): IPRegistryV3,
        }

    def get_model(self, registry_id: str, version: str) -> Type[BaseModel]:
        """Get the Pydantic model for a registry version.

        Args:
            registry_id: The registry identifier (e.g., "ip_registry").
            version: The schema version (e.g., "v2", "v3").

        Returns:
            The Pydantic model class.

        Raises:
            KeyError: If the registry/version combination is not found.
        """
        key = (registry_id, version)
        if key not in self._schemas:
            available = [f"{r}:{v}" for r, v in self._schemas.keys()]
            raise KeyError(
                f"Unknown registry schema: {registry_id}:{version}. "
                f"Available: {available}"
            )
        return self._schemas[key]

    def get_latest_version(self, registry_id: str) -> str:
        """Get the latest version for a registry.

        Args:
            registry_id: The registry identifier.

        Returns:
            The latest version string.
        """
        versions = [v for r, v in self._schemas.keys() if r == registry_id]
        if not versions:
            raise KeyError(f"Unknown registry: {registry_id}")

        # Sort versions (assumes v1, v2, v3, ... format)
        versions.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0, reverse=True)
        return versions[0]

    def list_versions(self, registry_id: str) -> list[str]:
        """List all versions for a registry.

        Args:
            registry_id: The registry identifier.

        Returns:
            List of version strings.
        """
        return sorted(
            [v for r, v in self._schemas.keys() if r == registry_id],
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
        )

    def register(self, registry_id: str, version: str, model: Type[BaseModel]) -> None:
        """Register a new schema version.

        Args:
            registry_id: The registry identifier.
            version: The schema version.
            model: The Pydantic model class.
        """
        self._schemas[(registry_id, version)] = model


# Default singleton instance
_default_registry: RegistrySchemaRegistry | None = None


def get_schema_registry() -> RegistrySchemaRegistry:
    """Get the default schema registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = RegistrySchemaRegistry()
    return _default_registry
