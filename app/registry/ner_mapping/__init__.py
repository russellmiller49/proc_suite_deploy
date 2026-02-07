"""NER-to-Registry mapping module.

Maps granular NER entities to RegistryRecord fields for deterministic
CPT code derivation.
"""

from app.registry.ner_mapping.entity_to_registry import (
    NERToRegistryMapper,
    RegistryMappingResult,
)
from app.registry.ner_mapping.station_extractor import EBUSStationExtractor
from app.registry.ner_mapping.procedure_extractor import ProcedureExtractor

__all__ = [
    "NERToRegistryMapper",
    "RegistryMappingResult",
    "EBUSStationExtractor",
    "ProcedureExtractor",
]
