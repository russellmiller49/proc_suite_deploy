"""Compatibility shim for granular registry models and helpers.

The canonical implementation lives in:
- `app.registry.schema.granular_models` (Pydantic models + validators)
- `app.registry.schema.granular_logic` (derivation/validation helpers)

This module preserves the historical import path `app.registry.schema_granular`.
"""

from __future__ import annotations

from app.registry.schema.granular_logic import (
    derive_aggregate_fields,
    derive_procedures_from_granular,
    validate_ebus_consistency,
)
from app.registry.schema.granular_models import (
    AirwayStentProcedure,
    BLVRChartisMeasurement,
    BLVRValvePlacement,
    CAOInterventionDetail,
    CAOModalityApplication,
    ClinicalContext,
    CryobiopsySite,
    EBUSStationDetail,
    EnhancedRegistryGranularData,
    IPCProcedure,
    NavigationTarget,
    PatientDemographics,
    SpecimenCollected,
    ThoracoscopyFinding,
)

__all__ = [
    "IPCProcedure",
    "ClinicalContext",
    "PatientDemographics",
    "AirwayStentProcedure",
    # Per-site models
    "EBUSStationDetail",
    "NavigationTarget",
    "CAOModalityApplication",
    "CAOInterventionDetail",
    "BLVRValvePlacement",
    "BLVRChartisMeasurement",
    "CryobiopsySite",
    "ThoracoscopyFinding",
    "SpecimenCollected",
    # Container
    "EnhancedRegistryGranularData",
    # Helpers
    "validate_ebus_consistency",
    "derive_aggregate_fields",
    "derive_procedures_from_granular",
]

