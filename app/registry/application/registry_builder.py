"""Registry entry builders using the Strategy Pattern.

This module provides versioned builders for constructing IP Registry entries,
allowing the service to easily support new schema versions without modifying
core logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, TypeVar

from proc_schemas.registry.ip_v2 import (
    IPRegistryV2,
    PatientInfo as PatientInfoV2,
    ProcedureInfo as ProcedureInfoV2,
)
from proc_schemas.registry.ip_v3 import (
    IPRegistryV3,
    PatientInfo as PatientInfoV3,
    ProcedureInfo as ProcedureInfoV3,
)

# Type variables for generic builder interface
PatientT = TypeVar("PatientT", PatientInfoV2, PatientInfoV3)
ProcedureT = TypeVar("ProcedureT", ProcedureInfoV2, ProcedureInfoV3)
EntryT = TypeVar("EntryT", IPRegistryV2, IPRegistryV3)


class RegistryBuilderProtocol(ABC):
    """Abstract base class for registry entry builders.

    Each schema version should have its own builder implementation
    to encapsulate version-specific logic.
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the schema version this builder supports."""
        ...

    @abstractmethod
    def build_patient(
        self,
        metadata: dict[str, Any],
        missing_fields: list[str],
    ) -> PatientInfoV2 | PatientInfoV3:
        """Build patient info from metadata.

        Args:
            metadata: Dict containing patient data under 'patient' key.
            missing_fields: List to append missing field names to.

        Returns:
            PatientInfo object for this schema version.
        """
        ...

    @abstractmethod
    def build_procedure(
        self,
        procedure_id: str,
        metadata: dict[str, Any],
        missing_fields: list[str],
    ) -> ProcedureInfoV2 | ProcedureInfoV3:
        """Build procedure info from metadata.

        Args:
            procedure_id: The procedure identifier.
            metadata: Dict containing procedure data under 'procedure' key.
            missing_fields: List to append missing field names to.

        Returns:
            ProcedureInfo object for this schema version.
        """
        ...

    @abstractmethod
    def build_entry(
        self,
        procedure_id: str,
        patient: PatientInfoV2 | PatientInfoV3,
        procedure: ProcedureInfoV2 | ProcedureInfoV3,
        registry_fields: dict[str, Any],
        metadata: dict[str, Any],
    ) -> IPRegistryV2 | IPRegistryV3:
        """Build the complete registry entry.

        Args:
            procedure_id: The procedure identifier.
            patient: Patient info object from build_patient().
            procedure: Procedure info object from build_procedure().
            registry_fields: Fields derived from CPT mappings.
            metadata: Original metadata dict for additional overrides.

        Returns:
            Complete registry entry for this schema version.
        """
        ...

    @abstractmethod
    def get_metadata_fields(self) -> list[str]:
        """Return list of metadata field names this version supports.

        Used to apply metadata overrides to the entry.
        """
        ...


class V2RegistryBuilder(RegistryBuilderProtocol):
    """Builder for IP Registry V2 schema entries."""

    @property
    def version(self) -> str:
        return "v2"

    def build_patient(
        self,
        metadata: dict[str, Any],
        missing_fields: list[str],
    ) -> PatientInfoV2:
        """Build PatientInfoV2 from metadata."""
        patient_data = metadata.get("patient", {})

        patient_id = patient_data.get("patient_id", "")
        mrn = patient_data.get("mrn", "")
        age = patient_data.get("age")
        sex = patient_data.get("sex")

        if not patient_id and not mrn:
            missing_fields.append("patient.patient_id or patient.mrn")

        return PatientInfoV2(
            patient_id=patient_id,
            mrn=mrn,
            age=age,
            sex=sex,
        )

    def build_procedure(
        self,
        procedure_id: str,
        metadata: dict[str, Any],
        missing_fields: list[str],
    ) -> ProcedureInfoV2:
        """Build ProcedureInfoV2 from metadata."""
        proc_data = metadata.get("procedure", {})

        procedure_date = proc_data.get("procedure_date")
        if isinstance(procedure_date, str):
            try:
                procedure_date = date.fromisoformat(procedure_date)
            except ValueError:
                procedure_date = None

        if not procedure_date:
            missing_fields.append("procedure.procedure_date")

        procedure_type = proc_data.get("procedure_type", "")
        indication = proc_data.get("indication", "")
        urgency = proc_data.get("urgency", "routine")

        if not indication:
            missing_fields.append("procedure.indication")

        return ProcedureInfoV2(
            procedure_id=procedure_id,
            procedure_date=procedure_date,
            procedure_type=procedure_type,
            indication=indication,
            urgency=urgency,
        )

    def build_entry(
        self,
        procedure_id: str,
        patient: PatientInfoV2,
        procedure: ProcedureInfoV2,
        registry_fields: dict[str, Any],
        metadata: dict[str, Any],
    ) -> IPRegistryV2:
        """Build an IPRegistryV2 entry."""
        entry_data: dict[str, Any] = {
            "patient": patient,
            "procedure": procedure,
        }

        # Apply registry fields from CPT mappings
        entry_data.update(registry_fields)

        # Apply metadata overrides for V2 fields
        for key in self.get_metadata_fields():
            if key in metadata:
                entry_data[key] = metadata[key]

        # Handle any_complications flag
        complications = metadata.get("complications", [])
        if complications:
            entry_data["any_complications"] = True

        return IPRegistryV2(**entry_data)

    def get_metadata_fields(self) -> list[str]:
        """Return V2-specific metadata field names."""
        return [
            "sedation",
            "ebus_stations",
            "tblb_sites",
            "bal_sites",
            "navigation_system",
            "stents",
            "findings",
            "complications",
            "disposition",
            "impression",
            "recommendations",
        ]


class V3RegistryBuilder(RegistryBuilderProtocol):
    """Builder for IP Registry V3 schema entries."""

    @property
    def version(self) -> str:
        return "v3"

    def build_patient(
        self,
        metadata: dict[str, Any],
        missing_fields: list[str],
    ) -> PatientInfoV3:
        """Build PatientInfoV3 from metadata (includes V3-specific fields)."""
        patient_data = metadata.get("patient", {})

        patient_id = patient_data.get("patient_id", "")
        mrn = patient_data.get("mrn", "")
        age = patient_data.get("age")
        sex = patient_data.get("sex")

        if not patient_id and not mrn:
            missing_fields.append("patient.patient_id or patient.mrn")

        # V3-specific patient fields
        bmi = patient_data.get("bmi")
        smoking_status = patient_data.get("smoking_status")

        return PatientInfoV3(
            patient_id=patient_id,
            mrn=mrn,
            age=age,
            sex=sex,
            bmi=bmi,
            smoking_status=smoking_status,
        )

    def build_procedure(
        self,
        procedure_id: str,
        metadata: dict[str, Any],
        missing_fields: list[str],
    ) -> ProcedureInfoV3:
        """Build ProcedureInfoV3 from metadata (includes V3-specific fields)."""
        proc_data = metadata.get("procedure", {})

        procedure_date = proc_data.get("procedure_date")
        if isinstance(procedure_date, str):
            try:
                procedure_date = date.fromisoformat(procedure_date)
            except ValueError:
                procedure_date = None

        if not procedure_date:
            missing_fields.append("procedure.procedure_date")

        procedure_type = proc_data.get("procedure_type", "")
        indication = proc_data.get("indication", "")
        urgency = proc_data.get("urgency", "routine")

        if not indication:
            missing_fields.append("procedure.indication")

        # V3-specific procedure fields
        operator = proc_data.get("operator", "")
        facility = proc_data.get("facility", "")

        return ProcedureInfoV3(
            procedure_id=procedure_id,
            procedure_date=procedure_date,
            procedure_type=procedure_type,
            indication=indication,
            urgency=urgency,
            operator=operator,
            facility=facility,
        )

    def build_entry(
        self,
        procedure_id: str,
        patient: PatientInfoV3,
        procedure: ProcedureInfoV3,
        registry_fields: dict[str, Any],
        metadata: dict[str, Any],
    ) -> IPRegistryV3:
        """Build an IPRegistryV3 entry."""
        entry_data: dict[str, Any] = {
            "patient": patient,
            "procedure": procedure,
        }

        # Apply registry fields from CPT mappings
        entry_data.update(registry_fields)

        # Apply metadata overrides for V3 fields
        for key in self.get_metadata_fields():
            if key in metadata:
                entry_data[key] = metadata[key]

        # Handle any_complications flag
        complications = metadata.get("complications", [])
        if complications:
            entry_data["any_complications"] = True

        return IPRegistryV3(**entry_data)

    def get_metadata_fields(self) -> list[str]:
        """Return V3-specific metadata field names."""
        return [
            "sedation",
            "events",
            "ebus_stations",
            "ebus_station_count",
            "tblb_sites",
            "tblb_technique",
            "navigation_target_reached",
            "radial_ebus_findings",
            "bal_sites",
            "bal_volume_ml",
            "bal_return_ml",
            "dilation_sites",
            "dilation_technique",
            "stents",
            "ablation_sites",
            "blvr_chartis_performed",
            "blvr_cv_result",
            "findings",
            "complications",
            "outcome",
            "disposition",
            "length_of_stay_hours",
            "impression",
            "recommendations",
        ]


def get_builder(version: str) -> RegistryBuilderProtocol:
    """Factory function to get the appropriate builder for a schema version.

    Args:
        version: Schema version string ("v2" or "v3").

    Returns:
        Builder instance for the specified version.

    Raises:
        ValueError: If version is not supported.
    """
    builders: dict[str, type[RegistryBuilderProtocol]] = {
        "v2": V2RegistryBuilder,
        "v3": V3RegistryBuilder,
    }

    if version not in builders:
        raise ValueError(
            f"Unsupported registry version: {version}. "
            f"Supported versions: {list(builders.keys())}"
        )

    return builders[version]()
