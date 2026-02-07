"""Main NER-to-Registry mapping module.

Combines station and procedure extractors to populate RegistryRecord fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ner.inference import NEREntity, NERExtractionResult
from app.registry.ner_mapping.station_extractor import EBUSStationExtractor
from app.registry.ner_mapping.procedure_extractor import ProcedureExtractor
from app.registry.schema import RegistryRecord, LinearEBUSProcedure, NodeInteraction
from app.common.logger import get_logger

logger = get_logger("registry.ner_mapping")


@dataclass
class RegistryMappingResult:
    """Result from mapping NER entities to RegistryRecord fields."""

    record: RegistryRecord
    """The populated RegistryRecord."""

    field_evidence: Dict[str, List[NEREntity]]
    """Which entities drove which fields."""

    unmapped_entities: List[NEREntity]
    """Entities not mapped to any field."""

    warnings: List[str] = field(default_factory=list)
    """Warnings during mapping."""

    @property
    def stations_sampled_count(self) -> int:
        """Number of unique stations sampled."""
        linear_ebus = self.record.procedures_performed.linear_ebus
        if linear_ebus and linear_ebus.stations_sampled:
            return len(linear_ebus.stations_sampled)
        return 0


class NERToRegistryMapper:
    """Maps NER entities to RegistryRecord fields.

    This is the central orchestrator for the NER-driven extraction pathway.
    It uses specialized extractors for different entity types and combines
    their results into a unified RegistryRecord.
    """

    def __init__(self) -> None:
        self.station_extractor = EBUSStationExtractor()
        self.procedure_extractor = ProcedureExtractor()

    def map_entities(
        self,
        ner_result: NERExtractionResult,
        existing_record: Optional[RegistryRecord] = None,
    ) -> RegistryMappingResult:
        """
        Map NER entities to RegistryRecord fields.

        Args:
            ner_result: NER extraction result with entities
            existing_record: Optional existing record to merge into

        Returns:
            RegistryMappingResult with populated record and evidence
        """
        field_evidence: Dict[str, List[NEREntity]] = {}
        unmapped_entities: List[NEREntity] = []
        warnings: List[str] = []

        # Start with existing record or empty
        if existing_record:
            record_dict = existing_record.model_dump()
        else:
            record_dict = {}

        # Ensure procedures_performed exists
        if "procedures_performed" not in record_dict:
            record_dict["procedures_performed"] = {}

        # 1. Extract EBUS stations
        station_result = self.station_extractor.extract(ner_result)
        warnings.extend(station_result.warnings)

        if station_result.node_events:
            # Populate linear_ebus
            linear_ebus_dict = record_dict.get("procedures_performed", {}).get("linear_ebus", {}) or {}
            linear_ebus_dict["performed"] = True
            linear_ebus_dict["stations_sampled"] = station_result.stations_sampled
            linear_ebus_dict["node_events"] = [
                {
                    "station": ne.station,
                    "action": ne.action,
                    "outcome": ne.outcome,
                    "evidence_quote": ne.evidence_quote,
                }
                for ne in station_result.node_events
            ]
            record_dict["procedures_performed"]["linear_ebus"] = linear_ebus_dict

            # Track evidence
            for entity in ner_result.entities_by_type.get("ANAT_LN_STATION", []):
                if "linear_ebus.stations_sampled" not in field_evidence:
                    field_evidence["linear_ebus.stations_sampled"] = []
                field_evidence["linear_ebus.stations_sampled"].append(entity)

        # 2. Extract procedures
        proc_result = self.procedure_extractor.extract(ner_result)
        warnings.extend(proc_result.warnings)

        for proc_name, performed in proc_result.procedure_flags.items():
            field_path = self.procedure_extractor.field_path_for(proc_name)
            if not field_path:
                field_path = f"procedures_performed.{proc_name}.performed"

            self._set_nested_field(record_dict, field_path, performed)

            attrs = proc_result.procedure_attributes.get(proc_name, {})
            if attrs:
                base_path = field_path[: -len(".performed")] if field_path.endswith(".performed") else field_path
                for attr_name, attr_value in attrs.items():
                    self._set_nested_field(record_dict, f"{base_path}.{attr_name}", attr_value)

            # Track evidence
            if proc_name in proc_result.evidence:
                for entity_text in proc_result.evidence[proc_name]:
                    # Find the entity by text
                    for entity in ner_result.entities:
                        if entity.text == entity_text:
                            if proc_name not in field_evidence:
                                field_evidence[proc_name] = []
                            field_evidence[proc_name].append(entity)
                            break

        # 3. Extract lobe locations for TBBx
        lobes = self.procedure_extractor.extract_lobe_locations(ner_result)
        if lobes:
            tbbx_dict = record_dict.get("procedures_performed", {}).get("transbronchial_biopsy", {}) or {}
            tbbx_dict["locations"] = lobes
            if not tbbx_dict.get("performed"):
                tbbx_dict["performed"] = True
            record_dict["procedures_performed"]["transbronchial_biopsy"] = tbbx_dict

            for entity in ner_result.entities_by_type.get("ANAT_LUNG_LOC", []):
                if "transbronchial_biopsy.locations" not in field_evidence:
                    field_evidence["transbronchial_biopsy.locations"] = []
                field_evidence["transbronchial_biopsy.locations"].append(entity)

        # 4. Extract valve count for BLVR
        valve_count = self.procedure_extractor.extract_valve_count(ner_result)
        if valve_count > 0:
            blvr_dict = record_dict.get("procedures_performed", {}).get("blvr", {}) or {}
            blvr_dict["number_of_valves"] = valve_count
            if not blvr_dict.get("performed"):
                blvr_dict["performed"] = True
            record_dict["procedures_performed"]["blvr"] = blvr_dict

        # 5. Track unmapped entities
        mapped_entity_ids = set()
        for entities in field_evidence.values():
            for entity in entities:
                mapped_entity_ids.add(id(entity))

        for entity in ner_result.entities:
            if id(entity) not in mapped_entity_ids:
                unmapped_entities.append(entity)

        # Build final record
        try:
            record = RegistryRecord(**record_dict)
        except Exception as e:
            logger.warning("Failed to build RegistryRecord: %s", e)
            record = RegistryRecord()
            warnings.append(f"RegistryRecord construction failed: {e}")

        return RegistryMappingResult(
            record=record,
            field_evidence=field_evidence,
            unmapped_entities=unmapped_entities,
            warnings=warnings,
        )

    def _set_nested_field(
        self,
        d: Dict[str, Any],
        path: str,
        value: Any,
    ) -> None:
        """Set a nested field in a dictionary using dot notation."""
        parts = path.split(".")
        current = d

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value
