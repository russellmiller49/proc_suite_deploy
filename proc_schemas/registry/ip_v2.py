"""IP Registry Schema v2.

This is the current production schema for IP procedure registry entries.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Patient demographic information."""

    patient_id: str = ""
    mrn: str = ""
    age: Optional[int] = None
    sex: Optional[Literal["M", "F", "O"]] = None


class ProcedureInfo(BaseModel):
    """Procedure metadata."""

    procedure_id: str = ""
    procedure_date: Optional[date] = None
    procedure_type: str = ""
    indication: str = ""
    urgency: Literal["routine", "urgent", "emergent"] = "routine"


class Sedation(BaseModel):
    """Sedation details."""

    type: Literal["none", "moderate", "deep", "general"] = "moderate"
    agents: List[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = None
    provider: str = ""


class BiopsySite(BaseModel):
    """A biopsy site with location and technique."""

    site: str  # e.g., "4R", "7", "RUL"
    technique: str = ""  # "EBUS-TBNA", "TBLB", "forceps"
    passes: Optional[int] = None
    rose_result: Optional[str] = None


class StentPlacement(BaseModel):
    """Stent placement details."""

    location: str  # "trachea", "RMS", "LMS"
    type: str = ""  # "silicone", "metal"
    size: str = ""  # "14x40mm"
    manufacturer: str = ""


class Finding(BaseModel):
    """A finding from the procedure."""

    category: str  # "anatomic", "pathologic", "incidental"
    description: str
    severity: Optional[str] = None
    action_taken: Optional[str] = None


class Complication(BaseModel):
    """A complication from the procedure."""

    type: str  # "bleeding", "pneumothorax", "hypoxia"
    severity: Literal["mild", "moderate", "severe"] = "mild"
    intervention: Optional[str] = None
    resolved: bool = True


class IPRegistryV2(BaseModel):
    """IP Registry Schema v2 - current production schema."""

    # Metadata
    schema_version: Literal["v2"] = "v2"
    registry_id: str = "ip_registry"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Patient and procedure
    patient: PatientInfo = Field(default_factory=PatientInfo)
    procedure: ProcedureInfo = Field(default_factory=ProcedureInfo)

    # Sedation
    sedation: Sedation = Field(default_factory=Sedation)

    # EBUS-TBNA
    ebus_performed: bool = False
    ebus_stations: List[BiopsySite] = Field(default_factory=list)

    # Transbronchial biopsy
    tblb_performed: bool = False
    tblb_sites: List[BiopsySite] = Field(default_factory=list)

    # Navigation
    navigation_performed: bool = False
    navigation_system: str = ""

    # Radial EBUS
    radial_ebus_performed: bool = False

    # BAL
    bal_performed: bool = False
    bal_sites: List[str] = Field(default_factory=list)

    # Therapeutic procedures
    dilation_performed: bool = False
    dilation_sites: List[str] = Field(default_factory=list)

    stent_placed: bool = False
    stents: List[StentPlacement] = Field(default_factory=list)

    ablation_performed: bool = False
    ablation_technique: str = ""

    blvr_performed: bool = False
    blvr_valves: int = 0
    blvr_target_lobe: str = ""

    # Findings
    findings: List[Finding] = Field(default_factory=list)

    # Complications
    complications: List[Complication] = Field(default_factory=list)
    any_complications: bool = False

    # Disposition
    disposition: str = ""  # "home", "observation", "admit"

    # Free text
    impression: str = ""
    recommendations: str = ""

    model_config = {"frozen": False}
