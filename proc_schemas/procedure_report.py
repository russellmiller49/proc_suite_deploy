from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class TargetSpecimen(BaseModel):
    lobe: Optional[str] = None
    segment: Optional[str] = None
    guidance: Optional[Literal["radial_ebus", "robotic", "enb", "fluoro", "none"]] = None
    specimens: Dict[Literal["fna", "forceps", "cryo", "brush", "bal"], int] = Field(default_factory=dict)


class ProcedureCore(BaseModel):
    type: Literal[
        "ebus_tbna",
        "bronchoscopy",
        "robotic_nav",
        "cryobiopsy",
        "thoracentesis",
        "ipc",
        "pleuroscopy",
        "stent",
    ]
    laterality: Optional[Literal["left", "right", "bilateral"]] = None
    stations_sampled: List[str] = Field(default_factory=list)
    targets: List[TargetSpecimen] = Field(default_factory=list)
    devices: Dict[str, Any] = Field(default_factory=dict)
    fluoro: Dict[str, Any] = Field(default_factory=dict)


class NLPTrace(BaseModel):
    paragraph_hashes: List[str] = Field(default_factory=list)
    umls: List[Dict[str, Any]] = Field(default_factory=list)


class ProcedureReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meta: Dict[str, Any] = Field(default_factory=dict)
    indication: Dict[str, Any] = Field(default_factory=dict)
    procedure_core: ProcedureCore
    intraop: Dict[str, Any] = Field(default_factory=dict)
    postop: Dict[str, Any] = Field(default_factory=dict)
    nlp: NLPTrace = Field(default_factory=NLPTrace)
    billing: Dict[str, Any] = Field(default_factory=dict)
