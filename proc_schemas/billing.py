from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BillingLine(BaseModel):
    cpt: str
    units: int = 1
    modifiers: List[str] = Field(default_factory=list)
    reason: str
    evidence: List[Dict[str, Any]]
    rvu: Optional[Dict[str, Any]] = None


class BillingResult(BaseModel):
    codes: List[BillingLine] = Field(default_factory=list)
    ncci_conflicts: List[Dict[str, str]] = Field(default_factory=list)
    confidence: float = 0.0
    audit_trail: Dict[str, Any] = Field(default_factory=dict)
    review_required: bool = False
    financials: Optional[Dict[str, Any]] = None
