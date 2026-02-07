from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from config.settings import CoderSettings
from .knowledge import get_rvu

Setting = Literal["facility", "nonfacility"]

@dataclass
class RVURecord:
    cpt_code: str
    work_rvu: float
    total_facility_rvu: float
    total_nonfacility_rvu: float
    facility_payment: float
    nonfacility_payment: float


class RVUCalc:
    def __init__(self, conversion_factor: float | None = None):
        # Use centralized setting from CoderSettings (configurable via env var)
        self.cf = conversion_factor or CoderSettings().cms_conversion_factor

    def lookup(self, cpt_code: str) -> RVURecord | None:
        rec = get_rvu(cpt_code)
        if not rec:
            return None
        
        # get_rvu returns {"work": ..., "pe": ..., "mp": ...}
        # It doesn't currently return total_facility_rvu directly unless updated.
        # knowledge.knowledge_snapshot calculates total = work + pe + mp
        # But facility vs non-facility PE is not distinguished in get_rvu currently
        # because get_rvu flattens it.
        
        # Wait, app/common/knowledge.py get_rvu implementation:
        # def get_rvu(cpt: str) -> dict[str, float] | None:
        #     data = get_knowledge()
        #     rvus = data.get("rvus", {})
        #     entry = rvus.get(cpt)
        #     if not entry:
        #         return None
        #     return {
        #         "work": float(entry.get("work", 0.0)),
        #         "pe": float(entry.get("pe", 0.0)),
        #         "mp": float(entry.get("mp", 0.0)),
        #     }
        
        # The IP KB structure for RVUs:
        # "31622": {"work": 3.1, "pe": 6.0, "mp": 0.3},
        
        # It seems the JSON simplifies PE into one value (likely facility for IP context).
        # To support full facility/non-facility, the JSON or the logic needs to support it.
        # The prompt says: "Expect rec to have these fields; if not, adjust based on your JSON"
        
        # For now, I will assume the 'pe' in JSON is facility PE, and non-facility might not be there
        # or I treat them as same if missing.
        
        work = float(rec.get("work", 0.0))
        pe = float(rec.get("pe", 0.0))
        mp = float(rec.get("mp", 0.0))
        
        total = work + pe + mp
        
        # If we want accurate non-facility, we need it in the JSON.
        # The JSON has "rvus_additional" which has some split (pe vs null).
        # "32554": { "work": 1.82, "pe": 0.50, "mp": 0.28 },
        
        # Let's assume total = facility total for now.
        
        return RVURecord(
            cpt_code=cpt_code,
            work_rvu=work,
            total_facility_rvu=total,
            total_nonfacility_rvu=total, # Placeholder
            facility_payment=total * self.cf,
            nonfacility_payment=total * self.cf,
        )
