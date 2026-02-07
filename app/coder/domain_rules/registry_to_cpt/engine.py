from __future__ import annotations

from app.coder.domain_rules.registry_to_cpt.coding_rules import derive_all_codes_with_meta
from app.coder.domain_rules.registry_to_cpt.types import DerivedCode, RegistryCPTDerivation
from app.registry.schema import RegistryRecord


class RegistryToCPTDerivationEngine:
    def apply(self, record: RegistryRecord) -> RegistryCPTDerivation:
        codes, rationales, warnings = derive_all_codes_with_meta(record)

        derived = [
            DerivedCode(
                code=code,
                rationale=rationales.get(code, "derived"),
                rule_id=f"registry_to_cpt:{code}",
                confidence=1.0,
            )
            for code in codes
        ]
        return RegistryCPTDerivation(codes=derived, warnings=warnings)


def apply(record: RegistryRecord) -> RegistryCPTDerivation:
    return RegistryToCPTDerivationEngine().apply(record)


__all__ = ["RegistryToCPTDerivationEngine", "apply"]

