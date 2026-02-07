from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from pydantic import BaseModel, Field

from proc_schemas.clinical.common import ProcedureBundle


class PatchResult(BaseModel):
    changes: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def merge(self, other: "PatchResult") -> "PatchResult":
        merged = deepcopy(self.changes)
        _merge_dicts(merged, other.changes)
        return PatchResult(changes=merged, notes=[*self.notes, *other.notes])


def _merge_dicts(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict):
            existing = target.setdefault(key, {})
            if isinstance(existing, dict):
                _merge_dicts(existing, value)
            else:
                target[key] = deepcopy(value)
        else:
            target[key] = value


class InferenceEngine:
    def infer_procedure(self, proc: Any, bundle: ProcedureBundle) -> PatchResult:
        # Placeholder for future procedure-specific inference rules.
        return PatchResult()

    def infer_bundle(self, bundle: ProcedureBundle) -> PatchResult:
        result = PatchResult()
        for proc in bundle.procedures:
            proc_result = self.infer_procedure(proc, bundle)
            result = result.merge(proc_result)

        anesthesia_patch = self._infer_anesthesia(bundle)
        result = result.merge(anesthesia_patch)
        return result

    def _infer_anesthesia(self, bundle: ProcedureBundle) -> PatchResult:
        result = PatchResult()
        sedation = bundle.sedation
        anesthesia = bundle.anesthesia
        if anesthesia and anesthesia.type:
            return result
        sedation_texts = []
        if sedation:
            if getattr(sedation, "description", None):
                sedation_texts.append(sedation.description)
            if getattr(sedation, "type", None):
                sedation_texts.append(sedation.type)
            meds = getattr(sedation, "medications", None)
            if meds:
                sedation_texts.extend(meds if isinstance(meds, list) else [meds])
        joined = " ".join([text or "" for text in sedation_texts]).lower()
        if "propofol" in joined:
            result.changes.setdefault("bundle", {}).setdefault("anesthesia", {})["type"] = "Deep Sedation / TIVA"
            result.notes.append("Inferred anesthesia type based on propofol use.")
        return result


__all__ = ["InferenceEngine", "PatchResult"]
