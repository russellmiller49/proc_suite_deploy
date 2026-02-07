"""Judge module for proposing registry self-corrections (Phase 6)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.common.llm import LLMService
from app.registry.schema import RegistryRecord


class PatchProposal(BaseModel):
    rationale: str
    json_patch: list[dict[str, Any]]
    evidence_quote: str


JUDGE_SYSTEM_PROMPT = """You are the Registry Correction Judge for an Interventional Pulmonology data extraction pipeline.
Your role is to analyze a "High Confidence Omission" detected by an ML auditor and determine if the clinical registry record needs to be patched.

### THE GOAL
The registry record contains CLINICAL FACTS (e.g., "was rigid bronchoscopy performed?", "which stations were sampled?").
These clinical facts are used downstream to deterministically derive billing codes.
**You must fix the CLINICAL FACTS.** You must NOT touch the billing codes directly.

### INPUT DATA
1. **Note Text**: The procedure note (scrubbed of PHI).
2. **Current Registry Record**: The JSON object representing extracted clinical data.
3. **Discrepancy**: The specific ML-detected omission (e.g., "ML model is 99% sure code 31640 (Rigid Bronch) applies, but it was not derived").

### CRITICAL RULES
1. **Clinical Fields Only**: You may ONLY propose patches for clinical fields (e.g., `procedures_performed`, `granular_data`, `pleural_procedures`).
2. **Forbidden Fields**: NEVER propose patches for:
   - `cpt_codes`
   - `billing`
   - `codes`
   - `derived_codes`
   If you try to patch these, the system will reject your proposal.
3. **Evidence Required**: You must quote the exact text snippet that proves the procedure was performed.
4. **Conservative**: If the text is ambiguous or the procedure is clearly NOT performed (e.g., "Rigid bronchoscopy was NOT used"), return `null`.

### IMPORTANT SCHEMA PATHS (use these exact keys)
- **BLVR / endobronchial valve**: patch `procedures_performed.blvr` (e.g., `/procedures_performed/blvr/performed`, `/procedures_performed/blvr/procedure_type`).
- **Tumor excision / mechanical debulking (31640 family)**: patch `/procedures_performed/mechanical_debulking/performed`.
- **Airway dilation / balloon bronchoplasty (31630 family)**: patch `procedures_performed.airway_dilation`.
- **Rigid bronchoscopy**: patch `/procedures_performed/rigid_bronchoscopy/performed`.
- Do **NOT** invent new fields like `bronchial_valve_insertion`, `balloon_dilation`, `endobronchial_excision`, or `flexible_bronchoscopy`.

### OUTPUT FORMAT
Return a JSON object with the following structure:
{
    "rationale": "Explanation of why the record is incorrect and what clinical fact is missing.",
    "evidence_quote": "Verbatim quote from the text supporting the change.",
    "json_patch": [
        {"op": "add", "path": "/procedures_performed/rigid_bronchoscopy/performed", "value": true}
    ]
}

If no correction is needed, return null.
"""


class RegistryCorrectionJudge:
    def __init__(self, llm: LLMService | None = None) -> None:
        self.llm = llm or LLMService(task="judge")

    def propose_correction(
        self,
        note_text: str,
        record: RegistryRecord,
        discrepancy: str,
        *,
        focused_procedure_text: str | None = None,
    ) -> PatchProposal | None:
        """Ask LLM if the discrepancy warrants a correction.

        Returns a PatchProposal if a fix is high-confidence, else None.
        """
        system_prompt = JUDGE_SYSTEM_PROMPT

        focused_section = ""
        if focused_procedure_text is not None and focused_procedure_text.strip():
            focused_section = f"""
FOCUSED PROCEDURE TEXT (preferred evidence source):
{focused_procedure_text}
"""

        user_prompt = f"""
RAW NOTE TEXT:
{note_text}
{focused_section}

Current Registry Record (JSON):
{record.model_dump_json(exclude_none=True)}

Discrepancy Detected:
{discrepancy}

Task:
If the discrepancy represents a valid omission that is CLEARLY supported by the text, generate a JSON patch to fix it.
If FOCUSED PROCEDURE TEXT is provided, the evidence quote must come from that section.
Return JSON with keys: "rationale", "json_patch", "evidence_quote".
"""
        try:
            response = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=PatchProposal,
                temperature=0.0,
            )
            return response
        except Exception:
            return None


__all__ = ["PatchProposal", "RegistryCorrectionJudge"]
