"""Adapter to convert CodingService output to legacy CoderOutput format.

This module provides backward compatibility for the /v1/coder/run endpoint
by converting CodingResult/CodeSuggestion from the new hexagonal architecture
to the legacy CoderOutput/CodeDecision format expected by existing clients.
"""

from __future__ import annotations

import re

from config.settings import CoderSettings
from app.coder.schema import (
    BundleDecision,
    CodeDecision,
    CoderOutput,
    FinancialSummary,
    LLMCodeSuggestion,
    PerCodeBilling,
)
from app.domain.knowledge_base.repository import KnowledgeBaseRepository
from proc_schemas.coding import CodeSuggestion, CodingResult

_BILLABLE_HYBRID_DECISIONS = {
    "accepted_agreement",
    "accepted_hybrid",
    "kept_rule_priority",
    "EXTRACTION_FIRST",
}


def _is_billable_suggestion(suggestion: CodeSuggestion) -> bool:
    """Return True if a suggestion should be treated as billable in legacy output."""
    # Defensive: older callers or manual suggestions may not set hybrid_decision.
    if suggestion.hybrid_decision is None:
        return True
    return suggestion.hybrid_decision in _BILLABLE_HYBRID_DECISIONS


def convert_suggestion_to_code_decision(
    suggestion: CodeSuggestion,
    kb_repo: KnowledgeBaseRepository | None = None,
) -> CodeDecision:
    """Convert a CodeSuggestion to the legacy CodeDecision format.

    Args:
        suggestion: CodeSuggestion from CodingService
        kb_repo: Optional KB repository for RVU lookups

    Returns:
        CodeDecision in the legacy format
    """
    # Build rationale from reasoning
    rationale: list[str] = []
    if suggestion.reasoning:
        if suggestion.reasoning.rule_paths:
            rationale.extend(suggestion.reasoning.rule_paths)
        if suggestion.reasoning.trigger_phrases:
            rationale.append(f"Triggered by: {', '.join(suggestion.reasoning.trigger_phrases)}")
        if suggestion.reasoning.ncci_notes:
            rationale.append(f"NCCI: {suggestion.reasoning.ncci_notes}")
        if suggestion.reasoning.mer_notes:
            rationale.append(f"MER: {suggestion.reasoning.mer_notes}")

    if not rationale:
        rationale = [f"Suggested by {suggestion.source} engine"]

    # Build context dict
    context: dict = {
        "hybrid_decision": suggestion.hybrid_decision,
        "review_flag": suggestion.review_flag,
        "evidence_verified": suggestion.evidence_verified,
    }

    raw_code = suggestion.code
    base_code = raw_code.lstrip("+")
    display_code = base_code

    # Add '+' prefix for add-on codes in legacy output (guard for stub KBs)
    if (
        kb_repo
        and callable(getattr(kb_repo, "is_addon_code", None))
        and kb_repo.is_addon_code(base_code)
    ):
        display_code = f"+{base_code}"

    # Add RVU data if KB available
    rvu_data: dict = {}
    if kb_repo:
        proc_info = kb_repo.get_procedure_info(base_code)
        if proc_info:
            rvu_data = {
                "work_rvu": proc_info.work_rvu,
                "facility_pe_rvu": proc_info.facility_pe_rvu,
                "malpractice_rvu": proc_info.malpractice_rvu,
                "total_rvu": proc_info.total_facility_rvu,
            }
            context["rvu_data"] = rvu_data

    return CodeDecision(
        cpt=display_code,
        description=suggestion.description,
        modifiers=[],  # Modifiers not tracked in CodeSuggestion
        rationale=rationale,
        confidence=suggestion.final_confidence,
        context=context,
        rule_trace=suggestion.reasoning.rule_paths if suggestion.reasoning else [],
    )


def convert_coding_result_to_coder_output(
    result: CodingResult,
    kb_repo: KnowledgeBaseRepository | None = None,
    locality: str = "00",
    conversion_factor: float | None = None,
) -> CoderOutput:
    """Convert a CodingResult to the legacy CoderOutput format.

    Args:
        result: CodingResult from CodingService.generate_result()
        kb_repo: Optional KB repository for RVU lookups
        locality: Geographic locality code for RVU calculations
        conversion_factor: Medicare conversion factor (uses CODER_CMS_CONVERSION_FACTOR if None)

    Returns:
        CoderOutput in the legacy format for API compatibility
    """
    # Use centralized setting if no explicit override provided
    if conversion_factor is None:
        conversion_factor = CoderSettings().cms_conversion_factor

    # Keep legacy output restricted to billable decisions so bundled codes are excluded.
    codes: list[CodeDecision] = []
    billable_codes: list[CodeDecision] = []
    for suggestion in result.suggestions:
        code_decision = convert_suggestion_to_code_decision(suggestion, kb_repo)
        if _is_billable_suggestion(suggestion):
            codes.append(code_decision)
            billable_codes.append(code_decision)

    # Build NCCI actions from warning notes (result-level and per-suggestion reasoning)
    ncci_actions: list[BundleDecision] = []
    ncci_notes: set[str] = set(result.ncci_notes or [])
    for suggestion in result.suggestions:
        if suggestion.reasoning and suggestion.reasoning.ncci_notes:
            for part in suggestion.reasoning.ncci_notes.split(";"):
                note = part.strip()
                if note:
                    ncci_notes.add(note)

    for note in sorted(ncci_notes):
        # Format from CodingService: "NCCI_BUNDLE: <removed> bundled into <primary> - <reason>"
        # Keep it resilient so non-standard notes still appear.
        primary = ""
        removed = ""
        match = re.match(
            r"^NCCI_BUNDLE:\s*(?P<removed>[+]?\d+)\s+bundled\s+into\s+(?P<primary>[+]?\d+)\s+-\s+(?P<reason>.+)$",
            note,
            re.IGNORECASE,
        )
        if match:
            primary = match.group("primary")
            removed = match.group("removed")
        ncci_actions.append(
            BundleDecision(
                pair=(primary, removed),
                action="bundled",
                reason=note,
            )
        )

    # Build financial summary if KB available
    financials: FinancialSummary | None = None
    if kb_repo and billable_codes:
        per_code_billing: list[PerCodeBilling] = []
        total_work_rvu = 0.0
        total_facility_payment = 0.0

        for code_decision in billable_codes:
            proc_info = kb_repo.get_procedure_info(code_decision.cpt.lstrip("+"))
            if proc_info:
                work_rvu = proc_info.work_rvu
                total_rvu = proc_info.total_facility_rvu
                payment = total_rvu * conversion_factor

                total_work_rvu += work_rvu
                total_facility_payment += payment

                per_code_billing.append(PerCodeBilling(
                    cpt_code=code_decision.cpt,
                    description=code_decision.description,
                    modifiers=code_decision.modifiers,
                    work_rvu=work_rvu,
                    total_facility_rvu=total_rvu,
                    facility_payment=payment,
                    allowed_facility_rvu=total_rvu,
                    allowed_facility_payment=payment,
                ))

        if per_code_billing:
            financials = FinancialSummary(
                conversion_factor=conversion_factor,
                locality=locality,
                per_code=per_code_billing,
                total_work_rvu=total_work_rvu,
                total_facility_payment=total_facility_payment,
                total_nonfacility_payment=0.0,
            )

    # Extract LLM suggestions (codes from LLM source)
    llm_suggestions: list[LLMCodeSuggestion] = []
    for suggestion in result.suggestions:
        if suggestion.source in ("llm", "hybrid"):
            rationale = ""
            if suggestion.reasoning and suggestion.reasoning.rule_paths:
                rationale = "; ".join(suggestion.reasoning.rule_paths)
            llm_suggestions.append(LLMCodeSuggestion(
                cpt=suggestion.code,
                description=suggestion.description,
                rationale=rationale,
            ))

    # Collect all warnings
    warnings = list(result.warnings)
    warnings.extend(sorted(ncci_notes))
    warnings.extend(result.mer_notes)

    return CoderOutput(
        codes=codes,
        intents=[],  # Intents not tracked in new architecture
        mer_summary=None,
        ncci_actions=ncci_actions,
        warnings=warnings,
        version=f"new_arch_{result.policy_version}",
        financials=financials,
        llm_suggestions=llm_suggestions,
        llm_disagreements=[],  # Could extract from hybrid decisions
    )
