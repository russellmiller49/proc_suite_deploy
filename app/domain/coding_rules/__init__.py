# Coding rules domain module
from .ncci import apply_ncci_edits, NCCIEdit
from .mer import apply_mer_rules, MERResult
from .rule_engine import RuleEngine, RuleEngineResult, RuleCandidate
from .evidence_context import EvidenceContext, RulesResult
from .coding_rules_engine import CodingRulesEngine
from .json_rules_evaluator import JSONRulesEvaluator, CodingRule

__all__ = [
    # NCCI edits
    "apply_ncci_edits",
    "NCCIEdit",
    # MER rules
    "apply_mer_rules",
    "MERResult",
    # Simplified rule engine (keyword-based)
    "RuleEngine",
    "RuleEngineResult",
    "RuleCandidate",
    # New coding rules engine (Plan 2 migration)
    "EvidenceContext",
    "RulesResult",
    "CodingRulesEngine",
    # JSON rules evaluator
    "JSONRulesEvaluator",
    "CodingRule",
]
