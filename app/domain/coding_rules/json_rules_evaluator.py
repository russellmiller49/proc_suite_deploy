"""JSON-based coding rules evaluator.

This module provides a declarative rules engine that evaluates JSON-defined
coding rules against an EvidenceContext. It extends the base DSL evaluator
with coding-specific operators for code manipulation.

Key concepts:
- Rules are loaded from coding_rules.v1.json
- Each rule has: id, phase, priority, when (predicate), then (action)
- Phases: filter, inference, validation, exclusion
- Actions: add_code, remove_code, upgrade_code
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.common.rules_engine.dsl import evaluate_predicate
from app.domain.coding_rules.evidence_context import EvidenceContext, RulesResult


@dataclass
class CodingRule:
    """A single coding rule loaded from JSON."""

    id: str
    name: str
    phase: str
    priority: int
    enabled: bool
    description: str
    when: Dict[str, Any]
    then: Dict[str, Any]
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingRule":
        """Create a CodingRule from a JSON dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            phase=data.get("phase", "validation"),
            priority=data.get("priority", 500),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            when=data.get("when", {}),
            then=data.get("then", {}),
            notes=data.get("notes", ""),
        )


@dataclass
class JSONRulesEvaluator:
    """Evaluator for JSON-based coding rules.

    Loads rules from coding_rules.v1.json and applies them to an EvidenceContext.
    Supports custom operators for coding domain:
    - has_group: Check if a group is present
    - has_candidate: Check if a code is in candidates
    - any_candidate: Check if any of a list of codes is in candidates
    - any_term: Check if any term is in text
    - length_gt: Check if list length > N
    - gte, gt, lt, lte: Numeric comparisons
    - eq: Equality check
    """

    rules: List[CodingRule] = field(default_factory=list)
    code_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    version: str = ""

    @classmethod
    def load_from_file(cls, path: Optional[Path] = None) -> "JSONRulesEvaluator":
        """Load rules from the default or specified JSON file."""
        if path is None:
            # Default path
            repo_root = Path(__file__).parent.parent.parent.parent
            path = repo_root / "data" / "rules" / "coding_rules.v1.json"

        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        rules = [CodingRule.from_dict(r) for r in data.get("rules", [])]
        # Sort by priority
        rules.sort(key=lambda r: r.priority)

        return cls(
            rules=rules,
            code_metadata=data.get("code_metadata", {}),
            version=data.get("meta", {}).get("version", "unknown"),
        )

    def apply_rules(
        self,
        context: EvidenceContext,
        valid_cpts: Optional[Set[str]] = None,
    ) -> RulesResult:
        """Apply all enabled rules to the context.

        Args:
            context: The evidence context to evaluate against
            valid_cpts: Set of valid CPT codes (for R001 filter)

        Returns:
            RulesResult with final codes, applied rules, and warnings
        """
        result = RulesResult()
        result.codes = set(context.candidates)

        # Build evaluation context
        eval_context = self._build_eval_context(context, result, valid_cpts)

        # Apply rules in priority order (already sorted)
        for rule in self.rules:
            if not rule.enabled:
                continue

            # Update candidates in context for each rule evaluation
            eval_context["candidates"] = result.codes

            try:
                if self._evaluate_when(rule.when, eval_context):
                    self._execute_then(rule, eval_context, result)
            except Exception as e:
                result.add_warning(f"Rule {rule.id} evaluation error: {e}")

        return result

    def _build_eval_context(
        self,
        context: EvidenceContext,
        result: RulesResult,
        valid_cpts: Optional[Set[str]],
    ) -> Dict[str, Any]:
        """Build the evaluation context dict for predicate evaluation."""
        return {
            # Core context fields
            "groups": context.groups,
            "evidence": context.evidence,
            "registry": context.registry,
            "candidates": result.codes,
            "term_hits": dict(context.term_hits),
            "navigation_context": context.navigation_context,
            "radial_context": context.radial_context,
            "text_lower": context.text_lower,
            "valid_cpts": valid_cpts or set(),
            # Computed fields
            "_stent_evidence_present": self._check_stent_evidence(context),
        }

    def _check_stent_evidence(self, context: EvidenceContext) -> bool:
        """Check if stent 4-gate evidence is present."""
        stent_ev = context.evidence.get("bronchoscopy_therapeutic_stent", {})
        return bool(
            stent_ev.get("stent_word")
            and stent_ev.get("placement_action")
            and (stent_ev.get("tracheal_location") or stent_ev.get("bronchial_location"))
            and not stent_ev.get("stent_negated")
        )

    def _evaluate_when(self, predicate: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate the 'when' predicate with coding-specific operators."""
        if not predicate:
            return True

        # Handle standard logical operators with recursion for custom operators
        if "and" in predicate:
            return all(self._evaluate_when(item, context) for item in predicate["and"])

        if "or" in predicate:
            return any(self._evaluate_when(item, context) for item in predicate["or"])

        if "not" in predicate:
            return not self._evaluate_when(predicate["not"], context)

        if "var" in predicate:
            return bool(self._get_nested(context, predicate["var"]))

        # Handle custom operators
        if "has_group" in predicate:
            group = predicate["has_group"]
            return group in context.get("groups", set())

        if "has_candidate" in predicate:
            code = predicate["has_candidate"]
            candidates = context.get("candidates", set())
            return code in candidates or f"+{code}" in candidates

        if "any_candidate" in predicate:
            codes = predicate["any_candidate"]
            candidates = context.get("candidates", set())
            return any(
                c in candidates or f"+{c}" in candidates or c.lstrip("+") in {x.lstrip("+") for x in candidates}
                for c in codes
            )

        if "any_term" in predicate:
            config = predicate["any_term"]
            terms = config.get("terms", [])
            text = context.get(config.get("in", "text_lower"), "")
            return any(term in text for term in terms)

        if "length_gt" in predicate:
            args = predicate["length_gt"]
            value = self._resolve_value(args[0], context)
            threshold = args[1]
            if isinstance(value, (list, tuple, set)):
                return len(value) > threshold
            return False

        if "count_gt" in predicate:
            args = predicate["count_gt"]
            matcher = args[0]
            threshold = args[1]
            if "candidates_matching" in matcher:
                codes = matcher["candidates_matching"]
                candidates = context.get("candidates", set())
                count = sum(1 for c in codes if c in candidates or f"+{c}" in candidates)
                return count > threshold
            return False

        if "gte" in predicate:
            args = predicate["gte"]
            left = self._resolve_value(args[0], context)
            right = self._resolve_value(args[1], context)
            return (left or 0) >= (right or 0)

        if "gt" in predicate:
            args = predicate["gt"]
            left = self._resolve_value(args[0], context)
            right = self._resolve_value(args[1], context)
            return (left or 0) > (right or 0)

        if "lt" in predicate:
            args = predicate["lt"]
            left = self._resolve_value(args[0], context)
            right = self._resolve_value(args[1], context)
            return (left or 0) < (right or 0)

        if "eq" in predicate:
            args = predicate["eq"]
            left = self._resolve_value(args[0], context)
            right = self._resolve_value(args[1], context)
            return left == right

        if "not_in" in predicate:
            args = predicate["not_in"]
            item = self._resolve_value(args[0], context)
            collection = self._resolve_value(args[1], context) or set()
            return item not in collection

        if "in" in predicate:
            args = predicate["in"]
            item = self._resolve_value(args[0], context)
            collection = self._resolve_value(args[1], context)
            if collection is None:
                return False
            return item in collection

        # Fall back to base DSL evaluator for standard operators
        return evaluate_predicate(predicate, context)

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value, handling var references."""
        if isinstance(value, dict) and "var" in value:
            path = value["var"]
            return self._get_nested(context, path)
        return value

    def _get_nested(self, context: Dict[str, Any], path: str) -> Any:
        """Get a nested value from context using dot notation."""
        parts = path.split(".")
        current = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _execute_then(
        self,
        rule: CodingRule,
        context: Dict[str, Any],
        result: RulesResult,
    ) -> None:
        """Execute the 'then' action(s) of a rule."""
        then = rule.then

        # Single action
        if "action" in then:
            self._execute_action(then, rule.id, context, result)

        # Multiple actions
        if "actions" in then:
            for action in then["actions"]:
                self._execute_action(action, rule.id, context, result)

        # Conditional actions (if/else-if/else chain - first match wins)
        if "conditional_actions" in then:
            any_condition_matched = False
            for cond_action in then["conditional_actions"]:
                if "if" in cond_action:
                    if self._evaluate_when(cond_action["if"], context):
                        self._execute_action(cond_action["then"], rule.id, context, result)
                        any_condition_matched = True
                        # Don't break - allow multiple independent if conditions to fire
                        # But track that at least one matched (to skip else)
                elif "else" in cond_action:
                    # Fallback action - only executes if NO if-conditions matched
                    if not any_condition_matched:
                        self._execute_action(cond_action["else"], rule.id, context, result)

        # Site priority selection (for thoracoscopy)
        if "site_priority_select" in then:
            self._execute_site_priority(then["site_priority_select"], rule.id, context, result)

    def _execute_action(
        self,
        action: Dict[str, Any],
        rule_id: str,
        context: Dict[str, Any],
        result: RulesResult,
    ) -> None:
        """Execute a single action."""
        action_type = action.get("action")
        code = action.get("code", "")
        reason = action.get("reason", "")

        if action_type == "remove_code":
            result.remove_code(code, rule_id, reason)
        elif action_type == "add_code":
            # Check condition if present
            condition = action.get("condition", {})
            if condition:
                if "was_candidate" in condition:
                    orig_code = condition["was_candidate"]
                    orig_candidates = context.get("_original_candidates", context.get("candidates", set()))
                    if orig_code not in orig_candidates and f"+{orig_code}" not in orig_candidates:
                        return
            result.add_code(code, rule_id)
        elif action_type == "upgrade_code":
            from_code = action.get("from_code", "")
            to_code = action.get("to_code", "")
            result.upgrade_code(from_code, to_code, rule_id)
        elif action_type == "filter_out_of_domain":
            # Special action for R001: remove codes not in valid_cpts
            valid_cpts = context.get("valid_cpts", set())
            if valid_cpts:
                codes_to_remove = []
                for c in result.codes:
                    norm_code = c.lstrip("+")
                    if norm_code not in valid_cpts:
                        codes_to_remove.append(c)
                for c in codes_to_remove:
                    result.remove_code(c, rule_id, reason)

    def _execute_site_priority(
        self,
        config: Dict[str, Any],
        rule_id: str,
        context: Dict[str, Any],
        result: RulesResult,
    ) -> None:
        """Execute site priority selection for thoracoscopy."""
        priority_order = config.get("priority_order", [])
        evidence_path = config.get("evidence_path", "")
        thoracoscopy_ev = self._get_nested(context, evidence_path) or {}

        candidates = result.codes
        remaining = {c for c in ["32601", "32604", "32606", "32607", "32609"] if c in candidates}

        if len(remaining) <= 1:
            return

        codes_to_keep: Set[str] = set()

        for priority_item in priority_order:
            if "site" in priority_item:
                site_key = priority_item["site"]
                code = priority_item.get("code")
                if thoracoscopy_ev.get(site_key) and code and code in remaining:
                    codes_to_keep.add(code)
            elif "default_priority" in priority_item:
                if thoracoscopy_ev.get("has_biopsy") and not codes_to_keep:
                    for preferred in priority_item["default_priority"]:
                        if preferred in remaining:
                            codes_to_keep.add(preferred)
                            break
            elif "fallback" in priority_item:
                fallback = priority_item["fallback"]
                if not codes_to_keep and fallback in remaining:
                    codes_to_keep.add(fallback)

        # Remove codes not in codes_to_keep
        for code in remaining - codes_to_keep:
            result.remove_code(code, rule_id, "Site priority selection")
