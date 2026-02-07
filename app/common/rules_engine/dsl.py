"""Minimal JSONLogic-like helper utilities for declarative rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

JSONValue = Any

__all__ = [
    "Rule",
    "evaluate_predicate",
    "run_rules",
]


@dataclass(slots=True)
class Rule:
    """Declarative rule definition."""

    name: str
    when: JSONValue
    action: Mapping[str, Any]


def run_rules(rules: Sequence[Rule], context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Evaluate *rules* against *context* returning matched actions."""

    results: list[Mapping[str, Any]] = []
    for rule in rules:
        if evaluate_predicate(rule.when, context):
            results.append(rule.action)
    return results


def evaluate_predicate(predicate: JSONValue, context: Mapping[str, Any]) -> Any:
    """Evaluate a JSONLogic-style predicate tree against *context*."""

    if isinstance(predicate, Mapping):
        if not predicate:
            return True
        if len(predicate) > 1:
            raise ValueError("Predicates must contain a single operator")
        operator, argument = next(iter(predicate.items()))
        return _evaluate_operator(operator, argument, context)

    if isinstance(predicate, (list, tuple)):
        return [evaluate_predicate(item, context) for item in predicate]

    return predicate


def _evaluate_operator(operator: str, value: Any, context: Mapping[str, Any]) -> Any:
    if operator == "var":
        return _resolve_var(value, context)
    if operator == "and":
        return all(evaluate_predicate(item, context) for item in value)
    if operator == "or":
        return any(evaluate_predicate(item, context) for item in value)
    if operator == "not":
        return not evaluate_predicate(value, context)
    if operator in {"==", "!=", ">", ">=", "<", "<="}:
        left, right = (evaluate_predicate(item, context) for item in value)
        return _compare(operator, left, right)
    if operator == "in":
        left, right = (evaluate_predicate(item, context) for item in value)
        return left in right
    if operator == "if":
        condition, when_true, when_false = value
        return evaluate_predicate(when_true, context) if evaluate_predicate(condition, context) else evaluate_predicate(when_false, context)
    raise KeyError(f"Unsupported operator: {operator}")


def _resolve_var(identifier: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(identifier, str):
        parts = identifier.split(".")
        current: Any = context
        for part in parts:
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None
        return current
    return identifier


def _compare(operator: str, left: Any, right: Any) -> bool:
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    raise KeyError(operator)
