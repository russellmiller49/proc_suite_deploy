"""Evidence context for coding rules evaluation.

This module defines the EvidenceContext dataclass that encapsulates all
contextual information needed to evaluate coding rules. This includes:
- Groups detected from text (KB groups)
- Evidence extracted from text analysis
- Registry data from procedure forms
- Initial code candidates
- Term hits from terminology extraction
- Navigation and radial EBUS context

The EvidenceContext is designed to be immutable and serves as the input
to the CodingRulesEngine.apply_rules() method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple


@dataclass(frozen=True)
class EvidenceContext:
    """Immutable context for coding rules evaluation.

    This dataclass captures all the information extracted from a procedure
    note and registry that is needed to evaluate coding rules.

    Attributes:
        groups: Set of KB groups detected from text (e.g., "bronchoscopy_ebus_linear")
        evidence: Evidence dictionary from ip_kb.last_group_evidence
        registry: Registry data from procedure_data.registry
        candidates: Initial code candidates from KB codes_for_groups()
        term_hits: Term extraction results from terminology normalizer
        navigation_context: Navigation-specific registry data
        radial_context: Radial EBUS-specific registry data
        text_lower: Lowercase version of the note text (for term matching)
    """

    groups: FrozenSet[str] = field(default_factory=frozenset)
    evidence: Dict[str, Any] = field(default_factory=dict)
    registry: Dict[str, Any] = field(default_factory=dict)
    candidates: FrozenSet[str] = field(default_factory=frozenset)
    term_hits: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    navigation_context: Dict[str, Any] = field(default_factory=dict)
    radial_context: Dict[str, Any] = field(default_factory=dict)
    text_lower: str = ""

    @classmethod
    def from_procedure_data(
        cls,
        groups_from_text: Set[str],
        evidence: Dict[str, Any],
        registry: Dict[str, Any],
        candidates_from_text: Set[str],
        term_hits: Dict[str, list],
        navigation_context: Dict[str, Any],
        radial_context: Dict[str, Any],
        note_text: str = "",
    ) -> "EvidenceContext":
        """Create an EvidenceContext from mutable procedure data.

        This factory method converts mutable types to immutable equivalents
        for safe storage in the frozen dataclass.

        Args:
            groups_from_text: Set of KB groups detected from text
            evidence: Evidence dictionary from ip_kb.last_group_evidence
            registry: Registry data from procedure_data
            candidates_from_text: Initial code candidates
            term_hits: Term extraction results (will be converted to tuples)
            navigation_context: Navigation-specific registry data
            radial_context: Radial EBUS-specific registry data
            note_text: Original note text

        Returns:
            An immutable EvidenceContext instance
        """
        # Convert mutable collections to immutable equivalents
        immutable_term_hits = {
            k: tuple(v) for k, v in term_hits.items()
        }

        return cls(
            groups=frozenset(groups_from_text),
            evidence=dict(evidence),  # Shallow copy
            registry=dict(registry),  # Shallow copy
            candidates=frozenset(candidates_from_text),
            term_hits=immutable_term_hits,
            navigation_context=dict(navigation_context),
            radial_context=dict(radial_context),
            text_lower=note_text.lower() if note_text else "",
        )

    def get_evidence(self, key: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """Get evidence for a specific group key.

        Args:
            key: Evidence key (e.g., "bronchoscopy_therapeutic_stent")
            default: Default value if key not found

        Returns:
            Evidence dictionary for the key
        """
        return self.evidence.get(key, default or {})

    def has_group(self, group: str) -> bool:
        """Check if a specific group is present.

        Args:
            group: Group name to check

        Returns:
            True if group is in the detected groups
        """
        return group in self.groups

    def has_candidate(self, code: str) -> bool:
        """Check if a code is in the initial candidates.

        Args:
            code: CPT code to check (with or without + prefix)

        Returns:
            True if code is in candidates
        """
        return code in self.candidates or f"+{code}" in self.candidates

    def registry_get(self, *path: str) -> Any:
        """Navigate nested registry data.

        Args:
            *path: Keys to traverse in the registry dict

        Returns:
            Value at the path, or None if not found
        """
        current = self.registry
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current


@dataclass
class RulesResult:
    """Result from applying coding rules.

    This is a mutable result class that accumulates the effects of
    rule application.

    Attributes:
        codes: Final set of CPT codes after rule application
        applied_rules: List of rule IDs that were applied
        removed_codes: Mapping of removed codes to their removal reason
        warnings: List of warning messages from rule evaluation
    """

    codes: Set[str] = field(default_factory=set)
    applied_rules: list[str] = field(default_factory=list)
    removed_codes: Dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_code(self, code: str, rule_id: str) -> None:
        """Add a code via a rule.

        Args:
            code: CPT code to add
            rule_id: ID of the rule that added this code
        """
        self.codes.add(code)
        self.applied_rules.append(f"{rule_id}:add:{code}")

    def remove_code(self, code: str, rule_id: str, reason: str) -> None:
        """Remove a code via a rule.

        Args:
            code: CPT code to remove
            rule_id: ID of the rule that removed this code
            reason: Human-readable reason for removal
        """
        if code in self.codes:
            self.codes.discard(code)
        # Also try with + prefix
        plus_code = f"+{code}"
        if plus_code in self.codes:
            self.codes.discard(plus_code)

        self.removed_codes[code] = reason
        self.applied_rules.append(f"{rule_id}:remove:{code}")

    def upgrade_code(self, from_code: str, to_code: str, rule_id: str) -> None:
        """Replace one code with another.

        Args:
            from_code: Code to remove
            to_code: Code to add in its place
            rule_id: ID of the rule performing the upgrade
        """
        if from_code in self.codes:
            self.codes.discard(from_code)
        if f"+{from_code}" in self.codes:
            self.codes.discard(f"+{from_code}")

        self.codes.add(to_code)
        self.applied_rules.append(f"{rule_id}:upgrade:{from_code}->{to_code}")

    def add_warning(self, warning: str) -> None:
        """Add a warning message.

        Args:
            warning: Warning message
        """
        self.warnings.append(warning)
