"""Deterministic rules engine for CPT code validation and veto logic.

Responsibilities:
- Hierarchy normalization: stent families, thoracoscopy precedence.
- Bundling: NCCI / payer-specific hard walls.
- Validation: Confirms combos don't violate hard rules.
- Veto logic: Detects impossible/suspicious code combinations.

This keeps LLMs focused on extraction; RulesEngine handles billable logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set

from app.coder.code_families import load_code_families
from app.coder.ncci import NCCIEngine, NCCI_BUNDLED_REASON_PREFIX
from app.coder.types import CodeCandidate


class RuleViolationError(Exception):
    """Raised when a code combination violates hard validation rules.

    This signals that the case requires human review or LLM fallback
    because the ML/rule combination produced an impossible state.
    """

    def __init__(self, message: str, codes_involved: List[str] | None = None):
        super().__init__(message)
        self.codes_involved = codes_involved or []


@dataclass
class ValidationResult:
    """Result from code validation."""

    codes: List[str]
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    removed_codes: Dict[str, str] = field(default_factory=dict)  # code -> reason


# Biopsy-related terms that indicate actual tissue sampling
BIOPSY_TERMS = [
    "forceps biopsy",
    "forcep biopsy",
    "cryobiopsy",
    "cryo biopsy",
    "biopsies obtained",
    "biopsy obtained",
    "transbronchial biopsy",
    "tbbx",
    "tbna",
    "needle aspiration",
    "tissue sample",
    "specimens obtained",
    "pathology specimens",
]

# Lavage-only terms (suggest BAL without biopsy)
LAVAGE_ONLY_TERMS = [
    "bal only",
    "lavage only",
    "wash only",
    "no biopsy",
    "bronchoalveolar lavage without biopsy",
]

# Mutual exclusion rules: (code_a, code_b, reason)
MUTUAL_EXCLUSIONS = [
    ("31652", "31653", "Cannot bill both 1-2 station and 3+ station EBUS-TBNA"),
    ("31640", "31641", "Cannot bill both mechanical debulking and ablation for same lesion"),
]


class CodingRulesEngine:
    """Deterministic rules layer with validation and veto logic.

    Provides:
    - Hierarchy normalization using code family definitions
    - NCCI bundling gatekeeper
    - Validation API for ML/LLM output checking
    - Veto logic for impossible code combinations
    """

    def __init__(
        self,
        families_cfg: Dict[str, Any] | None = None,
        ncci_engine: NCCIEngine | None = None,
    ):
        self._families_cfg = families_cfg or load_code_families()
        self._ncci = ncci_engine or NCCIEngine()

    def _compute_replacements(self, codes: Set[str]) -> Dict[str, str]:
        """Build mapping of target_code -> replacement_code."""
        replacements: Dict[str, str] = {}
        families = (self._families_cfg.get("families") if self._families_cfg else None) or {}

        for family in families.values():
            dominant_map = (family.get("dominant_codes") or {}) if isinstance(family, dict) else {}
            for dominant_code, rule in dominant_map.items():
                if dominant_code not in codes:
                    continue
                overrides = (rule.get("overrides") or {}) if isinstance(rule, dict) else {}
                for target_code, replacement_code in overrides.items():
                    if target_code in codes:
                        replacements[target_code] = replacement_code

        return replacements

    def _apply_hierarchy_normalization(
        self,
        candidates: Sequence[CodeCandidate],
    ) -> list[CodeCandidate]:
        if not candidates:
            return []

        present_codes = {candidate.code for candidate in candidates}
        replacements = self._compute_replacements(present_codes)

        normalized: list[CodeCandidate] = []
        seen: Set[str] = set()

        for candidate in candidates:
            original_code = candidate.code
            replacement_code = replacements.get(original_code)

            if replacement_code:
                if replacement_code in seen:
                    continue
                reason = candidate.reason or ""
                hierarchy_note = f"hierarchy:{original_code}->{replacement_code}"
                reason = f"{reason}|{hierarchy_note}" if reason else hierarchy_note
                normalized.append(
                    CodeCandidate(
                        code=replacement_code,
                        confidence=candidate.confidence,
                        reason=reason,
                        evidence=candidate.evidence,
                    )
                )
                seen.add(replacement_code)
                continue

            if original_code in seen:
                continue
            normalized.append(candidate)
            seen.add(original_code)

        return normalized

    def _apply_ncci_bundling(
        self,
        candidates: Sequence[CodeCandidate],
    ) -> list[CodeCandidate]:
        if not candidates:
            return []

        codes = {candidate.code for candidate in candidates}
        result = self._ncci.apply(codes)
        allowed = result.allowed
        bundled_map = result.bundled

        bundled_candidates: list[CodeCandidate] = []
        for candidate in candidates:
            code = candidate.code
            if code in allowed:
                bundled_candidates.append(candidate)
                continue
            if code in bundled_map:
                primary = bundled_map[code]
                marker = f"{NCCI_BUNDLED_REASON_PREFIX}{primary}"
                reason = candidate.reason or ""
                reason = f"{reason}|{marker}" if reason else marker
                bundled_candidates.append(
                    CodeCandidate(
                        code=code,
                        confidence=candidate.confidence,
                        reason=reason,
                        evidence=candidate.evidence,
                    )
                )
            else:
                bundled_candidates.append(candidate)

        return bundled_candidates

    def apply(self, candidates: Sequence[CodeCandidate], note_text: str) -> list[CodeCandidate]:
        """Apply deterministic rules to candidate codes."""
        normalized = self._apply_hierarchy_normalization(candidates)
        bundled = self._apply_ncci_bundling(normalized)
        return bundled

    # -------------------------------------------------------------------------
    # Validation API for ML/LLM output checking
    # -------------------------------------------------------------------------

    def validate(
        self,
        codes: List[str],
        note_text: str,
        context: Dict[str, Any] | None = None,
        strict: bool = False,
    ) -> List[str]:
        """
        Validate a list of CPT codes against hard rules.

        This is the primary entry point for ML/LLM output validation.

        Args:
            codes: List of CPT codes to validate
            note_text: The procedure note text for context-aware validation
            context: Optional dict with pre-parsed metadata (registry_entry, tools used, etc.)
                     TODO: Use context to avoid re-parsing note_text when available
            strict: If True, raise RuleViolationError on any violation.
                    If False, remove violating codes and return cleaned list.

        Returns:
            List of validated/cleaned CPT codes

        Raises:
            RuleViolationError: If strict=True and violations are found
        """
        if not codes:
            return []

        result = self.validate_detailed(codes, note_text, context)

        if strict and result.violations:
            raise RuleViolationError(
                "; ".join(result.violations),
                codes_involved=list(result.removed_codes.keys()),
            )

        return result.codes

    def validate_detailed(
        self,
        codes: List[str],
        note_text: str,
        context: Dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Detailed validation with full results.

        Args:
            codes: List of CPT codes to validate
            note_text: The procedure note text
            context: Optional pre-parsed metadata

        Returns:
            ValidationResult with codes, violations, warnings, and removed codes
        """
        cleaned = list(codes)
        violations: List[str] = []
        warnings: List[str] = []
        removed: Dict[str, str] = {}

        # Apply validation rules in order
        cleaned, v, w, r = self._check_mutual_exclusions(cleaned)
        violations.extend(v)
        warnings.extend(w)
        removed.update(r)

        cleaned, v, w, r = self._check_biopsy_vs_lavage(cleaned, note_text)
        violations.extend(v)
        warnings.extend(w)
        removed.update(r)

        cleaned, v, w, r = self._check_ebus_station_count(cleaned, note_text)
        violations.extend(v)
        warnings.extend(w)
        removed.update(r)

        cleaned, v, w, r = self._check_diagnostic_with_therapeutic(cleaned)
        violations.extend(v)
        warnings.extend(w)
        removed.update(r)

        # TODO: Add more validation rules as needed
        # - Stent rules (31631/31636/31637 combinations)
        # - Thoracoscopy bundling rules
        # - Navigation requirements for certain add-ons

        return ValidationResult(
            codes=cleaned,
            violations=violations,
            warnings=warnings,
            removed_codes=removed,
        )

    def _check_mutual_exclusions(
        self, codes: List[str]
    ) -> tuple[List[str], List[str], List[str], Dict[str, str]]:
        """Check for mutually exclusive code pairs."""
        violations: List[str] = []
        warnings: List[str] = []
        removed: Dict[str, str] = {}
        code_set = set(codes)

        for code_a, code_b, reason in MUTUAL_EXCLUSIONS:
            if code_a in code_set and code_b in code_set:
                # Keep the higher-value code (typically the one with more work)
                # For EBUS: keep 31653 (3+ stations) over 31652 (1-2 stations)
                # For debulking: this is an error - need human review
                if code_a == "31652" and code_b == "31653":
                    code_set.discard(code_a)
                    removed[code_a] = reason
                    warnings.append(f"Removed {code_a}: {reason}")
                elif code_a == "31640" and code_b == "31641":
                    # Both present is suspicious - flag for review
                    violations.append(reason)
                else:
                    # Default: remove first code
                    code_set.discard(code_a)
                    removed[code_a] = reason
                    warnings.append(f"Removed {code_a}: {reason}")

        return list(code_set), violations, warnings, removed

    def _check_biopsy_vs_lavage(
        self, codes: List[str], note_text: str
    ) -> tuple[List[str], List[str], List[str], Dict[str, str]]:
        """
        Check for biopsy vs lavage conflicts.

        If biopsy terms dominate the note but lavage code is present,
        flag for review.
        """
        violations: List[str] = []
        warnings: List[str] = []
        removed: Dict[str, str] = {}

        lavage_code = "31624"
        if lavage_code not in codes:
            return codes, violations, warnings, removed

        text_lower = note_text.lower()

        # Count biopsy evidence
        biopsy_evidence = sum(
            1 for term in BIOPSY_TERMS if term in text_lower
        )

        # Check for lavage-only indicators
        lavage_only_evidence = any(term in text_lower for term in LAVAGE_ONLY_TERMS)

        # Strong biopsy evidence with no lavage-only indicator = suspicious
        if biopsy_evidence >= 2 and not lavage_only_evidence:
            # Check if "lavage" or "bal" is actually mentioned
            has_lavage_mention = any(
                term in text_lower for term in ["lavage", "bal", "bronchoalveolar"]
            )
            if not has_lavage_mention:
                violations.append(
                    f"Lavage code {lavage_code} present but biopsy evidence dominates "
                    f"({biopsy_evidence} biopsy terms found, no lavage mention)"
                )

        return codes, violations, warnings, removed

    def _check_ebus_station_count(
        self, codes: List[str], note_text: str
    ) -> tuple[List[str], List[str], List[str], Dict[str, str]]:
        """
        Check EBUS station count consistency.

        31652 = 1-2 stations, 31653 = 3+ stations
        Verify the code matches the documented station count.
        """
        violations: List[str] = []
        warnings: List[str] = []
        removed: Dict[str, str] = {}

        # If neither EBUS code is present, nothing to check
        if "31652" not in codes and "31653" not in codes:
            return codes, violations, warnings, removed

        # Try to count stations from the note
        # TODO: Use parsed context if available instead of re-parsing
        text_lower = note_text.lower()

        # Common station patterns: 4R, 4L, 7, 10R, 10L, 11R, 11L, etc.
        station_pattern = r"\b(station\s+)?([247]|4[rl]|10[rl]|11[rl]|12[rl])\b"
        stations_found = set(re.findall(station_pattern, text_lower, re.IGNORECASE))

        # Also check for "X stations sampled" pattern
        count_pattern = r"(\d+)\s+(?:lymph\s+node\s+)?stations?\s+(?:sampled|biopsied|aspirated)"
        count_matches = re.findall(count_pattern, text_lower)

        station_count = len(stations_found)
        if count_matches:
            # Use explicit count if mentioned
            explicit_count = max(int(m) for m in count_matches)
            station_count = max(station_count, explicit_count)

        # Validate code against count
        if "31653" in codes and station_count > 0 and station_count < 3:
            warnings.append(
                f"Code 31653 (3+ stations) but only {station_count} station(s) documented. "
                "Verify station count."
            )
        elif "31652" in codes and station_count >= 3:
            warnings.append(
                f"Code 31652 (1-2 stations) but {station_count} stations documented. "
                "Consider upgrading to 31653."
            )

        return codes, violations, warnings, removed

    def _check_diagnostic_with_therapeutic(
        self, codes: List[str]
    ) -> tuple[List[str], List[str], List[str], Dict[str, str]]:
        """
        Check that diagnostic bronchoscopy (31622) isn't billed with therapeutic codes.

        Per coding guidelines, 31622 is bundled into therapeutic bronchoscopy codes.
        """
        violations: List[str] = []
        warnings: List[str] = []
        removed: Dict[str, str] = {}

        diagnostic_code = "31622"
        if diagnostic_code not in codes:
            return codes, violations, warnings, removed

        # Therapeutic codes that bundle diagnostic
        therapeutic_codes = {
            "31623", "31624", "31625", "31626", "31627", "31628", "31629",
            "31631", "31632", "31633", "31634", "31636", "31637", "31638",
            "31640", "31641", "31645", "31646", "31647",
            "31652", "31653", "31654",
        }

        code_set = set(codes)
        has_therapeutic = bool(code_set & therapeutic_codes)

        if has_therapeutic:
            code_set.discard(diagnostic_code)
            removed[diagnostic_code] = "Diagnostic bundled into therapeutic bronchoscopy"
            warnings.append(
                f"Removed {diagnostic_code}: bundled into therapeutic bronchoscopy codes"
            )

        return list(code_set), violations, warnings, removed
