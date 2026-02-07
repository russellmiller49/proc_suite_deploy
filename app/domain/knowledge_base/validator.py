"""Semantic validator for Knowledge Base integrity.

Treats the Knowledge Base JSON as deployable code by enforcing:
1) Referential integrity: code references must exist in master_code_index.
2) Bundling cycles: no circular "bundled into" relationships across KB rules.
3) RVU completeness: billable CPT codes must have non-negative work RVUs.
"""

from __future__ import annotations

import re
from typing import Any


class SemanticValidator:
    """Run semantic integrity checks over a loaded KB JSON document."""

    _CODE_TOKEN_RE = re.compile(r"^\+?(?:\d{5}|[A-Z]\d{4})$")

    def __init__(self, kb_data: dict[str, Any]):
        self.data = kb_data
        self.master_index = kb_data.get("master_code_index", {})
        self.issues: list[str] = []

    def validate(self) -> list[str]:
        """Run all semantic checks."""
        self.check_referential_integrity()
        self.check_bundling_cycles()
        self.check_rvu_completeness()
        return self.issues

    def _fail(self, msg: str) -> None:
        self.issues.append(msg)

    @staticmethod
    def _normalize_code(code: str) -> str:
        return code.strip().upper().lstrip("+")

    @classmethod
    def _is_code_token(cls, value: str) -> bool:
        return bool(cls._CODE_TOKEN_RE.match(value.strip().upper()))

    def _iter_code_tokens(self, obj: Any, path: tuple[str, ...] = ()) -> list[tuple[str, tuple[str, ...]]]:
        found: list[tuple[str, tuple[str, ...]]] = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and self._is_code_token(k):
                    found.append((k, path + (k,)))
                if isinstance(k, str):
                    found.extend(self._iter_code_tokens(v, path + (k,)))
                else:
                    found.extend(self._iter_code_tokens(v, path))
            return found

        if isinstance(obj, list):
            for i, item in enumerate(obj):
                found.extend(self._iter_code_tokens(item, path + (f"[{i}]",)))
            return found

        if isinstance(obj, str) and self._is_code_token(obj):
            found.append((obj, path))
            return found

        return found

    @staticmethod
    def _format_path(path: tuple[str, ...]) -> str:
        if not path:
            return "<root>"
        out = ""
        for part in path:
            if part.startswith("[") and part.endswith("]"):
                out += part
            else:
                out = f"{out}.{part}" if out else part
        return out

    def check_referential_integrity(self) -> None:
        """Ensure all referenced codes exist in master_code_index."""
        if not isinstance(self.master_index, dict) or not self.master_index:
            self._fail("INTEGRITY: master_code_index missing or empty")
            return

        master_codes = {self._normalize_code(code) for code in self.master_index.keys() if isinstance(code, str)}
        missing: set[tuple[str, str]] = set()

        for raw_code, path in self._iter_code_tokens(self.data):
            normalized = self._normalize_code(raw_code)
            if not normalized:
                continue
            if normalized in master_codes:
                continue
            loc = self._format_path(path)
            key = (normalized, loc)
            if key in missing:
                continue
            missing.add(key)
            self._fail(f"INTEGRITY: Missing code '{normalized}' referenced at {loc}")

    def _build_bundling_graph(self) -> dict[str, set[str]]:
        """Build a directed graph of 'bundled into' relationships (keep -> drop)."""
        graph: dict[str, set[str]] = {}

        def _codes_from(value: Any) -> set[str]:
            if value is None:
                return set()
            if isinstance(value, str):
                if self._is_code_token(value):
                    return {self._normalize_code(value)}
                return set()
            if isinstance(value, list):
                out: set[str] = set()
                for item in value:
                    out |= _codes_from(item)
                return out
            if isinstance(value, dict):
                out: set[str] = set()
                for item in value.values():
                    out |= _codes_from(item)
                return out
            return set()

        def _add_edges(keepers: set[str], drops: set[str]) -> None:
            for keep in keepers:
                if not keep:
                    continue
                for drop in drops:
                    if not drop or drop == keep:
                        continue
                    graph.setdefault(keep, set()).add(drop)

        # 1) NCCI pairs: primary keeps, secondary drops when modifier not allowed.
        for raw in self.data.get("ncci_pairs", []) or []:
            if not isinstance(raw, dict):
                continue
            if bool(raw.get("modifier_allowed", False)):
                continue
            primary = raw.get("primary")
            secondary = raw.get("secondary")
            if not isinstance(primary, str) or not isinstance(secondary, str):
                continue
            _add_edges(_codes_from(primary), _codes_from(secondary))

        # 2) Bundling rules (heuristics based on rule structure)
        bundling_rules = self.data.get("bundling_rules", {}) or {}
        if not isinstance(bundling_rules, dict):
            return graph

        for _name, rule in bundling_rules.items():
            if not isinstance(rule, dict):
                continue

            # Common pattern: drop_codes + trigger/therapeutic codes
            if "drop_codes" in rule:
                drops = _codes_from(rule.get("drop_codes"))
                keepers: set[str] = set()
                for key, value in rule.items():
                    if key in {
                        "drop_codes",
                        "description",
                        "inclusion_terms",
                        "requires_resection_terms",
                        "negative_terms",
                        "enabled",
                        "rule_id",
                        "trigger_code",
                        "requires_any",
                        "error_message",
                        "base_increment_minutes",
                        "rationale",
                        "pdt_terms",
                    }:
                        continue
                    keepers |= _codes_from(value)
                _add_edges(keepers, drops)

            # dominant + paired (e.g., 31641 dominates 31640)
            if "dominant" in rule and "paired" in rule:
                dom = _codes_from(rule.get("dominant"))
                paired = _codes_from(rule.get("paired"))
                _add_edges(dom, paired - dom)

            # dominant + suppressed_codes
            if "dominant" in rule and "suppressed_codes" in rule:
                _add_edges(_codes_from(rule.get("dominant")), _codes_from(rule.get("suppressed_codes")))

            # primary_code + exclusive_codes
            if "primary_code" in rule and "exclusive_codes" in rule:
                _add_edges(_codes_from(rule.get("primary_code")), _codes_from(rule.get("exclusive_codes")))

            # chartis_code bundled when valve codes present
            if "chartis_code" in rule and "valve_codes" in rule:
                _add_edges(_codes_from(rule.get("valve_codes")), _codes_from(rule.get("chartis_code")))

            # imaging bundled into pleural procedures
            if "pleural_codes" in rule and "bundled_imaging" in rule:
                _add_edges(_codes_from(rule.get("pleural_codes")), _codes_from(rule.get("bundled_imaging")))

            # thoracentesis bundled into IPC placement
            if "ipc_code" in rule and "thoracentesis_codes" in rule:
                _add_edges(_codes_from(rule.get("ipc_code")), _codes_from(rule.get("thoracentesis_codes")))

            # open chest tube bundled into thoracoscopy
            if "thoracoscopy_therapeutic" in rule and "open_chest_tube" in rule:
                _add_edges(_codes_from(rule.get("thoracoscopy_therapeutic")), _codes_from(rule.get("open_chest_tube")))

            # dilation bundled into stent in same segment
            if "stent_codes" in rule and "dilation_codes" in rule:
                _add_edges(_codes_from(rule.get("stent_codes")), _codes_from(rule.get("dilation_codes")))

            # conventional TBNA bundled into EBUS-TBNA (same stations)
            if "ebus_tbna" in rule and "conventional_tbna" in rule:
                _add_edges(_codes_from(rule.get("ebus_tbna")), _codes_from(rule.get("conventional_tbna")))

            # pleurodesis 32560 bundled into thoracoscopic pleurodesis 32650
            if "thoracoscopy_pleurodesis" in rule and "pleurodesis_code" in rule:
                _add_edges(_codes_from(rule.get("thoracoscopy_pleurodesis")), _codes_from(rule.get("pleurodesis_code")))

        return graph

    def check_bundling_cycles(self) -> None:
        """Detect cycles in KB bundling relationships."""
        graph = self._build_bundling_graph()
        if not graph:
            return

        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        stack: list[str] = []
        cycles: set[tuple[str, ...]] = set()

        def _canonical_rotation(nodes: list[str]) -> tuple[str, ...]:
            if not nodes:
                return tuple()
            rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes))]
            return min(rotations)

        def dfs(node: str) -> None:
            state[node] = 1
            stack.append(node)
            for neighbor in graph.get(node, set()):
                if neighbor not in state:
                    dfs(neighbor)
                    continue
                if state[neighbor] != 1:
                    continue
                try:
                    idx = stack.index(neighbor)
                except ValueError:
                    continue
                cycle_nodes = stack[idx:]
                cycles.add(_canonical_rotation(cycle_nodes))
            stack.pop()
            state[node] = 2

        for node in sorted(graph.keys()):
            if node not in state:
                dfs(node)

        for cycle in sorted(cycles):
            if not cycle:
                continue
            loop = " -> ".join([*cycle, cycle[0]])
            self._fail(f"CYCLE: Bundling cycle detected: {loop}")

    def check_rvu_completeness(self) -> None:
        """Ensure billable CPT codes have non-negative work RVUs."""
        if not isinstance(self.master_index, dict):
            return

        for code, entry in self.master_index.items():
            if not isinstance(code, str) or not isinstance(entry, dict):
                continue

            typ = str(entry.get("type") or "")
            if typ in {"reference", "hcpcs", "anesthesia"}:
                continue
            if typ != "cpt":
                continue

            attrs = entry.get("attributes")
            attrs = attrs if isinstance(attrs, dict) else {}
            if attrs.get("status") == "deleted":
                continue
            if attrs.get("status_code") != "A":
                continue

            normalized = self._normalize_code(code)
            simplified = entry.get("rvu_simplified") if isinstance(entry.get("rvu_simplified"), dict) else {}
            work = simplified.get("work") if isinstance(simplified, dict) else None

            if work is None:
                financials = entry.get("financials") if isinstance(entry.get("financials"), dict) else {}
                cms_candidates: list[tuple[int, dict[str, Any]]] = []
                for k, v in (financials or {}).items():
                    if not isinstance(k, str) or not isinstance(v, dict):
                        continue
                    m = re.match(r"^cms_pfs_(\d{4})$", k)
                    if not m:
                        continue
                    cms_candidates.append((int(m.group(1)), v))
                if cms_candidates:
                    _year, payload = max(cms_candidates, key=lambda item: item[0])
                    work = payload.get("work_rvu")

            if work is None:
                self._fail(f"DATA: Code '{normalized}' missing work_rvu")
                continue

            if not isinstance(work, (int, float)) or work < 0:
                self._fail(f"DATA: Code '{normalized}' has invalid work_rvu: {work!r}")
