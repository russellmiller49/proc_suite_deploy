"""EBUS station extraction from NER entities.

Maps ANAT_LN_STATION entities with PROC_ACTION entities in the same
sentence or bullet scope
to NodeInteraction records for CPT derivation (31652 vs 31653).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from app.ner.inference import NEREntity, NERExtractionResult
from app.ner.entity_types import (
    SAMPLING_ACTION_KEYWORDS,
    INSPECTION_ACTION_KEYWORDS,
    normalize_station,
)
from app.infra.nlp_warmup import get_spacy_model
from app.registry.schema import NodeInteraction, NodeActionType, NodeOutcomeType


@dataclass
class StationExtractionResult:
    """Result from station extraction."""

    node_events: List[NodeInteraction]
    stations_sampled: List[str]
    stations_inspected_only: List[str]
    warnings: List[str]


@dataclass(frozen=True)
class TextScope:
    start_char: int
    end_char: int
    text: str
    is_bullet: bool = False

    def contains(self, entity: NEREntity) -> bool:
        return self.start_char <= entity.start_char and entity.end_char <= self.end_char


class EBUSStationExtractor:
    """Extract EBUS station node_events from NER entities.

    Algorithm:
    1. Find all ANAT_LN_STATION entities
    2. For each station, find PROC_ACTION within the same sentence/bullet scope
    3. Classify action: aspiration → needle_aspiration, viewed → inspected_only
    4. Find OBS_ROSE within the same sentence/bullet scope
    5. Build NodeInteraction with evidence quote
    """

    # Legacy window sizes (unused; retained for API compatibility)
    ACTION_WINDOW_CHARS = 200
    ROSE_WINDOW_CHARS = 300

    # Station pattern for validation
    STATION_PATTERN = re.compile(
        r"^(2[RL]|4[RL]|5|7|8|9|10[RL]|11[RL]s?i?)$",
        re.IGNORECASE,
    )

    BULLET_LINE_PATTERN = re.compile(r"^\s*(?:[-*]|\u2022|\d+[.)])\s+")

    def __init__(
        self,
        action_window: int = ACTION_WINDOW_CHARS,
        rose_window: int = ROSE_WINDOW_CHARS,
    ) -> None:
        self.action_window = action_window
        self.rose_window = rose_window
        self._nlp = get_spacy_model()

    def extract(self, ner_result: NERExtractionResult) -> StationExtractionResult:
        """
        Extract NodeInteraction events from NER entities.

        Args:
            ner_result: NER extraction result with entities

        Returns:
            StationExtractionResult with node_events and station lists
        """
        stations = ner_result.entities_by_type.get("ANAT_LN_STATION", [])
        actions = ner_result.entities_by_type.get("PROC_ACTION", [])
        rose_entities = ner_result.entities_by_type.get("OBS_ROSE", [])

        node_events: List[NodeInteraction] = []
        stations_sampled: Set[str] = set()
        stations_inspected_only: Set[str] = set()
        warnings: List[str] = []
        best_by_station: dict[str, tuple[NodeInteraction, int]] = {}

        raw_text = ner_result.raw_text or ""
        bullet_scopes, line_scopes = self._collect_line_scopes(raw_text)
        paragraph_scopes = self._collect_paragraph_scopes(raw_text)
        sentence_scopes = self._collect_sentence_scopes(raw_text)
        if not sentence_scopes:
            sentence_scopes = line_scopes
            if raw_text:
                warnings.append("spaCy sentence boundaries unavailable; using line scopes.")

        for station_entity in stations:
            # Normalize station name
            station_name = self._normalize_station(station_entity.text)
            if not station_name:
                warnings.append(
                    f"Invalid station format: '{station_entity.text}'"
                )
                continue

            scope = self._find_scope_for_entity(
                station_entity,
                bullet_scopes,
                sentence_scopes,
                line_scopes,
            )
            if scope is None:
                warnings.append(
                    f"No scope found for station '{station_entity.text}'"
                )

            # Find action within the same scope; if missing, broaden to paragraph/window.
            nearby_action = self._find_nearest_in_scope(
                station_entity,
                actions,
                scope,
            )

            if nearby_action is None and paragraph_scopes:
                paragraph_scope = self._find_scope_for_entity(
                    station_entity,
                    paragraph_scopes,
                    [],
                    [],
                )
                nearby_action = self._find_nearest_in_scope(
                    station_entity,
                    actions,
                    paragraph_scope,
                )

            # Use a local context window as an additional backstop when the action
            # entity is missing or the note spans multiple lines.
            context_window = ""
            if raw_text:
                start = max(0, int(station_entity.start_char) - self.action_window)
                end = min(len(raw_text), int(station_entity.end_char) + self.action_window)
                context_window = raw_text[start:end]

            context_parts: list[str] = []
            if scope is not None and scope.text:
                context_parts.append(scope.text)
            if context_window:
                context_parts.append(context_window)
            action_type = self._classify_action(nearby_action, context_text=" ".join(context_parts) or None)

            # Find ROSE outcome within the same scope
            nearby_rose = self._find_nearest_in_scope(
                station_entity,
                rose_entities,
                scope,
            )
            outcome = self._classify_outcome(nearby_rose)

            # Build evidence quote
            evidence = station_entity.evidence_quote or ""

            candidate = NodeInteraction(
                station=station_name,
                action=action_type,
                outcome=outcome,
                evidence_quote=evidence,
            )
            candidate_pos = int(getattr(station_entity, "start_char", 0) or 0)

            existing = best_by_station.get(station_name)
            if existing is None:
                best_by_station[station_name] = (candidate, candidate_pos)
            else:
                existing_event, existing_pos = existing
                action_priority = {
                    "inspected_only": 0,
                    "needle_aspiration": 1,
                    "core_biopsy": 1,
                    "forceps_biopsy": 1,
                }
                existing_score = action_priority.get(existing_event.action, 0)
                candidate_score = action_priority.get(candidate.action, 0)

                replace = False
                if candidate_score > existing_score:
                    replace = True
                elif candidate_score == existing_score:
                    if existing_event.outcome is None and candidate.outcome is not None:
                        replace = True
                    elif len((candidate.evidence_quote or "").strip()) > len((existing_event.evidence_quote or "").strip()):
                        replace = True
                    elif candidate_pos > existing_pos:
                        # Stable tie-breaker: prefer later mention (often the sampled section).
                        replace = True

                if replace:
                    best_by_station[station_name] = (candidate, candidate_pos)

        node_events = [event for event, _pos in sorted(best_by_station.values(), key=lambda pair: pair[1])]
        for event in node_events:
            if event.action != "inspected_only":
                stations_sampled.add(event.station)
            else:
                stations_inspected_only.add(event.station)

        return StationExtractionResult(
            node_events=node_events,
            stations_sampled=sorted(stations_sampled),
            stations_inspected_only=sorted(stations_inspected_only),
            warnings=warnings,
        )

    def _normalize_station(self, text: str) -> Optional[str]:
        """Normalize a station string to canonical format."""
        raw = str(text or "")

        # Avoid false positives from enumerators like "Site 5:" (common note formatting).
        if re.search(r"(?i)^\s*(?:site|case|patient)\s*#?\s*\d{1,2}\b", raw):
            return None

        result = normalize_station(raw)
        if result:
            return result

        # Try to extract station from longer text
        # e.g., "station 4R" or "level 7"
        match = re.search(r"\b(2[RL]|4[RL]|10[RL]|11[RL]s?i?)\b", raw, re.IGNORECASE)
        if match:
            return normalize_station(match.group(1))

        match = re.search(r"(?i)\b(?:station|level)\s*(5|7|8|9)\b", raw)
        if match:
            return normalize_station(match.group(1))

        return None

    def _collect_line_scopes(self, text: str) -> tuple[List[TextScope], List[TextScope]]:
        bullet_scopes: List[TextScope] = []
        line_scopes: List[TextScope] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            line_start = offset
            line_end = offset + len(line)
            offset = line_end

            line_text = line.rstrip("\r\n")
            if not line_text.strip():
                continue

            is_bullet = bool(self.BULLET_LINE_PATTERN.match(line_text))
            scope = TextScope(
                start_char=line_start,
                end_char=line_end,
                text=line_text,
                is_bullet=is_bullet,
            )
            if is_bullet:
                bullet_scopes.append(scope)
            else:
                line_scopes.append(scope)

        return bullet_scopes, line_scopes

    def _collect_sentence_scopes(self, text: str) -> List[TextScope]:
        if not text or self._nlp is None:
            return []
        try:
            doc = self._nlp(text)
        except Exception:
            return []
        if not doc.has_annotation("SENT_START"):
            return []
        scopes: List[TextScope] = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            scopes.append(
                TextScope(
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    text=sent.text,
                    is_bullet=False,
                )
            )
        return scopes

    def _collect_paragraph_scopes(self, text: str) -> List[TextScope]:
        """Collect paragraph scopes separated by blank lines.

        This is a deterministic fallback when spaCy is unavailable and when
        station + action phrases are split across adjacent lines.
        """
        if not text:
            return []

        scopes: List[TextScope] = []
        offset = 0
        paragraph_start: int | None = None
        paragraph_chunks: list[str] = []

        for raw_line in text.splitlines(keepends=True):
            is_blank = not raw_line.strip()
            if is_blank:
                if paragraph_start is not None and paragraph_chunks:
                    paragraph_text = "".join(paragraph_chunks).rstrip("\r\n")
                    scopes.append(
                        TextScope(
                            start_char=paragraph_start,
                            end_char=offset,
                            text=paragraph_text,
                            is_bullet=False,
                        )
                    )
                paragraph_start = None
                paragraph_chunks = []
                offset += len(raw_line)
                continue

            if paragraph_start is None:
                paragraph_start = offset
            paragraph_chunks.append(raw_line)
            offset += len(raw_line)

        if paragraph_start is not None and paragraph_chunks:
            paragraph_text = "".join(paragraph_chunks).rstrip("\r\n")
            scopes.append(
                TextScope(
                    start_char=paragraph_start,
                    end_char=offset,
                    text=paragraph_text,
                    is_bullet=False,
                )
            )

        return scopes

    def _find_scope_for_entity(
        self,
        entity: NEREntity,
        bullet_scopes: List[TextScope],
        sentence_scopes: List[TextScope],
        line_scopes: List[TextScope],
    ) -> Optional[TextScope]:
        for scope in bullet_scopes:
            if scope.contains(entity):
                return scope
        for scope in sentence_scopes:
            if scope.contains(entity):
                return scope
        for scope in line_scopes:
            if scope.contains(entity):
                return scope
        return None

    def _find_nearest_in_scope(
        self,
        anchor: NEREntity,
        candidates: List[NEREntity],
        scope: Optional[TextScope],
    ) -> Optional[NEREntity]:
        """Find the nearest candidate entity within the same scope."""
        if not candidates or scope is None:
            return None

        best_candidate: Optional[NEREntity] = None
        best_distance = float("inf")

        anchor_center = (anchor.start_char + anchor.end_char) / 2

        for candidate in candidates:
            if not scope.contains(candidate):
                continue
            candidate_center = (candidate.start_char + candidate.end_char) / 2
            distance = abs(anchor_center - candidate_center)

            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate

        return best_candidate

    def _classify_action(
        self,
        action_entity: Optional[NEREntity],
        *,
        context_text: str | None = None,
    ) -> NodeActionType:
        """Classify PROC_ACTION entity into NodeActionType."""
        candidate_texts: list[str] = []
        if action_entity is not None and isinstance(action_entity.text, str):
            candidate_texts.append(action_entity.text)
        if context_text:
            candidate_texts.append(context_text)
        if not candidate_texts:
            return "inspected_only"

        text_lower = " ".join(candidate_texts).lower()

        # Check for sampling keywords
        if any(kw in text_lower for kw in SAMPLING_ACTION_KEYWORDS):
            # Distinguish between aspiration types
            if any(kw in text_lower for kw in ["core", "fnb"]):
                return "core_biopsy"
            if "forceps" in text_lower:
                return "forceps_biopsy"
            return "needle_aspiration"

        # Check for inspection keywords
        if any(kw in text_lower for kw in INSPECTION_ACTION_KEYWORDS):
            return "inspected_only"

        # Default to inspected_only if unclear
        return "inspected_only"

    def _classify_outcome(
        self,
        rose_entity: Optional[NEREntity],
    ) -> Optional[NodeOutcomeType]:
        """Classify OBS_ROSE entity into NodeOutcomeType."""
        if rose_entity is None:
            return None

        text_lower = rose_entity.text.lower()

        if any(kw in text_lower for kw in ["malignant", "positive", "tumor", "cancer", "carcinoma"]):
            return "malignant"
        if any(kw in text_lower for kw in ["suspicious", "atypical"]):
            return "suspicious"
        if any(kw in text_lower for kw in ["benign", "negative", "reactive", "lymphoid"]):
            return "benign"
        if any(kw in text_lower for kw in ["nondiagnostic", "inadequate", "qns"]):
            return "nondiagnostic"

        return "deferred_to_final_path"
