"""Keyword mapping loader for CPT evidence verification.

Loads keyword mappings from YAML files that define positive and negative
phrases for each CPT code, used by the smart hybrid policy to verify
LLM code suggestions against the actual note text.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class KeywordMapping:
    """Mapping of keywords/phrases for a CPT code."""

    code: str
    description: str
    positive_phrases: list[str]
    negative_phrases: list[str]
    context_window_chars: int
    version: str
    notes: str | None = None


class KeywordMappingRepository(ABC):
    """Abstract base class for keyword mapping repositories."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return a version hash for all mappings."""
        ...

    @abstractmethod
    def get_mapping(self, code: str) -> Optional[KeywordMapping]:
        """Get the keyword mapping for a CPT code."""
        ...

    @abstractmethod
    def get_all_codes(self) -> list[str]:
        """Get all CPT codes with mappings."""
        ...


class YamlKeywordMappingRepository(KeywordMappingRepository):
    """Repository that loads keyword mappings from YAML files."""

    DEFAULT_CONTEXT_WINDOW = 200

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)
        self._mappings: dict[str, KeywordMapping] = {}
        self._version_hash: str = ""
        self._load_all()

    @property
    def version(self) -> str:
        return self._version_hash

    def get_mapping(self, code: str) -> Optional[KeywordMapping]:
        # Handle codes with + prefix
        normalized = code.lstrip("+")
        return self._mappings.get(code) or self._mappings.get(normalized)

    def get_all_codes(self) -> list[str]:
        return list(self._mappings.keys())

    def _load_all(self) -> None:
        """Load all YAML mapping files from the directory."""
        if not self._directory.is_dir():
            # Directory doesn't exist yet - use defaults
            self._load_defaults()
            return

        yaml_files = list(self._directory.glob("*.yaml")) + list(
            self._directory.glob("*.yml")
        )

        if not yaml_files:
            self._load_defaults()
            return

        contents: list[str] = []

        for yaml_file in sorted(yaml_files):
            try:
                with yaml_file.open() as f:
                    content = f.read()
                    contents.append(content)
                    data = yaml.safe_load(content)

                if not data or not isinstance(data, dict):
                    continue

                code = data.get("code", "")
                if not code:
                    continue

                mapping = KeywordMapping(
                    code=code,
                    description=data.get("description", ""),
                    positive_phrases=data.get("positive_phrases", []),
                    negative_phrases=data.get("negative_phrases", []),
                    context_window_chars=data.get(
                        "context_window_chars", self.DEFAULT_CONTEXT_WINDOW
                    ),
                    version=data.get("version", ""),
                    notes=data.get("notes"),
                )
                self._mappings[code] = mapping

            except yaml.YAMLError:
                continue

        # Compute version hash from all file contents
        combined = "".join(contents)
        self._version_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _load_defaults(self) -> None:
        """Load default keyword mappings for common IP procedures."""
        defaults = self._get_default_mappings()
        for mapping in defaults:
            self._mappings[mapping.code] = mapping

        # Compute version hash from defaults
        content = str(defaults)
        self._version_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_default_mappings(self) -> list[KeywordMapping]:
        """Return default keyword mappings for common procedures."""
        return [
            KeywordMapping(
                code="31622",
                description="Diagnostic bronchoscopy with cell washing",
                positive_phrases=[
                    "bronchoscopy",
                    "diagnostic bronchoscopy",
                    "cell washing",
                    "bronchial washing",
                ],
                negative_phrases=["planned", "scheduled", "will consider", "no bronchoscopy"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31624",
                description="Bronchoalveolar lavage",
                positive_phrases=[
                    "bal",
                    "bronchoalveolar lavage",
                    "lavage",
                ],
                negative_phrases=["planned", "scheduled", "no bal", "bal deferred"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31628",
                description="Transbronchial lung biopsy",
                positive_phrases=[
                    "transbronchial biopsy",
                    "tblb",
                    "tbbx",
                    "forceps biopsy",
                    "transbronchial lung biopsy",
                ],
                negative_phrases=["planned", "scheduled", "will consider", "attempted biopsy", "no biopsy"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31629",
                description="Transbronchial needle aspiration",
                positive_phrases=[
                    "tbna",
                    "transbronchial needle aspiration",
                    "needle aspiration",
                ],
                negative_phrases=["planned", "scheduled", "no tbna"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31652",
                description="EBUS-TBNA 1-2 stations",
                positive_phrases=[
                    "ebus",
                    "ebus-tbna",
                    "endobronchial ultrasound",
                    "linear ebus",
                ],
                negative_phrases=["planned", "scheduled", "no ebus"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31653",
                description="EBUS-TBNA 3+ stations",
                positive_phrases=[
                    "ebus",
                    "ebus-tbna",
                    "endobronchial ultrasound",
                    "multiple stations",
                ],
                negative_phrases=["planned", "scheduled", "no ebus"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31627",
                description="Navigation bronchoscopy",
                positive_phrases=[
                    "navigation",
                    "electromagnetic navigation",
                    "enb",
                    "superDimension",
                    "Ion",
                    "Monarch",
                ],
                negative_phrases=["planned", "scheduled", "no navigation"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="+31654",
                description="Radial EBUS",
                positive_phrases=[
                    "radial ebus",
                    "r-ebus",
                    "radial probe",
                    "peripheral ebus",
                ],
                negative_phrases=["planned", "scheduled", "no radial"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31630",
                description="Bronchial dilation",
                positive_phrases=[
                    "dilation",
                    "balloon dilation",
                    "bronchial dilation",
                    "stenosis dilation",
                ],
                negative_phrases=["planned", "scheduled", "no dilation"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31631",
                description="Tracheal stent placement",
                positive_phrases=[
                    "tracheal stent",
                    "stent placed in trachea",
                    "tracheal stent placement",
                ],
                negative_phrases=["planned", "scheduled", "stent removal"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31636",
                description="Bronchial stent placement",
                positive_phrases=[
                    "bronchial stent",
                    "stent placed",
                    "stent deployment",
                    "stent insertion",
                ],
                negative_phrases=["planned", "scheduled", "stent removal", "tracheal stent"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31641",
                description="Tumor destruction/debulking",
                positive_phrases=[
                    "tumor destruction",
                    "debulking",
                    "cryotherapy",
                    "laser",
                    "electrocautery",
                    "ablation",
                    "APC",
                    "argon plasma",
                ],
                negative_phrases=["planned", "scheduled", "no ablation"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="31647",
                description="BLVR valve placement",
                positive_phrases=[
                    "valve",
                    "blvr",
                    "endobronchial valve",
                    "zephyr",
                    "spiration",
                ],
                negative_phrases=["planned", "scheduled", "valve removal"],
                context_window_chars=200,
                version="default",
            ),
            KeywordMapping(
                code="32555",
                description="Thoracentesis with imaging",
                positive_phrases=[
                    "thoracentesis",
                    "pleural aspiration",
                    "pleural tap",
                    "ultrasound-guided thoracentesis",
                ],
                negative_phrases=["planned", "scheduled", "no thoracentesis"],
                context_window_chars=200,
                version="default",
            ),
        ]
