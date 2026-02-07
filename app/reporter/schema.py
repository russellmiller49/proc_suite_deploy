"""Structured report schema for synoptic output."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class StructuredReport(BaseModel):
    """Represents the structured synoptic report payload."""

    model_config = ConfigDict(extra="forbid")

    indication: str
    anesthesia: str
    survey: List[str] = Field(default_factory=list)
    localization: str
    sampling: List[str] = Field(default_factory=list)
    therapeutics: List[str] = Field(default_factory=list)
    complications: List[str] = Field(default_factory=list)
    disposition: str
    metadata: dict = Field(default_factory=dict)
    version: str = "0.1.0"

    def summary(self) -> str:
        parts = [f"Indication: {self.indication}", f"Anesthesia: {self.anesthesia}"]
        if self.localization:
            parts.append(f"Localization: {self.localization}")
        return " | ".join(parts)


__all__ = ["StructuredReport"]

