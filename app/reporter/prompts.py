"""Prompt scaffolding and rail guards for the structured reporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import StructuredReport

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "data" / "pulm_procedure_templates.txt"

DEFAULT_EXAMPLES = [
    StructuredReport(
        indication="Peripheral pulmonary lesion",
        anesthesia="Moderate Sedation",
        survey=["Airway inspected to segmental bronchi"],
        localization="Navigation to RUL lesion confirmed",
        sampling=["TBNA 4R", "TBLB RUL"],
        therapeutics=["Stent LMB"],
        complications=[],
        disposition="PACU observation",
    ),
    StructuredReport(
        indication="Pleural effusion",
        anesthesia="MAC",
        survey=["Pleural cavity inspected"],
        localization="Ultrasound-guided entry",
        sampling=["Pleural biopsies"],
        therapeutics=["Talc pleurodesis"],
        complications=[],
        disposition="Admit to floor",
    ),
]


def load_examples() -> List[StructuredReport]:
    if TEMPLATE_PATH.exists():
        content = TEMPLATE_PATH.read_text(encoding="utf-8").strip()
        if content:
            return DEFAULT_EXAMPLES
    return DEFAULT_EXAMPLES


def build_prompt(note: str) -> str:
    examples = load_examples()
    example_blocks = []
    for example in examples:
        example_blocks.append(
            "Example:\nNote:\n{note}\nStructuredReport JSON:\n{json}\n".format(
                note=example.summary().replace(" | ", "\n"),
                json=json.dumps(example.model_dump(), indent=2),
            )
        )
    guard = rail_guard_text()
    prompt = (
        "You are a pulmonary procedures documentation expert. "
        "Read the procedure note and output ONLY valid JSON matching the StructuredReport schema.\n"
        f"{guard}\n"
        + "\n".join(example_blocks)
        + f"\nNew note:\n{note}\nJSON:"  # Request JSON only
    )
    return prompt


def rail_guard_text() -> str:
    return (
        "Required fields: indication, anesthesia, localization, sampling (array), therapeutics (array),"
        " complications (array), disposition."
        " If any field cannot be determined use the string 'Unknown' or an empty list."
        " If therapeutics includes stent placement there must be a sampling entry referencing the same site."
    )


__all__ = ["build_prompt", "rail_guard_text", "load_examples"]

