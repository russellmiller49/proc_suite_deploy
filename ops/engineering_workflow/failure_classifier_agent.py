"""Optional SDK agent for ambiguous validation failures."""

from __future__ import annotations

import json

from .contracts import FailureClassification, ValidationResult
from .sdk import load_agents_sdk


class FailureClassifierAgent:
    """Uses the Agents SDK only when deterministic classification is inconclusive."""

    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.model = model

    def classify(self, validation_result: ValidationResult) -> FailureClassification:
        agents = load_agents_sdk()
        agent = agents.Agent(
            name="Validation Failure Classifier",
            model=self.model,
            instructions="Classify the validation failure conservatively. Prefer non-repairable if uncertain.",
            output_type=FailureClassification,
        )
        result = agents.Runner.run_sync(
            agent,
            json.dumps(validation_result.model_dump(mode="json"), indent=2, sort_keys=True),
        )
        return result.final_output

