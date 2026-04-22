"""Optional report polish agent."""

from __future__ import annotations

from .sdk import load_agents_sdk


class ReporterAgent:
    """Optional narrative polisher for the deterministic handoff."""

    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.model = model

    def polish(self, markdown_report: str) -> str:
        agents = load_agents_sdk()
        agent = agents.Agent(
            name="Session Reporter",
            model=self.model,
            instructions="Polish the Markdown lightly without changing facts or adding speculation.",
        )
        result = agents.Runner.run_sync(agent, markdown_report)
        return result.final_output

