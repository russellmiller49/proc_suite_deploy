"""Planner agent wrapper using the Agents SDK."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import PlanPacket, PreflightReport
from .sdk import load_agents_sdk


def _load_context_excerpt(repo_root: Path) -> str:
    candidates = [
        repo_root / "AGENTS.md",
        repo_root / "docs" / "Multi_agent_collaboration" / "Session Startup Template.md",
        repo_root / "extraction_results_3_9_26" / "session_handoff_2026_03_10.md",
    ]
    excerpts: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        excerpts.append(f"# {path.name}\n{content[:3000]}")
    return "\n\n".join(excerpts)


class PlannerAgent:
    """Builds a typed execution plan from repo context and a user goal."""

    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.model = model

    def plan(self, *, goal: str, preflight: PreflightReport, repo_root: Path) -> PlanPacket:
        agents = load_agents_sdk()
        prompt = "\n\n".join(
            [
                "You are planning bounded engineering work for a deterministic code orchestrator.",
                "Return only valid structured output for the provided schema.",
                "Prefer 1-3 slices. Keep slices small, with explicit allowed paths, protected paths, validation commands, and diff budgets.",
                "Use repo-relative paths.",
                f"Goal:\n{goal}",
                "Preflight:",
                json.dumps(preflight.model_dump(mode="json"), indent=2, sort_keys=True),
                "Repo context excerpts:",
                _load_context_excerpt(repo_root),
            ]
        )
        agent = agents.Agent(
            name="Engineering Planner",
            model=self.model,
            instructions="Plan deterministic bounded coding work with strict typed output.",
            output_type=PlanPacket,
        )
        result = agents.Runner.run_sync(agent, prompt)
        return result.final_output

