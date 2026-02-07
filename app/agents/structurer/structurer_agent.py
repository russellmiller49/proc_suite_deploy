from app.agents.contracts import StructurerIn, StructurerOut, Trace

class StructurerAgent:
    """Maps summaries and entities into registry fields and codes."""
    def run(self, structurer_in: StructurerIn) -> StructurerOut:
        # Placeholder: produce empty registry and codes
        trace = Trace(trigger_phrases=[], rule_paths=["structurer.placeholder.v1"], confounders_checked=[], confidence=1.0)
        return StructurerOut(registry={}, codes=[], rationale={}, trace=trace)
