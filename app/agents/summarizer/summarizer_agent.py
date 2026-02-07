from app.agents.contracts import SummarizerIn, SummarizerOut, Trace

class SummarizerAgent:
    """Simple summarizer that returns the first 200 characters of each segment as a summary."""
    def run(self, summarizer_in: SummarizerIn) -> SummarizerOut:
        summaries = {}
        for segment in summarizer_in.parser_out.segments:
            text = segment.text.strip()
            if not text:
                summary = ""
            else:
                summary = text[:200] + ("..." if len(text) > 200 else "")
            summaries[segment.type] = summary
        trace = Trace(
            trigger_phrases=list(summaries.keys()),
            rule_paths=["summarizer.simple_summary.v1"],
            confounders_checked=[],
            confidence=1.0,
        )
        return SummarizerOut(summaries=summaries, trace=trace)
