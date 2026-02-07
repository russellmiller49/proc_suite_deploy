from app.agents.contracts import ParserIn, ParserOut, Segment, Entity, Trace
import re

class ParserAgent:
    """Deterministic parser that splits notes by common headings and extracts segments."""
    # Define possible section headings
    headings = [
        "HPI",
        "History",
        "Procedure",
        "Technique",
        "Findings",
        "Indication",
        "Impression",
        "Specimens",
        "Sedation",
        "Complications",
        "Disposition",
    ]

    def run(self, parser_in: ParserIn) -> ParserOut:
        text = parser_in.raw_text or ""
        # Regex to find headings followed by a colon at start of line
        pattern = re.compile(r"^([A-Za-z ]+):", re.MULTILINE)
        matches = list(pattern.finditer(text))
        segments = []
        # If headings are found, slice text between them
        for idx, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            seg_type = next((h for h in self.headings if h.lower() == header.lower()), "unknown")
            seg_text = text[start:end].strip()
            segments.append(
                Segment(
                    type=seg_type,
                    text=seg_text,
                    start_char=start,
                    end_char=end,
                )
            )
        # Fallback: treat entire note as one segment
        if not segments:
            segments.append(
                Segment(
                    type="full",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
            )
        trace = Trace(
            trigger_phrases=[seg.type for seg in segments],
            rule_paths=["parser.heading_split.v1"],
            confounders_checked=[],
            confidence=1.0,
        )
        return ParserOut(note_id=parser_in.note_id, segments=segments, entities=[], trace=trace)
