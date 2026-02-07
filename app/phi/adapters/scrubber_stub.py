"""Stub scrubber for demo/testing use only.

Performs a trivial redaction and returns a minimal entity list. This will
be replaced with a production scrubber (e.g., Presidio) in HIPAA-ready
deployments.
"""

from __future__ import annotations

from app.phi.ports import PHIScrubberPort, ScrubResult


class StubScrubber(PHIScrubberPort):
    def __init__(self, placeholder_token: str = "[[REDACTED]]"):
        self.placeholder_token = placeholder_token

    def scrub(self, text: str, document_type: str | None = None, specialty: str | None = None) -> ScrubResult:
        # Simple demo behavior: redact the first "Patient" token if present.
        entities = []
        scrubbed_text = text
        target = "Patient"
        if target in text:
            start = text.index(target)
            end = start + len(target)
            scrubbed_text = text.replace(target, self.placeholder_token, 1)
            entities.append(
                {
                    "placeholder": self.placeholder_token,
                    "entity_type": "PERSON",
                    "original_start": start,
                    "original_end": end,
                }
            )

        return ScrubResult(scrubbed_text=scrubbed_text, entities=entities)


__all__ = ["StubScrubber"]
