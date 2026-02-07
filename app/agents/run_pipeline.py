"""Pipeline orchestrator for parser, summarizer, and structurer agents.

This module orchestrates the 3-agent reporter pipeline with proper
status tracking, error handling, and graceful degradation.
"""

from typing import Literal

from app.agents.contracts import (
    ParserIn,
    ParserOut,
    SummarizerIn,
    SummarizerOut,
    StructurerIn,
    StructurerOut,
    PipelineResult,
    AgentError,
)
from app.agents.parser.parser_agent import ParserAgent
from app.agents.summarizer.summarizer_agent import SummarizerAgent
from app.agents.structurer.structurer_agent import StructurerAgent
from observability.timing import timed
from observability.logging_config import get_logger

logger = get_logger("pipeline")


def run_pipeline(note: dict) -> dict:
    """Run the full agent pipeline on a single note dict.

    Args:
        note: Dict with keys 'note_id' and 'raw_text'.

    Returns:
        Dict with pipeline_status, agent outputs, registry, and codes.
    """
    result = run_pipeline_typed(note)
    return result.model_dump()


def run_pipeline_typed(note: dict) -> PipelineResult:
    """Run the full agent pipeline with typed output.

    The pipeline runs through three stages:
    1. Parser: Splits raw text into segments and extracts entities
    2. Summarizer: Produces section summaries from segments/entities
    3. Structurer: Maps summaries to registry model and generates codes

    If any stage fails (status='failed'), the pipeline stops and returns
    the partial result. Degraded stages continue but mark the overall
    pipeline as degraded.

    Args:
        note: Dict with keys 'note_id' and 'raw_text'.

    Returns:
        PipelineResult with status, agent outputs, registry, and codes.
    """
    note_id = note.get("note_id", "")
    raw_text = note.get("raw_text", "")

    parser_ms = 0.0
    summarizer_ms = 0.0
    structurer_ms = 0.0

    with timed("pipeline.total") as timing:
        # Stage 1: Parser
        with timed("pipeline.parser") as t_parser:
            parser_out = _run_parser(note_id, raw_text)
        parser_ms = t_parser.elapsed_ms

        if parser_out.status == "failed":
            logger.warning(
                "Pipeline failed at parser stage",
                extra={"note_id": note_id, "errors": [e.model_dump() for e in parser_out.errors]},
            )
            return PipelineResult(
                pipeline_status="failed_parser",
                parser=parser_out,
            )

        # Stage 2: Summarizer
        with timed("pipeline.summarizer") as t_summarizer:
            summarizer_out = _run_summarizer(parser_out)
        summarizer_ms = t_summarizer.elapsed_ms

        if summarizer_out.status == "failed":
            logger.warning(
                "Pipeline failed at summarizer stage",
                extra={"note_id": note_id, "errors": [e.model_dump() for e in summarizer_out.errors]},
            )
            return PipelineResult(
                pipeline_status="failed_summarizer",
                parser=parser_out,
                summarizer=summarizer_out,
            )

        # Stage 3: Structurer
        with timed("pipeline.structurer") as t_structurer:
            structurer_out = _run_structurer(summarizer_out)
        structurer_ms = t_structurer.elapsed_ms

        if structurer_out.status == "failed":
            logger.warning(
                "Pipeline failed at structurer stage",
                extra={"note_id": note_id, "errors": [e.model_dump() for e in structurer_out.errors]},
            )
            return PipelineResult(
                pipeline_status="failed_structurer",
                parser=parser_out,
                summarizer=summarizer_out,
                structurer=structurer_out,
            )

        # Determine overall status
        statuses = [parser_out.status, summarizer_out.status, structurer_out.status]
        if all(s == "ok" for s in statuses):
            pipeline_status: Literal["ok", "degraded", "failed_parser", "failed_summarizer", "failed_structurer"] = "ok"
        else:
            pipeline_status = "degraded"

    logger.info(
        "Pipeline complete",
        extra={
            "note_id": note_id,
            "pipeline_status": pipeline_status,
            "processing_time_ms": timing.elapsed_ms,
            "parser_time_ms": parser_ms,
            "summarizer_time_ms": summarizer_ms,
            "structurer_time_ms": structurer_ms,
            "parser_status": parser_out.status,
            "summarizer_status": summarizer_out.status,
            "structurer_status": structurer_out.status,
        },
    )

    return PipelineResult(
        pipeline_status=pipeline_status,
        parser=parser_out,
        summarizer=summarizer_out,
        structurer=structurer_out,
        registry=structurer_out.registry,
        codes=structurer_out.codes,
    )


def _run_parser(note_id: str, raw_text: str) -> ParserOut:
    """Run the parser agent with error handling."""
    try:
        parser_in = ParserIn(note_id=note_id, raw_text=raw_text)
        parser_agent = ParserAgent()
        parser_out = parser_agent.run(parser_in)

        # Ensure note_id is set
        parser_out.note_id = note_id

        return parser_out

    except Exception as e:
        logger.error(f"Parser agent threw exception: {e}")
        return ParserOut(
            note_id=note_id,
            status="failed",
            errors=[
                AgentError(
                    code="PARSER_EXCEPTION",
                    message=str(e),
                )
            ],
        )


def _run_summarizer(parser_out: ParserOut) -> SummarizerOut:
    """Run the summarizer agent with error handling."""
    try:
        summarizer_in = SummarizerIn(parser_out=parser_out)
        summarizer_agent = SummarizerAgent()
        summarizer_out = summarizer_agent.run(summarizer_in)

        # Ensure note_id is set
        summarizer_out.note_id = parser_out.note_id

        return summarizer_out

    except Exception as e:
        logger.error(f"Summarizer agent threw exception: {e}")
        return SummarizerOut(
            note_id=parser_out.note_id,
            status="failed",
            errors=[
                AgentError(
                    code="SUMMARIZER_EXCEPTION",
                    message=str(e),
                )
            ],
        )


def _run_structurer(summarizer_out: SummarizerOut) -> StructurerOut:
    """Run the structurer agent with error handling."""
    try:
        structurer_in = StructurerIn(summarizer_out=summarizer_out)
        structurer_agent = StructurerAgent()
        structurer_out = structurer_agent.run(structurer_in)

        # Ensure note_id is set
        structurer_out.note_id = summarizer_out.note_id

        return structurer_out

    except Exception as e:
        logger.error(f"Structurer agent threw exception: {e}")
        return StructurerOut(
            note_id=summarizer_out.note_id,
            status="failed",
            errors=[
                AgentError(
                    code="STRUCTURER_EXCEPTION",
                    message=str(e),
                )
            ],
        )
