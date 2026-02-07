"""CLI for structured report generation and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.common.knowledge_cli import print_knowledge_info
from app.common.text_io import load_note

from .engine import ReportEngine
from .schema import StructuredReport

app = typer.Typer(help="Reporter CLI")
console = Console()

REPORT_PATH_ARGUMENT = typer.Argument(..., exists=True)


@app.callback()
def _cli_entry(
    _: typer.Context,
    knowledge_info: bool = typer.Option(
        False,
        "--knowledge-info",
        help="Print knowledge metadata and exit.",
        is_eager=True,
    ),
) -> None:
    if knowledge_info:
        print_knowledge_info(console)
        raise typer.Exit()


@app.command("gen")
def generate(
    from_free_text: str = typer.Option(..., "--from-free-text", help="Source note path or text"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output"),
    explain: bool = typer.Option(False, "--explain", help="Show structured fields"),
) -> None:
    note = load_note(from_free_text)
    engine = ReportEngine()
    report = engine.from_free_text(note)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        if explain:
            _print_report(report)
        return

    _print_report(report)
    if explain:
        console.print(Panel(report.model_dump_json(indent=2), title="Structured JSON"))


@app.command("render")
def render(
    report_path: Path = REPORT_PATH_ARGUMENT,
    template: str = typer.Option("bronchoscopy", "--template", help="Template key or filename"),
) -> None:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if isinstance(payload, str):
        payload = json.loads(payload)
    report = StructuredReport(**payload)
    engine = ReportEngine()
    report = engine.validate_and_autofix(report)
    output = engine.render(report, template=template)
    typer.echo(output)


def _print_report(report: StructuredReport) -> None:
    table = Table(title="Structured Report", show_lines=False)
    table.add_column("Section", style="cyan")
    table.add_column("Content")
    table.add_row("Indication", report.indication)
    table.add_row("Anesthesia", report.anesthesia)
    table.add_row("Localization", report.localization)
    table.add_row("Survey", "\n".join(report.survey) or "—")
    table.add_row("Sampling", "\n".join(report.sampling) or "—")
    table.add_row("Therapeutics", "\n".join(report.therapeutics) or "—")
    table.add_row("Complications", "\n".join(report.complications) or "—")
    table.add_row("Disposition", report.disposition)
    console.print(table)


def main() -> None:  # pragma: no cover - CLI entry point
    app()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
