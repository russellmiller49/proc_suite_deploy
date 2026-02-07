"""Typer CLI for registry extraction."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from app.common.knowledge_cli import print_knowledge_info
from app.common.text_io import load_note

from .engine import RegistryEngine
from .schema import RegistryRecord

app = typer.Typer(help="Run the registry extractor.")
console = Console()


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


@app.command()
def run(
    note: str,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    explain: bool = typer.Option(False, "--explain", help="Show field evidence."),
) -> None:
    text = load_note(note)
    engine = RegistryEngine()
    result = engine.run(text, explain=explain)
    if isinstance(result, tuple):
        record, evidence = result
        record.evidence = evidence
    else:
        record = result
    if json_output:
        typer.echo(json.dumps(record.model_dump(), indent=2, default=str))
        if explain:
            _print_evidence(record)
        return

    _print_registry(record)
    if explain:
        _print_evidence(record)


def _print_registry(record: RegistryRecord) -> None:
    table = Table(title="Registry Record", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    data = record.model_dump(exclude={"evidence"})
    for field, value in data.items():
        table.add_row(field, _format_value(value))
    console.print(table)


def _print_evidence(record: RegistryRecord) -> None:
    if not record.evidence:
        console.print("No evidence captured.")
        return
    table = Table(title="Evidence", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Spans")
    for field, spans in record.evidence.items():
        formatted = [
            (
                f"{span.section or 'Unknown'}: “{span.text.strip()}” "
                f"({span.start}-{span.end})"
            )
            for span in spans
        ]
        table.add_row(field, "\n".join(formatted))
    console.print(table)


def _format_value(value: object) -> str:
    """Format a registry value for display.

    Per specification:
    - Arrays of primitives (strings, numbers): join with commas
    - Arrays of objects (e.g., ebus_stations_detail): expand per item as structured lines
    - None: show as "—" (not "null")
    - Booleans: show as "Yes"/"No"
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        if not value:
            return "—"
        # Check if this is a list of dicts (objects) vs primitives
        if value and isinstance(value[0], dict):
            # Format each object as a structured line
            # Special handling for ebus_stations_detail
            lines = []
            for item in value:
                if isinstance(item, dict):
                    line = _format_station_detail(item)
                    if line:
                        lines.append(line)
            return "\n".join(lines) if lines else "—"
        else:
            # List of primitives - join with commas
            return ", ".join(str(item) for item in value)
    return str(value)


def _format_station_detail(detail: dict) -> str:
    """Format a single ebus_stations_detail entry as a structured line.

    Per specification format:
      - 11L: size 5.4 mm; ROSE Nondiagnostic
      - 4R: size 5.5 mm; passes 3; round; distinct margins; heterogeneous; CHS present; ROSE Benign
    """
    parts = []

    station = detail.get("station")
    if not station:
        return ""

    # Start with station name
    line = f"  - {station}:"

    # Add size
    size_mm = detail.get("size_mm")
    if size_mm is not None:
        parts.append(f"size {size_mm} mm")

    # Add passes
    passes = detail.get("passes")
    if passes is not None:
        parts.append(f"passes {passes}")

    # Add morphology fields if present (not null)
    shape = detail.get("shape")
    if shape:
        parts.append(shape)

    margin = detail.get("margin")
    if margin:
        parts.append(f"{margin} margins")

    echogenicity = detail.get("echogenicity")
    if echogenicity:
        parts.append(echogenicity)

    chs_present = detail.get("chs_present")
    if chs_present is True:
        parts.append("CHS present")
    elif chs_present is False:
        parts.append("CHS absent")

    # Add ROSE result
    rose_result = detail.get("rose_result")
    if rose_result:
        parts.append(f"ROSE {rose_result}")

    if parts:
        line += " " + "; ".join(parts)

    return line


def main() -> None:  # pragma: no cover - CLI entrypoint
    app()


if __name__ == "__main__":
    main()
