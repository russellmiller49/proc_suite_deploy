"""Helpers for rendering knowledge metadata in CLI contexts."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .knowledge import knowledge_snapshot


def print_knowledge_info(console: Console, *, top_n: int = 20) -> None:
    """Render the knowledge metadata using Rich tables."""

    snapshot = knowledge_snapshot(top_n=top_n)
    console.print(f"Knowledge version: {snapshot.version}")
    console.print(f"Knowledge SHA256: {snapshot.sha256}")

    table = Table(title=f"RVU Table (top {top_n})", show_lines=False)
    table.add_column("CPT", style="cyan", no_wrap=True)
    table.add_column("Work", justify="right")
    table.add_column("PE", justify="right")
    table.add_column("MP", justify="right")
    table.add_column("Total", justify="right")

    for entry in snapshot.rvus:
        table.add_row(
            entry.code,
            f"{entry.work:.2f}",
            f"{entry.pe:.2f}",
            f"{entry.mp:.2f}",
            f"{entry.total:.2f}",
        )

    console.print(table)

    add_on_display = ", ".join(snapshot.add_on_codes) or "—"
    console.print(f"Add-on whitelist: {add_on_display}")
    rule_display = ", ".join(snapshot.bundling_rules) or "—"
    console.print(f"Active bundling rules: {rule_display}")


__all__ = ["print_knowledge_info"]
