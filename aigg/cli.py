"""The short command line entry point for ai-gov-graph."""

from __future__ import annotations

import typer

from aigg.acquire import app as acquire_app
from aigg.graph import app as graph_app

app = typer.Typer(
    help="Acquire GOV.UK evidence and construct graph experiment lineages.",
    no_args_is_help=True,
)
app.add_typer(acquire_app, name="acquire")
app.add_typer(graph_app, name="graph")


if __name__ == "__main__":
    app()
