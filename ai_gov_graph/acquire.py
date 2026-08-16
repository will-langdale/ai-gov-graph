"""Commands for acquiring GOV.UK source documents into an evidence corpus."""

from __future__ import annotations

import typer

app = typer.Typer(
    help="Acquire GOV.UK source documents into a local evidence corpus.",
    no_args_is_help=True,
)
documents_app = typer.Typer(help="Acquire source documents into an evidence corpus.")
app.add_typer(documents_app, name="documents")


@documents_app.command()
def fetch() -> None:
    """Describe the acquisition boundary until document retrieval is implemented."""
    msg = "Document acquisition is not implemented yet. No GOV.UK content was fetched."
    typer.echo(msg, err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
