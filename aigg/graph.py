"""Commands for constructing a graph from a local evidence corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from aigg.artefacts import (
    ARTEFACT_SCHEMA_VERSION,
    ArtefactReference,
    ArtefactStore,
    JsonValue,
)

LINEAGE_SCHEMA_VERSION = "1"

app = typer.Typer(
    help="Run graph construction against locally acquired evidence.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(help="Create and inspect experiment lineages.")
app.add_typer(experiment_app, name="experiment")
documents_app = typer.Typer(help="Run graph construction against source documents.")
app.add_typer(documents_app, name="documents")


def initialise_lineage(
    lineage_directory: Path, configuration: dict[str, JsonValue]
) -> Path:
    """Create one lineage with a content-addressed configuration artefact."""
    if lineage_directory.exists() and any(lineage_directory.iterdir()):
        msg = f"Lineage directory is not empty: {lineage_directory}"
        raise ValueError(msg)

    store = ArtefactStore(lineage_directory / "artefacts")
    reference = store.write_json("configuration", configuration)
    manifest = {
        "artefact_schema_version": ARTEFACT_SCHEMA_VERSION,
        "configuration": _reference_data(reference),
        "lineage_schema_version": LINEAGE_SCHEMA_VERSION,
    }
    manifest_path = lineage_directory / "lineage.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


@experiment_app.command("initialise")
def initialise(
    lineage_directory: Annotated[
        Path,
        typer.Option(help="New directory that will hold the experiment lineage."),
    ],
    configuration: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON configuration recorded as the lineage's first artefact.",
        ),
    ],
) -> None:
    """Initialise a durable experiment lineage from a JSON configuration."""
    try:
        content = configuration.read_text(encoding="utf-8")
        parsed_configuration = json.loads(content)
    except json.JSONDecodeError as error:
        raise typer.BadParameter("Configuration must contain valid JSON.") from error
    if not isinstance(parsed_configuration, dict):
        raise typer.BadParameter("Configuration must be a JSON object.")

    try:
        manifest_path = initialise_lineage(lineage_directory, parsed_configuration)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Initialised experiment lineage: {manifest_path}")


@documents_app.command()
def run() -> None:
    """Describe the graph construction boundary until processing is implemented."""
    msg = "Graph construction is not implemented yet. No local evidence was processed."
    typer.echo(msg, err=True)
    raise typer.Exit(1)


def _reference_data(reference: ArtefactReference) -> dict[str, str]:
    """Return the durable representation of an artefact reference."""
    return {
        "identity": reference.identity,
        "kind": reference.kind,
        "schema_version": reference.schema_version,
    }


if __name__ == "__main__":
    app()
