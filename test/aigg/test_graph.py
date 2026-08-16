"""Tests for graph construction commands."""

import json
from pathlib import Path

from aigg.graph import app
from typer.testing import CliRunner


def test_experiment_lineage_configuration(tmp_path: Path) -> None:
    """An operator can inspect configuration.

    Guards the lineage's reproducible start.
    """
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(
        json.dumps({"reasoning_mode": "exact-replay"}), encoding="utf-8"
    )
    lineage_path = tmp_path / "lineage"

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "initialise",
            "--lineage-directory",
            str(lineage_path),
            "--configuration",
            str(configuration_path),
        ],
    )

    assert result.exit_code == 0, result.output
    lineage = json.loads((lineage_path / "lineage.json").read_text(encoding="utf-8"))
    assert lineage == {
        "artefact_schema_version": "1",
        "configuration": {
            "identity": (
                "sha256:642a0ca2ac943fb20038a8dcb4f2bcf4237ccd00450256cb6c744b14515f1d5e"
            ),
            "kind": "configuration",
            "schema_version": "1",
        },
        "lineage_schema_version": "1",
    }


def test_source_document_graph_construction_unimplemented() -> None:
    """A graph command reports its boundary, guarding against unrecorded processing."""
    result = CliRunner().invoke(app, ["documents", "run"])

    assert result.exit_code == 1
    assert result.output == (
        "Graph construction is not implemented yet. No local evidence was processed.\n"
    )
