"""Tests for source document acquisition commands."""

from ai_gov_graph.acquire import app
from typer.testing import CliRunner


def test_source_document_acquisition_unimplemented() -> None:
    """An acquisition command reports its boundary.

    Guards against an invisible fetch.
    """
    result = CliRunner().invoke(app, ["documents", "fetch"])

    assert result.exit_code == 1
    assert result.output == (
        "Document acquisition is not implemented yet. No GOV.UK content was fetched.\n"
    )
