"""Tests for the short command line entry point."""

import pytest
from typer.testing import CliRunner

from aigg.cli import app


@pytest.mark.parametrize(
    ("command",),
    [
        pytest.param("acquire", id="acquire"),
        pytest.param("graph", id="graph"),
    ],
)
def test_short_command_workflow(command: str) -> None:
    """An operator can discover each workflow through the short command.

    Guards the public command from losing a workflow while the separate
    applications evolve.
    """
    result = CliRunner().invoke(app, [command, "--help"])

    assert result.exit_code == 0, result.output
