"""Tests for the public package interface."""

import tomllib
from pathlib import Path

import aigg


def test_package_project_version() -> None:
    """The package exposes the configured project version.

    Guards callers of the public package interface against a stale version.
    """
    configuration = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert aigg.__version__ == configuration["project"]["version"]
