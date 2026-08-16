"""Tests for durable experiment artefacts."""

import json
from pathlib import Path

import pytest
from aigg.artefacts import ArtefactIntegrityError, ArtefactStore


def test_experiment_lineage_integrity_changed_content(tmp_path: Path) -> None:
    """Changed content is rejected before use, guarding the recorded hash invariant."""
    store = ArtefactStore(tmp_path / "artefacts")
    reference = store.write_json("configuration", {"model": "recorded"})

    artefact_path = store.path_for(reference)
    artefact_path.write_text(json.dumps({"model": "changed"}), encoding="utf-8")

    with pytest.raises(ArtefactIntegrityError, match="does not match"):
        store.read_json(reference)
