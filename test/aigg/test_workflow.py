"""Tests for single-document resolution workflow adapters."""

from dataclasses import dataclass
from pathlib import Path

from aigg.artefacts import ArtefactStore, JsonValue
from aigg.entity_resolution import EntityResolutionService
from aigg.reasoning import ModelConfiguration, StructuredModel
from aigg.temporal_resolution import TemporalResolutionService
from aigg.workflow import entity_resolution_node, temporal_resolution_node


@dataclass
class EmptyModel(StructuredModel):
    """Fail if a deterministic workflow test unexpectedly invokes a model."""

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Reject an unexpected model call at the external boundary."""
        del configuration, structured_input
        msg = "The workflow test must not invoke a model."
        raise AssertionError(msg)


def test_plan_resolution_nodes_return_durable_references(tmp_path: Path) -> None:
    """Each LangGraph adapter returns only a durable result reference.

    Guards workflow state from becoming the authority for a stage result.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    entity_service = EntityResolutionService(
        store, EmptyModel(), _configuration(), maximum_attempts=1
    )
    temporal_service = TemporalResolutionService(
        store, EmptyModel(), _configuration(), maximum_attempts=1
    )

    assert callable(entity_resolution_node(entity_service))
    assert callable(temporal_resolution_node(temporal_service))


def _configuration() -> ModelConfiguration:
    """Return a stable model identity for adapter construction."""
    return ModelConfiguration("openrouter", "example/model", {"temperature": 0})
