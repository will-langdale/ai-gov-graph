"""LangGraph adapters for one document's durable resolution stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NotRequired, TypeAlias, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from aigg.artefacts import ArtefactReference
from aigg.entity_resolution import EntityResolutionService
from aigg.temporal_resolution import TemporalResolutionService


class SingleDocumentResolutionState(TypedDict):
    """The durable references that move through single-document resolution."""

    entity_resolution_request: ArtefactReference
    temporal_resolution_request: ArtefactReference
    entity_resolution: NotRequired[ArtefactReference]
    replay_entity_resolution: NotRequired[ArtefactReference]
    temporal_resolution: NotRequired[ArtefactReference]
    replay_temporal_resolution: NotRequired[ArtefactReference]


ResolutionStage: TypeAlias = Callable[
    [SingleDocumentResolutionState], dict[str, ArtefactReference]
]


def entity_resolution_node(
    service: EntityResolutionService,
) -> ResolutionStage:
    """Adapt one durable entity request to a minimal LangGraph stage node."""

    def resolve(state: SingleDocumentResolutionState) -> dict[str, ArtefactReference]:
        """Write only the durable entity result reference into graph state."""
        return {
            "entity_resolution": service.resolve_request(
                state["entity_resolution_request"],
                replay=state.get("replay_entity_resolution"),
            ).reference
        }

    return resolve


def temporal_resolution_node(
    service: TemporalResolutionService,
) -> ResolutionStage:
    """Adapt one durable temporal request to a minimal LangGraph stage node."""

    def resolve(state: SingleDocumentResolutionState) -> dict[str, ArtefactReference]:
        """Write only the durable temporal result reference into graph state."""
        return {
            "temporal_resolution": service.resolve_request(
                state["temporal_resolution_request"],
                replay=state.get("replay_temporal_resolution"),
            ).reference
        }

    return resolve


def single_document_resolution_workflow(
    entity_service: EntityResolutionService,
    temporal_service: TemporalResolutionService,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the single-document resolution workflow from thin stage adapters."""
    workflow = StateGraph(cast(Any, SingleDocumentResolutionState))
    workflow.add_node(
        "resolve_entities", cast(Any, entity_resolution_node(entity_service))
    )
    workflow.add_node(
        "resolve_temporal", cast(Any, temporal_resolution_node(temporal_service))
    )
    workflow.add_edge(START, "resolve_entities")
    workflow.add_edge("resolve_entities", "resolve_temporal")
    workflow.add_edge("resolve_temporal", END)
    return workflow.compile()
