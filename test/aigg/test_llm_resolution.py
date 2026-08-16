"""Tests for LLM-backed entity and temporal resolution."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from aigg.artefacts import ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor
from aigg.entity_resolution import (
    Entity,
    EntityResolutionService,
    ResolutionContext,
)
from aigg.open_extraction import ExtractedMention
from aigg.reasoning import ModelConfiguration, ReasoningValidationError, StructuredModel
from aigg.temporal_resolution import (
    ExtractedTemporalExpression,
    ResolvedTemporalExpression,
    TemporalConstraint,
    TemporalResolutionContext,
    TemporalResolutionService,
    UnresolvedTemporalExpression,
)
from aigg.workflow import single_document_resolution_workflow


@dataclass
class SequencedModel(StructuredModel):
    """Control the external, non-deterministic model boundary in tests."""

    outputs: list[JsonValue]
    calls: list[JsonValue]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the next configured structured output."""
        self.calls.append(structured_input)
        return self.outputs.pop(0)


def _evidence(text: str) -> EvidenceAnchor:
    """Return retained Evidence for one source expression."""
    return EvidenceAnchor(
        canonical_text_sha256="a" * 64,
        canonicaliser_version="1",
        content_id="source-id",
        end_line=1,
        end_offset=len(text),
        prefix="",
        selected_text=text,
        source_json_sha256="b" * 64,
        source_url="https://www.gov.uk/example",
        start_line=1,
        start_offset=0,
        suffix="",
    )


def test_entity_resolution_openrouter_existing_exact_replay(tmp_path: Path) -> None:
    """An existing Entity decision replays without another model invocation.

    Guards exact replay from treating an already-recorded entity decision as a
    fresh request to an external, non-deterministic provider.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    model = SequencedModel(
        [
            {
                "confidence": 0.9,
                "entity_id": "entity:trade",
                "evidence": [_evidence("Department for Business and Trade").as_json()],
                "kind": "existing",
                "rationale": "The source uses the candidate's full label.",
            }
        ],
        [],
    )
    service = EntityResolutionService(
        store, model, _configuration(), maximum_attempts=2
    )
    request = service.create_request(
        ResolutionContext(
            _mention("Department for Business and Trade"),
            (Entity("entity:trade", "Department for Business and Trade"),),
            maximum_candidates=1,
        )
    )

    recorded = service.resolve_request(request)
    replay_model = SequencedModel([], [])
    replayed = EntityResolutionService(
        store, replay_model, _configuration(), maximum_attempts=2
    ).resolve_request(request, replay=recorded.reference)

    assert recorded.outcome == replayed.outcome
    assert recorded.history == replayed.history
    assert replay_model.calls == []
    invocation = _json_object(store.read_json(recorded.reasoning_invocation))
    assert invocation["provider"] == "openrouter"
    assert invocation["stage"] == "entity-resolution"
    structured_input = _json_object(invocation["structured_input"])
    assert structured_input["candidates"] == [
        {"entity_id": "entity:trade", "label": "Department for Business and Trade"}
    ]


def test_entity_resolution_openrouter_retry_bound(tmp_path: Path) -> None:
    """Malformed entity output uses only the configured validation retry bound.

    Guards resolution from accepting an unsupported outcome or retrying an
    external model indefinitely. Uses a sequenced model at that boundary.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    model = SequencedModel([{"kind": "existing"}, {"kind": "existing"}], [])
    service = EntityResolutionService(
        store, model, _configuration(), maximum_attempts=2
    )
    request = service.create_request(
        ResolutionContext(
            _mention("Department for Business and Trade"),
            (Entity("entity:trade", "Department for Business and Trade"),),
            maximum_candidates=1,
        )
    )

    with pytest.raises(ReasoningValidationError) as error:
        service.resolve_request(request)

    assert len(model.calls) == 2
    invocation = _json_object(store.read_json(error.value.reference))
    assert len(_json_array(invocation["retry_history"])) == 2


@pytest.mark.parametrize(
    ("output", "expected_type", "expected_constraint"),
    [
        pytest.param(
            {
                "constraint": {
                    "lower_bound": "2026-04-01",
                    "lower_inclusive": True,
                    "upper_bound": "2026-05-01",
                    "upper_inclusive": False,
                },
                "evidence": [_evidence("Published 1 March 2026").as_json()],
                "kind": "resolved",
                "rationale": "The source publication date establishes the year.",
            },
            ResolvedTemporalExpression,
            TemporalConstraint.during(date(2026, 4, 1), date(2026, 5, 1)),
            id="justified_relative_time",
        ),
        pytest.param(
            {
                "kind": "unresolved",
                "rationale": "Neither context identifies the reference year.",
            },
            UnresolvedTemporalExpression,
            None,
            id="unresolved",
        ),
    ],
)
def test_temporal_resolution_openrouter_outcomes(
    tmp_path: Path,
    output: JsonValue,
    expected_type: type[ResolvedTemporalExpression | UnresolvedTemporalExpression],
    expected_constraint: TemporalConstraint | None,
) -> None:
    """Temporal judgement records either a justified constraint or unresolved result.

    Guards the hybrid resolver from fabricating a comparable time when bounded
    Source and graph context do not support one. Uses a static external model.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    model = SequencedModel([output], [])
    service = TemporalResolutionService(
        store, model, _configuration(), maximum_attempts=1
    )
    request = service.create_request(
        TemporalResolutionContext(
            _expression("the following April"),
            reference_time=datetime(2026, 3, 1, tzinfo=UTC),
            source_context=(
                {
                    "constraint": {
                        "lower_bound": "2026-04-01",
                        "lower_inclusive": True,
                        "upper_bound": "2026-05-01",
                        "upper_inclusive": False,
                    },
                    "evidence": [_evidence("Published 1 March 2026").as_json()],
                    "publication_date": "2026-03-01",
                },
            ),
            graph_context=({"known_event": "scheme announcement"},),
            maximum_source_context=1,
            maximum_graph_context=1,
        )
    )

    recorded = service.resolve_request(request)

    assert isinstance(recorded.outcome, expected_type)
    if expected_constraint is not None:
        assert isinstance(recorded.outcome, ResolvedTemporalExpression)
        assert recorded.outcome.constraint == expected_constraint
    assert recorded.reasoning_invocation is not None
    invocation = _json_object(store.read_json(recorded.reasoning_invocation))
    structured_input = _json_object(invocation["structured_input"])
    assert structured_input["source_context"] == [
        {
            "constraint": {
                "lower_bound": "2026-04-01",
                "lower_inclusive": True,
                "upper_bound": "2026-05-01",
                "upper_inclusive": False,
            },
            "evidence": [_evidence("Published 1 March 2026").as_json()],
            "publication_date": "2026-03-01",
        }
    ]
    assert structured_input["graph_context"] == [{"known_event": "scheme announcement"}]
    replay_model = SequencedModel([], [])
    replayed = TemporalResolutionService(
        store, replay_model, _configuration(), maximum_attempts=1
    ).resolve_request(request, replay=recorded.reference)
    assert replayed.outcome == recorded.outcome
    assert replay_model.calls == []


def test_single_document_resolution_workflow_passes_only_references(
    tmp_path: Path,
) -> None:
    """Resolution nodes exchange durable references through LangGraph state.

    Guards a single-document workflow from making its in-memory state the
    authority for entity or temporal resolution results.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    entity_service = EntityResolutionService(
        store,
        SequencedModel(
            [
                {
                    "confidence": 0.9,
                    "entity_id": "entity:trade",
                    "evidence": [
                        _evidence("Department for Business and Trade").as_json()
                    ],
                    "kind": "existing",
                    "rationale": "The source uses the candidate's full label.",
                }
            ],
            [],
        ),
        _configuration(),
        maximum_attempts=1,
    )
    temporal_service = TemporalResolutionService(
        store, SequencedModel([], []), _configuration(), maximum_attempts=1
    )
    entity_request = entity_service.create_request(
        ResolutionContext(
            _mention("Department for Business and Trade"),
            (Entity("entity:trade", "Department for Business and Trade"),),
            maximum_candidates=1,
        )
    )
    temporal_request = temporal_service.create_request(
        TemporalResolutionContext(_expression("2026-04"))
    )

    result = single_document_resolution_workflow(
        entity_service, temporal_service
    ).invoke(
        {
            "entity_resolution_request": entity_request,
            "temporal_resolution_request": temporal_request,
        }
    )

    assert set(result) == {
        "entity_resolution_request",
        "entity_resolution",
        "temporal_resolution_request",
        "temporal_resolution",
    }
    assert result["entity_resolution"].kind == "entity-resolution"
    assert result["temporal_resolution"].kind == "temporal-resolution"


def _configuration() -> ModelConfiguration:
    """Return the stable OpenRouter identity used by the resolver tests."""
    return ModelConfiguration("openrouter", "example/model", {"temperature": 0})


def _mention(text: str) -> ExtractedMention:
    """Return an evidence-backed mention awaiting identity resolution."""
    return ExtractedMention((_evidence(text),), text)


def _expression(text: str) -> ExtractedTemporalExpression:
    """Return an evidence-backed expression awaiting temporal resolution."""
    return ExtractedTemporalExpression((_evidence(text),), text)


def _json_object(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow one durable JSON payload to the object test fixtures expect."""
    assert isinstance(value, dict)
    return value


def _json_array(value: JsonValue) -> list[JsonValue]:
    """Narrow one durable JSON payload to the list test fixtures expect."""
    assert isinstance(value, list)
    return value
