"""Tests for entity-resolution decisions and their history."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from aigg.artefacts import ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor
from aigg.entity_resolution import (
    Entity,
    EntityDecisionHistory,
    EntityResolutionService,
    ExistingEntityResolution,
    ProvisionalEntityResolution,
    ResolutionContext,
    ResolutionProvenance,
    Resolver,
    UnresolvedResolution,
)
from aigg.open_extraction import ExtractedMention
from aigg.reasoning import ModelConfiguration, StructuredModel


@dataclass
class StaticResolver(Resolver):
    """Return a configured resolution without making an external decision."""

    outcome: (
        ExistingEntityResolution | ProvisionalEntityResolution | UnresolvedResolution
    )

    def resolve(
        self, context: ResolutionContext
    ) -> ExistingEntityResolution | ProvisionalEntityResolution | UnresolvedResolution:
        """Return the configured outcome."""
        return self.outcome


@dataclass
class StaticModel(StructuredModel):
    """Return one configured result from the external model boundary."""

    output: JsonValue
    calls: list[JsonValue]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Record one model input and return its configured structured output."""
        del configuration
        self.calls.append(structured_input)
        return self.output


@pytest.mark.parametrize(
    ("outcome_factory",),
    [
        pytest.param(
            lambda: ExistingEntityResolution("entity:business", 0.9, _provenance()),
            id="existing_entity",
        ),
        pytest.param(
            lambda: ProvisionalEntityResolution(
                Entity("entity:department-for-business", "Department for Business"),
                0.7,
                _provenance(),
            ),
            id="provisional_entity",
        ),
        pytest.param(lambda: UnresolvedResolution(0.2, _provenance()), id="unresolved"),
    ],
)
def test_entity_resolution_outcome(
    tmp_path: Path,
    outcome_factory: Callable[
        [],
        ExistingEntityResolution | ProvisionalEntityResolution | UnresolvedResolution,
    ],
) -> None:
    """A Resolver records one explicit entity-resolution outcome.

    Guards the decision contract from replacing unresolved identity with an
    implicit null or collapsing provisional Entity creation into an existing
    Entity decision.
    """
    history = EntityDecisionHistory(ArtefactStore(tmp_path / "artefacts"))
    context = ResolutionContext(
        _mention(), (Entity("entity:business", "Business"),), maximum_candidates=1
    )

    outcome = outcome_factory()
    recorded = history.resolve(StaticResolver(outcome), context)

    assert recorded.outcome == outcome
    assert history.inspect(recorded.history) == (recorded.decision,)


def test_entity_resolution_context_candidate_bound() -> None:
    """A Resolver receives no more candidate Entities than its stated bound.

    Guards a resolver implementation from relying on an unbounded, unstable
    view of the Entity collection.
    """
    with pytest.raises(ValueError, match="maximum"):
        ResolutionContext(
            _mention(),
            (
                Entity("entity:business", "Business"),
                Entity("entity:trade", "Trade"),
            ),
            maximum_candidates=1,
        )


def test_entity_resolution_history_merge_split_reversal(tmp_path: Path) -> None:
    """Entity merges and splits remain inspectable after a later reversal.

    Guards new identity evidence from overwriting an earlier decision that an
    auditor must be able to review and correct.
    """
    history = EntityDecisionHistory(ArtefactStore(tmp_path / "artefacts"))
    merge = history.merge(
        ("entity:department-for-business", "entity:bis"),
        Entity("entity:business", "Business"),
        _provenance(),
    )
    split = history.split(
        "entity:business",
        (
            Entity("entity:department-for-business", "Department for Business"),
            Entity("entity:bis", "Department for Business, Innovation and Skills"),
        ),
        _provenance(),
        history=merge.history,
    )
    reversed_split = history.reverse(
        split.decision.decision_id, _provenance(), split.history
    )

    decisions = history.inspect(reversed_split.history)

    assert [decision.kind for decision in decisions] == ["merge", "split", "reversal"]
    assert decisions[-1].reversed_decision_id == split.decision.decision_id
    assert decisions[0] == merge.decision
    assert decisions[1] == split.decision


def test_entity_resolution_openrouter_exact_replay(tmp_path: Path) -> None:
    """A recorded Entity decision replays without an external model call.

    Guards exact replay from treating a retained decision as a new provider request.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = EntityResolutionService(
        store,
        StaticModel(
            {
                "confidence": 0.9,
                "entity_id": "entity:business",
                "evidence": [_evidence().as_json()],
                "kind": "existing",
                "rationale": "The label matches the supplied candidate.",
            },
            [],
        ),
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        maximum_attempts=1,
    )
    request = service.create_request(
        ResolutionContext(
            _mention(), (Entity("entity:business", "Business"),), maximum_candidates=1
        )
    )
    recorded = service.resolve_request(request)
    replay_model = StaticModel({}, [])

    replayed = EntityResolutionService(
        store,
        replay_model,
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        maximum_attempts=1,
    ).resolve_request(request, replay=recorded.reference)

    assert replayed.outcome == recorded.outcome
    assert replay_model.calls == []


def _mention() -> ExtractedMention:
    """Return one evidence-backed mention awaiting Entity resolution."""
    return ExtractedMention((_evidence(),), "Business")


def _provenance() -> ResolutionProvenance:
    """Return the recorded basis for one entity-resolution decision."""
    return ResolutionProvenance("exact-label", "The labels match.", (_evidence(),))


def _evidence() -> EvidenceAnchor:
    """Return a minimal retained Evidence anchor for the decision fixture."""
    return EvidenceAnchor(
        canonical_text_sha256="a" * 64,
        canonicaliser_version="1",
        content_id="source-id",
        end_line=1,
        end_offset=8,
        prefix="",
        selected_text="Business",
        source_json_sha256="b" * 64,
        source_url="https://www.gov.uk/example",
        start_line=1,
        start_offset=0,
        suffix="",
    )
