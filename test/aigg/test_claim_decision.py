"""Tests for bounded candidate Claim decisions."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from aigg.artefacts import ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor
from aigg.claim_decision import (
    ClaimChangeValidator,
    ClaimDecisionContext,
    ClaimDecisionService,
)
from aigg.claim_mapping import (
    ClaimDisposition,
    ClaimMapping,
    SemanticAssertion,
    StageReason,
)
from aigg.open_extraction import CandidateClaim
from aigg.reasoning import (
    ModelConfiguration,
    ReasoningValidationError,
    StructuredModel,
)


@dataclass
class StaticModel(StructuredModel):
    """Return one configured result and retain the supplied model inputs."""

    output: JsonValue
    calls: list[JsonValue]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Record one bounded request at the external reasoning boundary."""
        del configuration
        self.calls.append(structured_input)
        return self.output


def test_claim_decision_accepted_exact_replay(tmp_path: Path) -> None:
    """An accepted Claim replays its recorded mapping without a model call.

    Guards exact replay from treating a retained Claim decision as a new external
    reasoning request.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    model = StaticModel(
        {
            "acceptance": "The assertion is ready for projection.",
            "conflict": "No conflicting accepted assertion was supplied.",
            "disposition": "accepted",
            "mapping": "The active Ontology has the required predicate.",
            "scope": "The Claim is within the current experiment scope.",
            "semantic_assertions": [
                {
                    "object": "available",
                    "object_kind": "literal",
                    "predicate": "https://example.test/hasStatus",
                    "subject": "https://example.test/scheme",
                }
            ],
            "validation": "The proposed assertion satisfies the active SHACL release.",
        },
        [],
    )
    service = ClaimDecisionService(
        store,
        model,
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        maximum_attempts=1,
    )
    context = _context(store)
    request = service.create_request(context)

    recorded = service.decide_request(request)
    replay_model = StaticModel({}, [])
    replayed = ClaimDecisionService(
        store,
        replay_model,
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        maximum_attempts=1,
    ).decide_request(request, replay=recorded.reference)

    assert recorded.mapping.disposition is ClaimDisposition.ACCEPTED
    assert recorded.mapping.candidate == context.candidate
    assert replayed.mapping == recorded.mapping
    assert replay_model.calls == []


def test_claim_decision_non_accepted_outcome(tmp_path: Path) -> None:
    """An Ontology gap remains a normal recorded Claim disposition.

    Guards an unresolved mapping requirement from being represented as a failed
    reasoning invocation or an accepted assertion.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = ClaimDecisionService(
        store,
        StaticModel(
            {
                "acceptance": "The Claim awaits Ontology review.",
                "conflict": "No assertion is available for conflict assessment.",
                "disposition": "ontology_gap",
                "mapping": "The active Ontology has no status predicate.",
                "scope": "The Claim is in scope.",
                "semantic_assertions": [],
                "validation": "No assertion can be validated before the gap closes.",
            },
            [],
        ),
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        maximum_attempts=1,
    )

    recorded = service.decide_request(service.create_request(_context(store)))

    assert recorded.mapping.disposition is ClaimDisposition.ONTOLOGY_GAP
    assert recorded.mapping.semantic_assertions == ()


def test_claim_decision_rejects_unbounded_model_evidence(tmp_path: Path) -> None:
    """A model cannot add unrecognised Evidence fields to a Claim decision.

    Guards the reasoning boundary from accepting GOV.UK evidence other than the
    Candidate Claim and completed decisions supplied in its durable request.
    """
    model = StaticModel(
        {
            "evidence": [],
        },
        [],
    )
    store = ArtefactStore(tmp_path / "artefacts")
    service = ClaimDecisionService(
        store,
        model,
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        maximum_attempts=1,
    )

    with pytest.raises(ReasoningValidationError):
        service.decide_request(service.create_request(_context(store)))

    structured_input = model.calls[0]
    assert isinstance(structured_input, dict)
    assert set(structured_input) == {
        "accepted_mappings",
        "candidate",
        "claim_id",
        "entity_decisions",
        "instructions",
        "maximum_accepted_context",
        "ontology_turtle",
        "shacl_turtle",
        "temporal_decisions",
    }
    assert "semantic_assertions" in structured_input["instructions"]


def test_claim_change_validation_shacl_diagnostics(tmp_path: Path) -> None:
    """A proposed assertion that breaks SHACL retains constraint diagnostics.

    Guards a SHACL failure from being reported as a model or workflow failure.
    """
    mapping = _accepted_mapping("available")
    assessment = ClaimChangeValidator().assess(
        _context(
            ArtefactStore(tmp_path / "artefacts"),
            shacl_turtle=(
                "@prefix example: <https://example.test/> .\n"
                "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
                "example:SchemeShape a sh:NodeShape ;\n"
                "    sh:targetNode example:scheme ;\n"
                "    sh:property [ a sh:PropertyShape ;\n"
                "        sh:path example:required ; sh:minCount 1 ] .\n"
            )
        ),
        mapping,
    )

    assert assessment.constraint_diagnostics
    assert assessment.conflict_diagnostics == ()


def test_claim_change_validation_conflict_diagnostics(tmp_path: Path) -> None:
    """An incompatible overlapping assertion retains its conflicting Claim ID.

    Guards two source-supported Claims from being treated as a provider failure
    merely because they make incompatible statements about the same property.
    """
    assessment = ClaimChangeValidator().assess(
        _context(
            ArtefactStore(tmp_path / "artefacts"),
            accepted_mappings=(_accepted_mapping("closed"),),
        ),
        _accepted_mapping("available"),
    )

    assert assessment.constraint_diagnostics == ()
    assert assessment.conflict_diagnostics[0].reason == (
        "Conflicts with accepted Claim 'scheme-status' on "
        "'https://example.test/scheme' and 'https://example.test/hasStatus'."
    )
    assert assessment.conflict_diagnostics[0].evidence == (_evidence(),)


def _context(
    store: ArtefactStore,
    *,
    accepted_mappings: tuple[ClaimMapping, ...] = (),
    shacl_turtle: str | None = None,
) -> ClaimDecisionContext:
    """Return one Claim with only its completed bounded decision context."""
    return ClaimDecisionContext(
        claim_id="scheme-status",
        candidate=CandidateClaim(
            "The scheme is available.",
            0.9,
            (_evidence(),),
            "The source states that the scheme is available.",
        ),
        entity_decisions=(
            store.write_json(
                "entity-resolution",
                {"history": {}, "reasoning_invocation": {}, "request": {}},
            ),
        ),
        temporal_decisions=(
            store.write_json(
                "temporal-resolution",
                {"outcome": {}, "reasoning_invocation": {}, "request": {}},
            ),
        ),
        ontology_turtle=(
            "@prefix example: <https://example.test/> .\n"
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "example:Scheme a example:Class .\n"
            "example:hasStatus a owl:DatatypeProperty .\n"
        ),
        shacl_turtle=shacl_turtle
        or (
            "@prefix example: <https://example.test/> .\n"
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
            "example:SchemeShape a sh:NodeShape ; sh:targetClass example:Scheme .\n"
        ),
        accepted_mappings=accepted_mappings,
        maximum_accepted_context=16,
    )


def _accepted_mapping(value: str) -> ClaimMapping:
    """Return one accepted mapping that projects a scheme-status assertion."""
    return ClaimMapping(
        claim_id="scheme-status",
        candidate=_context_candidate(),
        disposition=ClaimDisposition.ACCEPTED,
        mapping=StageReason("The active Ontology has the required predicate."),
        validation=StageReason("The assertion satisfies the active SHACL release."),
        conflict=StageReason("No conflicting accepted assertion was supplied."),
        scope=StageReason("The Claim is in scope."),
        acceptance=StageReason("The Claim is ready for projection."),
        semantic_assertions=(
            SemanticAssertion(
                "https://example.test/scheme",
                "https://example.test/hasStatus",
                value,
            ),
        ),
    )


def _context_candidate() -> CandidateClaim:
    """Return the source-supported candidate used by mapping fixtures."""
    return CandidateClaim(
        "The scheme is available.",
        0.9,
        (_evidence(),),
        "The source states that the scheme is available.",
    )


def _evidence() -> EvidenceAnchor:
    """Return the retained Evidence that supports the candidate Claim."""
    return EvidenceAnchor(
        canonical_text_sha256="a" * 64,
        canonicaliser_version="1",
        content_id="source-id",
        end_line=1,
        end_offset=24,
        prefix="",
        selected_text="The scheme is available.",
        source_json_sha256="b" * 64,
        source_url="https://www.gov.uk/example",
        start_line=1,
        start_offset=0,
        suffix="",
    )
