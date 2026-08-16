"""Tests for the bounded Ontology-review decision flow."""

from dataclasses import dataclass
from pathlib import Path

from aigg.artefacts import ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor
from aigg.claim_mapping import (
    ClaimDisposition,
    ClaimMapping,
    ClaimMappingService,
    StageReason,
)
from aigg.ontology_evolution import ExternalOntologyArtefact
from aigg.ontology_review import OntologyReviewService
from aigg.open_extraction import CandidateClaim
from aigg.reasoning import ModelConfiguration, StructuredModel


@dataclass
class SequencedModel(StructuredModel):
    """Return recorded mock decisions in researcher-to-synthesiser order."""

    outputs: list[JsonValue]
    calls: list[JsonValue]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Record the bounded input for the next autonomous review role."""
        del configuration
        self.calls.append(structured_input)
        return self.outputs.pop(0)


@dataclass
class StaticExternalOntologyRetriever:
    """Return configured artefacts at the explicit external Ontology boundary."""

    artefacts: tuple[ExternalOntologyArtefact, ...]
    queries: list[str]

    def retrieve(self, query: str) -> tuple[ExternalOntologyArtefact, ...]:
        """Record the query without retrieving any source-document evidence."""
        self.queries.append(query)
        return self.artefacts


def test_ontology_review_accepted(tmp_path: Path) -> None:
    """An Ontology gap records every role before proposal consideration.

    Guards an Ontology revision from bypassing external-term research or any of
    the researcher, proposer, critic and synthesiser decision records.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    ontology_gap = ClaimMappingService(store).record(_ontology_gap())
    retriever = StaticExternalOntologyRetriever((_external_ontology(),), [])
    model = SequencedModel(
        [
            {
                "query": "scheme status ontology",
                "rationale": "The gap needs an external-term assessment.",
            },
            {
                "assessments": [
                    {
                        "artefact_index": 0,
                        "rationale": "The generic term does not model scheme status.",
                        "suitable": False,
                        "term": "https://example.test/GenericScheme",
                    }
                ],
                "conclusion": "A local status term is needed.",
                "rationale": "A local status term is the smallest change.",
            },
            {
                "rationale": (
                    "The proposed term does not duplicate a suitable external term."
                )
            },
            {
                "changes": [
                    {
                        "description": "Adds a status predicate for schemes.",
                        "external_terms": ["https://example.test/GenericScheme"],
                        "kind": "local_invention",
                        "term": "https://example.test/hasStatus",
                    }
                ],
                "mapping": {
                    "acceptance": "The assertion is ready for projection.",
                    "conflict": "No conflicting accepted assertion was supplied.",
                    "disposition": "accepted",
                    "mapping": "The proposed Ontology supplies the status predicate.",
                    "scope": "The Claim is in scope.",
                    "semantic_assertions": [
                        {
                            "object": "available",
                            "object_kind": "literal",
                            "predicate": "https://example.test/hasStatus",
                            "subject": "https://example.test/scheme",
                        }
                    ],
                    "validation": "The assertion satisfies the proposed SHACL release.",
                },
                "ontology_turtle": (
                    "@prefix example: <https://example.test/> .\n"
                    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
                    "example:hasStatus a owl:DatatypeProperty .\n"
                ),
                "rationale": (
                    "The reviewed change is ready for immutable-release validation."
                ),
                "reconsideration_reason": "The new predicate represents the gap Claim.",
                "shacl_turtle": (
                    "@prefix example: <https://example.test/> .\n"
                    "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
                    "example:SchemeShape a sh:NodeShape ; "
                    "sh:targetNode example:scheme .\n"
                ),
            },
        ],
        [],
    )
    service = OntologyReviewService(
        store,
        model,
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        retriever,
        maximum_attempts=1,
    )
    request = service.create_request(
        ontology_gap.reference,
        ontology_turtle="@prefix example: <https://example.test/> .\n",
        shacl_turtle=(
            "@prefix example: <https://example.test/> .\n"
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
            "example:ExistingShape a sh:NodeShape ; sh:targetNode example:existing .\n"
        ),
    )

    recorded = service.review_request(request)
    replay_model = SequencedModel([], [])
    replayed = OntologyReviewService(
        store,
        replay_model,
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        retriever,
        maximum_attempts=1,
    ).review_request(request, replay=recorded.reference)

    assert recorded.outcome.accepted
    assert replayed.outcome == recorded.outcome
    assert len(model.calls) == 4
    assert all(
        isinstance(call, dict) and "instructions" in call for call in model.calls
    )
    assert replay_model.calls == []
    assert retriever.queries == ["scheme status ontology"]
    research = store.read_json(recorded.research)
    assert isinstance(research, dict)
    assert research["ontology_gap"] == {
        "identity": ontology_gap.reference.identity,
        "kind": "claim-mapping",
        "schema_version": "1",
    }


def _ontology_gap() -> ClaimMapping:
    """Return a source-supported Claim whose current Ontology has a gap."""
    return ClaimMapping(
        claim_id="scheme-status",
        candidate=CandidateClaim(
            "The scheme is available.",
            0.9,
            (_evidence(),),
            "The source states that the scheme is available.",
        ),
        disposition=ClaimDisposition.ONTOLOGY_GAP,
        mapping=StageReason("The active Ontology has no status predicate."),
        validation=StageReason(
            "No assertion can be validated before the gap is resolved."
        ),
        conflict=StageReason(
            "No conflict assessment is possible without an assertion."
        ),
        scope=StageReason("The Claim is in scope."),
        acceptance=StageReason("The Claim awaits Ontology review."),
    )


def _external_ontology() -> ExternalOntologyArtefact:
    """Return the sole result from the explicit external Ontology boundary."""
    return ExternalOntologyArtefact(
        source_url="https://example.test/ontology.ttl",
        retrieved_at="2026-08-16T12:00:00Z",
        available_version="2026-08",
        licence="CC0-1.0",
        turtle=(
            "@prefix example: <https://example.test/> .\n"
            "example:GenericScheme a example:Class .\n"
        ),
    )


def _evidence() -> EvidenceAnchor:
    """Return the Evidence retained with the Claim that prompted review."""
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
