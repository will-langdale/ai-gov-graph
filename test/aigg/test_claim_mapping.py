"""Tests for mapping candidate Claims into accepted knowledge."""

from pathlib import Path

import pytest
from aigg.artefacts import ArtefactStore
from aigg.canonical import EvidenceAnchor
from aigg.claim_mapping import (
    ClaimDisposition,
    ClaimMapping,
    ClaimMappingService,
    SemanticAssertion,
    StageReason,
    project_claim_mappings,
)
from aigg.open_extraction import CandidateClaim
from pyoxigraph import QuerySolutions, Store


@pytest.mark.parametrize(
    ("disposition",),
    [
        pytest.param(ClaimDisposition.ONTOLOGY_GAP, id="ontology_gap"),
        pytest.param(ClaimDisposition.CONSTRAINT_VIOLATION, id="constraint_violation"),
        pytest.param(ClaimDisposition.CONFLICT, id="conflict"),
        pytest.param(ClaimDisposition.UNRESOLVED_ENTITY, id="unresolved_entity"),
        pytest.param(ClaimDisposition.UNRESOLVED_TIME, id="unresolved_time"),
        pytest.param(ClaimDisposition.OUT_OF_SCOPE, id="out_of_scope"),
        pytest.param(ClaimDisposition.LOW_CONFIDENCE, id="low_confidence"),
        pytest.param(ClaimDisposition.REJECTED, id="rejected"),
        pytest.param(ClaimDisposition.SUPERSEDED, id="superseded"),
    ],
)
def test_claim_mapping_non_accepted_dispositions(
    tmp_path: Path, disposition: ClaimDisposition
) -> None:
    """A non-accepted Claim retains its distinct disposition and stage reasons.

    Guards non-acceptance from being flattened into a rejected Claim or losing
    the mapping, validation, scope and acceptance reasoning needed to revisit it.
    """
    service = ClaimMappingService(ArtefactStore(tmp_path / "artefacts"))
    mapping = _mapping(disposition)

    recorded = service.record(mapping)

    assert service.inspect(recorded.reference) == mapping


def test_claim_mapping_accepted_knowledge_projection(tmp_path: Path) -> None:
    """An accepted Claim projects its assertion into canonical semantic knowledge.

    Guards an accepted Claim from appearing in its decision record without an
    observable assertion in the canonical semantic graph.
    """
    mapping = _mapping(
        ClaimDisposition.ACCEPTED,
        (
            SemanticAssertion(
                "https://example.test/scheme",
                "https://example.test/hasStatus",
                "available",
            ),
        ),
    )
    service = ClaimMappingService(ArtefactStore(tmp_path / "artefacts"))
    recorded = service.record(mapping)
    store = Store()
    store.extend(project_claim_mappings((service.inspect(recorded.reference),)))

    accepted = store.query(
        """
        SELECT ?status WHERE {
            GRAPH <https://w3id.org/aigg/graph/accepted-knowledge> {
                <https://example.test/scheme> <https://example.test/hasStatus> ?status .
            }
        }
        """
    )
    assert isinstance(accepted, QuerySolutions)
    assert [row["status"].value for row in accepted] == ["available"]


def test_claim_mapping_accepted_claim_provenance(tmp_path: Path) -> None:
    """An accepted Claim retains its decision history and Evidence.

    Guards the canonical semantic projection from replacing the Claim-level
    provenance that explains why the assertion became accepted knowledge.
    """
    mapping = _mapping(
        ClaimDisposition.ACCEPTED,
        (
            SemanticAssertion(
                "https://example.test/scheme",
                "https://example.test/hasStatus",
                "available",
            ),
        ),
    )
    service = ClaimMappingService(ArtefactStore(tmp_path / "artefacts"))
    recorded = service.record(mapping)
    store = Store()
    store.extend(project_claim_mappings((service.inspect(recorded.reference),)))

    claim = store.query(
        """
        PREFIX aigg: <https://w3id.org/aigg/>
        SELECT
            ?disposition ?mappingReason ?validationReason ?scopeReason ?acceptanceReason
        WHERE {
            GRAPH <https://w3id.org/aigg/graph/claims> {
                <https://w3id.org/aigg/claim/scheme-status> a aigg:Claim ;
                    aigg:disposition ?disposition ;
                    aigg:mappingReason ?mappingReason ;
                    aigg:validationReason ?validationReason ;
                    aigg:scopeReason ?scopeReason ;
                    aigg:acceptanceReason ?acceptanceReason .
            }
        }
        """
    )
    provenance = store.query(
        """
        PREFIX aigg: <https://w3id.org/aigg/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        ASK {
            GRAPH <https://w3id.org/aigg/graph/claims> {
                <https://w3id.org/aigg/claim/scheme-status>
                    aigg:evidence ?evidence ;
                    aigg:projectsAssertion ?statement .
                ?statement rdf:subject <https://example.test/scheme> ;
                    rdf:predicate <https://example.test/hasStatus> ;
                    rdf:object "available" .
            }
        }
        """
    )

    assert isinstance(claim, QuerySolutions)
    assert [
        tuple(
            row[name].value
            for name in (
                "disposition",
                "mappingReason",
                "validationReason",
                "scopeReason",
                "acceptanceReason",
            )
        )
        for row in claim
    ] == [
        (
            "accepted",
            "The ontology supplies a status predicate.",
            "The proposed assertion satisfies the active constraints.",
            "Availability supports content discovery.",
            "The Claim is ready for the accepted knowledge projection.",
        )
    ]
    assert bool(provenance)


def test_claim_mapping_accepted_requires_semantic_assertion() -> None:
    """An accepted Claim has at least one canonical semantic assertion.

    Guards acceptance from reporting a Claim as projected when it cannot add any
    observable knowledge to the canonical semantic graph.
    """
    with pytest.raises(ValueError, match="semantic assertion"):
        _mapping(ClaimDisposition.ACCEPTED)


def _mapping(
    disposition: ClaimDisposition,
    assertions: tuple[SemanticAssertion, ...] = (),
) -> ClaimMapping:
    """Return one complete candidate Claim mapping fixture."""
    return ClaimMapping(
        claim_id="scheme-status",
        candidate=_candidate(),
        disposition=disposition,
        mapping=StageReason("The ontology supplies a status predicate."),
        validation=StageReason(
            "The proposed assertion satisfies the active constraints."
        ),
        scope=StageReason("Availability supports content discovery."),
        acceptance=StageReason(
            "The Claim is ready for the accepted knowledge projection."
        ),
        semantic_assertions=assertions,
    )


def _candidate() -> CandidateClaim:
    """Return one source-supported candidate Claim for mapping."""
    evidence = EvidenceAnchor(
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
    return CandidateClaim(
        "The scheme is available.",
        0.9,
        (evidence,),
        "The source states that the scheme is available.",
    )
