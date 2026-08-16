"""Tests for durable Ontology evolution."""

from dataclasses import replace
from pathlib import Path

import pytest
from aigg.artefacts import (
    ArtefactReference,
    ArtefactStore,
    JsonValue,
    reference_from_json,
)
from aigg.canonical import EvidenceAnchor
from aigg.claim_mapping import (
    ClaimDisposition,
    ClaimMapping,
    ClaimMappingService,
    SemanticAssertion,
    StageReason,
)
from aigg.ontology_evolution import (
    ClaimReconsideration,
    ExternalOntologyArtefact,
    ExternalTermAssessment,
    OntologyChange,
    OntologyChangeKind,
    OntologyDecision,
    OntologyDecisionStage,
    OntologyEvolutionService,
    OntologyEvolutionValidationError,
    OntologyProposal,
    OntologyResearch,
)
from aigg.open_extraction import CandidateClaim

EXTERNAL_ONTOLOGY_TURTLE = (
    "@prefix example: <https://example.test/> .\n"
    "example:GenericScheme a example:Class .\n"
    "example:Scheme a example:Class .\n"
)
EXTERNAL_ONTOLOGY_SHA256 = (
    "870471242cb8e8ef41f015bdf590c3278155d56ef59fe39f6b84a9940bbd4568"
)


def test_ontology_evolution_vendor_external_artefact(tmp_path: Path) -> None:
    """External Ontology content remains available with retrieval metadata.

    Guards later replay from depending on a changing remote Ontology document.
    """
    service = OntologyEvolutionService(ArtefactStore(tmp_path / "artefacts"))

    reference = service.vendor(
        ExternalOntologyArtefact(
            source_url="https://example.test/ontology.ttl",
            retrieved_at="2026-08-16T12:00:00Z",
            available_version="2026-08",
            licence="CC0-1.0",
            turtle=EXTERNAL_ONTOLOGY_TURTLE,
        )
    )

    assert service.inspect_vendored(reference) == {
        "available_version": "2026-08",
        "content_sha256": EXTERNAL_ONTOLOGY_SHA256,
        "licence": "CC0-1.0",
        "retrieved_at": "2026-08-16T12:00:00Z",
        "source_url": "https://example.test/ontology.ttl",
        "turtle": EXTERNAL_ONTOLOGY_TURTLE,
    }


def test_ontology_evolution_valid_release(tmp_path: Path) -> None:
    """A validated proposal activates independently identified Ontology releases.

    Guards accepted knowledge from using a release before its RDF, reference and
    reconsideration checks have completed.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)

    proposal, _ = _proposal(service, store)
    outcome = service.consider(proposal)

    assert outcome.accepted
    assert outcome.current_release == outcome.activated_release
    assert outcome.activated_release is not None
    record = _record(service.inspect(outcome.reference))
    assert record["status"] == "accepted"
    assert record["ontology_release"] != record["shacl_release"]
    assert (
        _record(store.read_json(outcome.activated_release))["shacl_release"]
        == record["shacl_release"]
    )
    assert record["diagnostics"] == []


@pytest.mark.parametrize(
    ("ontology_turtle", "shacl_turtle", "diagnostic"),
    (
        pytest.param(
            "this is not Turtle",
            None,
            "Ontology RDF is invalid.",
            id="invalid_ontology",
        ),
        pytest.param(
            None,
            "this is not Turtle",
            "SHACL RDF is invalid.",
            id="invalid_shacl",
        ),
        pytest.param(
            None,
            (
                "@prefix example: <https://example.test/> .\n"
                "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
                "example:SchemeShape a sh:NodeShape ;\n"
                "    sh:targetClass example:Scheme ;\n"
                '    sh:property [ sh:path example:status ; sh:minCount "many" ] .\n'
            ),
            "SHACL release is invalid.",
            id="invalid_shacl_constraint",
        ),
    ),
)
def test_ontology_evolution_failed_proposal(
    tmp_path: Path,
    ontology_turtle: str | None,
    shacl_turtle: str | None,
    diagnostic: str,
) -> None:
    """A rejected proposal keeps diagnostics without replacing the active release.

    Guards a malformed Ontology proposal from corrupting the current Ontology.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    valid_proposal, _ = _proposal(service, store)
    activated = service.consider(valid_proposal)
    service = OntologyEvolutionService(store)

    invalid_proposal, _ = _proposal(
        service,
        store,
        ontology_turtle=ontology_turtle,
        shacl_turtle=shacl_turtle,
    )
    outcome = service.consider(
        invalid_proposal,
    )

    assert not outcome.accepted
    assert outcome.activated_release is None
    assert outcome.current_release == activated.current_release
    record = _record(service.inspect(outcome.reference))
    assert record["status"] == "failed"
    assert record["current_release"] == _reference_data(activated.current_release)
    assert record["diagnostics"] == [diagnostic]


def test_ontology_evolution_success_records_causation(
    tmp_path: Path,
) -> None:
    """A release records the evidence and changes that caused it.

    Guards Ontology history from hiding why a successful revision occurred.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)

    proposal, previous_mapping = _proposal(service, store)
    outcome = service.consider(proposal)

    record = _record(service.inspect(outcome.reference))
    assert record["research"] == _reference_data(proposal.research)
    assert record["changes"] == [
        {
            "description": "Adds the Scheme class needed by retained Claims.",
            "external_terms": ["https://example.test/GenericScheme"],
            "kind": "local_invention",
            "term": "https://example.test/Scheme",
        }
    ]
    reconsiderations = _list(record["reconsiderations"])
    reconsidered_record = _record(reconsiderations[0])
    assert reconsiderations == [
        {
            "claim_id": "scheme-status",
            "new_disposition": "accepted",
            "previous_mapping": _reference_data(previous_mapping),
            "reason": "The active Ontology now represents scheme status.",
            "reconsidered_mapping": reconsidered_record["reconsidered_mapping"],
        }
    ]


def test_ontology_evolution_success_reconsiders_claim(tmp_path: Path) -> None:
    """A successful release writes the affected Claim's reconsidered mapping.

    Guards a Claim reconsideration from remaining only proposal metadata after
    the related Ontology release activates.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    proposal, _ = _proposal(service, store)

    outcome = service.consider(proposal)

    record = _record(service.inspect(outcome.reference))
    reconsiderations = _list(record["reconsiderations"])
    reconsidered_record = _record(reconsiderations[0])
    reconsidered = reference_from_json(reconsidered_record["reconsidered_mapping"])
    assert reconsidered is not None
    assert (
        ClaimMappingService(store).inspect(reconsidered).disposition
        is ClaimDisposition.ACCEPTED
    )


def test_ontology_evolution_gap_prompts_research(tmp_path: Path) -> None:
    """An Ontology-gap Claim starts durable external-term research.

    Guards local Ontology invention from being considered without a recorded
    research prompt tied to the Claim that exposed the gap.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    ontology_gap = ClaimMappingService(store).record(
        _mapping(ClaimDisposition.ONTOLOGY_GAP)
    )
    external_artefact = _external_artefact(service)

    reference = service.prompt_research(
        ontology_gap.reference,
        _research(external_artefact),
    )

    research = _record(store.read_json(reference))
    assert research["ontology_gap"] == _reference_data(ontology_gap.reference)


def test_ontology_evolution_research_requires_ontology_gap(tmp_path: Path) -> None:
    """Research cannot be prompted by a Claim outside the Ontology-gap state.

    Guards an accepted Claim from manufacturing a causal record for a later
    local Ontology invention.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    accepted = ClaimMappingService(store).record(_mapping(ClaimDisposition.ACCEPTED))

    with pytest.raises(OntologyEvolutionValidationError, match="Ontology-gap"):
        service.prompt_research(
            accepted.reference, _research(_external_artefact(service))
        )


def test_ontology_evolution_proposal_reconsiders_claim(tmp_path: Path) -> None:
    """An Ontology proposal requires at least one affected Claim reconsideration.

    Guards a successful release from activating without retaining the affected
    Claim outcome required to explain its causal evidence.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    proposal, _ = _proposal(service, store)

    with pytest.raises(OntologyEvolutionValidationError, match="reconsider"):
        replace(proposal, reconsiderations=())


def test_ontology_evolution_external_reuse_requires_evidenced_term(
    tmp_path: Path,
) -> None:
    """External reuse cannot introduce a different local Ontology term.

    Guards a local invention from bypassing external-term research by claiming
    unrelated external reuse.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    proposal, ontology_gap = _proposal(service, store)
    research = service.prompt_research(
        ontology_gap,
        _research(_external_artefact(service), suitable=True),
    )
    external_reuse = replace(
        proposal,
        changes=(
            OntologyChange(
                "https://example.test/LocalReuse",
                "Claims reuse of an unrelated external term.",
                OntologyChangeKind.EXTERNAL_REUSE,
                ("https://example.test/GenericScheme",),
            ),
        ),
        research=research,
    )

    outcome = service.consider(external_reuse)

    assert not outcome.accepted
    record = _record(service.inspect(outcome.reference))
    assert record["diagnostics"] == [
        "Ontology change 'https://example.test/LocalReuse' must reuse an "
        "evidenced external term."
    ]


def test_ontology_evolution_reconsideration_satisfies_shacl(tmp_path: Path) -> None:
    """A reconsidered Claim must satisfy the proposed SHACL release.

    Guards the active Ontology from accepting a Claim mapping that violates the
    deterministic constraints of the release it would activate.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    proposal, _ = _proposal(service, store)
    constrained = replace(
        proposal,
        shacl_turtle=(
            "@prefix example: <https://example.test/> .\n"
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
            "example:SchemeShape a sh:NodeShape ;\n"
            "    sh:targetNode example:scheme ;\n"
            "    sh:property [ a sh:PropertyShape ;\n"
            "        sh:path example:required ;\n"
            "        sh:minCount 1 ] .\n"
        ),
    )

    outcome = service.consider(constrained)

    assert not outcome.accepted
    record = _record(service.inspect(outcome.reference))
    assert record["diagnostics"] == [
        "Reconsidered Claim mappings violate the SHACL release."
    ]


def test_ontology_evolution_unresearched_local_invention(tmp_path: Path) -> None:
    """A local term is rejected when research did not assess its external terms.

    Guards local invention from skipping the recorded external-term research that
    must precede it.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    proposal, previous_mapping = _proposal(service, store)
    unresearched = replace(
        proposal,
        changes=(
            OntologyChange(
                "https://example.test/LocalScheme",
                "Adds a local class without researching alternatives.",
                OntologyChangeKind.LOCAL_INVENTION,
                ("https://example.test/UnresearchedScheme",),
            ),
        ),
    )

    outcome = service.consider(unresearched)

    assert not outcome.accepted
    record = _record(service.inspect(outcome.reference))
    assert record["diagnostics"] == [
        "Ontology change 'https://example.test/LocalScheme' lacks recorded "
        "external-term research."
    ]


def test_ontology_evolution_unevidenced_external_term(tmp_path: Path) -> None:
    """A local term is rejected when research cannot evidence an external term.

    Guards a recorded assessment from presenting a term as external when its
    vendored Ontology artefact does not contain it.
    """
    store = ArtefactStore(tmp_path / "artefacts")
    service = OntologyEvolutionService(store)
    proposal, previous_mapping = _proposal(service, store)
    unrelated_artefact = service.vendor(
        ExternalOntologyArtefact(
            source_url="https://example.test/unrelated.ttl",
            retrieved_at="2026-08-16T12:00:00Z",
            available_version="2026-08",
            licence="CC0-1.0",
            turtle=(
                "@prefix example: <https://example.test/> .\n"
                "example:Other a example:Class .\n"
            ),
        )
    )
    research = service.prompt_research(
        previous_mapping,
        _research(unrelated_artefact),
    )

    outcome = service.consider(replace(proposal, research=research))

    assert not outcome.accepted
    record = _record(service.inspect(outcome.reference))
    assert record["diagnostics"] == [
        "External-term assessment is not evidenced by its artefact."
    ]


def _proposal(
    service: OntologyEvolutionService,
    store: ArtefactStore,
    *,
    ontology_turtle: str | None = None,
    shacl_turtle: str | None = None,
) -> tuple[OntologyProposal, ArtefactReference]:
    """Return one proposal with all recorded autonomous decisions."""
    if ontology_turtle is None:
        ontology_turtle = (
            "@prefix example: <https://example.test/> .\n"
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "example:Scheme a owl:Class .\n"
        )
    if shacl_turtle is None:
        shacl_turtle = (
            "@prefix example: <https://example.test/> .\n"
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
            "example:SchemeShape a sh:NodeShape ; sh:targetClass example:Scheme .\n"
        )
    external_artefact = _external_artefact(service)
    decisions = tuple(
        service.record_decision(
            OntologyDecision(stage, f"The {stage.value} supports this revision.")
        )
        for stage in OntologyDecisionStage
    )
    previous_mapping = ClaimMappingService(store).record(
        _mapping(ClaimDisposition.ONTOLOGY_GAP)
    )
    research = _research_reference(
        service, external_artefact, previous_mapping.reference
    )
    proposal = OntologyProposal(
        research=research,
        researcher=decisions[0],
        proposer=decisions[1],
        critic=decisions[2],
        synthesiser=decisions[3],
        ontology_turtle=ontology_turtle,
        shacl_turtle=shacl_turtle,
        changes=(
            OntologyChange(
                "https://example.test/Scheme",
                "Adds the Scheme class needed by retained Claims.",
                OntologyChangeKind.LOCAL_INVENTION,
                ("https://example.test/GenericScheme",),
            ),
        ),
        reconsiderations=(
            ClaimReconsideration(
                previous_mapping.reference,
                _mapping(ClaimDisposition.ACCEPTED),
                reason="The active Ontology now represents scheme status.",
            ),
        ),
    )
    return proposal, previous_mapping.reference


def _research_reference(
    service: OntologyEvolutionService,
    external_artefact: ArtefactReference,
    ontology_gap: ArtefactReference,
) -> ArtefactReference:
    """Record research before every proposal in the fixture."""
    return service.prompt_research(ontology_gap, _research(external_artefact))


def _research(
    external_artefact: ArtefactReference,
    *,
    suitable: bool = False,
) -> OntologyResearch:
    """Return one recorded assessment of an external term."""
    return OntologyResearch(
        query="GOV.UK scheme status",
        conclusion="No reusable term models the required GOV.UK concept.",
        external_artefacts=(external_artefact,),
        assessments=(
            ExternalTermAssessment(
                term="https://example.test/GenericScheme",
                artefact=external_artefact,
                suitable=suitable,
                rationale="The external term lacks the required status relation.",
            ),
        ),
    )


def _external_artefact(service: OntologyEvolutionService) -> ArtefactReference:
    """Vendor the external Ontology used by the fixture research."""
    return service.vendor(
        ExternalOntologyArtefact(
            source_url="https://example.test/generic-scheme.ttl",
            retrieved_at="2026-08-16T12:00:00Z",
            available_version="2026-08",
            licence="CC0-1.0",
            turtle=EXTERNAL_ONTOLOGY_TURTLE,
        )
    )


def _mapping(disposition: ClaimDisposition) -> ClaimMapping:
    """Return one source-supported Claim mapping for reconsideration."""
    assertions = (
        (
            SemanticAssertion(
                "https://example.test/scheme",
                "https://example.test/hasStatus",
                "available",
            ),
        )
        if disposition is ClaimDisposition.ACCEPTED
        else ()
    )
    return ClaimMapping(
        claim_id="scheme-status",
        candidate=CandidateClaim(
            assertion="The scheme is available.",
            confidence=0.9,
            evidence=(
                EvidenceAnchor(
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
                ),
            ),
            rationale="The source states that the scheme is available.",
        ),
        disposition=disposition,
        mapping=StageReason("The Ontology supplies a status predicate."),
        validation=StageReason("The proposed assertion satisfies the constraints."),
        scope=StageReason("Availability supports content discovery."),
        acceptance=StageReason("The Claim is ready for projection."),
        semantic_assertions=assertions,
    )


def _record(value: JsonValue) -> dict[str, JsonValue]:
    """Return a durable record after checking its JSON object shape."""
    assert isinstance(value, dict)
    return value


def _list(value: JsonValue) -> list[JsonValue]:
    """Return a durable list after checking its JSON array shape."""
    assert isinstance(value, list)
    return value


def _reference_data(reference: ArtefactReference | None) -> dict[str, str] | None:
    """Return the durable data for an optional artefact reference."""
    if reference is None:
        return None
    return {
        "identity": reference.identity,
        "kind": reference.kind,
        "schema_version": reference.schema_version,
    }
