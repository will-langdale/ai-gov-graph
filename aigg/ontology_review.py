"""Recorded Ontology review without access to additional GOV.UK evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from aigg.artefacts import (
    ArtefactIntegrityError,
    ArtefactReference,
    ArtefactStore,
    JsonValue,
    reference_as_json,
    reference_from_json,
)
from aigg.claim_mapping import (
    ClaimDisposition,
    ClaimMapping,
    ClaimMappingService,
    claim_mapping_from_json,
)
from aigg.ontology_evolution import (
    ClaimReconsideration,
    ExternalOntologyArtefact,
    ExternalTermAssessment,
    OntologyChange,
    OntologyChangeKind,
    OntologyDecision,
    OntologyDecisionStage,
    OntologyEvolutionOutcome,
    OntologyEvolutionService,
    OntologyProposal,
    OntologyResearch,
)
from aigg.reasoning import ModelConfiguration, ReasoningRunner, StructuredModel


class OntologyReviewValidationError(ValueError):
    """Raised when an Ontology-review record cannot preserve its provenance."""


class OntologyReviewOperationalError(RuntimeError):
    """Raised when the explicit external Ontology boundary cannot respond."""


class ExternalOntologyRetriever(Protocol):
    """Retrieve only external Ontology artefacts for a supplied research query."""

    def retrieve(self, query: str) -> tuple[ExternalOntologyArtefact, ...]:
        """Return external Ontology artefacts without retrieving GOV.UK evidence."""


@dataclass(frozen=True)
class RecordedOntologyReview:
    """One complete Ontology-review flow and its immutable outcome."""

    reference: ArtefactReference
    research: ArtefactReference
    researcher: ArtefactReference
    proposer: ArtefactReference
    critic: ArtefactReference
    synthesiser: ArtefactReference
    outcome: OntologyEvolutionOutcome


class OntologyReviewService:
    """Coordinate bounded autonomous review for one recorded Ontology-gap Claim."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        retriever: ExternalOntologyRetriever,
        *,
        maximum_attempts: int,
    ) -> None:
        """Create a review service around durable decisions and retrieval artefacts."""
        self._store = store
        self._runner = ReasoningRunner(
            store, model, configuration, maximum_attempts=maximum_attempts
        )
        self._retriever = retriever
        self._mappings = ClaimMappingService(store)
        self._evolution = OntologyEvolutionService(store)

    def create_request(
        self,
        ontology_gap: ArtefactReference,
        *,
        ontology_turtle: str,
        shacl_turtle: str,
    ) -> ArtefactReference:
        """Store a review request that carries no additional source evidence."""
        self._inspect_gap(ontology_gap)
        _require_text(ontology_turtle, "Active Ontology RDF")
        _require_text(shacl_turtle, "Active SHACL RDF")
        return self._store.write_json(
            "ontology-review-request",
            {
                "ontology_gap": reference_as_json(ontology_gap),
                "ontology_turtle": ontology_turtle,
                "shacl_turtle": shacl_turtle,
            },
        )

    def review_request(
        self,
        request: ArtefactReference,
        *,
        replay: ArtefactReference | None = None,
    ) -> RecordedOntologyReview:
        """Run and retain researcher, proposer, critic and synthesiser decisions."""
        if replay is not None:
            return self._replay(request, replay)
        ontology_gap, ontology_turtle, shacl_turtle = _request_from_json(
            self._store.read_json(request)
        )
        mapping = self._inspect_gap(ontology_gap)
        research_output, researcher_invocation = self._run(
            "ontology-review-researcher",
            {
                "ontology_gap": mapping.as_json(),
                "ontology_turtle": ontology_turtle,
                "shacl_turtle": shacl_turtle,
            },
            _validated_research_output,
        )
        research_details = _research_output_from_json(research_output)
        researcher = self._evolution.record_decision(
            OntologyDecision(
                OntologyDecisionStage.RESEARCHER, research_details.rationale
            )
        )
        artefacts = self._retrieve_and_vendor(research_details.query)
        proposer_output, proposer_invocation = self._run(
            "ontology-review-proposer",
            {
                "external_artefacts": _vendored_artefacts_json(self._store, artefacts),
                "ontology_gap": mapping.as_json(),
                "ontology_turtle": ontology_turtle,
            },
            _validated_proposal_output,
        )
        proposal_details = _proposal_output_from_json(proposer_output)
        proposer = self._evolution.record_decision(
            OntologyDecision(OntologyDecisionStage.PROPOSER, proposal_details.rationale)
        )
        research = self._evolution.prompt_research(
            ontology_gap,
            OntologyResearch(
                research_details.query,
                proposal_details.conclusion,
                tuple(reference for reference, _ in artefacts),
                _assessments_from_output(proposal_details.assessments, artefacts),
            ),
        )
        critic, critic_invocation = self._record_rationale_decision(
            OntologyDecisionStage.CRITIC,
            "ontology-review-critic",
            {
                "external_artefacts": _vendored_artefacts_json(self._store, artefacts),
                "ontology_gap": mapping.as_json(),
                "ontology_turtle": ontology_turtle,
                "proposer": self._store.read_json(proposer),
                "research": self._store.read_json(research),
            },
        )
        synthesis_output, synthesiser_invocation = self._run(
            "ontology-review-synthesiser",
            {
                "critic": self._store.read_json(critic),
                "external_artefacts": _vendored_artefacts_json(self._store, artefacts),
                "ontology_gap": mapping.as_json(),
                "ontology_turtle": ontology_turtle,
                "proposer": self._store.read_json(proposer),
                "research": self._store.read_json(research),
                "shacl_turtle": shacl_turtle,
            },
            _validated_synthesis_output,
        )
        synthesis = _synthesis_from_json(synthesis_output, mapping, ontology_gap)
        synthesiser = self._evolution.record_decision(
            OntologyDecision(OntologyDecisionStage.SYNTHESISER, synthesis.rationale)
        )
        outcome = self._evolution.consider(
            OntologyProposal(
                research,
                researcher,
                proposer,
                critic,
                synthesiser,
                synthesis.ontology_turtle,
                synthesis.shacl_turtle,
                synthesis.changes,
                (synthesis.reconsideration,),
            )
        )
        reference = self._store.write_json(
            "ontology-review",
            {
                "critic": reference_as_json(critic),
                "evolution": reference_as_json(outcome.reference),
                "invocations": {
                    "critic": reference_as_json(critic_invocation),
                    "proposer": reference_as_json(proposer_invocation),
                    "researcher": reference_as_json(researcher_invocation),
                    "synthesiser": reference_as_json(synthesiser_invocation),
                },
                "proposer": reference_as_json(proposer),
                "request": reference_as_json(request),
                "research": reference_as_json(research),
                "researcher": reference_as_json(researcher),
                "synthesiser": reference_as_json(synthesiser),
            },
        )
        return RecordedOntologyReview(
            reference, research, researcher, proposer, critic, synthesiser, outcome
        )

    def _replay(
        self, request: ArtefactReference, replay: ArtefactReference
    ) -> RecordedOntologyReview:
        """Read a complete recorded review without calling a model or retriever."""
        value = self._store.read_json(replay)
        if not isinstance(value, dict) or set(value) != {
            "critic",
            "evolution",
            "invocations",
            "proposer",
            "request",
            "research",
            "researcher",
            "synthesiser",
        }:
            msg = "Ontology review artefact has an invalid shape."
            raise OntologyReviewValidationError(msg)
        if _required_reference(value["request"], "Ontology review request") != request:
            msg = "Ontology review replay belongs to a different request."
            raise OntologyReviewValidationError(msg)
        evolution = _required_reference(value["evolution"], "Ontology evolution")
        return RecordedOntologyReview(
            replay,
            _required_reference(value["research"], "Ontology research"),
            _required_reference(value["researcher"], "Researcher decision"),
            _required_reference(value["proposer"], "Proposer decision"),
            _required_reference(value["critic"], "Critic decision"),
            _required_reference(value["synthesiser"], "Synthesiser decision"),
            _evolution_outcome_from_record(
                evolution, self._evolution.inspect(evolution)
            ),
        )

    def _run(
        self,
        stage: str,
        structured_input: dict[str, JsonValue],
        validate_output: Callable[[JsonValue], JsonValue],
    ) -> tuple[JsonValue, ArtefactReference]:
        """Run one durable structured decision without exposing broader context."""
        invocation = self._runner.run(
            stage=stage,
            structured_input=structured_input,
            validate_output=validate_output,
        )
        return invocation.output, invocation.reference

    def _record_rationale_decision(
        self,
        decision_stage: OntologyDecisionStage,
        reasoning_stage: str,
        structured_input: dict[str, JsonValue],
    ) -> tuple[ArtefactReference, ArtefactReference]:
        """Run one role whose contribution is an inspectable rationale."""
        output, invocation = self._run(
            reasoning_stage, structured_input, _validated_rationale_output
        )
        rationale = _rationale_from_json(output)
        return (
            self._evolution.record_decision(
                OntologyDecision(decision_stage, rationale)
            ),
            invocation,
        )

    def _retrieve_and_vendor(
        self, query: str
    ) -> tuple[tuple[ArtefactReference, ExternalOntologyArtefact], ...]:
        """Vendor each retrieved external artefact before it influences a proposal."""
        try:
            retrieved = self._retriever.retrieve(query)
        except Exception as error:
            msg = "External Ontology retrieval failed."
            raise OntologyReviewOperationalError(msg) from error
        if not retrieved:
            msg = "External Ontology retrieval returned no artefacts."
            raise OntologyReviewOperationalError(msg)
        return tuple(
            (self._evolution.vendor(artefact), artefact) for artefact in retrieved
        )

    def _inspect_gap(self, reference: ArtefactReference) -> ClaimMapping:
        """Require the recorded Claim that starts review to retain its gap state."""
        if reference.kind != "claim-mapping":
            msg = "Ontology review must start from a Claim mapping."
            raise OntologyReviewValidationError(msg)
        mapping = self._mappings.inspect(reference)
        if mapping.disposition is not ClaimDisposition.ONTOLOGY_GAP:
            msg = "Ontology review must start from an Ontology-gap Claim."
            raise OntologyReviewValidationError(msg)
        return mapping


@dataclass(frozen=True)
class _ResearchOutput:
    """The validated researcher decision before external retrieval occurs."""

    query: str
    rationale: str


@dataclass(frozen=True)
class _ProposalOutput:
    """The proposer assessment made after external artefacts are vendored."""

    conclusion: str
    rationale: str
    assessments: tuple[dict[str, JsonValue], ...]


@dataclass(frozen=True)
class _Synthesis:
    """The validated proposal content supplied to immutable-release validation."""

    rationale: str
    ontology_turtle: str
    shacl_turtle: str
    changes: tuple[OntologyChange, ...]
    reconsideration: ClaimReconsideration


def _validated_research_output(value: JsonValue) -> JsonValue:
    """Validate researcher JSON while retaining it unchanged in its invocation."""
    _research_output_from_json(value)
    return value


def _research_output_from_json(value: JsonValue) -> _ResearchOutput:
    """Parse the researcher output before the retrieval boundary is crossed."""
    if not isinstance(value, dict) or set(value) != {"query", "rationale"}:
        msg = "Ontology researcher output has an invalid shape."
        raise OntologyReviewValidationError(msg)
    return _ResearchOutput(
        _text(value["query"], "Ontology research query"),
        _text(value["rationale"], "Ontology researcher rationale"),
    )


def _validated_proposal_output(value: JsonValue) -> JsonValue:
    """Validate proposer research assessment after external retrieval completes."""
    _proposal_output_from_json(value)
    return value


def _proposal_output_from_json(value: JsonValue) -> _ProposalOutput:
    """Parse the assessment of the vendored external Ontology artefacts."""
    if not isinstance(value, dict) or set(value) != {
        "assessments",
        "conclusion",
        "rationale",
    }:
        msg = "Ontology proposer output has an invalid shape."
        raise OntologyReviewValidationError(msg)
    assessments = value["assessments"]
    if not isinstance(assessments, list) or not assessments:
        msg = "Ontology proposer output must assess at least one external term."
        raise OntologyReviewValidationError(msg)
    return _ProposalOutput(
        _text(value["conclusion"], "Ontology research conclusion"),
        _text(value["rationale"], "Ontology proposer rationale"),
        tuple(_assessment_output_from_json(item) for item in assessments),
    )


def _assessment_output_from_json(value: JsonValue) -> dict[str, JsonValue]:
    """Validate an assessment that can only select a retrieved artefact by index."""
    if not isinstance(value, dict) or set(value) != {
        "artefact_index",
        "rationale",
        "suitable",
        "term",
    }:
        msg = "External-term assessment has an invalid shape."
        raise OntologyReviewValidationError(msg)
    if not isinstance(value["artefact_index"], int) or isinstance(
        value["artefact_index"], bool
    ):
        msg = "External-term assessment artefact index must be an integer."
        raise OntologyReviewValidationError(msg)
    if not isinstance(value["suitable"], bool):
        msg = "External-term assessment suitability must be a boolean."
        raise OntologyReviewValidationError(msg)
    _text(value["term"], "External Ontology term")
    _text(value["rationale"], "External-term assessment rationale")
    return value


def _assessments_from_output(
    assessments: tuple[dict[str, JsonValue], ...],
    artefacts: tuple[tuple[ArtefactReference, ExternalOntologyArtefact], ...],
) -> tuple[ExternalTermAssessment, ...]:
    """Bind research assessments to vendored artefacts before they influence change."""
    parsed: list[ExternalTermAssessment] = []
    for assessment in assessments:
        index = assessment["artefact_index"]
        assert isinstance(index, int) and not isinstance(index, bool)
        if not 0 <= index < len(artefacts):
            msg = (
                "External-term assessment selects an artefact outside retrieval "
                "results."
            )
            raise OntologyReviewValidationError(msg)
        suitable = assessment["suitable"]
        assert isinstance(suitable, bool)
        parsed.append(
            ExternalTermAssessment(
                _text(assessment["term"], "External Ontology term"),
                artefacts[index][0],
                suitable,
                _text(assessment["rationale"], "External-term assessment rationale"),
            )
        )
    return tuple(parsed)


def _vendored_artefacts_json(
    store: ArtefactStore,
    artefacts: tuple[tuple[ArtefactReference, ExternalOntologyArtefact], ...],
) -> list[JsonValue]:
    """Expose only already-vendored external artefacts to later review roles."""
    return [
        {
            "artefact": reference_as_json(reference),
            "content": store.read_json(reference),
        }
        for reference, _ in artefacts
    ]


def _validated_rationale_output(value: JsonValue) -> JsonValue:
    """Validate a role rationale while preserving the exact structured output."""
    _rationale_from_json(value)
    return value


def _rationale_from_json(value: JsonValue) -> str:
    """Parse the one permitted proposer or critic decision field."""
    if not isinstance(value, dict) or set(value) != {"rationale"}:
        msg = "Ontology decision output must contain only a rationale."
        raise OntologyReviewValidationError(msg)
    return _text(value["rationale"], "Ontology decision rationale")


def _validated_synthesis_output(value: JsonValue) -> JsonValue:
    """Validate synthesis JSON before creating an immutable-release proposal."""
    if not isinstance(value, dict) or set(value) != {
        "changes",
        "mapping",
        "ontology_turtle",
        "rationale",
        "reconsideration_reason",
        "shacl_turtle",
    }:
        msg = "Ontology synthesiser output has an invalid shape."
        raise OntologyReviewValidationError(msg)
    _text(value["ontology_turtle"], "Proposed Ontology RDF")
    _text(value["shacl_turtle"], "Proposed SHACL RDF")
    _text(value["rationale"], "Ontology synthesiser rationale")
    _text(value["reconsideration_reason"], "Claim reconsideration reason")
    if not isinstance(value["changes"], list) or not value["changes"]:
        msg = "Ontology synthesis must contain at least one change."
        raise OntologyReviewValidationError(msg)
    if not isinstance(value["mapping"], dict):
        msg = "Ontology synthesis must contain one reconsidered Claim mapping."
        raise OntologyReviewValidationError(msg)
    return value


def _synthesis_from_json(
    value: JsonValue, ontology_gap: ClaimMapping, gap_reference: ArtefactReference
) -> _Synthesis:
    """Bind a synthesised proposal only to the Claim that prompted review."""
    _validated_synthesis_output(value)
    assert isinstance(value, dict)
    changes = tuple(_change_from_json(item) for item in value["changes"])
    mapping = _reconsidered_mapping_from_json(value["mapping"], ontology_gap)
    return _Synthesis(
        _text(value["rationale"], "Ontology synthesiser rationale"),
        _text(value["ontology_turtle"], "Proposed Ontology RDF"),
        _text(value["shacl_turtle"], "Proposed SHACL RDF"),
        changes,
        ClaimReconsideration(
            gap_reference,
            mapping,
            _text(value["reconsideration_reason"], "Claim reconsideration reason"),
        ),
    )


def _change_from_json(value: JsonValue) -> OntologyChange:
    """Parse one minimal Ontology change from the synthesiser decision."""
    if not isinstance(value, dict) or set(value) != {
        "description",
        "external_terms",
        "kind",
        "term",
    }:
        msg = "Ontology change has an invalid shape."
        raise OntologyReviewValidationError(msg)
    terms = value["external_terms"]
    if not isinstance(terms, list) or not terms:
        msg = "Ontology change external terms must be a non-empty list."
        raise OntologyReviewValidationError(msg)
    try:
        kind = OntologyChangeKind(_text(value["kind"], "Ontology change kind"))
    except ValueError as error:
        msg = "Ontology change kind is not recognised."
        raise OntologyReviewValidationError(msg) from error
    return OntologyChange(
        _text(value["term"], "Ontology change term"),
        _text(value["description"], "Ontology change description"),
        kind,
        tuple(_text(term, "External Ontology term") for term in terms),
    )


def _reconsidered_mapping_from_json(
    value: JsonValue, ontology_gap: ClaimMapping
) -> ClaimMapping:
    """Preserve the original candidate and Evidence during reconsideration."""
    if not isinstance(value, dict) or set(value) != {
        "acceptance",
        "conflict",
        "disposition",
        "mapping",
        "scope",
        "semantic_assertions",
        "validation",
    }:
        msg = "Reconsidered Claim mapping has an invalid shape."
        raise OntologyReviewValidationError(msg)
    record: dict[str, JsonValue] = {
        "acceptance": value["acceptance"],
        "candidate": ontology_gap.candidate.as_json(),
        "claim_id": ontology_gap.claim_id,
        "conflict": value["conflict"],
        "disposition": value["disposition"],
        "mapping": value["mapping"],
        "scope": value["scope"],
        "semantic_assertions": value["semantic_assertions"],
        "validation": value["validation"],
    }
    return claim_mapping_from_json(record)


def _request_from_json(value: JsonValue) -> tuple[ArtefactReference, str, str]:
    """Parse one durable Ontology-review request."""
    if not isinstance(value, dict) or set(value) != {
        "ontology_gap",
        "ontology_turtle",
        "shacl_turtle",
    }:
        msg = "Ontology review request has an invalid shape."
        raise OntologyReviewValidationError(msg)
    return (
        _required_reference(value["ontology_gap"], "Ontology-gap Claim"),
        _text(value["ontology_turtle"], "Active Ontology RDF"),
        _text(value["shacl_turtle"], "Active SHACL RDF"),
    )


def _evolution_outcome_from_record(
    reference: ArtefactReference, value: JsonValue
) -> OntologyEvolutionOutcome:
    """Recreate the immutable-release outcome needed for exact review replay."""
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        msg = "Ontology evolution outcome has an invalid shape."
        raise OntologyReviewValidationError(msg)
    status = value["status"]
    if status not in {"accepted", "failed"}:
        msg = "Ontology evolution outcome has an invalid status."
        raise OntologyReviewValidationError(msg)
    current_release = _optional_reference(
        value.get("current_release"), "Current release"
    )
    ontology_release = _required_reference(
        value.get("ontology_release"), "Ontology release"
    )
    return OntologyEvolutionOutcome(
        reference,
        current_release,
        ontology_release if status == "accepted" else None,
        status == "accepted",
    )


def _required_reference(value: JsonValue, name: str) -> ArtefactReference:
    """Require a complete durable reference in one review record."""
    try:
        reference = reference_from_json(value)
    except ArtefactIntegrityError as error:
        raise OntologyReviewValidationError(f"{name} reference is invalid.") from error
    if reference is None:
        msg = f"{name} reference is required."
        raise OntologyReviewValidationError(msg)
    return reference


def _optional_reference(value: JsonValue | None, name: str) -> ArtefactReference | None:
    """Parse an optional durable reference in an evolution outcome."""
    try:
        return reference_from_json(value)
    except ArtefactIntegrityError as error:
        raise OntologyReviewValidationError(f"{name} reference is invalid.") from error


def _text(value: JsonValue, name: str) -> str:
    """Require meaningful JSON text for a durable decision record."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be non-empty text."
        raise OntologyReviewValidationError(msg)
    return value


def _require_text(value: str, name: str) -> None:
    """Reject empty dataclass input text before writing a review request."""
    if not value.strip():
        msg = f"{name} must be non-empty text."
        raise OntologyReviewValidationError(msg)
