"""Durable, validated evolution of an emergent Ontology."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from pyoxigraph import NamedNode, Quad, RdfFormat, Store
from pyshacl import validate as validate_shacl
from pyshacl.errors import ReportableRuntimeError

from aigg.artefacts import (
    ArtefactIntegrityError,
    ArtefactReference,
    ArtefactStore,
    JsonValue,
    reference_as_json,
    reference_from_json,
)
from aigg.claim_mapping import (
    ACCEPTED_KNOWLEDGE_GRAPH,
    ClaimDisposition,
    ClaimMapping,
    ClaimMappingService,
    RecordedClaimMapping,
    claim_mapping_from_json,
    project_claim_mappings,
)


class OntologyEvolutionValidationError(ValueError):
    """Raised when an Ontology evolution input is incomplete."""


class OntologyDecisionStage(StrEnum):
    """The autonomous roles that inspect an Ontology revision."""

    RESEARCHER = "researcher"
    PROPOSER = "proposer"
    CRITIC = "critic"
    SYNTHESISER = "synthesiser"


@dataclass(frozen=True)
class ExternalOntologyArtefact:
    """One vendored external Ontology artefact and its retrieval metadata."""

    source_url: str
    retrieved_at: str
    available_version: str
    licence: str
    turtle: str

    def __post_init__(self) -> None:
        """Require enough metadata to replay an external Ontology judgement."""
        for value, name in (
            (self.source_url, "External Ontology source URL"),
            (self.retrieved_at, "External Ontology retrieval time"),
            (self.available_version, "External Ontology version"),
            (self.licence, "External Ontology licence"),
            (self.turtle, "External Ontology content"),
        ):
            _require_text(value, name)

    def as_json(self) -> dict[str, str]:
        """Return the durable vendor record."""
        return {
            "available_version": self.available_version,
            "content_sha256": sha256(self.turtle.encode("utf-8")).hexdigest(),
            "licence": self.licence,
            "retrieved_at": self.retrieved_at,
            "source_url": self.source_url,
            "turtle": self.turtle,
        }


@dataclass(frozen=True)
class OntologyResearch:
    """Recorded research completed before a local Ontology revision."""

    query: str
    conclusion: str
    external_artefacts: tuple[ArtefactReference, ...]
    assessments: tuple[ExternalTermAssessment, ...]

    def __post_init__(self) -> None:
        """Require an inspectable search and its conclusion."""
        _require_text(self.query, "Ontology research query")
        _require_text(self.conclusion, "Ontology research conclusion")
        if not self.assessments:
            msg = "Ontology research must assess at least one external term."
            raise OntologyEvolutionValidationError(msg)

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable research record."""
        return {
            "conclusion": self.conclusion,
            "assessments": [assessment.as_json() for assessment in self.assessments],
            "external_artefacts": [
                reference_as_json(reference) for reference in self.external_artefacts
            ],
            "query": self.query,
        }


@dataclass(frozen=True)
class ExternalTermAssessment:
    """One external term assessed before deciding whether to invent locally."""

    term: str
    artefact: ArtefactReference
    suitable: bool
    rationale: str

    def __post_init__(self) -> None:
        """Require an inspectable external-term judgement."""
        _require_text(self.term, "External Ontology term")
        _require_text(self.rationale, "External Ontology term rationale")

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable external-term assessment."""
        return {
            "artefact": reference_as_json(self.artefact),
            "rationale": self.rationale,
            "suitable": self.suitable,
            "term": self.term,
        }


@dataclass(frozen=True)
class OntologyDecision:
    """One autonomous role's recorded decision about an Ontology revision."""

    stage: OntologyDecisionStage
    rationale: str

    def __post_init__(self) -> None:
        """Reject decisions that cannot explain their conclusion."""
        _require_text(self.rationale, "Ontology decision rationale")

    def as_json(self) -> dict[str, str]:
        """Return the durable decision record."""
        return {"rationale": self.rationale, "stage": self.stage.value}


class OntologyChangeKind(StrEnum):
    """The origin of a term introduced by an Ontology change."""

    LOCAL_INVENTION = "local_invention"
    EXTERNAL_REUSE = "external_reuse"


@dataclass(frozen=True)
class OntologyChange:
    """One minimal Ontology change included in a proposal."""

    term: str
    description: str
    kind: OntologyChangeKind
    external_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require an inspectable changed term and description."""
        _require_text(self.term, "Ontology change term")
        _require_text(self.description, "Ontology change description")
        if not self.external_terms:
            msg = "An Ontology change must record assessed external terms."
            raise OntologyEvolutionValidationError(msg)

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable change record."""
        return {
            "description": self.description,
            "external_terms": list(self.external_terms),
            "kind": self.kind.value,
            "term": self.term,
        }


@dataclass(frozen=True)
class ClaimReconsideration:
    """One Claim reconsidered after an Ontology proposal."""

    previous_mapping: ArtefactReference
    mapping: ClaimMapping
    reason: str

    def __post_init__(self) -> None:
        """Require the before, after and reason for reconsideration."""
        for value, name in ((self.reason, "Claim reconsideration reason"),):
            _require_text(value, name)

    def as_json(
        self, reconsidered_mapping: ArtefactReference | None
    ) -> dict[str, JsonValue]:
        """Return the durable Claim reconsideration record."""
        return {
            "claim_id": self.mapping.claim_id,
            "new_disposition": self.mapping.disposition.value,
            "previous_mapping": reference_as_json(self.previous_mapping),
            "reconsidered_mapping": reference_as_json(reconsidered_mapping),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OntologyProposal:
    """A proposed Ontology and SHACL release with its causal record."""

    research: ArtefactReference
    researcher: ArtefactReference
    proposer: ArtefactReference
    critic: ArtefactReference
    synthesiser: ArtefactReference
    ontology_turtle: str
    shacl_turtle: str
    changes: tuple[OntologyChange, ...]
    reconsiderations: tuple[ClaimReconsideration, ...]

    def __post_init__(self) -> None:
        """Require the proposed release and its autonomous decision history."""
        _require_text(self.ontology_turtle, "Ontology release RDF")
        _require_text(self.shacl_turtle, "SHACL release RDF")
        if not self.changes:
            msg = "An Ontology proposal must record at least one change."
            raise OntologyEvolutionValidationError(msg)
        if not self.reconsiderations:
            msg = "An Ontology proposal must reconsider at least one Claim."
            raise OntologyEvolutionValidationError(msg)


@dataclass(frozen=True)
class OntologyEvolutionOutcome:
    """The durable outcome from considering one Ontology proposal."""

    reference: ArtefactReference
    current_release: ArtefactReference | None
    activated_release: ArtefactReference | None
    accepted: bool


class OntologyEvolutionService:
    """Vendor artefacts and activate only validated immutable Ontology releases."""

    def __init__(self, store: ArtefactStore) -> None:
        """Create a service backed by the durable artefact authority."""
        self._store = store
        self._current_release = self._load_current_release()

    def vendor(self, artefact: ExternalOntologyArtefact) -> ArtefactReference:
        """Store an external Ontology artefact before it can influence a proposal."""
        return self._store.write_json("external-ontology", artefact.as_json())

    def inspect_vendored(self, reference: ArtefactReference) -> JsonValue:
        """Return one verified vendored external Ontology artefact."""
        if reference.kind != "external-ontology":
            msg = "Reference does not identify an external Ontology artefact."
            raise OntologyEvolutionValidationError(msg)
        return self._store.read_json(reference)

    def prompt_research(
        self,
        ontology_gap: ArtefactReference,
        research: OntologyResearch,
    ) -> ArtefactReference:
        """Record the research prompted by one persisted Ontology-gap Claim."""
        self._inspect_ontology_gap(ontology_gap)
        for reference in research.external_artefacts:
            self.inspect_vendored(reference)
        for assessment in research.assessments:
            self.inspect_vendored(assessment.artefact)
        record = research.as_json()
        record["ontology_gap"] = reference_as_json(ontology_gap)
        return self._store.write_json("ontology-research", record)

    def record_decision(self, decision: OntologyDecision) -> ArtefactReference:
        """Record a researcher, proposer, critic or synthesiser decision."""
        return self._store.write_json("ontology-decision", decision.as_json())

    def consider(
        self,
        proposal: OntologyProposal,
    ) -> OntologyEvolutionOutcome:
        """Validate a proposal and activate it only when every check succeeds."""
        shacl_release = self._store.write_json(
            "shacl-release", {"turtle": proposal.shacl_turtle}
        )
        ontology_release = self._store.write_json(
            "ontology-release",
            {
                "shacl_release": reference_as_json(shacl_release),
                "turtle": proposal.ontology_turtle,
            },
        )
        diagnostics = self._diagnostics(proposal)
        accepted = not diagnostics
        reconsiderations = self._record_reconsiderations(proposal) if accepted else []
        active_release = ontology_release if accepted else self._current_release
        record = self._store.write_json(
            "ontology-evolution",
            {
                "changes": [change.as_json() for change in proposal.changes],
                "critic": reference_as_json(proposal.critic),
                "current_release": reference_as_json(active_release),
                "diagnostics": diagnostics,
                "ontology_release": reference_as_json(ontology_release),
                "proposer": reference_as_json(proposal.proposer),
                "reconsiderations": [
                    reconsideration.as_json(recorded.reference)
                    for reconsideration, recorded in reconsiderations
                ],
                "research": reference_as_json(proposal.research),
                "researcher": reference_as_json(proposal.researcher),
                "shacl_release": reference_as_json(shacl_release),
                "status": "accepted" if accepted else "failed",
                "synthesiser": reference_as_json(proposal.synthesiser),
            },
        )
        if accepted:
            self._current_release = ontology_release
            self._write_current_release(ontology_release)
        return OntologyEvolutionOutcome(
            reference=record,
            current_release=active_release,
            activated_release=ontology_release if accepted else None,
            accepted=accepted,
        )

    def inspect(self, reference: ArtefactReference) -> JsonValue:
        """Return one verified Ontology evolution outcome."""
        if reference.kind != "ontology-evolution":
            msg = "Reference does not identify an Ontology evolution outcome."
            raise OntologyEvolutionValidationError(msg)
        return self._store.read_json(reference)

    def _diagnostics(self, proposal: OntologyProposal) -> list[str]:
        """Return every deterministic validation failure for a proposal."""
        diagnostics = self._reference_diagnostics(proposal)
        diagnostics.extend(self._research_diagnostics(proposal))
        diagnostics.extend(self._claim_diagnostics(proposal))
        if not _is_valid_turtle(proposal.ontology_turtle):
            diagnostics.append("Ontology RDF is invalid.")
        if not _is_valid_turtle(proposal.shacl_turtle):
            diagnostics.append("SHACL RDF is invalid.")
        elif not _has_node_shape(proposal.shacl_turtle):
            diagnostics.append("SHACL release has no NodeShape.")
        else:
            shacl_diagnostic = _reconsideration_shacl_diagnostic(proposal)
            if shacl_diagnostic is not None:
                diagnostics.append(shacl_diagnostic)
        claim_ids = [item.mapping.claim_id for item in proposal.reconsiderations]
        if len(claim_ids) != len(set(claim_ids)):
            diagnostics.append("Claim reconsideration IDs must be unique.")
        return diagnostics

    def _research_diagnostics(self, proposal: OntologyProposal) -> list[str]:
        """Check that local invention follows recorded external-term research."""
        try:
            research = self._store.read_json(proposal.research)
        except ArtefactIntegrityError:
            return []
        if not isinstance(research, dict):
            return ["Ontology research record has an invalid shape."]
        diagnostics = self._research_gap_diagnostics(proposal, research)
        assessments = research.get("assessments")
        if not isinstance(assessments, list):
            diagnostics.append(
                "Ontology research record has no external-term assessments."
            )
            return diagnostics
        assessments_by_term = {
            assessment.get("term"): assessment
            for assessment in assessments
            if isinstance(assessment, dict)
            and isinstance(assessment.get("term"), str)
            and isinstance(assessment.get("suitable"), bool)
        }
        for change in proposal.changes:
            missing_terms = [
                term
                for term in change.external_terms
                if term not in assessments_by_term
            ]
            if missing_terms:
                diagnostics.append(
                    f"Ontology change {change.term!r} lacks recorded external-term "
                    "research."
                )
            if change.kind is OntologyChangeKind.LOCAL_INVENTION and any(
                _assessment_is_suitable(assessments_by_term[term])
                for term in change.external_terms
                if term in assessments_by_term
            ):
                diagnostics.append(
                    f"Ontology change {change.term!r} invents locally despite a "
                    "suitable external term."
                )
            if change.kind is OntologyChangeKind.EXTERNAL_REUSE and not any(
                _assessment_is_suitable(assessments_by_term[term])
                for term in change.external_terms
                if term in assessments_by_term
            ):
                diagnostics.append(
                    f"Ontology change {change.term!r} does not identify a suitable "
                    "reused term."
                )
            if change.kind is OntologyChangeKind.EXTERNAL_REUSE and (
                change.term not in assessments_by_term
                or not _assessment_is_suitable(assessments_by_term[change.term])
            ):
                diagnostics.append(
                    f"Ontology change {change.term!r} must reuse an evidenced "
                    "external term."
                )
            for term in change.external_terms:
                assessment = assessments_by_term.get(term)
                if assessment is not None:
                    diagnostics.extend(self._assessment_diagnostics(assessment))
        return diagnostics

    def _research_gap_diagnostics(
        self,
        proposal: OntologyProposal,
        research: dict[str, JsonValue],
    ) -> list[str]:
        """Check that research and reconsideration share one Ontology-gap Claim."""
        try:
            ontology_gap = reference_from_json(research.get("ontology_gap"))
        except ArtefactIntegrityError:
            return ["Ontology research record has an invalid Ontology-gap Claim."]
        if ontology_gap is None:
            return ["Ontology research record has no Ontology-gap Claim."]
        try:
            self._inspect_ontology_gap(ontology_gap)
        except (ArtefactIntegrityError, OntologyEvolutionValidationError):
            return ["Ontology research record has an invalid Ontology-gap Claim."]
        if ontology_gap not in {
            reconsideration.previous_mapping
            for reconsideration in proposal.reconsiderations
        }:
            return [
                "The Ontology-gap Claim that prompted research must be reconsidered."
            ]
        return []

    def _inspect_ontology_gap(self, reference: ArtefactReference) -> ClaimMapping:
        """Return the Claim mapping that is validly marked as an Ontology gap."""
        if reference.kind != "claim-mapping":
            msg = "Research must be prompted by a Claim mapping."
            raise OntologyEvolutionValidationError(msg)
        mapping = claim_mapping_from_json(self._store.read_json(reference))
        if mapping.disposition is not ClaimDisposition.ONTOLOGY_GAP:
            msg = "Research must be prompted by an Ontology-gap Claim."
            raise OntologyEvolutionValidationError(msg)
        return mapping

    def _assessment_diagnostics(self, assessment: dict[str, JsonValue]) -> list[str]:
        """Check that one external-term assessment cites its vendored artefact."""
        reference = assessment.get("artefact")
        try:
            artefact = reference_from_json(reference)
        except ArtefactIntegrityError:
            return ["External-term assessment has an invalid artefact reference."]
        if artefact is None:
            return ["External-term assessment has no vendored artefact."]
        try:
            record = self.inspect_vendored(artefact)
        except (ArtefactIntegrityError, OntologyEvolutionValidationError):
            return ["External-term assessment has an invalid vendored artefact."]
        if not isinstance(record, dict) or not isinstance(record.get("turtle"), str):
            return ["Vendored external Ontology artefact has an invalid shape."]
        term = assessment.get("term")
        if not isinstance(term, str):
            return ["External-term assessment has an invalid term."]
        try:
            node = NamedNode(term)
            store = Store()
            store.load(input=record["turtle"], format=RdfFormat.TURTLE)
        except (SyntaxError, ValueError):
            return ["External-term assessment has an invalid term or artefact."]
        if not bool(
            store.query(
                f"""
                ASK {{
                    {{ <{node.value}> ?predicate ?object }}
                    UNION {{ ?subject <{node.value}> ?object }}
                    UNION {{ ?subject ?predicate <{node.value}> }}
                }}
                """
            )
        ):
            return ["External-term assessment is not evidenced by its artefact."]
        return []

    def _load_current_release(self) -> ArtefactReference | None:
        """Return the verified current release retained by this artefact store."""
        path = self._current_release_path()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as error:
            msg = f"Current Ontology release is not valid JSON: {path}."
            raise OntologyEvolutionValidationError(msg) from error
        try:
            reference = reference_from_json(value)
        except ArtefactIntegrityError as error:
            msg = f"Current Ontology release is invalid: {path}."
            raise OntologyEvolutionValidationError(msg) from error
        if reference is None or reference.kind != "ontology-release":
            msg = f"Current Ontology release is invalid: {path}."
            raise OntologyEvolutionValidationError(msg)
        try:
            self._store.read_json(reference)
        except ArtefactIntegrityError as error:
            msg = f"Current Ontology release is invalid: {path}."
            raise OntologyEvolutionValidationError(msg) from error
        return reference

    def _write_current_release(self, reference: ArtefactReference) -> None:
        """Persist the validated active release for later service instances."""
        path = self._current_release_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(reference_as_json(reference), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _current_release_path(self) -> Path:
        """Return the durable location of the active Ontology release reference."""
        return self._store.root / "ontology-evolution" / "current-release.json"

    def _claim_diagnostics(self, proposal: OntologyProposal) -> list[str]:
        """Check every affected Claim against its durable prior mapping."""
        diagnostics: list[str] = []
        for reconsideration in proposal.reconsiderations:
            if reconsideration.previous_mapping.kind != "claim-mapping":
                diagnostics.append(
                    "Claim reconsideration must reference a Claim mapping."
                )
                continue
            try:
                previous = claim_mapping_from_json(
                    self._store.read_json(reconsideration.previous_mapping)
                )
            except (ArtefactIntegrityError, ValueError):
                diagnostics.append(
                    f"Claim reconsideration for {reconsideration.mapping.claim_id!r} "
                    "has an invalid prior mapping."
                )
                continue
            if previous.claim_id != reconsideration.mapping.claim_id:
                diagnostics.append("Claim reconsideration must preserve the Claim ID.")
            elif previous.candidate != reconsideration.mapping.candidate:
                diagnostics.append(
                    "Claim reconsideration must preserve the candidate Claim."
                )
        return diagnostics

    def _record_reconsiderations(
        self, proposal: OntologyProposal
    ) -> list[tuple[ClaimReconsideration, RecordedClaimMapping]]:
        """Persist the validated new mapping for every affected Claim."""
        service = ClaimMappingService(self._store)
        return [
            (reconsideration, service.record(reconsideration.mapping))
            for reconsideration in proposal.reconsiderations
        ]

    def _reference_diagnostics(self, proposal: OntologyProposal) -> list[str]:
        """Return diagnostics for missing or misclassified causal artefacts."""
        expected_references = (
            (proposal.research, "ontology-research", None),
            (
                proposal.researcher,
                "ontology-decision",
                OntologyDecisionStage.RESEARCHER,
            ),
            (proposal.proposer, "ontology-decision", OntologyDecisionStage.PROPOSER),
            (proposal.critic, "ontology-decision", OntologyDecisionStage.CRITIC),
            (
                proposal.synthesiser,
                "ontology-decision",
                OntologyDecisionStage.SYNTHESISER,
            ),
        )
        diagnostics: list[str] = []
        for reference, expected_kind, expected_stage in expected_references:
            if reference.kind != expected_kind:
                diagnostics.append(f"Expected {expected_kind} causal artefact.")
                continue
            try:
                record = self._store.read_json(reference)
            except ArtefactIntegrityError:
                diagnostics.append(f"Causal artefact {reference.identity} is invalid.")
                continue
            if expected_stage is not None and (
                not isinstance(record, dict)
                or record.get("stage") != expected_stage.value
            ):
                diagnostics.append(
                    f"Causal artefact {reference.identity} is not a "
                    f"{expected_stage.value} decision."
                )
        return diagnostics


def _is_valid_turtle(turtle: str) -> bool:
    """Return whether text is valid Turtle RDF."""
    try:
        Store().load(input=turtle, format=RdfFormat.TURTLE)
    except (SyntaxError, ValueError):
        return False
    return True


def _has_node_shape(turtle: str) -> bool:
    """Return whether a SHACL release declares at least one NodeShape."""
    store = Store()
    store.load(input=turtle, format=RdfFormat.TURTLE)
    return bool(
        store.query(
            """
            ASK {
                ?shape a <http://www.w3.org/ns/shacl#NodeShape> .
            }
            """
        )
    )


def _reconsideration_shacl_diagnostic(proposal: OntologyProposal) -> str | None:
    """Return the SHACL diagnostic for reconsidered Claim assertions, if any."""
    store = Store()
    store.extend(
        Quad(quad.subject, quad.predicate, quad.object)
        for quad in project_claim_mappings(
            tuple(item.mapping for item in proposal.reconsiderations)
        )
        if quad.graph_name == ACCEPTED_KNOWLEDGE_GRAPH
    )
    data = BytesIO()
    store.dump(output=data, format=RdfFormat.TRIG)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="Dataset.default_context is deprecated",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="ConjunctiveGraph is deprecated",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="Dataset.identifier is deprecated",
            )
            conforms, _, _ = validate_shacl(
                data_graph=data.getvalue()
                or b"@prefix aigg: <https://w3id.org/aigg/> .\n",
                data_graph_format="trig",
                meta_shacl=True,
                shacl_graph=proposal.shacl_turtle,
                shacl_graph_format="turtle",
            )
    except (ReportableRuntimeError, ValueError):
        return "SHACL release is invalid."
    if not conforms:
        return "Reconsidered Claim mappings violate the SHACL release."
    return None


def _assessment_is_suitable(assessment: dict[str, JsonValue]) -> bool:
    """Return whether a recorded external-term assessment found a suitable term."""
    return assessment.get("suitable") is True


def _require_text(value: str, name: str) -> None:
    """Reject blank durable values before they become artefacts."""
    if not value.strip():
        msg = f"{name} must not be blank."
        raise OntologyEvolutionValidationError(msg)
