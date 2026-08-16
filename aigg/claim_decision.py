"""Bounded Claim decisions and deterministic candidate-change assessment."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyshacl import validate as validate_shacl
from rdflib import Graph, URIRef
from rdflib import Literal as RdfLiteral

from aigg.artefacts import (
    ArtefactIntegrityError,
    ArtefactReference,
    ArtefactStore,
    JsonValue,
    reference_as_json,
    reference_from_json,
)
from aigg.canonical import EvidenceAnchor
from aigg.claim_mapping import (
    ClaimDisposition,
    ClaimMapping,
    ClaimMappingService,
    SemanticAssertion,
    StageReason,
    candidate_claim_from_json,
    claim_mapping_from_json,
)
from aigg.open_extraction import CandidateClaim
from aigg.reasoning import ModelConfiguration, ReasoningRunner, StructuredModel
from aigg.temporal_resolution import ResolvedTemporalExpression, TemporalConstraint

CLAIM_DECISION_STAGE = "claim-decision"
CLAIM_DECISION_INSTRUCTIONS = """Map the supplied candidate Claim using only its context.
Return one JSON object with exactly these fields: acceptance, conflict,
disposition, mapping, scope, semantic_assertions and validation. Each reason
field is non-empty text. disposition is one of accepted, ontology_gap,
constraint_violation, conflict, unresolved_entity, unresolved_time,
out_of_scope, low_confidence, rejected or superseded. semantic_assertions is a
list of objects with subject, predicate, object and object_kind; object_kind is
iri or literal. Include semantic_assertions only when disposition is accepted.
Do not add Evidence, source material, fields or assertions outside the supplied
context."""


class ClaimDecisionValidationError(ValueError):
    """Raised when a Claim decision cannot retain its bounded contract."""


class ClaimDecisionOperationalError(RuntimeError):
    """Raised when deterministic decision infrastructure cannot operate."""


@dataclass(frozen=True)
class ClaimDecisionContext:
    """The complete bounded context permitted for one candidate Claim decision."""

    claim_id: str
    candidate: CandidateClaim
    entity_decisions: tuple[ArtefactReference, ...]
    temporal_decisions: tuple[ArtefactReference, ...]
    ontology_turtle: str
    shacl_turtle: str
    accepted_mappings: tuple[ClaimMapping, ...] = ()
    maximum_accepted_context: int = 16

    def __post_init__(self) -> None:
        """Reject unbounded context before it can reach a reasoning model."""
        _require_text(self.claim_id, "Claim ID")
        if not self.ontology_turtle.strip():
            msg = "Active Ontology RDF must be non-empty."
            raise ClaimDecisionValidationError(msg)
        if not self.shacl_turtle.strip():
            msg = "Active SHACL RDF must be non-empty."
            raise ClaimDecisionValidationError(msg)
        if self.maximum_accepted_context < 0:
            msg = "Accepted knowledge context bound cannot be negative."
            raise ClaimDecisionValidationError(msg)
        if len(self.accepted_mappings) > self.maximum_accepted_context:
            msg = "Accepted knowledge context exceeds its stated maximum."
            raise ClaimDecisionValidationError(msg)
        if any(
            mapping.disposition is not ClaimDisposition.ACCEPTED
            for mapping in self.accepted_mappings
        ):
            msg = (
                "Accepted knowledge context must contain only accepted Claim mappings."
            )
            raise ClaimDecisionValidationError(msg)
        if any(item.kind != "entity-resolution" for item in self.entity_decisions):
            msg = "Entity decisions must reference completed Entity resolutions."
            raise ClaimDecisionValidationError(msg)
        if any(item.kind != "temporal-resolution" for item in self.temporal_decisions):
            msg = "Temporal decisions must reference completed temporal resolutions."
            raise ClaimDecisionValidationError(msg)

    def as_json(self) -> dict[str, JsonValue]:
        """Return exactly the bounded model input and durable request record."""
        return {
            "accepted_mappings": [
                mapping.as_json() for mapping in self.accepted_mappings
            ],
            "candidate": self.candidate.as_json(),
            "claim_id": self.claim_id,
            "entity_decisions": [reference_as_json(item) for item in self.entity_decisions],
            "maximum_accepted_context": self.maximum_accepted_context,
            "ontology_turtle": self.ontology_turtle,
            "shacl_turtle": self.shacl_turtle,
            "temporal_decisions": [reference_as_json(item) for item in self.temporal_decisions],
        }


@dataclass(frozen=True)
class ClaimChangeAssessment:
    """The retained deterministic checks for one proposed accepted change."""

    ontology_diagnostics: tuple[str, ...]
    constraint_diagnostics: tuple[str, ...]
    conflict_diagnostics: tuple[ConflictDiagnostic, ...]

    @property
    def accepted(self) -> bool:
        """Return whether the proposal satisfies every deterministic policy."""
        return (
            not self.ontology_diagnostics
            and not self.constraint_diagnostics
            and not self.conflict_diagnostics
        )


@dataclass(frozen=True)
class ConflictDiagnostic:
    """The conflicting accepted Claim and retained Evidence that support it."""

    claim_id: str
    assertion: SemanticAssertion
    evidence: tuple[EvidenceAnchor, ...]

    @property
    def reason(self) -> str:
        """Describe the incompatible assertion without discarding its evidence."""
        return (
            f"Conflicts with accepted Claim {self.claim_id!r} on "
            f"{self.assertion.subject!r} and {self.assertion.predicate!r}."
        )

    def as_json(self) -> dict[str, JsonValue]:
        """Return durable conflict evidence and the incompatible assertion."""
        return {
            "assertion": self.assertion.as_json(),
            "claim_id": self.claim_id,
            "evidence": [item.as_json() for item in self.evidence],
            "reason": self.reason,
        }


class ClaimChangeValidator:
    """Apply active SHACL constraints and conflict policy to proposed assertions."""

    def assess(
        self, context: ClaimDecisionContext, mapping: ClaimMapping
    ) -> ClaimChangeAssessment:
        """Return distinct SHACL and epistemic-conflict diagnostics for a mapping."""
        if mapping.disposition is not ClaimDisposition.ACCEPTED:
            return ClaimChangeAssessment((), (), ())
        return ClaimChangeAssessment(
            self._ontology_diagnostics(context, mapping),
            self._constraint_diagnostics(context, mapping),
            self._conflict_diagnostics(context, mapping),
        )

    def _ontology_diagnostics(
        self, context: ClaimDecisionContext, mapping: ClaimMapping
    ) -> tuple[str, ...]:
        """Require every proposed predicate to occur in the active Ontology."""
        ontology = Graph()
        try:
            ontology.parse(data=context.ontology_turtle, format="turtle")
        except Exception as error:
            msg = "Active Ontology RDF could not be evaluated."
            raise ClaimDecisionOperationalError(msg) from error
        terms = set(ontology.all_nodes())
        return tuple(
            f"Active Ontology does not represent predicate {assertion.predicate!r}."
            for assertion in mapping.semantic_assertions
            if URIRef(assertion.predicate) not in terms
        )

    def _constraint_diagnostics(
        self, context: ClaimDecisionContext, mapping: ClaimMapping
    ) -> tuple[str, ...]:
        """Validate the accepted graph plus the candidate assertions through SHACL."""
        data = Graph()
        for existing in context.accepted_mappings:
            _add_assertions(data, existing.semantic_assertions)
        _add_assertions(data, mapping.semantic_assertions)
        shapes = Graph()
        try:
            shapes.parse(data=context.shacl_turtle, format="turtle")
            conforms, _, report = validate_shacl(
                data_graph=data,
                shacl_graph=shapes,
                inference="none",
                advanced=True,
            )
        except Exception as error:
            msg = "Active SHACL release could not be evaluated."
            raise ClaimDecisionOperationalError(msg) from error
        if conforms:
            return ()
        return (str(report).strip(),)

    def _conflict_diagnostics(
        self, context: ClaimDecisionContext, mapping: ClaimMapping
    ) -> tuple[ConflictDiagnostic, ...]:
        """Find incompatible assertions whose Claim times overlap."""
        diagnostics: list[ConflictDiagnostic] = []
        for existing in context.accepted_mappings:
            if not _claim_times_overlap(existing, mapping):
                continue
            for proposed in mapping.semantic_assertions:
                for accepted in existing.semantic_assertions:
                    if _assertions_conflict(proposed, accepted):
                        diagnostics.append(
                            ConflictDiagnostic(
                                existing.claim_id,
                                accepted,
                                existing.candidate.evidence,
                            )
                        )
        return tuple(diagnostics)


@dataclass(frozen=True)
class RecordedClaimDecision:
    """One complete Claim decision and its durable references."""

    mapping: ClaimMapping
    mapping_reference: ArtefactReference
    reasoning_invocation: ArtefactReference
    assessment: ClaimChangeAssessment
    reference: ArtefactReference


class ClaimDecisionService:
    """Record and replay bounded Claim-mapping decisions."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        *,
        maximum_attempts: int,
        validator: ClaimChangeValidator | None = None,
    ) -> None:
        """Create a service whose artefacts remain the decision authority."""
        self._store = store
        self._runner = ReasoningRunner(
            store, model, configuration, maximum_attempts=maximum_attempts
        )
        self._mappings = ClaimMappingService(store)
        self._validator = validator or ClaimChangeValidator()

    def create_request(self, context: ClaimDecisionContext) -> ArtefactReference:
        """Store one Claim-decision request with no ambient graph access."""
        self._decision_records(context)
        return self._store.write_json("claim-decision-request", context.as_json())

    def decide_request(
        self,
        request: ArtefactReference,
        *,
        replay: ArtefactReference | None = None,
    ) -> RecordedClaimDecision:
        """Decide a durable request or read its exact previous decision."""
        context = _context_from_json(self._store.read_json(request))
        if replay is not None:
            return self._replay(request, replay)
        invocation = self._runner.run(
            stage=CLAIM_DECISION_STAGE,
            structured_input=self._structured_input(context),
            validate_output=lambda value: _validated_mapping_output(value, context),
        )
        mapping = _mapping_output_from_json(invocation.output, context)
        assessment = self._validator.assess(context, mapping)
        mapping = _apply_assessment(mapping, assessment)
        recorded_mapping = self._mappings.record(mapping)
        reference = self._store.write_json(
            "claim-decision",
            {
                "assessment": {
                    "conflict_diagnostics": [
                        diagnostic.as_json()
                        for diagnostic in assessment.conflict_diagnostics
                    ],
                    "constraint_diagnostics": list(assessment.constraint_diagnostics),
                    "ontology_diagnostics": list(assessment.ontology_diagnostics),
                },
                "mapping": reference_as_json(recorded_mapping.reference),
                "reasoning_invocation": reference_as_json(invocation.reference),
                "request": reference_as_json(request),
            },
        )
        return RecordedClaimDecision(
            mapping,
            recorded_mapping.reference,
            invocation.reference,
            assessment,
            reference,
        )

    def _replay(
        self, request: ArtefactReference, replay: ArtefactReference
    ) -> RecordedClaimDecision:
        """Read a prior Claim decision without invoking a configured model."""
        value = self._store.read_json(replay)
        if not isinstance(value, dict) or set(value) != {
            "assessment",
            "mapping",
            "reasoning_invocation",
            "request",
        }:
            msg = "Claim decision artefact has an invalid shape."
            raise ClaimDecisionValidationError(msg)
        if _required_reference(value["request"], "Claim decision request") != request:
            msg = "Claim decision replay belongs to a different request."
            raise ClaimDecisionValidationError(msg)
        mapping_reference = _required_reference(value["mapping"], "Claim mapping")
        invocation = _required_reference(
            value["reasoning_invocation"], "Reasoning invocation"
        )
        assessment = _assessment_from_json(value["assessment"])
        return RecordedClaimDecision(
            self._mappings.inspect(mapping_reference),
            mapping_reference,
            invocation,
            assessment,
            replay,
        )

    def _structured_input(self, context: ClaimDecisionContext) -> dict[str, JsonValue]:
        """Pass only verified completed decisions, rather than caller JSON, to the model."""
        structured_input = context.as_json()
        entity, temporal = self._decision_records(context)
        structured_input["entity_decisions"] = entity
        structured_input["temporal_decisions"] = temporal
        structured_input["instructions"] = CLAIM_DECISION_INSTRUCTIONS
        return structured_input

    def _decision_records(
        self, context: ClaimDecisionContext
    ) -> tuple[list[JsonValue], list[JsonValue]]:
        """Read and verify the durable result shapes required by Claim mapping."""
        entity = [self._store.read_json(item) for item in context.entity_decisions]
        temporal = [self._store.read_json(item) for item in context.temporal_decisions]
        if any(
            not isinstance(item, dict)
            or set(item) != {"history", "reasoning_invocation", "request"}
            for item in entity
        ):
            msg = "Entity decision is not a completed Entity-resolution result."
            raise ClaimDecisionValidationError(msg)
        if any(
            not isinstance(item, dict)
            or set(item) != {"outcome", "reasoning_invocation", "request"}
            for item in temporal
        ):
            msg = "Temporal decision is not a completed temporal-resolution result."
            raise ClaimDecisionValidationError(msg)
        return entity, temporal


def _mapping_output_from_json(
    value: JsonValue, context: ClaimDecisionContext
) -> ClaimMapping:
    """Validate one model result while keeping its Claim and evidence immutable."""
    if not isinstance(value, dict) or set(value) != {
        "acceptance",
        "conflict",
        "disposition",
        "mapping",
        "scope",
        "semantic_assertions",
        "validation",
    }:
        msg = "Claim decision output has an invalid shape."
        raise ClaimDecisionValidationError(msg)
    try:
        disposition = ClaimDisposition(_text(value["disposition"], "Claim disposition"))
    except ValueError as error:
        msg = "Claim decision output has an unrecognised disposition."
        raise ClaimDecisionValidationError(msg) from error
    assertions_value = value["semantic_assertions"]
    if not isinstance(assertions_value, list):
        msg = "Claim decision semantic assertions must be a list."
        raise ClaimDecisionValidationError(msg)
    try:
        assertions = tuple(_assertion_from_json(item) for item in assertions_value)
        return ClaimMapping(
            claim_id=context.claim_id,
            candidate=context.candidate,
            disposition=disposition,
            mapping=StageReason(_text(value["mapping"], "Claim mapping reason")),
            validation=StageReason(
                _text(value["validation"], "Claim validation reason")
            ),
            scope=StageReason(_text(value["scope"], "Claim scope reason")),
            acceptance=StageReason(
                _text(value["acceptance"], "Claim acceptance reason")
            ),
            conflict=StageReason(_text(value["conflict"], "Claim conflict reason")),
            semantic_assertions=assertions,
        )
    except ValueError as error:
        raise ClaimDecisionValidationError(str(error)) from error


def _validated_mapping_output(
    value: JsonValue, context: ClaimDecisionContext
) -> JsonValue:
    """Validate model output without replacing the durable JSON invocation output."""
    _mapping_output_from_json(value, context)
    return value


def _apply_assessment(
    mapping: ClaimMapping, assessment: ClaimChangeAssessment
) -> ClaimMapping:
    """Turn deterministic diagnostics into ordinary recorded Claim dispositions."""
    if assessment.ontology_diagnostics:
        return replace(
            mapping,
            disposition=ClaimDisposition.ONTOLOGY_GAP,
            mapping=StageReason(" ".join(assessment.ontology_diagnostics)),
            acceptance=StageReason("The Claim awaits Ontology review."),
            semantic_assertions=(),
        )
    if assessment.constraint_diagnostics:
        return replace(
            mapping,
            disposition=ClaimDisposition.CONSTRAINT_VIOLATION,
            validation=StageReason(" ".join(assessment.constraint_diagnostics)),
            conflict=StageReason(
                " ".join(item.reason for item in assessment.conflict_diagnostics)
                or "No conflict was detected against accepted knowledge."
            ),
            acceptance=StageReason("The Claim cannot enter accepted knowledge."),
            semantic_assertions=(),
        )
    if assessment.conflict_diagnostics:
        return replace(
            mapping,
            disposition=ClaimDisposition.CONFLICT,
            conflict=StageReason(
                " ".join(item.reason for item in assessment.conflict_diagnostics)
            ),
            acceptance=StageReason("The Claim conflicts with accepted knowledge."),
            semantic_assertions=(),
        )
    return mapping


def _context_from_json(value: JsonValue) -> ClaimDecisionContext:
    """Parse one stored Claim-decision request before it reaches the model."""
    if not isinstance(value, dict) or set(value) != {
        "accepted_mappings",
        "candidate",
        "claim_id",
        "entity_decisions",
        "maximum_accepted_context",
        "ontology_turtle",
        "shacl_turtle",
        "temporal_decisions",
    }:
        msg = "Claim decision request has an invalid shape."
        raise ClaimDecisionValidationError(msg)
    mappings = value["accepted_mappings"]
    if not isinstance(mappings, list):
        msg = "Claim decision accepted knowledge must be a list."
        raise ClaimDecisionValidationError(msg)
    return ClaimDecisionContext(
        claim_id=_text(value["claim_id"], "Claim ID"),
        candidate=_candidate_from_json(value["candidate"]),
        entity_decisions=_references_from_json(value["entity_decisions"], "entity-resolution"),
        temporal_decisions=_references_from_json(value["temporal_decisions"], "temporal-resolution"),
        ontology_turtle=_text(value["ontology_turtle"], "Active Ontology RDF"),
        shacl_turtle=_text(value["shacl_turtle"], "Active SHACL RDF"),
        accepted_mappings=tuple(claim_mapping_from_json(item) for item in mappings),
        maximum_accepted_context=_integer(
            value["maximum_accepted_context"], "Accepted knowledge context bound"
        ),
    )


def _candidate_from_json(value: JsonValue) -> CandidateClaim:
    """Parse a source-supported candidate without fabricating a Claim mapping."""
    return candidate_claim_from_json(value)


def _assessment_from_json(value: JsonValue) -> ClaimChangeAssessment:
    """Parse stored deterministic diagnostics for exact replay."""
    if not isinstance(value, dict) or set(value) != {
        "conflict_diagnostics",
        "constraint_diagnostics",
        "ontology_diagnostics",
    }:
        msg = "Claim change assessment has an invalid shape."
        raise ClaimDecisionValidationError(msg)
    return ClaimChangeAssessment(
        _diagnostics_from_json(value["ontology_diagnostics"], "Ontology"),
        _diagnostics_from_json(value["constraint_diagnostics"], "Constraint"),
        _conflict_diagnostics_from_json(value["conflict_diagnostics"]),
    )


def _diagnostics_from_json(value: JsonValue, name: str) -> tuple[str, ...]:
    """Require a serialised diagnostics list with meaningful messages."""
    if not isinstance(value, list):
        msg = f"{name} diagnostics must be a list."
        raise ClaimDecisionValidationError(msg)
    return tuple(_text(item, f"{name} diagnostic") for item in value)


def _conflict_diagnostics_from_json(value: JsonValue) -> tuple[ConflictDiagnostic, ...]:
    """Parse conflict details with the accepted Claim's supporting Evidence."""
    if not isinstance(value, list):
        msg = "Conflict diagnostics must be a list."
        raise ClaimDecisionValidationError(msg)
    diagnostics: list[ConflictDiagnostic] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "assertion",
            "claim_id",
            "evidence",
            "reason",
        }:
            msg = "Conflict diagnostic has an invalid shape."
            raise ClaimDecisionValidationError(msg)
        evidence = item["evidence"]
        if not isinstance(evidence, list):
            msg = "Conflict diagnostic Evidence must be a list."
            raise ClaimDecisionValidationError(msg)
        diagnostic = ConflictDiagnostic(
            _text(item["claim_id"], "Conflicting Claim ID"),
            _assertion_from_json(item["assertion"]),
            tuple(EvidenceAnchor.from_json(anchor) for anchor in evidence),
        )
        if item["reason"] != diagnostic.reason:
            msg = "Conflict diagnostic reason does not match its evidence."
            raise ClaimDecisionValidationError(msg)
        diagnostics.append(diagnostic)
    return tuple(diagnostics)


def _assertion_from_json(value: JsonValue) -> SemanticAssertion:
    """Parse one proposed semantic assertion from structured model output."""
    if not isinstance(value, dict) or set(value) != {
        "object",
        "object_kind",
        "predicate",
        "subject",
    }:
        msg = "Semantic assertion has an invalid shape."
        raise ClaimDecisionValidationError(msg)
    object_kind = _text(value["object_kind"], "Semantic assertion object kind")
    if object_kind not in {"iri", "literal"}:
        msg = "Semantic assertion object kind must be iri or literal."
        raise ClaimDecisionValidationError(msg)
    fields = (
        _text(value["subject"], "Semantic assertion subject"),
        _text(value["predicate"], "Semantic assertion predicate"),
        _text(value["object"], "Semantic assertion object"),
    )
    if object_kind == "iri":
        return SemanticAssertion(*fields, "iri")
    return SemanticAssertion(*fields, "literal")


def _add_assertions(graph: Graph, assertions: tuple[SemanticAssertion, ...]) -> None:
    """Add the canonical form of semantic assertions to an RDFLib graph."""
    for assertion in assertions:
        object_ = (
            URIRef(assertion.object)
            if assertion.object_kind == "iri"
            else RdfLiteral(assertion.object)
        )
        graph.add((URIRef(assertion.subject), URIRef(assertion.predicate), object_))


def _assertions_conflict(
    proposed: SemanticAssertion, accepted: SemanticAssertion
) -> bool:
    """Apply the explicit initial conflict policy to two semantic assertions."""
    return (
        proposed.subject == accepted.subject
        and proposed.predicate == accepted.predicate
        and (proposed.object, proposed.object_kind)
        != (accepted.object, accepted.object_kind)
    )


def _claim_times_overlap(first: ClaimMapping, second: ClaimMapping) -> bool:
    """Return whether Claim applicability or event intervals could overlap."""
    first_constraint = _claim_constraint(first)
    second_constraint = _claim_constraint(second)
    if first_constraint is None or second_constraint is None:
        return True
    return _constraints_overlap(first_constraint, second_constraint)


def _claim_constraint(mapping: ClaimMapping) -> TemporalConstraint | None:
    """Select the strongest resolved Claim time for the initial conflict policy."""
    for value in (mapping.candidate.times.applicability, mapping.candidate.times.event):
        if isinstance(value, ResolvedTemporalExpression):
            return value.constraint
    return None


def _constraints_overlap(first: TemporalConstraint, second: TemporalConstraint) -> bool:
    """Compare two half-open-or-closed temporal intervals without normalising them."""
    if first.upper_bound is not None and second.lower_bound is not None:
        if first.upper_bound < second.lower_bound:
            return False
        if first.upper_bound == second.lower_bound and not (
            first.upper_inclusive and second.lower_inclusive
        ):
            return False
    if second.upper_bound is not None and first.lower_bound is not None:
        if second.upper_bound < first.lower_bound:
            return False
        if second.upper_bound == first.lower_bound and not (
            second.upper_inclusive and first.lower_inclusive
        ):
            return False
    return True


def _references_from_json(
    value: JsonValue, kind: str
) -> tuple[ArtefactReference, ...]:
    """Parse a bounded collection of completed decision-result references."""
    if not isinstance(value, list):
        msg = "Completed decisions must be a list of durable references."
        raise ClaimDecisionValidationError(msg)
    references = tuple(_required_reference(item, "Completed decision") for item in value)
    if any(item.kind != kind for item in references):
        msg = "Completed decision has the wrong durable result kind."
        raise ClaimDecisionValidationError(msg)
    return references


def _required_reference(value: JsonValue, name: str) -> ArtefactReference:
    """Require one durable reference in a replay record."""
    try:
        reference = reference_from_json(value)
    except ArtefactIntegrityError as error:
        raise ClaimDecisionValidationError(f"{name} reference is invalid.") from error
    if reference is None:
        msg = f"{name} reference is required."
        raise ClaimDecisionValidationError(msg)
    return reference


def _integer(value: JsonValue, name: str) -> int:
    """Require a non-boolean integer from durable JSON."""
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} must be an integer."
        raise ClaimDecisionValidationError(msg)
    return value


def _text(value: JsonValue, name: str) -> str:
    """Require non-empty durable text."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be non-empty text."
        raise ClaimDecisionValidationError(msg)
    return value


def _require_text(value: str, name: str) -> None:
    """Reject empty dataclass text before it enters an artefact."""
    if not value.strip():
        msg = f"{name} must be non-empty text."
        raise ClaimDecisionValidationError(msg)
