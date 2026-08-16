"""Record candidate Claim dispositions and project accepted knowledge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from urllib.parse import quote

from pyoxigraph import Literal as RdfLiteral
from pyoxigraph import NamedNode, Quad

from aigg.artefacts import ArtefactReference, ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor
from aigg.open_extraction import CandidateClaim
from aigg.temporal_resolution import (
    ExtractedTemporalExpression,
    claim_times_from_json,
)

AIGG = "https://w3id.org/aigg/"
CLAIMS_GRAPH = NamedNode(f"{AIGG}graph/claims")
ACCEPTED_KNOWLEDGE_GRAPH = NamedNode(f"{AIGG}graph/accepted-knowledge")
RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDF_SUBJECT = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")
RDF_PREDICATE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")
RDF_OBJECT = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#object")
RDF_STATEMENT = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")
RDF_JSON = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON")
CLAIM = NamedNode(f"{AIGG}Claim")
CLAIM_ASSERTION = NamedNode(f"{AIGG}claimAssertion")
CLAIM_CONFIDENCE = NamedNode(f"{AIGG}claimConfidence")
CLAIM_RATIONALE = NamedNode(f"{AIGG}claimRationale")
CLAIM_TIME = NamedNode(f"{AIGG}claimTime")
DISPOSITION = NamedNode(f"{AIGG}disposition")
EVIDENCE = NamedNode(f"{AIGG}evidence")
MAPPING_REASON = NamedNode(f"{AIGG}mappingReason")
VALIDATION_REASON = NamedNode(f"{AIGG}validationReason")
SCOPE_REASON = NamedNode(f"{AIGG}scopeReason")
ACCEPTANCE_REASON = NamedNode(f"{AIGG}acceptanceReason")
PROJECTS_ASSERTION = NamedNode(f"{AIGG}projectsAssertion")


class ClaimMappingValidationError(ValueError):
    """Raised when a Claim mapping cannot preserve its decision history."""


class ClaimDisposition(StrEnum):
    """The primary outcome of mapping one candidate Claim."""

    ACCEPTED = "accepted"
    ONTOLOGY_GAP = "ontology_gap"
    CONSTRAINT_VIOLATION = "constraint_violation"
    CONFLICT = "conflict"
    UNRESOLVED_ENTITY = "unresolved_entity"
    UNRESOLVED_TIME = "unresolved_time"
    OUT_OF_SCOPE = "out_of_scope"
    LOW_CONFIDENCE = "low_confidence"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class StageReason:
    """The retained reason from one stage of a Claim's acceptance decision."""

    reason: str

    def __post_init__(self) -> None:
        """Reject empty stage reasoning before it becomes an artefact."""
        _require_text(self.reason, "Claim stage reason")


@dataclass(frozen=True)
class SemanticAssertion:
    """One RDF assertion that can appear in accepted semantic knowledge."""

    subject: str
    predicate: str
    object: str
    object_kind: Literal["iri", "literal"] = "literal"

    def __post_init__(self) -> None:
        """Require valid RDF names while retaining a simple durable contract."""
        _require_text(self.subject, "Semantic assertion subject")
        _require_text(self.predicate, "Semantic assertion predicate")
        _require_text(self.object, "Semantic assertion object")
        if self.object_kind not in ("iri", "literal"):
            msg = "Semantic assertion object kind must be iri or literal."
            raise ClaimMappingValidationError(msg)
        try:
            NamedNode(self.subject)
            NamedNode(self.predicate)
            if self.object_kind == "iri":
                NamedNode(self.object)
        except ValueError as error:
            msg = "Semantic assertion contains an invalid IRI."
            raise ClaimMappingValidationError(msg) from error

    def as_json(self) -> dict[str, str]:
        """Return the durable assertion representation."""
        return {
            "object": self.object,
            "object_kind": self.object_kind,
            "predicate": self.predicate,
            "subject": self.subject,
        }


@dataclass(frozen=True)
class ClaimMapping:
    """One candidate Claim's mapping, validation, scope and acceptance record."""

    claim_id: str
    candidate: CandidateClaim
    disposition: ClaimDisposition
    mapping: StageReason
    validation: StageReason
    scope: StageReason
    acceptance: StageReason
    semantic_assertions: tuple[SemanticAssertion, ...] = ()

    def __post_init__(self) -> None:
        """Keep acceptance and the canonical semantic projection consistent."""
        _require_text(self.claim_id, "Claim ID")
        if not self.candidate.evidence:
            msg = "A mapped Claim must retain Evidence."
            raise ClaimMappingValidationError(msg)
        if (
            self.disposition is ClaimDisposition.ACCEPTED
            and not self.semantic_assertions
        ):
            msg = "An accepted Claim must have a semantic assertion."
            raise ClaimMappingValidationError(msg)
        if (
            self.disposition is not ClaimDisposition.ACCEPTED
            and self.semantic_assertions
        ):
            msg = "Only an accepted Claim can have semantic assertions."
            raise ClaimMappingValidationError(msg)

    def as_json(self) -> dict[str, JsonValue]:
        """Return the complete durable Claim decision record."""
        return {
            "acceptance": self.acceptance.reason,
            "candidate": self.candidate.as_json(),
            "claim_id": self.claim_id,
            "disposition": self.disposition.value,
            "mapping": self.mapping.reason,
            "scope": self.scope.reason,
            "semantic_assertions": [
                assertion.as_json() for assertion in self.semantic_assertions
            ],
            "validation": self.validation.reason,
        }


@dataclass(frozen=True)
class RecordedClaimMapping:
    """One Claim mapping and its durable artefact reference."""

    mapping: ClaimMapping
    reference: ArtefactReference


class ClaimMappingService:
    """Record and inspect durable candidate Claim mapping decisions."""

    def __init__(self, store: ArtefactStore) -> None:
        """Create a service whose artefacts are the decision authority."""
        self._store = store

    def record(self, mapping: ClaimMapping) -> RecordedClaimMapping:
        """Persist one complete Claim mapping without altering its disposition."""
        reference = self._store.write_json("claim-mapping", mapping.as_json())
        return RecordedClaimMapping(mapping, reference)

    def inspect(self, reference: ArtefactReference) -> ClaimMapping:
        """Return one verified Claim mapping from its durable artefact."""
        return claim_mapping_from_json(self._store.read_json(reference))


def claim_mapping_from_json(value: JsonValue) -> ClaimMapping:
    """Parse a durable Claim mapping before it enters a projection."""
    if not isinstance(value, dict) or set(value) != {
        "acceptance",
        "candidate",
        "claim_id",
        "disposition",
        "mapping",
        "scope",
        "semantic_assertions",
        "validation",
    }:
        msg = "Claim mapping has an invalid shape."
        raise ClaimMappingValidationError(msg)
    try:
        disposition = ClaimDisposition(_text(value["disposition"], "Claim disposition"))
    except ValueError as error:
        msg = "Claim disposition is not recognised."
        raise ClaimMappingValidationError(msg) from error
    return ClaimMapping(
        claim_id=_text(value["claim_id"], "Claim ID"),
        candidate=_candidate_from_json(value["candidate"]),
        disposition=disposition,
        mapping=StageReason(_text(value["mapping"], "Claim mapping reason")),
        validation=StageReason(_text(value["validation"], "Claim validation reason")),
        scope=StageReason(_text(value["scope"], "Claim scope reason")),
        acceptance=StageReason(_text(value["acceptance"], "Claim acceptance reason")),
        semantic_assertions=_semantic_assertions_from_json(
            value["semantic_assertions"]
        ),
    )


def project_claim_mappings(mappings: tuple[ClaimMapping, ...]) -> list[Quad]:
    """Return Claim records and accepted assertions in their named graphs."""
    quads: list[Quad] = []
    for mapping in mappings:
        claim = NamedNode(f"{AIGG}claim/{quote(mapping.claim_id, safe='')}")
        quads.extend(
            [
                Quad(claim, RDF_TYPE, CLAIM, CLAIMS_GRAPH),
                Quad(
                    claim,
                    CLAIM_ASSERTION,
                    RdfLiteral(mapping.candidate.assertion),
                    CLAIMS_GRAPH,
                ),
                Quad(
                    claim,
                    CLAIM_CONFIDENCE,
                    RdfLiteral(str(mapping.candidate.confidence)),
                    CLAIMS_GRAPH,
                ),
                Quad(
                    claim,
                    CLAIM_RATIONALE,
                    RdfLiteral(mapping.candidate.rationale),
                    CLAIMS_GRAPH,
                ),
                Quad(
                    claim,
                    DISPOSITION,
                    RdfLiteral(mapping.disposition.value),
                    CLAIMS_GRAPH,
                ),
                Quad(
                    claim,
                    MAPPING_REASON,
                    RdfLiteral(mapping.mapping.reason),
                    CLAIMS_GRAPH,
                ),
                Quad(
                    claim,
                    VALIDATION_REASON,
                    RdfLiteral(mapping.validation.reason),
                    CLAIMS_GRAPH,
                ),
                Quad(
                    claim, SCOPE_REASON, RdfLiteral(mapping.scope.reason), CLAIMS_GRAPH
                ),
                Quad(
                    claim,
                    ACCEPTANCE_REASON,
                    RdfLiteral(mapping.acceptance.reason),
                    CLAIMS_GRAPH,
                ),
                Quad(
                    claim,
                    CLAIM_TIME,
                    _json_literal(mapping.candidate.times.as_json()),
                    CLAIMS_GRAPH,
                ),
            ]
        )
        quads.extend(
            Quad(claim, EVIDENCE, _json_literal(anchor.as_json()), CLAIMS_GRAPH)
            for anchor in mapping.candidate.evidence
        )
        if mapping.disposition is ClaimDisposition.ACCEPTED:
            quads.extend(_accepted_assertion_quads(claim, mapping.semantic_assertions))
    return quads


def _accepted_assertion_quads(
    claim: NamedNode, assertions: tuple[SemanticAssertion, ...]
) -> list[Quad]:
    """Return canonical triples and their Claim-level RDF reification."""
    quads: list[Quad] = []
    for index, assertion in enumerate(assertions):
        subject = NamedNode(assertion.subject)
        predicate = NamedNode(assertion.predicate)
        object_ = (
            NamedNode(assertion.object)
            if assertion.object_kind == "iri"
            else RdfLiteral(assertion.object)
        )
        statement = NamedNode(f"{claim.value}/assertion/{index}")
        quads.extend(
            [
                Quad(subject, predicate, object_, ACCEPTED_KNOWLEDGE_GRAPH),
                Quad(claim, PROJECTS_ASSERTION, statement, CLAIMS_GRAPH),
                Quad(statement, RDF_TYPE, RDF_STATEMENT, CLAIMS_GRAPH),
                Quad(statement, RDF_SUBJECT, subject, CLAIMS_GRAPH),
                Quad(statement, RDF_PREDICATE, predicate, CLAIMS_GRAPH),
                Quad(statement, RDF_OBJECT, object_, CLAIMS_GRAPH),
            ]
        )
    return quads


def _candidate_from_json(value: JsonValue) -> CandidateClaim:
    """Recreate a source-supported candidate Claim from its durable record."""
    if not isinstance(value, dict) or set(value) != {
        "assertion",
        "confidence",
        "evidence",
        "rationale",
        "times",
    }:
        msg = "Claim mapping candidate has an invalid shape."
        raise ClaimMappingValidationError(msg)
    confidence = value["confidence"]
    if (
        not isinstance(confidence, int | float)
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        msg = "Candidate Claim confidence must be a number from 0 to 1."
        raise ClaimMappingValidationError(msg)
    evidence = value["evidence"]
    if not isinstance(evidence, list):
        msg = "Candidate Claim Evidence must be a list."
        raise ClaimMappingValidationError(msg)
    return CandidateClaim(
        assertion=_text(value["assertion"], "Candidate Claim assertion"),
        confidence=float(confidence),
        evidence=tuple(EvidenceAnchor.from_json(anchor) for anchor in evidence),
        rationale=_text(value["rationale"], "Candidate Claim rationale"),
        times=claim_times_from_json(value["times"], _temporal_expression_from_json),
    )


def _temporal_expression_from_json(value: JsonValue) -> ExtractedTemporalExpression:
    """Recreate an extracted temporal expression without losing its Evidence."""
    if not isinstance(value, dict) or set(value) != {"evidence", "text"}:
        msg = "Extracted temporal expression has an invalid shape."
        raise ClaimMappingValidationError(msg)
    evidence = value["evidence"]
    if not isinstance(evidence, list):
        msg = "Extracted temporal expression Evidence must be a list."
        raise ClaimMappingValidationError(msg)
    return ExtractedTemporalExpression(
        tuple(EvidenceAnchor.from_json(anchor) for anchor in evidence),
        _text(value["text"], "Extracted temporal expression text"),
    )


def _semantic_assertions_from_json(value: JsonValue) -> tuple[SemanticAssertion, ...]:
    """Parse all canonical assertions without inferring any unrecorded terms."""
    if not isinstance(value, list):
        msg = "Claim semantic assertions must be a list."
        raise ClaimMappingValidationError(msg)
    assertions: list[SemanticAssertion] = []
    for assertion in value:
        if not isinstance(assertion, dict) or set(assertion) != {
            "object",
            "object_kind",
            "predicate",
            "subject",
        }:
            msg = "Claim semantic assertion has an invalid shape."
            raise ClaimMappingValidationError(msg)
        assertions.append(
            SemanticAssertion(
                subject=_text(assertion["subject"], "Semantic assertion subject"),
                predicate=_text(assertion["predicate"], "Semantic assertion predicate"),
                object=_text(assertion["object"], "Semantic assertion object"),
                object_kind=_semantic_object_kind(assertion["object_kind"]),
            )
        )
    return tuple(assertions)


def _semantic_object_kind(value: JsonValue) -> Literal["iri", "literal"]:
    """Parse the object kind of one durable semantic assertion."""
    if isinstance(value, str) and value in ("iri", "literal"):
        return value
    msg = "Semantic assertion object kind must be iri or literal."
    raise ClaimMappingValidationError(msg)


def _json_literal(value: JsonValue) -> RdfLiteral:
    """Represent retained structured provenance as one RDF JSON value."""
    return RdfLiteral(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")), datatype=RDF_JSON
    )


def _text(value: JsonValue, field: str) -> str:
    """Return one non-empty text field from a durable contract."""
    if not isinstance(value, str):
        msg = f"{field} must be text."
        raise ClaimMappingValidationError(msg)
    _require_text(value, field)
    return value


def _require_text(value: str, field: str) -> None:
    """Reject blank text that cannot explain a Claim decision."""
    if not value.strip():
        msg = f"{field} must not be empty."
        raise ClaimMappingValidationError(msg)
