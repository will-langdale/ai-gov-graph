"""Resolve extracted mentions to Entities with reversible decision history."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal, Protocol, TypeAlias

from aigg.artefacts import ArtefactReference, ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor
from aigg.open_extraction import ExtractedMention
from aigg.reasoning import ModelConfiguration, ReasoningRunner, StructuredModel

ENTITY_RESOLUTION_STAGE = "entity-resolution"


class EntityResolutionValidationError(ValueError):
    """Raised when an entity-resolution contract or history is invalid."""


@dataclass(frozen=True)
class Entity:
    """An Entity that a Resolver can select or provision."""

    entity_id: str
    label: str

    def __post_init__(self) -> None:
        """Reject ambiguous Entity identifiers and labels at the boundary."""
        _require_text(self.entity_id, "Entity ID")
        _require_text(self.label, "Entity label")

    def as_json(self) -> dict[str, str]:
        """Return the durable Entity representation."""
        return {"entity_id": self.entity_id, "label": self.label}


@dataclass(frozen=True)
class ResolutionProvenance:
    """The inspectable basis for one identity decision."""

    methodology: str
    rationale: str
    evidence: tuple[EvidenceAnchor, ...]
    reasoning_invocation: ArtefactReference | None = None

    def __post_init__(self) -> None:
        """Require a stated method, rationale and retained supporting Evidence."""
        _require_text(self.methodology, "Resolution methodology")
        _require_text(self.rationale, "Resolution rationale")
        if not self.evidence:
            msg = "Resolution provenance must contain Evidence."
            raise EntityResolutionValidationError(msg)

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable provenance representation."""
        value: dict[str, JsonValue] = {
            "evidence": [anchor.as_json() for anchor in self.evidence],
            "methodology": self.methodology,
            "rationale": self.rationale,
        }
        if self.reasoning_invocation is not None:
            value["reasoning_invocation"] = _reference_json(self.reasoning_invocation)
        return value


@dataclass(frozen=True)
class ExistingEntityResolution:
    """Resolve a mention to one Entity in the supplied candidate context."""

    entity_id: str
    confidence: float
    provenance: ResolutionProvenance

    def __post_init__(self) -> None:
        """Validate the existing-Entity outcome."""
        _require_text(self.entity_id, "Resolved Entity ID")
        _validate_confidence(self.confidence)


@dataclass(frozen=True)
class ProvisionalEntityResolution:
    """Resolve a mention by creating a provisional Entity."""

    entity: Entity
    confidence: float
    provenance: ResolutionProvenance

    def __post_init__(self) -> None:
        """Validate the provisional-Entity outcome."""
        _validate_confidence(self.confidence)


@dataclass(frozen=True)
class UnresolvedResolution:
    """Record that the supplied context does not justify an Entity decision."""

    confidence: float
    provenance: ResolutionProvenance

    def __post_init__(self) -> None:
        """Validate the unresolved outcome."""
        _validate_confidence(self.confidence)


ResolutionOutcome: TypeAlias = (
    ExistingEntityResolution | ProvisionalEntityResolution | UnresolvedResolution
)


@dataclass(frozen=True)
class ResolutionContext:
    """The bounded evidence and candidate context supplied to a Resolver."""

    mention: ExtractedMention
    candidates: tuple[Entity, ...]
    maximum_candidates: int

    def __post_init__(self) -> None:
        """Keep resolver input bounded and tied to retained mention Evidence."""
        _require_text(self.mention.text, "Mention text")
        if not self.mention.evidence:
            msg = "A Resolver context must contain mention Evidence."
            raise EntityResolutionValidationError(msg)
        if self.maximum_candidates < 1:
            msg = "Resolver maximum candidates must be at least one."
            raise EntityResolutionValidationError(msg)
        if len(self.candidates) > self.maximum_candidates:
            msg = "Resolver candidates exceed the stated maximum."
            raise EntityResolutionValidationError(msg)
        candidate_ids = [candidate.entity_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            msg = "Resolver candidates must have unique Entity IDs."
            raise EntityResolutionValidationError(msg)


class Resolver(Protocol):
    """A replaceable methodology for resolving one bounded mention context."""

    def resolve(self, context: ResolutionContext) -> ResolutionOutcome:
        """Return exactly one Entity-resolution outcome for ``context``."""


@dataclass(frozen=True)
class EntityDecision:
    """One immutable entity-resolution or identity-migration decision."""

    decision_id: str
    kind: Literal["resolution", "merge", "split", "reversal"]
    provenance: ResolutionProvenance
    mention: ExtractedMention | None = None
    candidate_entity_ids: tuple[str, ...] = ()
    outcome: ResolutionOutcome | None = None
    source_entity_ids: tuple[str, ...] = ()
    target_entities: tuple[Entity, ...] = ()
    reversed_decision_id: str | None = None

    def as_json(self) -> dict[str, JsonValue]:
        """Return the stable, inspectable decision representation."""
        return {
            "candidate_entity_ids": list(self.candidate_entity_ids),
            "decision_id": self.decision_id,
            "kind": self.kind,
            "mention": _mention_json(self.mention),
            "outcome": _outcome_json(self.outcome),
            "provenance": self.provenance.as_json(),
            "reversed_decision_id": self.reversed_decision_id,
            "source_entity_ids": list(self.source_entity_ids),
            "target_entities": [entity.as_json() for entity in self.target_entities],
        }


@dataclass(frozen=True)
class RecordedEntityDecision:
    """One stored identity-migration decision and the resulting history head."""

    decision: EntityDecision
    history: ArtefactReference


@dataclass(frozen=True)
class RecordedResolution(RecordedEntityDecision):
    """One stored Resolver result and the resulting history head."""

    outcome: ResolutionOutcome


@dataclass(frozen=True)
class RecordedEntityResolution:
    """One resolver result represented by durable result and history references."""

    history: ArtefactReference
    outcome: ResolutionOutcome
    reasoning_invocation: ArtefactReference
    reference: ArtefactReference


@dataclass(frozen=True)
class _RecordedReasonedEntityOutcome:
    """One validated structured outcome and its invocation record."""

    invocation: ArtefactReference
    outcome: ResolutionOutcome


class OpenRouterEntityResolver:
    """Resolve bounded Entity candidates through recorded structured reasoning."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        *,
        maximum_attempts: int,
    ) -> None:
        """Create an OpenRouter resolver with an explicit retry bound."""
        self._runner = ReasoningRunner(
            store, model, configuration, maximum_attempts=maximum_attempts
        )

    def resolve(self, context: ResolutionContext) -> ResolutionOutcome:
        """Return one validated outcome for a bounded candidate context."""
        return self.resolve_recorded(context).outcome

    def resolve_recorded(
        self, context: ResolutionContext
    ) -> _RecordedReasonedEntityOutcome:
        """Run and retain the one structured Entity decision for ``context``."""
        invocation = self._runner.run(
            stage=ENTITY_RESOLUTION_STAGE,
            structured_input=_context_json(context),
            validate_output=lambda value: _validated_structured_output(value, context),
        )
        return _RecordedReasonedEntityOutcome(
            invocation.reference,
            _outcome_from_structured(invocation.output, context, invocation.reference),
        )


class EntityResolutionService:
    """Record and replay OpenRouter Entity decisions from durable requests."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        *,
        maximum_attempts: int,
    ) -> None:
        """Create a service whose durable store is the resolution authority."""
        self._store = store
        self._resolver = OpenRouterEntityResolver(
            store, model, configuration, maximum_attempts=maximum_attempts
        )
        self._history = EntityDecisionHistory(store)

    def create_request(
        self,
        context: ResolutionContext,
        history: ArtefactReference | None = None,
    ) -> ArtefactReference:
        """Store one bounded resolution request for a graph stage to reference."""
        if history is not None:
            self._history.inspect(history)
        return self._store.write_json(
            "entity-resolution-request",
            {"context": _context_json(context), "history": _reference_json(history)},
        )

    def resolve_request(
        self,
        request: ArtefactReference,
        *,
        replay: ArtefactReference | None = None,
    ) -> RecordedEntityResolution:
        """Resolve a durable request, or consume its exact prior result."""
        context, history = _request_from_json(self._store.read_json(request))
        if replay is not None:
            return self._replay(request, replay, context)

        resolver_result = self._resolver.resolve_recorded(context)
        recorded = self._history.resolve(
            _OutcomeResolver(resolver_result.outcome), context, history
        )
        reference = self._store.write_json(
            "entity-resolution",
            {
                "history": _reference_json(recorded.history),
                "reasoning_invocation": _reference_json(
                    resolver_result.outcome.provenance.reasoning_invocation
                ),
                "request": _reference_json(request),
            },
        )
        invocation = resolver_result.outcome.provenance.reasoning_invocation
        assert invocation is not None
        return RecordedEntityResolution(
            recorded.history, recorded.outcome, invocation, reference
        )

    def _replay(
        self,
        request: ArtefactReference,
        replay: ArtefactReference,
        context: ResolutionContext,
    ) -> RecordedEntityResolution:
        """Read an exact prior result without invoking the configured model."""
        value = self._store.read_json(replay)
        if not isinstance(value, dict) or set(value) != {
            "history",
            "reasoning_invocation",
            "request",
        }:
            msg = "Entity resolution artefact has an invalid shape."
            raise EntityResolutionValidationError(msg)
        if _reference_from_json(value["request"]) != request:
            msg = "Entity resolution replay belongs to a different request."
            raise EntityResolutionValidationError(msg)
        history = _reference_from_json(value["history"])
        invocation = _reference_from_json(value["reasoning_invocation"])
        if history is None or invocation is None:
            msg = "Entity resolution replay must retain durable references."
            raise EntityResolutionValidationError(msg)
        decisions = self._history.inspect(history)
        if not decisions or decisions[-1].kind != "resolution":
            msg = "Entity resolution replay has no recorded resolution decision."
            raise EntityResolutionValidationError(msg)
        outcome = decisions[-1].outcome
        if outcome is None:
            msg = "Entity resolution replay has no recorded outcome."
            raise EntityResolutionValidationError(msg)
        self._history._validate_outcome(outcome, context)
        if outcome.provenance.reasoning_invocation != invocation:
            msg = "Entity resolution replay has mismatched reasoning provenance."
            raise EntityResolutionValidationError(msg)
        return RecordedEntityResolution(history, outcome, invocation, replay)


@dataclass(frozen=True)
class _OutcomeResolver:
    """Present one recorded outcome at the Entity history's Resolver seam."""

    outcome: ResolutionOutcome

    def resolve(self, context: ResolutionContext) -> ResolutionOutcome:
        """Return the outcome that the reasoning boundary already validated."""
        del context
        return self.outcome


class EntityDecisionHistory:
    """Store append-only Entity decisions as a durable, reversible history."""

    def __init__(self, store: ArtefactStore) -> None:
        """Create a history backed by immutable experiment artefacts."""
        self._store = store

    def resolve(
        self,
        resolver: Resolver,
        context: ResolutionContext,
        history: ArtefactReference | None = None,
    ) -> RecordedResolution:
        """Run a Resolver and append its typed decision to ``history``."""
        outcome = resolver.resolve(context)
        self._validate_outcome(outcome, context)
        decision = _new_decision(
            "resolution",
            outcome.provenance,
            mention=context.mention,
            candidate_entity_ids=tuple(
                candidate.entity_id for candidate in context.candidates
            ),
            outcome=outcome,
        )
        reference = self._append(decision, history)
        return RecordedResolution(decision, reference, outcome)

    def merge(
        self,
        source_entity_ids: tuple[str, ...],
        target_entity: Entity,
        provenance: ResolutionProvenance,
        history: ArtefactReference | None = None,
    ) -> RecordedEntityDecision:
        """Append a merge without erasing the preceding Entity identities."""
        _validate_merge(source_entity_ids, target_entity)
        decision = _new_decision(
            "merge",
            provenance,
            source_entity_ids=source_entity_ids,
            target_entities=(target_entity,),
        )
        return RecordedEntityDecision(decision, self._append(decision, history))

    def split(
        self,
        source_entity_id: str,
        target_entities: tuple[Entity, ...],
        provenance: ResolutionProvenance,
        history: ArtefactReference | None = None,
    ) -> RecordedEntityDecision:
        """Append a split without discarding the earlier Entity decision."""
        _validate_split(source_entity_id, target_entities)
        decision = _new_decision(
            "split",
            provenance,
            source_entity_ids=(source_entity_id,),
            target_entities=target_entities,
        )
        return RecordedEntityDecision(decision, self._append(decision, history))

    def reverse(
        self,
        decision_id: str,
        provenance: ResolutionProvenance,
        history: ArtefactReference,
    ) -> RecordedEntityDecision:
        """Append a reversal of a prior merge or split without deleting it."""
        previous = {
            decision.decision_id: decision for decision in self.inspect(history)
        }
        try:
            decision = previous[decision_id]
        except KeyError as error:
            msg = f"Entity decision is absent from this history: {decision_id}."
            raise EntityResolutionValidationError(msg) from error
        if decision.kind not in {"merge", "split"}:
            msg = "Only Entity merges and splits can be reversed."
            raise EntityResolutionValidationError(msg)
        reversal = _new_decision(
            "reversal", provenance, reversed_decision_id=decision_id
        )
        return RecordedEntityDecision(reversal, self._append(reversal, history))

    def inspect(self, history: ArtefactReference) -> tuple[EntityDecision, ...]:
        """Return every retained decision from oldest to newest."""
        decisions: list[EntityDecision] = []
        current: ArtefactReference | None = history
        while current is not None:
            value = self._store.read_json(current)
            if not isinstance(value, dict) or set(value) != {"decision", "previous"}:
                msg = (
                    f"Entity history artefact {current.identity} has an invalid shape."
                )
                raise EntityResolutionValidationError(msg)
            decisions.append(_decision_from_json(value["decision"]))
            current = _reference_from_json(value["previous"])
        decisions.reverse()
        return tuple(decisions)

    def _append(
        self, decision: EntityDecision, history: ArtefactReference | None
    ) -> ArtefactReference:
        """Write one immutable history link after checking its predecessor."""
        if history is not None:
            self.inspect(history)
        return self._store.write_json(
            "entity-decision",
            {"decision": decision.as_json(), "previous": _reference_json(history)},
        )

    @staticmethod
    def _validate_outcome(outcome: object, context: ResolutionContext) -> None:
        """Ensure Resolver output is one allowed outcome for its input context."""
        if isinstance(outcome, ExistingEntityResolution):
            candidate_ids = {candidate.entity_id for candidate in context.candidates}
            if outcome.entity_id not in candidate_ids:
                msg = "Existing Entity resolution must select a supplied candidate."
                raise EntityResolutionValidationError(msg)
            return
        if isinstance(outcome, ProvisionalEntityResolution):
            candidate_ids = {candidate.entity_id for candidate in context.candidates}
            if outcome.entity.entity_id in candidate_ids:
                msg = "Provisional Entity resolution must not reuse a candidate ID."
                raise EntityResolutionValidationError(msg)
            return
        if isinstance(outcome, UnresolvedResolution):
            return
        msg = "Resolver must return an existing, provisional or unresolved outcome."
        raise EntityResolutionValidationError(msg)


def _new_decision(
    kind: Literal["resolution", "merge", "split", "reversal"],
    provenance: ResolutionProvenance,
    *,
    mention: ExtractedMention | None = None,
    candidate_entity_ids: tuple[str, ...] = (),
    outcome: ResolutionOutcome | None = None,
    source_entity_ids: tuple[str, ...] = (),
    target_entities: tuple[Entity, ...] = (),
    reversed_decision_id: str | None = None,
) -> EntityDecision:
    """Create a deterministic decision ID from its complete durable content."""
    provisional = EntityDecision(
        decision_id="",
        kind=kind,
        provenance=provenance,
        mention=mention,
        candidate_entity_ids=candidate_entity_ids,
        outcome=outcome,
        source_entity_ids=source_entity_ids,
        target_entities=target_entities,
        reversed_decision_id=reversed_decision_id,
    )
    canonical = json.dumps(
        provisional.as_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return replace(
        provisional,
        decision_id=f"entity-decision:sha256:{sha256(canonical).hexdigest()}",
    )


def _validate_merge(source_entity_ids: tuple[str, ...], target_entity: Entity) -> None:
    """Require a meaningful many-to-one identity migration."""
    if len(source_entity_ids) < 2 or len(source_entity_ids) != len(
        set(source_entity_ids)
    ):
        msg = "An Entity merge requires at least two unique source Entity IDs."
        raise EntityResolutionValidationError(msg)
    if target_entity.entity_id in source_entity_ids:
        msg = "An Entity merge target must differ from every source Entity."
        raise EntityResolutionValidationError(msg)


def _validate_split(source_entity_id: str, target_entities: tuple[Entity, ...]) -> None:
    """Require a meaningful one-to-many identity migration."""
    _require_text(source_entity_id, "Split source Entity ID")
    target_ids = [entity.entity_id for entity in target_entities]
    if len(target_ids) < 2 or len(target_ids) != len(set(target_ids)):
        msg = "An Entity split requires at least two unique target Entities."
        raise EntityResolutionValidationError(msg)
    if source_entity_id in target_ids:
        msg = "An Entity split target must differ from its source Entity."
        raise EntityResolutionValidationError(msg)


def _validate_confidence(confidence: object) -> None:
    """Require confidence values on the closed zero-to-one interval."""
    if (
        not isinstance(confidence, int | float)
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        msg = "Resolution confidence must be a number from 0 to 1."
        raise EntityResolutionValidationError(msg)


def _require_text(value: object, field: str) -> str:
    """Require meaningful text in a public entity-resolution contract."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{field} must be a non-empty string."
        raise EntityResolutionValidationError(msg)
    return value


def _mention_json(mention: ExtractedMention | None) -> dict[str, JsonValue] | None:
    """Return a durable mention, retaining its supporting Evidence."""
    if mention is None:
        return None
    return {
        "evidence": [anchor.as_json() for anchor in mention.evidence],
        "text": mention.text,
    }


def _outcome_json(outcome: ResolutionOutcome | None) -> dict[str, JsonValue] | None:
    """Return one tagged resolution outcome for the durable decision record."""
    if outcome is None:
        return None
    output: dict[str, JsonValue] = {
        "confidence": outcome.confidence,
        "provenance": outcome.provenance.as_json(),
    }
    if isinstance(outcome, ExistingEntityResolution):
        return {"entity_id": outcome.entity_id, "kind": "existing", **output}
    if isinstance(outcome, ProvisionalEntityResolution):
        return {"entity": outcome.entity.as_json(), "kind": "provisional", **output}
    return {"kind": "unresolved", **output}


def _reference_json(reference: ArtefactReference | None) -> dict[str, str] | None:
    """Return a durable history-link reference."""
    if reference is None:
        return None
    return {
        "identity": reference.identity,
        "kind": reference.kind,
        "schema_version": reference.schema_version,
    }


def _reference_from_json(value: JsonValue) -> ArtefactReference | None:
    """Parse one optional predecessor reference before following it."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "identity",
        "kind",
        "schema_version",
    }:
        msg = "Entity history contains an invalid predecessor reference."
        raise EntityResolutionValidationError(msg)
    identity = value["identity"]
    kind = value["kind"]
    schema_version = value["schema_version"]
    if not all(isinstance(field, str) for field in (identity, kind, schema_version)):
        msg = "Entity history predecessor fields must be strings."
        raise EntityResolutionValidationError(msg)
    if not identity.startswith("sha256:") or len(identity) != 71:
        msg = "Entity history predecessor has an invalid identity."
        raise EntityResolutionValidationError(msg)
    return ArtefactReference(kind, identity.removeprefix("sha256:"), schema_version)


def _decision_from_json(value: JsonValue) -> EntityDecision:
    """Parse a retained decision record before presenting it for inspection."""
    if not isinstance(value, dict) or set(value) != {
        "candidate_entity_ids",
        "decision_id",
        "kind",
        "mention",
        "outcome",
        "provenance",
        "reversed_decision_id",
        "source_entity_ids",
        "target_entities",
    }:
        msg = "Entity history contains an invalid decision shape."
        raise EntityResolutionValidationError(msg)
    kind = value["kind"]
    if kind not in {"resolution", "merge", "split", "reversal"}:
        msg = "Entity history contains an unknown decision kind."
        raise EntityResolutionValidationError(msg)
    decision_id = _string(value["decision_id"], "Entity decision ID")
    decision = EntityDecision(
        decision_id=decision_id,
        kind=kind,
        provenance=_provenance_from_json(value["provenance"]),
        mention=_mention_from_json(value["mention"]),
        candidate_entity_ids=_string_tuple(
            value["candidate_entity_ids"], "Entity candidate IDs"
        ),
        outcome=_outcome_from_json(value["outcome"]),
        source_entity_ids=_string_tuple(
            value["source_entity_ids"], "Entity source IDs"
        ),
        target_entities=_entities_from_json(value["target_entities"]),
        reversed_decision_id=_optional_string(
            value["reversed_decision_id"], "Reversed Entity decision ID"
        ),
    )
    expected = _new_decision(
        decision.kind,
        decision.provenance,
        mention=decision.mention,
        candidate_entity_ids=decision.candidate_entity_ids,
        outcome=decision.outcome,
        source_entity_ids=decision.source_entity_ids,
        target_entities=decision.target_entities,
        reversed_decision_id=decision.reversed_decision_id,
    )
    if decision.decision_id != expected.decision_id:
        msg = "Entity history decision ID does not match its recorded content."
        raise EntityResolutionValidationError(msg)
    return decision


def _provenance_from_json(value: JsonValue) -> ResolutionProvenance:
    """Parse retained resolution provenance."""
    if not isinstance(value, dict) or set(value) not in (
        {"evidence", "methodology", "rationale"},
        {"evidence", "methodology", "rationale", "reasoning_invocation"},
    ):
        msg = "Entity decision contains invalid provenance."
        raise EntityResolutionValidationError(msg)
    evidence = _evidence_from_json(value["evidence"])
    return ResolutionProvenance(
        _string(value["methodology"], "Resolution methodology"),
        _string(value["rationale"], "Resolution rationale"),
        evidence,
        (
            None
            if "reasoning_invocation" not in value
            else _reference_from_json(value["reasoning_invocation"])
        ),
    )


def _mention_from_json(value: JsonValue) -> ExtractedMention | None:
    """Parse an optional evidence-backed extracted mention."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"evidence", "text"}:
        msg = "Entity decision contains an invalid mention."
        raise EntityResolutionValidationError(msg)
    return ExtractedMention(
        _evidence_from_json(value["evidence"]),
        _string(value["text"], "Mention text"),
    )


def _outcome_from_json(value: JsonValue) -> ResolutionOutcome | None:
    """Parse one tagged Resolver outcome."""
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = "Entity decision contains an invalid resolution outcome."
        raise EntityResolutionValidationError(msg)
    kind = value.get("kind")
    provenance = _provenance_from_json(value.get("provenance"))
    confidence = _confidence_from_json(value.get("confidence"))
    if kind == "existing" and set(value) == {
        "confidence",
        "entity_id",
        "kind",
        "provenance",
    }:
        return ExistingEntityResolution(
            _string(value["entity_id"], "Resolved Entity ID"), confidence, provenance
        )
    if kind == "provisional" and set(value) == {
        "confidence",
        "entity",
        "kind",
        "provenance",
    }:
        return ProvisionalEntityResolution(
            _entity_from_json(value["entity"]), confidence, provenance
        )
    if kind == "unresolved" and set(value) == {"confidence", "kind", "provenance"}:
        return UnresolvedResolution(confidence, provenance)
    msg = "Entity decision contains an invalid resolution outcome."
    raise EntityResolutionValidationError(msg)


def _entities_from_json(value: JsonValue) -> tuple[Entity, ...]:
    """Parse the Entity targets in an identity migration."""
    if not isinstance(value, list):
        msg = "Entity decision target Entities must be a list."
        raise EntityResolutionValidationError(msg)
    return tuple(_entity_from_json(entity) for entity in value)


def _entity_from_json(value: JsonValue) -> Entity:
    """Parse one durable Entity representation."""
    if not isinstance(value, dict) or set(value) != {"entity_id", "label"}:
        msg = "Entity decision contains an invalid Entity."
        raise EntityResolutionValidationError(msg)
    return Entity(
        _string(value["entity_id"], "Entity ID"),
        _string(value["label"], "Entity label"),
    )


def _evidence_from_json(value: JsonValue) -> tuple[EvidenceAnchor, ...]:
    """Parse retained Evidence anchors without reinterpreting them."""
    if not isinstance(value, list) or not value:
        msg = "Entity decision Evidence must be a non-empty list."
        raise EntityResolutionValidationError(msg)
    try:
        return tuple(EvidenceAnchor.from_json(anchor) for anchor in value)
    except ValueError as error:
        raise EntityResolutionValidationError(str(error)) from error


def _string_tuple(value: JsonValue, field: str) -> tuple[str, ...]:
    """Parse a list of meaningful identifiers."""
    if not isinstance(value, list):
        msg = f"{field} must be a list."
        raise EntityResolutionValidationError(msg)
    return tuple(_string(item, field) for item in value)


def _optional_string(value: JsonValue, field: str) -> str | None:
    """Parse optional meaningful text."""
    if value is None:
        return None
    return _string(value, field)


def _confidence_from_json(value: JsonValue | object) -> float:
    """Parse a confidence value from a durable decision record."""
    _validate_confidence(value)
    assert isinstance(value, int | float)
    return float(value)


def _string(value: JsonValue | object, field: str) -> str:
    """Parse meaningful text from a durable JSON value."""
    return _require_text(value, field)


def _context_json(context: ResolutionContext) -> dict[str, JsonValue]:
    """Return the complete bounded input supplied to entity reasoning."""
    return {
        "candidates": [candidate.as_json() for candidate in context.candidates],
        "maximum_candidates": context.maximum_candidates,
        "mention": _mention_json(context.mention),
    }


def _request_from_json(
    value: JsonValue,
) -> tuple[ResolutionContext, ArtefactReference | None]:
    """Load one stored entity-resolution request before following its history."""
    if not isinstance(value, dict) or set(value) != {"context", "history"}:
        msg = "Entity resolution request has an invalid shape."
        raise EntityResolutionValidationError(msg)
    context = value["context"]
    if not isinstance(context, dict) or set(context) != {
        "candidates",
        "maximum_candidates",
        "mention",
    }:
        msg = "Entity resolution request has an invalid context."
        raise EntityResolutionValidationError(msg)
    mention = _mention_from_json(context["mention"])
    if mention is None:
        msg = "Entity resolution request needs a mention."
        raise EntityResolutionValidationError(msg)
    candidates = _entities_from_json(context["candidates"])
    maximum_candidates = context["maximum_candidates"]
    if not isinstance(maximum_candidates, int) or isinstance(maximum_candidates, bool):
        msg = "Entity resolution request maximum candidates must be an integer."
        raise EntityResolutionValidationError(msg)
    return ResolutionContext(
        mention, candidates, maximum_candidates
    ), _reference_from_json(value["history"])


def _outcome_from_structured(
    value: JsonValue,
    context: ResolutionContext,
    invocation: ArtefactReference | None = None,
) -> ResolutionOutcome:
    """Validate one model outcome against its bounded candidates and Evidence."""
    if not isinstance(value, dict):
        msg = "Entity resolution output must be an object."
        raise EntityResolutionValidationError(msg)
    kind = value.get("kind")
    required = {"confidence", "evidence", "kind", "rationale"}
    if kind == "existing":
        required.add("entity_id")
    elif kind == "provisional":
        required.add("entity")
    elif kind != "unresolved":
        msg = "Entity resolution output has an invalid kind."
        raise EntityResolutionValidationError(msg)
    if set(value) != required:
        msg = "Entity resolution output has an invalid shape."
        raise EntityResolutionValidationError(msg)
    evidence = _evidence_from_json(value["evidence"])
    if not set(evidence).issubset(context.mention.evidence):
        msg = "Entity resolution evidence must support the supplied mention."
        raise EntityResolutionValidationError(msg)
    provenance = ResolutionProvenance(
        "openrouter-entity-resolution",
        _string(value["rationale"], "Entity resolution rationale"),
        evidence,
        invocation,
    )
    confidence = _confidence_from_json(value["confidence"])
    if kind == "existing":
        outcome: ResolutionOutcome = ExistingEntityResolution(
            _string(value["entity_id"], "Resolved Entity ID"), confidence, provenance
        )
    elif kind == "provisional":
        outcome = ProvisionalEntityResolution(
            _entity_from_json(value["entity"]), confidence, provenance
        )
    else:
        outcome = UnresolvedResolution(confidence, provenance)
    EntityDecisionHistory._validate_outcome(outcome, context)
    return outcome


def _validated_structured_output(
    value: JsonValue, context: ResolutionContext
) -> JsonValue:
    """Keep validated model JSON in the reasoning record without coercing it."""
    _outcome_from_structured(value, context)
    return value
