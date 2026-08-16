"""Resolve temporal expressions without conflating Claim time kinds."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol, TypeAlias, cast

from aigg.artefacts import (
    ArtefactIntegrityError,
    ArtefactReference,
    ArtefactStore,
    JsonValue,
    reference_from_json,
)
from aigg.artefacts import (
    reference_as_json as _reference_json,
)
from aigg.canonical import EvidenceAnchor
from aigg.reasoning import ModelConfiguration, ReasoningRunner, StructuredModel

TEMPORAL_RESOLUTION_STAGE = "temporal-resolution"


class TemporalResolutionValidationError(ValueError):
    """Raised when a temporal-resolution contract is incomplete or ambiguous."""


TemporalBound: TypeAlias = date | datetime


@dataclass(frozen=True)
class ExtractedTemporalExpression:
    """A source-supported temporal expression awaiting normalisation."""

    evidence: tuple[EvidenceAnchor, ...]
    text: str

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable temporal-expression representation."""
        return {
            "evidence": [anchor.as_json() for anchor in self.evidence],
            "text": self.text,
        }


@dataclass(frozen=True)
class TemporalConstraint:
    """A machine-comparable interval with only evidence-supported boundaries."""

    lower_bound: TemporalBound | None = None
    lower_inclusive: bool = True
    upper_bound: TemporalBound | None = None
    upper_inclusive: bool = True

    def __post_init__(self) -> None:
        """Reject empty, naive or contradictory temporal constraints."""
        if self.lower_bound is None and self.upper_bound is None:
            msg = "A temporal constraint must contain at least one bound."
            raise TemporalResolutionValidationError(msg)
        _validate_boundary(self.lower_bound, "Lower temporal bound")
        _validate_boundary(self.upper_bound, "Upper temporal bound")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and type(self.lower_bound) is not type(self.upper_bound)
        ):
            msg = "Temporal bounds must use the same precision."
            raise TemporalResolutionValidationError(msg)
        _validate_inclusivity(
            self.lower_bound, self.lower_inclusive, "Lower temporal bound"
        )
        _validate_inclusivity(
            self.upper_bound, self.upper_inclusive, "Upper temporal bound"
        )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            msg = "A temporal lower bound cannot follow its upper bound."
            raise TemporalResolutionValidationError(msg)
        if self.lower_bound == self.upper_bound and not (
            self.lower_inclusive and self.upper_inclusive
        ):
            msg = "Equal temporal bounds must both be inclusive."
            raise TemporalResolutionValidationError(msg)

    @classmethod
    def exactly(cls, instant: TemporalBound) -> TemporalConstraint:
        """Return the precise instant justified by the evidence."""
        return cls(instant, True, instant, True)

    @classmethod
    def during(cls, start: TemporalBound, end: TemporalBound) -> TemporalConstraint:
        """Return a period that includes its start and excludes its end."""
        return cls(start, True, end, False)

    def as_json(self) -> dict[str, JsonValue]:
        """Return a durable temporal constraint without changing its precision."""
        return {
            "lower_bound": _datetime_json(self.lower_bound),
            "lower_inclusive": self.lower_inclusive,
            "upper_bound": _datetime_json(self.upper_bound),
            "upper_inclusive": self.upper_inclusive,
        }

    @classmethod
    def from_json(cls, value: JsonValue) -> TemporalConstraint:
        """Parse a durable comparable temporal constraint."""
        if not isinstance(value, dict) or set(value) != {
            "lower_bound",
            "lower_inclusive",
            "upper_bound",
            "upper_inclusive",
        }:
            msg = "Temporal constraint has an invalid shape."
            raise TemporalResolutionValidationError(msg)
        return cls(
            _bound_from_json(value["lower_bound"], "Lower temporal bound"),
            _boolean(value["lower_inclusive"], "Lower temporal bound inclusivity"),
            _bound_from_json(value["upper_bound"], "Upper temporal bound"),
            _boolean(value["upper_inclusive"], "Upper temporal bound inclusivity"),
        )


@dataclass(frozen=True)
class ResolvedTemporalExpression:
    """One temporal expression resolved to a supported comparable constraint."""

    expression: ExtractedTemporalExpression
    constraint: TemporalConstraint
    methodology: str
    rationale: str

    def __post_init__(self) -> None:
        """Keep a resolution tied to its source language and stated basis."""
        _validate_expression(self.expression)
        _require_text(self.methodology, "Temporal resolution methodology")
        _require_text(self.rationale, "Temporal resolution rationale")

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable resolved temporal-expression representation."""
        return {
            "constraint": self.constraint.as_json(),
            "expression": self.expression.as_json(),
            "kind": "resolved",
            "methodology": self.methodology,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class UnresolvedTemporalExpression:
    """One retained temporal expression that current context cannot resolve."""

    expression: ExtractedTemporalExpression
    methodology: str
    rationale: str

    def __post_init__(self) -> None:
        """Keep unresolved source language available for later reconsideration."""
        _validate_expression(self.expression)
        _require_text(self.methodology, "Temporal resolution methodology")
        _require_text(self.rationale, "Temporal resolution rationale")

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable unresolved temporal-expression representation."""
        return {
            "expression": self.expression.as_json(),
            "kind": "unresolved",
            "methodology": self.methodology,
            "rationale": self.rationale,
        }


TemporalResolution: TypeAlias = (
    ResolvedTemporalExpression | UnresolvedTemporalExpression
)


@dataclass(frozen=True)
class TemporalResolutionContext:
    """The expression and bounded Source and graph context for a Resolver."""

    expression: ExtractedTemporalExpression
    reference_time: datetime | None = None
    source_context: tuple[dict[str, JsonValue], ...] = ()
    graph_context: tuple[dict[str, JsonValue], ...] = ()
    maximum_source_context: int = 8
    maximum_graph_context: int = 8

    def __post_init__(self) -> None:
        """Ensure resolver input has retained source language and valid context."""
        _validate_expression(self.expression)
        _validate_boundary(self.reference_time, "Temporal reference time")
        _validate_context_bound(
            self.source_context, self.maximum_source_context, "Source context"
        )
        _validate_context_bound(
            self.graph_context, self.maximum_graph_context, "Graph context"
        )
        object.__setattr__(
            self, "source_context", _normalise_context(self.source_context, "Source")
        )
        object.__setattr__(
            self, "graph_context", _normalise_context(self.graph_context, "Graph")
        )


class TemporalResolver(Protocol):
    """A replaceable methodology for resolving one temporal expression."""

    def resolve(self, context: TemporalResolutionContext) -> TemporalResolution:
        """Return a comparable constraint or an explicit unresolved outcome."""


class CalendarTemporalResolver:
    """Resolve explicit and reference-relative calendar expressions conservatively."""

    def resolve(self, context: TemporalResolutionContext) -> TemporalResolution:
        """Resolve a year, month or date, otherwise retain the expression."""
        match = re.fullmatch(
            r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", context.expression.text
        )
        if match is None:
            if (
                context.expression.text in {"next April", "the following April"}
                and context.reference_time is not None
            ):
                year = context.reference_time.date().year
                if context.reference_time.date().month >= 4:
                    year += 1
                start, end = _calendar_period(year, 4, None)
                return ResolvedTemporalExpression(
                    context.expression,
                    TemporalConstraint.during(start, end),
                    "reference-calendar",
                    "The reference time establishes which April the expression names.",
                )
            return UnresolvedTemporalExpression(
                context.expression,
                "iso-calendar",
                "The expression does not state an ISO calendar period.",
            )
        year, month, day = (
            int(value) if value is not None else None for value in match.groups()
        )
        assert year is not None
        try:
            start, end = _calendar_period(year, month, day)
        except ValueError:
            return UnresolvedTemporalExpression(
                context.expression,
                "iso-calendar",
                "The expression is not a valid ISO calendar period.",
            )
        return ResolvedTemporalExpression(
            context.expression,
            TemporalConstraint.during(start, end),
            "iso-calendar",
            "The expression states its calendar period explicitly.",
        )


@dataclass(frozen=True)
class RecordedTemporalResolution:
    """One durable temporal result and its optional model invocation."""

    outcome: TemporalResolution
    reasoning_invocation: ArtefactReference | None
    reference: ArtefactReference


class OpenRouterTemporalResolver:
    """Use calendar calculation first and structured model judgement when needed."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        *,
        maximum_attempts: int,
    ) -> None:
        """Create a hybrid resolver with an explicit model retry bound."""
        self._calendar = CalendarTemporalResolver()
        self._runner = ReasoningRunner(
            store, model, configuration, maximum_attempts=maximum_attempts
        )

    def resolve(self, context: TemporalResolutionContext) -> TemporalResolution:
        """Return one deterministic or model-judged temporal outcome."""
        return self.resolve_recorded(context)[0]

    def resolve_recorded(
        self, context: TemporalResolutionContext
    ) -> tuple[TemporalResolution, ArtefactReference | None]:
        """Return an outcome and the durable invocation when judgement was needed."""
        deterministic = self._calendar.resolve(context)
        if isinstance(deterministic, ResolvedTemporalExpression):
            return deterministic, None
        invocation = self._runner.run(
            stage=TEMPORAL_RESOLUTION_STAGE,
            structured_input=_context_json(context),
            validate_output=lambda value: _validated_structured_output(value, context),
        )
        return _resolution_from_structured(
            invocation.output, context
        ), invocation.reference


class TemporalResolutionService:
    """Record and replay hybrid temporal results from durable stage requests."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        *,
        maximum_attempts: int,
    ) -> None:
        """Create a service whose result artefacts are workflow authority."""
        self._store = store
        self._resolver = OpenRouterTemporalResolver(
            store, model, configuration, maximum_attempts=maximum_attempts
        )

    def create_request(self, context: TemporalResolutionContext) -> ArtefactReference:
        """Store one bounded temporal request for a graph node to reference."""
        return self._store.write_json(
            "temporal-resolution-request", _context_json(context)
        )

    def resolve_request(
        self,
        request: ArtefactReference,
        *,
        replay: ArtefactReference | None = None,
    ) -> RecordedTemporalResolution:
        """Resolve a durable request, or consume an exact stored result."""
        context = _context_from_json(self._store.read_json(request))
        if replay is not None:
            return self._replay(request, replay)
        outcome, invocation = self._resolver.resolve_recorded(context)
        reference = self._store.write_json(
            "temporal-resolution",
            {
                "outcome": outcome.as_json(),
                "reasoning_invocation": _reference_json(invocation),
                "request": _reference_json(request),
            },
        )
        return RecordedTemporalResolution(outcome, invocation, reference)

    def _replay(
        self, request: ArtefactReference, replay: ArtefactReference
    ) -> RecordedTemporalResolution:
        """Read an exact result without calling the configured model."""
        value = self._store.read_json(replay)
        if not isinstance(value, dict) or set(value) != {
            "outcome",
            "reasoning_invocation",
            "request",
        }:
            msg = "Temporal resolution artefact has an invalid shape."
            raise TemporalResolutionValidationError(msg)
        if _reference_from_json(value["request"]) != request:
            msg = "Temporal resolution replay belongs to a different request."
            raise TemporalResolutionValidationError(msg)
        outcome = _resolution_from_json(
            value["outcome"], _expression_from_json_required
        )
        if outcome is None:
            msg = "Temporal resolution replay must retain an outcome."
            raise TemporalResolutionValidationError(msg)
        return RecordedTemporalResolution(
            outcome, _reference_from_json(value["reasoning_invocation"]), replay
        )


@dataclass(frozen=True)
class ClaimTimes:
    """Separate temporal qualifications associated with one Claim."""

    publication: TemporalResolution | None = None
    processing: TemporalResolution | None = None
    event: TemporalResolution | None = None
    applicability: TemporalResolution | None = None

    def __post_init__(self) -> None:
        """Keep each Claim time kind explicit and independently inspectable."""
        for name, value in (
            ("publication", self.publication),
            ("processing", self.processing),
            ("event", self.event),
            ("applicability", self.applicability),
        ):
            if value is not None and not isinstance(value, TemporalResolution):
                msg = f"Claim {name} time must be a temporal resolution."
                raise TemporalResolutionValidationError(msg)

    def as_json(self) -> dict[str, JsonValue]:
        """Return all Claim time kinds without conflating absent values."""
        return {
            "applicability": _resolution_json(self.applicability),
            "event": _resolution_json(self.event),
            "processing": _resolution_json(self.processing),
            "publication": _resolution_json(self.publication),
        }


def claim_times_from_json(
    value: JsonValue,
    expression_from_json: Callable[[JsonValue], ExtractedTemporalExpression],
) -> ClaimTimes:
    """Parse all Claim time kinds while validating their source expressions."""
    if not isinstance(value, dict) or set(value) != {
        "applicability",
        "event",
        "processing",
        "publication",
    }:
        msg = "Claim times have an invalid shape."
        raise TemporalResolutionValidationError(msg)
    return ClaimTimes(
        _resolution_from_json(value["publication"], expression_from_json),
        _resolution_from_json(value["processing"], expression_from_json),
        _resolution_from_json(value["event"], expression_from_json),
        _resolution_from_json(value["applicability"], expression_from_json),
    )


def _validate_expression(expression: ExtractedTemporalExpression) -> None:
    """Require source text and Evidence for every temporal decision."""
    _require_text(expression.text, "Temporal expression text")
    if not expression.evidence:
        msg = "A temporal expression must contain Evidence."
        raise TemporalResolutionValidationError(msg)


def _validate_boundary(value: TemporalBound | None, field: str) -> None:
    """Require temporal bounds to be dates or timezone-aware datetimes."""
    if value is not None and not isinstance(value, date):
        msg = f"{field} must be a date or datetime."
        raise TemporalResolutionValidationError(msg)
    if isinstance(value, datetime) and (
        value.tzinfo is None or value.utcoffset() is None
    ):
        msg = f"{field} must be timezone-aware."
        raise TemporalResolutionValidationError(msg)


def _validate_inclusivity(
    bound: TemporalBound | None, inclusive: bool, field: str
) -> None:
    """Reject ambiguous inclusivity for an absent temporal bound."""
    if not isinstance(inclusive, bool):
        msg = f"{field} inclusivity must be a boolean."
        raise TemporalResolutionValidationError(msg)
    if bound is None and not inclusive:
        msg = f"An absent {field.lower()} cannot be exclusive."
        raise TemporalResolutionValidationError(msg)


def _datetime_json(value: TemporalBound | None) -> str | None:
    """Return a stable JSON representation for an optional temporal boundary."""
    if value is None:
        return None
    return value.isoformat()


def _bound_from_json(value: JsonValue, field: str) -> TemporalBound | None:
    """Parse one optional ISO date or datetime before validating its precision."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{field} must be an ISO date, datetime or null."
        raise TemporalResolutionValidationError(msg)
    try:
        return (
            datetime.fromisoformat(value) if "T" in value else date.fromisoformat(value)
        )
    except ValueError as error:
        msg = f"{field} must be an ISO date, datetime or null."
        raise TemporalResolutionValidationError(msg) from error


def _boolean(value: JsonValue, field: str) -> bool:
    """Require one JSON boolean in a durable temporal representation."""
    if not isinstance(value, bool):
        msg = f"{field} must be a boolean."
        raise TemporalResolutionValidationError(msg)
    return value


def _resolution_from_json(
    value: JsonValue,
    expression_from_json: Callable[[JsonValue], ExtractedTemporalExpression],
) -> TemporalResolution | None:
    """Parse one optional tagged temporal resolution."""
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = "Claim time must be a temporal resolution or null."
        raise TemporalResolutionValidationError(msg)
    kind = value.get("kind")
    if kind == "resolved" and set(value) == {
        "constraint",
        "expression",
        "kind",
        "methodology",
        "rationale",
    }:
        return ResolvedTemporalExpression(
            expression_from_json(value["expression"]),
            TemporalConstraint.from_json(value["constraint"]),
            _require_text(value["methodology"], "Temporal resolution methodology"),
            _require_text(value["rationale"], "Temporal resolution rationale"),
        )
    if kind == "unresolved" and set(value) == {
        "expression",
        "kind",
        "methodology",
        "rationale",
    }:
        return UnresolvedTemporalExpression(
            expression_from_json(value["expression"]),
            _require_text(value["methodology"], "Temporal resolution methodology"),
            _require_text(value["rationale"], "Temporal resolution rationale"),
        )
    msg = "Claim time has an invalid temporal resolution shape."
    raise TemporalResolutionValidationError(msg)


def _calendar_period(
    year: int, month: int | None, day: int | None
) -> tuple[date, date]:
    """Return the exact calendar period that an ISO expression names."""
    if month is None:
        return date(year, 1, 1), date(year + 1, 1, 1)
    if day is None:
        if month == 12:
            return (
                date(year, month, 1),
                date(year + 1, 1, 1),
            )
        return (
            date(year, month, 1),
            date(year, month + 1, 1),
        )
    start = date(year, month, day)
    return start, start + timedelta(days=1)


def _resolution_json(value: TemporalResolution | None) -> dict[str, JsonValue] | None:
    """Return one optional temporal resolution in its durable representation."""
    if value is None:
        return None
    return value.as_json()


def _require_text(value: object, field: str) -> str:
    """Require meaningful text in a public temporal-resolution contract."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{field} must be a non-empty string."
        raise TemporalResolutionValidationError(msg)
    return value


def _validate_context_bound(
    context: tuple[dict[str, JsonValue], ...], maximum: int, name: str
) -> None:
    """Require a small, explicit context bound at the model boundary."""
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
        msg = f"{name} maximum must be a non-negative integer."
        raise TemporalResolutionValidationError(msg)
    if len(context) > maximum:
        msg = f"{name} exceeds its stated maximum."
        raise TemporalResolutionValidationError(msg)


def _normalise_context(
    context: tuple[dict[str, JsonValue], ...], name: str
) -> tuple[dict[str, JsonValue], ...]:
    """Copy JSON context so later caller mutation cannot expand the request."""
    normalised: list[dict[str, JsonValue]] = []
    for item in context:
        if not isinstance(item, dict):
            msg = f"{name} context items must be JSON objects."
            raise TemporalResolutionValidationError(msg)
        try:
            copied = json.loads(json.dumps(item, allow_nan=False))
        except (TypeError, ValueError) as error:
            msg = f"{name} context items must be JSON-compatible."
            raise TemporalResolutionValidationError(msg) from error
        if not isinstance(copied, dict):
            msg = f"{name} context items must be JSON objects."
            raise TemporalResolutionValidationError(msg)
        normalised.append(cast(dict[str, JsonValue], copied))
    return tuple(normalised)


def _context_json(context: TemporalResolutionContext) -> dict[str, JsonValue]:
    """Return the complete bounded temporal input supplied to reasoning."""
    return {
        "expression": context.expression.as_json(),
        "graph_context": list(context.graph_context),
        "maximum_graph_context": context.maximum_graph_context,
        "maximum_source_context": context.maximum_source_context,
        "reference_time": (
            None
            if context.reference_time is None
            else context.reference_time.isoformat()
        ),
        "source_context": list(context.source_context),
    }


def _context_from_json(value: JsonValue) -> TemporalResolutionContext:
    """Load one durable temporal request before resolving it."""
    if not isinstance(value, dict) or set(value) != {
        "expression",
        "graph_context",
        "maximum_graph_context",
        "maximum_source_context",
        "reference_time",
        "source_context",
    }:
        msg = "Temporal resolution request has an invalid shape."
        raise TemporalResolutionValidationError(msg)
    reference_time = value["reference_time"]
    if reference_time is not None:
        if not isinstance(reference_time, str):
            msg = "Temporal reference time must be an ISO datetime or null."
            raise TemporalResolutionValidationError(msg)
        try:
            parsed_reference = datetime.fromisoformat(reference_time)
        except ValueError as error:
            msg = "Temporal reference time must be an ISO datetime or null."
            raise TemporalResolutionValidationError(msg) from error
    else:
        parsed_reference = None
    return TemporalResolutionContext(
        _expression_from_json_required(value["expression"]),
        parsed_reference,
        _context_items(value["source_context"], "Source"),
        _context_items(value["graph_context"], "Graph"),
        _maximum_from_json(value["maximum_source_context"], "Source"),
        _maximum_from_json(value["maximum_graph_context"], "Graph"),
    )


def _context_items(value: JsonValue, name: str) -> tuple[dict[str, JsonValue], ...]:
    """Parse bounded JSON context items from a durable request."""
    if not isinstance(value, list):
        msg = f"{name} context must be a list."
        raise TemporalResolutionValidationError(msg)
    return _normalise_context(tuple(value), name)


def _maximum_from_json(value: JsonValue, name: str) -> int:
    """Parse one explicit temporal context maximum."""
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} context maximum must be an integer."
        raise TemporalResolutionValidationError(msg)
    return value


def _expression_from_json_required(value: JsonValue) -> ExtractedTemporalExpression:
    """Parse source-supported temporal language from a durable record."""
    if not isinstance(value, dict) or set(value) != {"evidence", "text"}:
        msg = "Temporal expression has an invalid shape."
        raise TemporalResolutionValidationError(msg)
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        msg = "Temporal expression must contain Evidence."
        raise TemporalResolutionValidationError(msg)
    try:
        anchors = tuple(EvidenceAnchor.from_json(anchor) for anchor in evidence)
    except ValueError as error:
        raise TemporalResolutionValidationError(str(error)) from error
    return ExtractedTemporalExpression(
        anchors, _require_text(value["text"], "Temporal expression text")
    )


def _resolution_from_structured(
    value: JsonValue, context: TemporalResolutionContext
) -> TemporalResolution:
    """Validate one model constraint against the original source expression."""
    if not isinstance(value, dict):
        msg = "Temporal resolution output must be an object."
        raise TemporalResolutionValidationError(msg)
    kind = value.get("kind")
    if kind == "resolved" and set(value) == {
        "constraint",
        "evidence",
        "kind",
        "rationale",
    }:
        evidence = _model_evidence(value["evidence"], context)
        constraint = TemporalConstraint.from_json(value["constraint"])
        supporting_records = _contextual_constraints(context)
        if not any(
            candidate == constraint and set(evidence).issubset(record_evidence)
            for candidate, record_evidence in supporting_records
        ):
            msg = (
                "Temporal resolution must cite the contextual Evidence that supports "
                "its selected constraint."
            )
            raise TemporalResolutionValidationError(msg)
        if any(
            _is_strictly_stronger(candidate, constraint)
            for candidate, _record_evidence in supporting_records
        ):
            msg = "Temporal resolution must select the strongest supported constraint."
            raise TemporalResolutionValidationError(msg)
        return ResolvedTemporalExpression(
            context.expression,
            constraint,
            "openrouter-temporal-resolution",
            _require_text(value["rationale"], "Temporal resolution rationale"),
        )
    if kind == "unresolved" and set(value) == {"kind", "rationale"}:
        return UnresolvedTemporalExpression(
            context.expression,
            "openrouter-temporal-resolution",
            _require_text(value["rationale"], "Temporal resolution rationale"),
        )
    msg = "Temporal resolution output has an invalid shape."
    raise TemporalResolutionValidationError(msg)


def _model_evidence(
    value: JsonValue, context: TemporalResolutionContext
) -> tuple[EvidenceAnchor, ...]:
    """Require a model result to cite Evidence supplied in its bounded input."""
    if not isinstance(value, list) or not value:
        msg = "Temporal resolution output must cite Evidence."
        raise TemporalResolutionValidationError(msg)
    try:
        evidence = tuple(EvidenceAnchor.from_json(anchor) for anchor in value)
    except ValueError as error:
        raise TemporalResolutionValidationError(str(error)) from error
    contextual_evidence = set().union(
        *(
            record_evidence
            for _constraint, record_evidence in _contextual_constraints(context)
        )
    )
    available = set(context.expression.evidence).union(contextual_evidence)
    if not set(evidence).issubset(available):
        msg = "Temporal resolution evidence is absent from the bounded context."
        raise TemporalResolutionValidationError(msg)
    return evidence


def _contextual_constraints(
    context: TemporalResolutionContext,
) -> tuple[tuple[TemporalConstraint, set[EvidenceAnchor]], ...]:
    """Return each comparable contextual constraint with its own Evidence."""
    records: list[tuple[TemporalConstraint, set[EvidenceAnchor]]] = []
    for item in (*context.source_context, *context.graph_context):
        raw_evidence = item.get("evidence")
        raw_constraint = item.get("constraint")
        if raw_evidence is None and raw_constraint is None:
            continue
        if raw_evidence is None or raw_constraint is None:
            msg = "Temporal contextual constraints must retain their Evidence."
            raise TemporalResolutionValidationError(msg)
        if not isinstance(raw_evidence, list) or not raw_evidence:
            msg = "Temporal contextual constraint Evidence must be a non-empty list."
            raise TemporalResolutionValidationError(msg)
        try:
            evidence = {EvidenceAnchor.from_json(anchor) for anchor in raw_evidence}
        except ValueError as error:
            raise TemporalResolutionValidationError(str(error)) from error
        records.append((TemporalConstraint.from_json(raw_constraint), evidence))
    return tuple(records)


def _is_strictly_stronger(
    candidate: TemporalConstraint, selected: TemporalConstraint
) -> bool:
    """Return whether ``candidate`` is a strictly narrower comparable interval."""
    if (
        candidate.lower_bound is None
        or candidate.upper_bound is None
        or selected.lower_bound is None
        or selected.upper_bound is None
        or type(candidate.lower_bound) is not type(selected.lower_bound)
        or type(candidate.upper_bound) is not type(selected.upper_bound)
    ):
        return False
    contains = (
        candidate.lower_bound >= selected.lower_bound
        and candidate.upper_bound <= selected.upper_bound
    )
    return contains and candidate != selected


def _validated_structured_output(
    value: JsonValue, context: TemporalResolutionContext
) -> JsonValue:
    """Keep validated model JSON in the reasoning record without coercing it."""
    _resolution_from_structured(value, context)
    return value


def _reference_from_json(value: JsonValue) -> ArtefactReference | None:
    """Parse one optional artefact reference before following it."""
    try:
        return reference_from_json(value)
    except ArtefactIntegrityError as error:
        raise TemporalResolutionValidationError(str(error)) from error
