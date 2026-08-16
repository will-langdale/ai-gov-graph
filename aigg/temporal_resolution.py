"""Resolve temporal expressions without conflating Claim time kinds."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol, TypeAlias

from aigg.artefacts import JsonValue
from aigg.canonical import EvidenceAnchor


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
    """The expression and known reference time supplied to a temporal Resolver."""

    expression: ExtractedTemporalExpression
    reference_time: datetime | None = None

    def __post_init__(self) -> None:
        """Ensure resolver input has retained source language and valid context."""
        _validate_expression(self.expression)
        _validate_boundary(self.reference_time, "Temporal reference time")


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
                context.expression.text == "next April"
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
                    "The reference time establishes which April is next.",
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
