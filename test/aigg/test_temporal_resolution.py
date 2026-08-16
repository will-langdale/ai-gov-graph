"""Tests for temporal resolution and Claim timing."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from aigg.artefacts import ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor
from aigg.open_extraction import CandidateClaim, ExtractedTemporalExpression
from aigg.reasoning import ModelConfiguration, StructuredModel
from aigg.temporal_resolution import (
    CalendarTemporalResolver,
    ClaimTimes,
    ResolvedTemporalExpression,
    TemporalConstraint,
    TemporalResolutionContext,
    TemporalResolutionService,
    UnresolvedTemporalExpression,
    claim_times_from_json,
)


@dataclass
class StaticModel(StructuredModel):
    """Return one configured result from the external model boundary."""

    output: JsonValue

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the configured structured output."""
        del configuration, structured_input
        return self.output


def _calendar_cases() -> list[object]:
    """Generate ISO calendar cases from independent calendar boundaries."""
    cases: list[object] = []
    for year in (2000, 2026):
        cases.append(
            pytest.param(
                f"{year}",
                _constraint_for_dates(date(year, 1, 1), date(year + 1, 1, 1)),
                id=f"year_{year}",
            )
        )
        for month in (1, 4, 12):
            start = date(year, month, 1)
            end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            cases.append(
                pytest.param(
                    f"{year}-{month:02}",
                    _constraint_for_dates(start, end),
                    id=f"month_{year}_{month:02}",
                )
            )
            for day in (1, monthrange(year, month)[1]):
                start = date(year, month, day)
                cases.append(
                    pytest.param(
                        start.isoformat(),
                        _constraint_for_dates(start, start + timedelta(days=1)),
                        id=f"day_{start.isoformat()}",
                    )
                )
    return cases


def _constraint_for_dates(start: date, end: date) -> TemporalConstraint:
    """Return the interval corresponding to independently calculated dates."""
    return TemporalConstraint.during(start, end)


@pytest.mark.parametrize(
    ("expression", "constraint"),
    _calendar_cases(),
)
def test_claim_temporal_resolution_calendar_period(
    expression: str, constraint: TemporalConstraint
) -> None:
    """An explicit calendar expression resolves to precisely its stated period.

    Guards the resolver from replacing a calendar period with a fabricated
    instant or a broader interval than the source justifies.
    """
    outcome = CalendarTemporalResolver().resolve(
        TemporalResolutionContext(_expression(expression))
    )

    assert isinstance(outcome, ResolvedTemporalExpression)
    assert outcome.constraint == constraint


def test_claim_temporal_resolution_unresolved_relative_expression() -> None:
    """A relative expression remains unresolved when its reference date is absent.

    Guards temporal resolution from fabricating a calendar date for source
    language that needs context the resolver has not received.
    """
    outcome = CalendarTemporalResolver().resolve(
        TemporalResolutionContext(_expression("next April"))
    )

    assert isinstance(outcome, UnresolvedTemporalExpression)
    assert outcome.expression.text == "next April"
    assert outcome.expression.evidence == (_evidence("next April"),)


def test_claim_temporal_resolution_relative_expression_reference_time() -> None:
    """A reference time resolves a relative expression to the next calendar month.

    Guards contextual temporal resolution from ignoring the supplied reference
    time or choosing a month before that reference.
    """
    outcome = CalendarTemporalResolver().resolve(
        TemporalResolutionContext(
            _expression("next April"),
            datetime(2026, 3, 31, 23, 30, tzinfo=timezone(timedelta(hours=-1))),
        )
    )

    assert isinstance(outcome, ResolvedTemporalExpression)
    assert outcome.constraint == TemporalConstraint.during(
        date(2026, 4, 1),
        date(2026, 5, 1),
    )


def test_claim_temporal_resolution_following_month_reference_time() -> None:
    """The following April resolves without model judgement when a reference exists.

    Guards the hybrid resolver from sending an equivalent calendar expression to
    an external model when deterministic calculation already has sufficient data.
    """
    outcome = CalendarTemporalResolver().resolve(
        TemporalResolutionContext(
            _expression("the following April"), datetime(2026, 3, 1, tzinfo=UTC)
        )
    )

    assert isinstance(outcome, ResolvedTemporalExpression)
    assert outcome.constraint == TemporalConstraint.during(
        date(2026, 4, 1),
        date(2026, 5, 1),
    )


def test_claim_temporal_resolution_rejects_unlinked_constraint_evidence(
    tmp_path: Path,
) -> None:
    """A model cannot combine one record's Evidence with another record's time.

    Guards temporal resolution from returning a comparable constraint that the
    model has not tied to its retained supporting Evidence.
    """
    first = _evidence("The scheme starts in April.")
    second = _evidence("The scheme ends in May.")
    service = TemporalResolutionService(
        ArtefactStore(tmp_path / "artefacts"),
        StaticModel(
            {
                "constraint": TemporalConstraint.during(
                    date(2026, 5, 1), date(2026, 6, 1)
                ).as_json(),
                "evidence": [first.as_json()],
                "kind": "resolved",
                "rationale": "The source establishes the later period.",
            }
        ),
        ModelConfiguration("openrouter", "example/model", {"temperature": 0}),
        maximum_attempts=1,
    )
    request = service.create_request(
        TemporalResolutionContext(
            _expression("after the announcement"),
            source_context=(
                {
                    "constraint": TemporalConstraint.during(
                        date(2026, 4, 1), date(2026, 5, 1)
                    ).as_json(),
                    "evidence": [first.as_json()],
                },
                {
                    "constraint": TemporalConstraint.during(
                        date(2026, 5, 1), date(2026, 6, 1)
                    ).as_json(),
                    "evidence": [second.as_json()],
                },
            ),
            maximum_source_context=2,
        )
    )

    with pytest.raises(ValueError, match="failed validation"):
        service.resolve_request(request)


def test_claim_temporal_resolution_distinguishes_time_kinds() -> None:
    """A Claim retains publication, processing, event and applicability time apart.

    Guards Claim timing from conflating source publication with system processing
    or replacing an unresolved event expression with applicability time.
    """
    publication = ResolvedTemporalExpression(
        _expression("published on 1 January 2026"),
        TemporalConstraint.exactly(datetime(2026, 1, 1, tzinfo=UTC)),
        "source-metadata",
        "The source gives its publication date.",
    )
    processing = ResolvedTemporalExpression(
        _expression("processed on 2 January 2026"),
        TemporalConstraint.exactly(datetime(2026, 1, 2, tzinfo=UTC)),
        "system-clock",
        "The system recorded its processing date.",
    )
    event = UnresolvedTemporalExpression(
        _expression("next April"),
        "contextual-date",
        "The source does not establish a reference year.",
    )
    applicability = ResolvedTemporalExpression(
        _expression("during 2026"),
        TemporalConstraint.during(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2027, 1, 1, tzinfo=UTC),
        ),
        "calendar-year",
        "The language names the whole calendar year.",
    )
    claim = CandidateClaim(
        "The scheme is available.",
        0.9,
        (_evidence("The scheme is available."),),
        "The source states availability.",
        ClaimTimes(publication, processing, event, applicability),
    )

    assert claim.times.publication == publication
    assert claim.times.processing == processing
    assert claim.times.event == event
    assert claim.times.applicability == applicability


def test_claim_temporal_resolution_retains_unresolved_expression() -> None:
    """An unresolved Claim time retains its source language after serialisation.

    Guards later reconsideration from losing the original expression or Evidence
    while preserving the explicit unresolved outcome.
    """
    times = ClaimTimes(
        event=UnresolvedTemporalExpression(
            _expression("next April"),
            "contextual-date",
            "The source does not establish a reference year.",
        )
    )

    reloaded = claim_times_from_json(times.as_json(), _expression_from_json)

    assert reloaded.as_json()["event"] == {
        "expression": {
            "evidence": [_evidence("next April").as_json()],
            "text": "next April",
        },
        "kind": "unresolved",
        "methodology": "contextual-date",
        "rationale": "The source does not establish a reference year.",
    }


def _expression(text: str) -> ExtractedTemporalExpression:
    """Return an evidence-backed temporal expression for one Claim."""
    return ExtractedTemporalExpression((_evidence(text),), text)


def _evidence(text: str) -> EvidenceAnchor:
    """Return Evidence for the temporal expression fixture."""
    return EvidenceAnchor(
        canonical_text_sha256="a" * 64,
        canonicaliser_version="1",
        content_id="source-id",
        end_line=1,
        end_offset=len(text),
        prefix="",
        selected_text=text,
        source_json_sha256="b" * 64,
        source_url="https://www.gov.uk/example",
        start_line=1,
        start_offset=0,
        suffix="",
    )


def _expression_from_json(value: object) -> ExtractedTemporalExpression:
    """Recreate one temporal expression without altering its retained Evidence."""
    assert isinstance(value, dict)
    evidence = value["evidence"]
    text = value["text"]
    assert isinstance(evidence, list)
    assert isinstance(text, str)
    return ExtractedTemporalExpression(
        tuple(EvidenceAnchor.from_json(anchor) for anchor in evidence), text
    )
