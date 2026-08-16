"""Extract and retain open candidate Claims for immutable Source versions."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from ai_gov_graph.artefacts import ArtefactReference, ArtefactStore, JsonValue
from ai_gov_graph.canonical import (
    EvidenceAnchor,
    EvidenceValidationError,
    canonicalise_source_document,
    validate_evidence_anchor,
)
from ai_gov_graph.reasoning import (
    ModelConfiguration,
    ReasoningRunner,
    StructuredModel,
)

OPEN_EXTRACTION_STAGE = "open-claim-extraction"


class OpenExtractionValidationError(ValueError):
    """Raised when an open-extraction result is not a valid durable contract."""


@dataclass(frozen=True)
class SourceVersion:
    """The exact local Source document version available to open extraction."""

    canonical_text: str
    canonical_text_sha256: str
    canonicaliser_version: str
    content_id: str
    source_json: bytes
    source_json_sha256: str
    source_url: str

    @property
    def identity(self) -> str:
        """Return the immutable identity used to deduplicate extraction."""
        return f"content_id:{self.content_id}:sha256:{self.source_json_sha256}"


@dataclass(frozen=True)
class CandidateClaim:
    """One source-supported proposition before later graph decisions."""

    assertion: str
    confidence: float
    evidence: tuple[EvidenceAnchor, ...]
    rationale: str

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable candidate Claim representation."""
        return {
            "assertion": self.assertion,
            "confidence": self.confidence,
            "evidence": [anchor.as_json() for anchor in self.evidence],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ExtractedMention:
    """A source mention retained separately from later entity resolution."""

    evidence: tuple[EvidenceAnchor, ...]
    text: str

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable mention representation."""
        return {
            "evidence": [anchor.as_json() for anchor in self.evidence],
            "text": self.text,
        }


@dataclass(frozen=True)
class ExtractedTemporalExpression:
    """A temporal expression retained before temporal normalisation."""

    evidence: tuple[EvidenceAnchor, ...]
    text: str

    def as_json(self) -> dict[str, JsonValue]:
        """Return the durable temporal-expression representation."""
        return {
            "evidence": [anchor.as_json() for anchor in self.evidence],
            "text": self.text,
        }


@dataclass(frozen=True)
class OpenExtraction:
    """The single open-extraction result for one Source document version."""

    candidate_claims: tuple[CandidateClaim, ...]
    mentions: tuple[ExtractedMention, ...]
    reference: ArtefactReference
    source_version: str
    temporal_expressions: tuple[ExtractedTemporalExpression, ...]


class OpenExtractionService:
    """Extract open candidates once and reuse the retained Source-version result."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        *,
        maximum_attempts: int,
    ) -> None:
        """Create an extraction service with explicit reasoning retry bounds."""
        self._store = store
        self._runner = ReasoningRunner(
            store,
            model,
            configuration,
            maximum_attempts=maximum_attempts,
        )

    def extract(self, source: SourceVersion) -> OpenExtraction:
        """Return the one durable result for ``source``, without ontology input."""
        _validate_source_version(source)
        with self._source_lock(source):
            existing = self._load_existing(source)
            if existing is not None:
                return existing

            invocation = self._runner.run(
                stage=OPEN_EXTRACTION_STAGE,
                structured_input={
                    "canonical_text": source.canonical_text,
                    "source_version": source.identity,
                },
                validate_output=lambda value: _normalise_output(value, source),
            )
            output = cast(dict[str, JsonValue], invocation.output)
            reference = self._store.write_json(
                "open-extraction",
                {
                    "candidate_claims": output["candidate_claims"],
                    "mentions": output["mentions"],
                    "reasoning_invocation": _reference_data(invocation.reference),
                    "source_version": source.identity,
                    "temporal_expressions": output["temporal_expressions"],
                },
            )
            self._write_index(source, reference)
            return _parse_extraction(
                self._store.read_json(reference), reference, source
            )

    @contextmanager
    def _source_lock(self, source: SourceVersion) -> Iterator[None]:
        """Serialise all extraction work for one Source document version."""
        lock_path = self._lock_path(source)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load_existing(self, source: SourceVersion) -> OpenExtraction | None:
        """Load a verified prior extraction record, if this version has one."""
        index_path = self._index_path(source)
        try:
            raw_index = index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            index = json.loads(raw_index)
        except json.JSONDecodeError as error:
            msg = f"Open-extraction index is not valid JSON: {index_path}."
            raise OpenExtractionValidationError(msg) from error
        if not isinstance(index, dict) or set(index) != {"reference", "source_version"}:
            msg = f"Open-extraction index has an invalid shape: {index_path}."
            raise OpenExtractionValidationError(msg)
        if index["source_version"] != source.identity:
            msg = (
                f"Open-extraction index has a mismatched Source version: {index_path}."
            )
            raise OpenExtractionValidationError(msg)
        reference = _parse_reference(index["reference"])
        return _parse_extraction(self._store.read_json(reference), reference, source)

    def _write_index(self, source: SourceVersion, reference: ArtefactReference) -> None:
        """Create the immutable lookup that prevents a second extraction call."""
        path = self._index_path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            {
                "reference": _reference_data(reference),
                "source_version": source.identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with path.open("x", encoding="utf-8") as index:
                index.write(content)
        except FileExistsError as error:
            msg = f"Open-extraction index appeared while locked: {path}."
            raise OpenExtractionValidationError(msg) from error

    def _index_path(self, source: SourceVersion) -> Path:
        """Return the stable lookup path for one Source version identity."""
        identity_hash = sha256(source.identity.encode("utf-8")).hexdigest()
        return self._store.root / "open-extraction-index" / f"{identity_hash}.json"

    def _lock_path(self, source: SourceVersion) -> Path:
        """Return the inter-process lock path for one Source version identity."""
        identity_hash = sha256(source.identity.encode("utf-8")).hexdigest()
        return self._store.root / "open-extraction-lock" / f"{identity_hash}.lock"


def _validate_source_version(source: SourceVersion) -> None:
    """Verify that extraction is bound to exactly the supplied source version."""
    if sha256(source.source_json).hexdigest() != source.source_json_sha256:
        msg = "Source JSON does not match its declared Source version hash."
        raise OpenExtractionValidationError(msg)
    canonical = canonicalise_source_document(
        source.source_json, source.canonicaliser_version
    )
    if canonical.text != source.canonical_text:
        msg = "Canonical text does not match the retained Source JSON."
        raise OpenExtractionValidationError(msg)
    if canonical.sha256 != source.canonical_text_sha256:
        msg = "Canonical text hash does not match the retained Source JSON."
        raise OpenExtractionValidationError(msg)


def _normalise_output(value: JsonValue, source: SourceVersion) -> JsonValue:
    """Validate and canonicalise structured open-extraction model output."""
    if not isinstance(value, dict) or set(value) != {
        "candidate_claims",
        "mentions",
        "temporal_expressions",
    }:
        msg = (
            "Open extraction must contain candidate_claims, mentions and "
            "temporal_expressions."
        )
        raise OpenExtractionValidationError(msg)
    candidate_claims = _parse_candidate_claims(value["candidate_claims"], source)
    mentions = _parse_mentions(value["mentions"], source)
    temporal_expressions = _parse_temporal_expressions(
        value["temporal_expressions"], source
    )
    return {
        "candidate_claims": [claim.as_json() for claim in candidate_claims],
        "mentions": [mention.as_json() for mention in mentions],
        "temporal_expressions": [
            expression.as_json() for expression in temporal_expressions
        ],
    }


def _parse_extraction(
    value: JsonValue, reference: ArtefactReference, source: SourceVersion
) -> OpenExtraction:
    """Parse one verified extraction artefact into the public typed result."""
    if not isinstance(value, dict) or set(value) != {
        "candidate_claims",
        "mentions",
        "reasoning_invocation",
        "source_version",
        "temporal_expressions",
    }:
        msg = f"Open-extraction artefact {reference.identity} has an invalid shape."
        raise OpenExtractionValidationError(msg)
    if value["source_version"] != source.identity:
        msg = (
            f"Open-extraction artefact {reference.identity} has a mismatched "
            "Source version."
        )
        raise OpenExtractionValidationError(msg)
    _parse_reference(value["reasoning_invocation"])
    return OpenExtraction(
        candidate_claims=_parse_candidate_claims(value["candidate_claims"], source),
        mentions=_parse_mentions(value["mentions"], source),
        reference=reference,
        source_version=source.identity,
        temporal_expressions=_parse_temporal_expressions(
            value["temporal_expressions"], source
        ),
    )


def _parse_candidate_claims(
    value: JsonValue, source: SourceVersion
) -> tuple[CandidateClaim, ...]:
    """Parse and validate all typed candidate Claims from model output."""
    if not isinstance(value, list):
        msg = "Open extraction candidate_claims must be a list."
        raise OpenExtractionValidationError(msg)
    claims: list[CandidateClaim] = []
    for claim in value:
        if not isinstance(claim, dict) or set(claim) != {
            "assertion",
            "confidence",
            "evidence",
            "rationale",
        }:
            msg = (
                "Each candidate Claim has assertion, confidence, evidence and "
                "rationale."
            )
            raise OpenExtractionValidationError(msg)
        confidence = claim["confidence"]
        if (
            not isinstance(confidence, int | float)
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            msg = "Candidate Claim confidence must be a number from 0 to 1."
            raise OpenExtractionValidationError(msg)
        claims.append(
            CandidateClaim(
                assertion=_non_empty_string(
                    claim["assertion"], "Candidate Claim assertion"
                ),
                confidence=float(confidence),
                evidence=_parse_evidence(claim["evidence"], source),
                rationale=_non_empty_string(
                    claim["rationale"], "Candidate Claim rationale"
                ),
            )
        )
    return tuple(claims)


def _parse_mentions(
    value: JsonValue, source: SourceVersion
) -> tuple[ExtractedMention, ...]:
    """Parse and validate mentions without resolving their identity."""
    if not isinstance(value, list):
        msg = "Open extraction mentions must be a list."
        raise OpenExtractionValidationError(msg)
    mentions: list[ExtractedMention] = []
    for mention in value:
        if not isinstance(mention, dict) or set(mention) != {"evidence", "text"}:
            msg = "Each mention has text and evidence."
            raise OpenExtractionValidationError(msg)
        mentions.append(
            ExtractedMention(
                evidence=_parse_evidence(mention["evidence"], source),
                text=_non_empty_string(mention["text"], "Mention text"),
            )
        )
    return tuple(mentions)


def _parse_temporal_expressions(
    value: JsonValue, source: SourceVersion
) -> tuple[ExtractedTemporalExpression, ...]:
    """Parse temporal language without assigning a normalised time value."""
    if not isinstance(value, list):
        msg = "Open extraction temporal_expressions must be a list."
        raise OpenExtractionValidationError(msg)
    expressions: list[ExtractedTemporalExpression] = []
    for expression in value:
        if not isinstance(expression, dict) or set(expression) != {
            "evidence",
            "text",
        }:
            msg = "Each temporal expression has text and evidence."
            raise OpenExtractionValidationError(msg)
        expressions.append(
            ExtractedTemporalExpression(
                evidence=_parse_evidence(expression["evidence"], source),
                text=_non_empty_string(expression["text"], "Temporal expression text"),
            )
        )
    return tuple(expressions)


def _parse_evidence(
    value: JsonValue, source: SourceVersion
) -> tuple[EvidenceAnchor, ...]:
    """Parse anchors and ensure they support this exact Source version."""
    if not isinstance(value, list) or not value:
        msg = "Open extraction evidence must be a non-empty list."
        raise OpenExtractionValidationError(msg)
    anchors: list[EvidenceAnchor] = []
    for raw_anchor in value:
        try:
            anchor = EvidenceAnchor.from_json(raw_anchor)
            validate_evidence_anchor(anchor, source.source_json)
        except EvidenceValidationError as error:
            raise OpenExtractionValidationError(str(error)) from error
        if (
            anchor.canonical_text_sha256 != source.canonical_text_sha256
            or anchor.canonicaliser_version != source.canonicaliser_version
            or anchor.content_id != source.content_id
            or anchor.source_json_sha256 != source.source_json_sha256
            or anchor.source_url != source.source_url
        ):
            msg = (
                "Open extraction Evidence does not identify the supplied Source "
                "version."
            )
            raise OpenExtractionValidationError(msg)
        anchors.append(anchor)
    return tuple(anchors)


def _non_empty_string(value: JsonValue, field: str) -> str:
    """Require a meaningful text field in a structured extraction result."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{field} must be a non-empty string."
        raise OpenExtractionValidationError(msg)
    return value


def _reference_data(reference: ArtefactReference) -> dict[str, str]:
    """Return one durable artefact reference in its JSON representation."""
    return {
        "identity": reference.identity,
        "kind": reference.kind,
        "schema_version": reference.schema_version,
    }


def _parse_reference(value: JsonValue) -> ArtefactReference:
    """Parse a durable artefact reference before following it."""
    if not isinstance(value, dict) or set(value) != {
        "identity",
        "kind",
        "schema_version",
    }:
        msg = "Open extraction contains an invalid artefact reference."
        raise OpenExtractionValidationError(msg)
    identity = value["identity"]
    kind = value["kind"]
    schema_version = value["schema_version"]
    if not all(isinstance(field, str) for field in (identity, kind, schema_version)):
        msg = "Open extraction artefact reference fields must be strings."
        raise OpenExtractionValidationError(msg)
    if not identity.startswith("sha256:") or len(identity) != 71:
        msg = "Open extraction artefact reference has an invalid identity."
        raise OpenExtractionValidationError(msg)
    return ArtefactReference(kind, identity.removeprefix("sha256:"), schema_version)
