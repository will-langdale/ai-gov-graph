"""Tests for one-pass open candidate Claim extraction."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Thread
from time import sleep
from typing import cast

from aigg.artefacts import ArtefactStore, JsonValue
from aigg.canonical import EvidenceAnchor, canonicalise_source_document
from aigg.open_extraction import (
    CandidateClaim,
    OpenExtraction,
    OpenExtractionService,
    SourceVersion,
)
from aigg.reasoning import ModelConfiguration, StructuredModel


@dataclass
class StaticModel(StructuredModel):
    """Control the external, non-deterministic model boundary in tests."""

    output: JsonValue
    calls: list[JsonValue]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the configured output."""
        self.calls.append(structured_input)
        return self.output


@dataclass
class DelayedModel(StructuredModel):
    """Keep one model call active while concurrent extraction begins."""

    output: JsonValue
    calls: list[JsonValue]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Delay the configured output at the external model boundary."""
        self.calls.append(structured_input)
        sleep(0.1)
        return self.output


def test_claim_extraction_source_document_one_pass(tmp_path: Path) -> None:
    """A Source document version reuses its original extraction result.

    Guards later reconsideration from silently creating a new candidate Claim
    from the same source prose.

    Uses a static model because model invocation is an external,
    non-deterministic boundary.
    """
    source_json = _source_json()
    source = _source_version(source_json)
    model = StaticModel(
        {
            "candidate_claims": [
                {
                    "assertion": "The department has a minister.",
                    "confidence": 0.9,
                    "evidence": [_evidence(source_json).as_json()],
                    "rationale": "The source states this directly.",
                }
            ],
            "mentions": [
                {"text": "department", "evidence": [_evidence(source_json).as_json()]}
            ],
            "temporal_expressions": [
                {
                    "text": "2026",
                    "evidence": [_temporal_evidence(source_json).as_json()],
                }
            ],
        },
        [],
    )
    store = ArtefactStore(tmp_path / "artefacts")
    service = OpenExtractionService(
        store,
        model,
        ModelConfiguration("example", "example-1", {"temperature": 0}),
        maximum_attempts=1,
    )

    extraction = service.extract(source)
    reconsideration_model = StaticModel(
        {"candidate_claims": [], "mentions": [], "temporal_expressions": []}, []
    )
    reconsidered = OpenExtractionService(
        store,
        reconsideration_model,
        ModelConfiguration("example", "example-1", {"temperature": 0}),
        maximum_attempts=1,
    ).extract(source)

    assert len(model.calls) == 1
    assert reconsideration_model.calls == []
    assert extraction == reconsidered


def test_claim_extraction_source_document_concurrent_one_pass(tmp_path: Path) -> None:
    """Concurrent requests invoke the model once for one Source document version.

    Guards simultaneous graph workers from creating competing candidate Claims
    from the same source prose. Uses a delayed model because model invocation is
    an external, non-deterministic boundary.
    """
    source_json = _source_json()
    source = _source_version(source_json)
    model = DelayedModel(_candidate_output(source_json), [])
    service = _service(tmp_path, model)
    start = Barrier(3)
    extractions: list[OpenExtraction] = []

    def extract() -> None:
        """Begin extraction with the other graph worker."""
        start.wait()
        extractions.append(service.extract(source))

    workers = [Thread(target=extract), Thread(target=extract)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join()

    assert len(model.calls) == 1
    assert extractions[0] == extractions[1]


def test_claim_extraction_candidate_claim(tmp_path: Path) -> None:
    """Open extraction returns a typed candidate Claim with its Evidence.

    Guards the candidate Claim contract from losing its assertion, confidence,
    rationale or source support before later graph decisions.

    Uses a static model because model invocation is an external,
    non-deterministic boundary.
    """
    source_json = _source_json()
    model = StaticModel(
        {
            "candidate_claims": [
                {
                    "assertion": "The department has a minister.",
                    "confidence": 0.9,
                    "evidence": [_evidence(source_json).as_json()],
                    "rationale": "The source states this directly.",
                }
            ],
            "mentions": [],
            "temporal_expressions": [],
        },
        [],
    )

    extraction = _service(tmp_path, model).extract(_source_version(source_json))

    assert extraction.candidate_claims == (
        CandidateClaim(
            assertion="The department has a minister.",
            confidence=0.9,
            evidence=(_evidence(source_json),),
            rationale="The source states this directly.",
        ),
    )


def test_claim_extraction_mention(tmp_path: Path) -> None:
    """Open extraction retains a source mention without resolving its Entity.

    Guards a later Resolver from having to recover an extracted mention from a
    candidate Claim assertion.

    Uses a static model because model invocation is an external,
    non-deterministic boundary.
    """
    source_json = _source_json()
    model = StaticModel(
        {
            "candidate_claims": [],
            "mentions": [
                {"text": "department", "evidence": [_evidence(source_json).as_json()]}
            ],
            "temporal_expressions": [],
        },
        [],
    )

    extraction = _service(tmp_path, model).extract(_source_version(source_json))

    assert extraction.mentions[0].text == "department"
    assert extraction.mentions[0].evidence == (_evidence(source_json),)


def test_claim_extraction_temporal_expression(tmp_path: Path) -> None:
    """Open extraction retains temporal language without normalising it.

    Guards later temporal resolution from needing to infer an expression that
    open extraction did not preserve.

    Uses a static model because model invocation is an external,
    non-deterministic boundary.
    """
    source_json = _source_json()
    model = StaticModel(
        {
            "candidate_claims": [],
            "mentions": [],
            "temporal_expressions": [
                {
                    "text": "2026",
                    "evidence": [_temporal_evidence(source_json).as_json()],
                }
            ],
        },
        [],
    )

    extraction = _service(tmp_path, model).extract(_source_version(source_json))

    assert extraction.temporal_expressions[0].text == "2026"
    assert extraction.temporal_expressions[0].evidence == (
        _temporal_evidence(source_json),
    )


def test_claim_extraction_source_document_no_candidates(tmp_path: Path) -> None:
    """A Source document version records an empty extraction result.

    Guards later processing from treating no candidate Claims as an unprocessed
    version that it may extract again.

    Uses a static model because model invocation is an external,
    non-deterministic boundary.
    """
    source_json = _source_json()
    source = _source_version(source_json)
    model = StaticModel(
        {"candidate_claims": [], "mentions": [], "temporal_expressions": []}, []
    )
    store = ArtefactStore(tmp_path / "artefacts")
    service = OpenExtractionService(
        store,
        model,
        ModelConfiguration("example", "example-1", {"temperature": 0}),
        maximum_attempts=1,
    )

    extraction = service.extract(source)
    repeated = service.extract(source)

    assert len(model.calls) == 1
    assert extraction == repeated
    assert extraction.candidate_claims == ()
    assert extraction.mentions == ()
    assert extraction.temporal_expressions == ()
    record = cast(dict[str, JsonValue], store.read_json(extraction.reference))
    assert record["candidate_claims"] == []


def _service(tmp_path: Path, model: StructuredModel) -> OpenExtractionService:
    """Create the open-extraction public seam with a static model boundary."""
    return OpenExtractionService(
        ArtefactStore(tmp_path / "artefacts"),
        model,
        ModelConfiguration("example", "example-1", {"temperature": 0}),
        maximum_attempts=1,
    )


def _candidate_output(source_json: bytes) -> JsonValue:
    """Return the smallest valid candidate Claim output for a model test double."""
    return {
        "candidate_claims": [
            {
                "assertion": "The department has a minister.",
                "confidence": 0.9,
                "evidence": [_evidence(source_json).as_json()],
                "rationale": "The source states this directly.",
            }
        ],
        "mentions": [],
        "temporal_expressions": [],
    }


def _source_json() -> bytes:
    """Return one retained Source document fixture."""
    return json.dumps(
        {
            "base_path": "/government/organisations/example",
            "content_id": "example-id",
            "details": {"body": "The department has a minister."},
            "locale": "en",
            "public_updated_at": "2026-08-16T11:00:00Z",
        }
    ).encode("utf-8")


def _source_version(source_json: bytes) -> SourceVersion:
    """Build the open-extraction input for the fixture Source document."""
    canonical = canonicalise_source_document(source_json)
    return SourceVersion(
        canonical_text=canonical.text,
        canonical_text_sha256=canonical.sha256,
        canonicaliser_version=canonical.canonicaliser_version,
        content_id="example-id",
        source_json=source_json,
        source_json_sha256=_sha256(source_json),
        source_url="https://www.gov.uk/government/organisations/example",
    )


def _evidence(source_json: bytes) -> EvidenceAnchor:
    """Return evidence for the fixture's only source-supported proposition."""
    canonical = canonicalise_source_document(source_json)
    selected_text = "The department has a minister."
    start_offset = canonical.text.index(selected_text)
    end_offset = start_offset + len(selected_text)
    return EvidenceAnchor(
        canonical_text_sha256=canonical.sha256,
        canonicaliser_version=canonical.canonicaliser_version,
        content_id="example-id",
        end_line=7,
        end_offset=end_offset,
        prefix="/details/body:\n",
        selected_text=selected_text,
        source_json_sha256=_sha256(source_json),
        source_url="https://www.gov.uk/government/organisations/example",
        start_line=7,
        start_offset=start_offset,
        suffix="\n",
    )


def _sha256(value: bytes) -> str:
    """Return the SHA-256 digest used in Source version identities."""
    return sha256(value).hexdigest()


def _temporal_evidence(source_json: bytes) -> EvidenceAnchor:
    """Return evidence for the temporal expression in the fixture source."""
    canonical = canonicalise_source_document(source_json)
    selected_text = "2026"
    start_offset = canonical.text.index(selected_text)
    end_offset = start_offset + len(selected_text)
    return EvidenceAnchor(
        canonical_text_sha256=canonical.sha256,
        canonicaliser_version=canonical.canonicaliser_version,
        content_id="example-id",
        end_line=13,
        end_offset=end_offset,
        prefix="/public_updated_at:\n",
        selected_text=selected_text,
        source_json_sha256=_sha256(source_json),
        source_url="https://www.gov.uk/government/organisations/example",
        start_line=13,
        start_offset=start_offset,
        suffix="-08-16T11:00:00Z\n",
    )
