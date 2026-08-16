"""Tests for canonical source text and Evidence anchors."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from ai_gov_graph.canonical import (
    CANONICALISER_VERSION,
    EvidenceAnchor,
    EvidenceValidationError,
    canonicalise_source_document,
    validate_canonical_text,
    validate_evidence_anchor,
)


def test_source_document_canonicalisation_deterministic() -> None:
    """A source version has one stable canonical text and hash.

    Guards Evidence positions against changes in JSON object order or platform
    line endings.
    """
    source = b'{"body":"<h1>Export goods</h1><p>Apply online.</p>","title":"Export"}'
    reordered_source = (
        b'{"title":"Export","body":"<h1>Export goods</h1><p>Apply online.</p>"}'
    )

    canonical = canonicalise_source_document(source)
    reordered_canonical = canonicalise_source_document(reordered_source)

    assert canonical.canonicaliser_version == CANONICALISER_VERSION
    assert canonical.text == (
        "/body:\n# Export goods\n\nApply online.\n\n/title:\nExport\n"
    )
    assert (
        canonical.sha256
        == "4537b95d4da91be86fb28e44f4af1825bb15d746ad8bdff70abb3694b706cedb"
    )
    assert reordered_canonical == canonical


@pytest.mark.parametrize(
    ("source", "expected_text"),
    (
        pytest.param(
            b'{"body":"<h2>Help</h2>"}',
            "/body:\n## Help\n",
            id="heading",
        ),
        pytest.param(
            b'{"body":"<ul><li>First</li><li>Second</li></ul>"}',
            "/body:\n- First\n\n- Second\n",
            id="list",
        ),
        pytest.param(
            b'{"body":"<a href=\'/apply\'>Apply</a>"}',
            "/body:\nApply </apply>\n",
            id="link",
        ),
        pytest.param(
            b'{"body":"<table><tr><th>Step</th><th>Action</th></tr></table>"}',
            "/body:\nStep | Action\n",
            id="table",
        ),
        pytest.param(b'{"body":"Caf\\u00e9"}', "/body:\nCafé\n", id="unicode"),
        pytest.param(
            b'{"body":"Apply online."}',
            "/body:\nApply online.\n",
            id="absent-fields",
        ),
    ),
)
def test_source_document_canonicalisation_content_feature(
    source: bytes, expected_text: str
) -> None:
    """A GOV.UK content feature has a stable readable canonical rendering.

    Guards each common source-content feature against losing its readable text or
    stable Evidence position during canonicalisation.
    """
    canonical = canonicalise_source_document(source)

    assert canonical.text == expected_text


def test_evidence_anchor_validation_exact_passage() -> None:
    """An Evidence anchor resolves to its selected text and context.

    Guards a candidate Claim against being attributed to a nearby passage in the
    same Source document.
    """
    source = (
        b'{"base_path":"/export","body":"<h1>Export goods</h1><p>Apply online.</p>",'
        b'"content_id":"export-id","title":"Export"}'
    )
    canonical = canonicalise_source_document(source)
    anchor = EvidenceAnchor(
        canonical_text_sha256=canonical.sha256,
        canonicaliser_version=CANONICALISER_VERSION,
        content_id="export-id",
        end_line=6,
        end_offset=57,
        prefix="# Export goods\n\n",
        selected_text="Apply online.",
        source_json_sha256=sha256(source).hexdigest(),
        source_url="https://www.gov.uk/export",
        start_line=6,
        start_offset=44,
        suffix="\n\n/content_id:\nexport-id\n\n/title:\nExport\n",
    )

    assert validate_evidence_anchor(anchor, source) == canonical


def test_evidence_anchor_validation_repeated_text() -> None:
    """An anchor uses context to locate the intended repeated passage.

    Guards a Claim from being attributed to the first matching quotation when the
    supporting text occurs more than once in its Source document.
    """
    source = (
        b'{"base_path":"/apply","body":"<p>Apply now.</p><p>Apply now.</p>",'
        b'"content_id":"apply-id"}'
    )
    canonical = canonicalise_source_document(source)
    anchor = EvidenceAnchor(
        canonical_text_sha256=canonical.sha256,
        canonicaliser_version=CANONICALISER_VERSION,
        content_id="apply-id",
        end_line=6,
        end_offset=49,
        prefix="Apply now.\n\n",
        selected_text="Apply now.",
        source_json_sha256=sha256(source).hexdigest(),
        source_url="https://www.gov.uk/apply",
        start_line=6,
        start_offset=39,
        suffix="\n\n/content_id:\napply-id\n",
    )

    assert validate_evidence_anchor(anchor, source) == canonical


def test_canonical_text_validation_rejects_altered_text() -> None:
    """A retained canonical artefact must be the declared rendering of its Source.

    Guards Evidence anchors from passing against a text file altered after corpus
    acquisition.
    """
    source = b'{"body":"Apply online."}'
    canonical = canonicalise_source_document(source)
    altered_text = canonical.text.replace("online", "offline").encode("utf-8")

    with pytest.raises(EvidenceValidationError, match="canonical_text"):
        validate_canonical_text(
            source,
            altered_text,
            sha256(altered_text).hexdigest(),
            CANONICALISER_VERSION,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param("source_json_sha256", "not-a-hash", id="source-json-hash"),
        pytest.param("canonical_text_sha256", "not-a-hash", id="canonical-text-hash"),
        pytest.param("start_offset", 22, id="offset"),
        pytest.param("end_offset", 56, id="end-offset"),
        pytest.param("start_line", 2, id="start-line"),
        pytest.param("selected_text", "Apply offline.", id="quotation"),
        pytest.param("prefix", "wrong context", id="prefix-context"),
        pytest.param("suffix", "wrong context", id="suffix-context"),
        pytest.param("content_id", "another-document", id="content-identifier"),
        pytest.param("source_url", "https://www.gov.uk/another", id="source-url"),
        pytest.param("canonicaliser_version", "999", id="canonicaliser-version"),
    ),
)
def test_evidence_anchor_validation_rejects_invalid_fields(
    field: str, value: int | str
) -> None:
    """Invalid anchor identities or passages fail with their responsible field.

    Guards later Claim processing from accepting stale, malformed or altered
    Evidence.
    """
    source = (
        b'{"base_path":"/export","body":"<h1>Export goods</h1><p>Apply online.</p>",'
        b'"content_id":"export-id","title":"Export"}'
    )
    canonical = canonicalise_source_document(source)
    valid_anchor = EvidenceAnchor(
        canonical_text_sha256=canonical.sha256,
        canonicaliser_version=CANONICALISER_VERSION,
        content_id="export-id",
        end_line=6,
        end_offset=57,
        prefix="# Export goods\n\n",
        selected_text="Apply online.",
        source_json_sha256=sha256(source).hexdigest(),
        source_url="https://www.gov.uk/export",
        start_line=6,
        start_offset=44,
        suffix="\n\n/content_id:\nexport-id\n\n/title:\nExport\n",
    )

    with pytest.raises(EvidenceValidationError, match=field):
        validate_evidence_anchor(replace(valid_anchor, **{field: value}), source)
