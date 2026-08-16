"""Tests for source document acquisition commands."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from ai_gov_graph.acquire import app
from typer.testing import CliRunner


class RecordedResponse:
    """One recorded response from the external GOV.UK API boundary."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> RecordedResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        """Return the recorded response bytes exactly as supplied."""
        return self.body


def test_source_document_acquisition_manifest_order(tmp_path: Path) -> None:
    """A corpus manifest orders source documents by base path.

    Guards against acquisition inheriting an unstable search or filesystem order.
    """
    alpha = (
        b'{"base_path":"/alpha","content_id":"alpha-id","locale":"en",'
        b'"updated_at":"2026-08-16T10:00:00.000+00:00"}'
    )
    zebra = (
        b'{"base_path":"/zebra","content_id":"zebra-id","locale":"en",'
        b'"updated_at":"2026-08-16T11:00:00.000+00:00"}'
    )
    search = json.dumps(
        {
            "results": [
                {"link": "/zebra"},
                {"link": "/alpha"},
            ],
            "start": 0,
            "total": 2,
        }
    ).encode()
    responses = iter(
        [RecordedResponse(search), RecordedResponse(alpha), RecordedResponse(zebra)]
    )
    corpus_directory = tmp_path / "corpus"

    with patch("ai_gov_graph.acquire.urlopen", side_effect=responses):
        result = CliRunner().invoke(
            app,
            [
                "documents",
                "fetch",
                "--corpus-directory",
                str(corpus_directory),
                "--organisation",
                "department-for-business-and-trade",
                "--maximum",
                "2",
            ],
        )

    assert result.exit_code == 0, result.output
    manifest = json.loads((corpus_directory / "manifest.json").read_text())
    assert [
        (document["sequence"], document["base_path"])
        for document in manifest["source_documents"]
    ] == [(0, "/alpha"), (1, "/zebra")]


def test_source_document_acquisition_source_version(tmp_path: Path) -> None:
    """A source document version retains the exact Content API response bytes.

    Guards evidence against being attributed to changed or reserialised source JSON.
    """
    source = (
        b'{"base_path":"/alpha","content_id":"alpha-id","locale":"en",'
        b'"updated_at":"2026-08-16T10:00:00.000+00:00"}'
    )
    search = json.dumps(
        {"results": [{"link": "/alpha"}], "start": 0, "total": 1}
    ).encode()
    responses = iter([RecordedResponse(search), RecordedResponse(source)])
    corpus_directory = tmp_path / "corpus"

    with patch("ai_gov_graph.acquire.urlopen", side_effect=responses):
        result = CliRunner().invoke(
            app,
            [
                "documents",
                "fetch",
                "--corpus-directory",
                str(corpus_directory),
                "--organisation",
                "department-for-business-and-trade",
                "--maximum",
                "1",
            ],
        )

    assert result.exit_code == 0, result.output
    source_hash = sha256(source).hexdigest()
    manifest = json.loads((corpus_directory / "manifest.json").read_text())
    document = manifest["source_documents"][0]
    assert document["source_json"] == f"source-documents/{source_hash}.json"
    assert document["source_json_sha256"] == source_hash
    assert document["source_version"] == f"content_id:alpha-id:sha256:{source_hash}"
    assert (
        corpus_directory / f"source-documents/{source_hash}.json"
    ).read_bytes() == source


def test_source_document_acquisition_canonical_text(tmp_path: Path) -> None:
    """A source document retains its canonical text and immutable identity.

    Guards later Evidence validation from relying on a text rendering that was
    absent from, or different to, the acquired corpus.
    """
    source = (
        b'{"base_path":"/alpha","content_id":"alpha-id","locale":"en",'
        b'"updated_at":"2026-08-16T10:00:00.000+00:00"}'
    )
    search = json.dumps(
        {"results": [{"link": "/alpha"}], "start": 0, "total": 1}
    ).encode()
    responses = iter([RecordedResponse(search), RecordedResponse(source)])
    corpus_directory = tmp_path / "corpus"

    with patch("ai_gov_graph.acquire.urlopen", side_effect=responses):
        result = CliRunner().invoke(
            app,
            [
                "documents",
                "fetch",
                "--corpus-directory",
                str(corpus_directory),
                "--organisation",
                "department-for-business-and-trade",
                "--maximum",
                "1",
            ],
        )

    assert result.exit_code == 0, result.output
    canonical_hash = "3d7b21d55cb48e4ae9d7ca9278ddd6771b8c9d0379f58764f5543805ad724ca1"
    manifest = json.loads((corpus_directory / "manifest.json").read_text())
    document = manifest["source_documents"][0]
    assert {
        "canonical_text": document["canonical_text"],
        "canonical_text_sha256": document["canonical_text_sha256"],
        "canonicaliser_version": document["canonicaliser_version"],
    } == {
        "canonical_text": f"canonical-documents/{canonical_hash}.txt",
        "canonical_text_sha256": canonical_hash,
        "canonicaliser_version": "1",
    }
    assert (corpus_directory / f"canonical-documents/{canonical_hash}.txt").read_text(
        encoding="utf-8"
    ) == (
        "/base_path:\n"
        "/alpha\n"
        "\n"
        "/content_id:\n"
        "alpha-id\n"
        "\n"
        "/locale:\n"
        "en\n"
        "\n"
        "/updated_at:\n"
        "2026-08-16T10:00:00.000+00:00\n"
    )


def test_source_document_acquisition_incomplete_download(tmp_path: Path) -> None:
    """A failed source download leaves a visible incomplete acquisition record.

    Guards against treating partially acquired evidence as a complete corpus.
    """
    alpha = (
        b'{"base_path":"/alpha","content_id":"alpha-id","locale":"en",'
        b'"updated_at":"2026-08-16T10:00:00.000+00:00"}'
    )
    search = json.dumps(
        {
            "results": [
                {"link": "/alpha"},
                {"link": "/zebra"},
            ],
            "start": 0,
            "total": 2,
        }
    ).encode()
    responses = iter(
        [
            RecordedResponse(search),
            RecordedResponse(alpha),
            URLError("source unavailable"),
        ]
    )
    corpus_directory = tmp_path / "corpus"

    with patch("ai_gov_graph.acquire.urlopen", side_effect=responses):
        result = CliRunner().invoke(
            app,
            [
                "documents",
                "fetch",
                "--corpus-directory",
                str(corpus_directory),
                "--organisation",
                "department-for-business-and-trade",
                "--maximum",
                "2",
            ],
        )

    assert result.exit_code == 1
    assert "Acquisition incomplete. No corpus manifest was written." in result.output
    assert not (corpus_directory / "manifest.json").exists()
    failure = json.loads((corpus_directory / "acquisition-failure.json").read_text())
    assert failure == {
        "failures": [
            {
                "message": "<urlopen error source unavailable>",
                "source_path": "/zebra",
                "stage": "download",
            }
        ],
        "schema_version": "1",
        "status": "incomplete",
    }


def test_source_document_acquisition_existing_manifest(tmp_path: Path) -> None:
    """An existing corpus manifest is never replaced.

    Guards an already complete experiment from an accidental second acquisition.
    """
    corpus_directory = tmp_path / "corpus"
    corpus_directory.mkdir()
    manifest_path = corpus_directory / "manifest.json"
    manifest_path.write_text('{"status":"complete"}\n', encoding="utf-8")

    with patch("ai_gov_graph.acquire.urlopen") as request:
        result = CliRunner().invoke(
            app,
            [
                "documents",
                "fetch",
                "--corpus-directory",
                str(corpus_directory),
                "--organisation",
                "department-for-business-and-trade",
            ],
        )

    assert result.exit_code == 2
    assert "Corpus directory is not empty" in result.output
    assert manifest_path.read_text(encoding="utf-8") == '{"status":"complete"}\n'
    request.assert_not_called()
