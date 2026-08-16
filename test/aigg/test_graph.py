"""Tests for graph construction commands."""

import json
from hashlib import sha256
from pathlib import Path

from aigg.graph import app
from pyoxigraph import NamedNode, QuerySolutions, RdfFormat, Store
from typer.testing import CliRunner


def test_experiment_lineage_configuration(tmp_path: Path) -> None:
    """An operator can inspect configuration.

    Guards the lineage's reproducible start.
    """
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(
        json.dumps({"reasoning_mode": "exact-replay"}), encoding="utf-8"
    )
    lineage_path = tmp_path / "lineage"

    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "initialise",
            "--lineage-directory",
            str(lineage_path),
            "--configuration",
            str(configuration_path),
        ],
    )

    assert result.exit_code == 0, result.output
    lineage = json.loads((lineage_path / "lineage.json").read_text(encoding="utf-8"))
    assert lineage == {
        "artefact_schema_version": "1",
        "configuration": {
            "identity": (
                "sha256:642a0ca2ac943fb20038a8dcb4f2bcf4237ccd00450256cb6c744b14515f1d5e"
            ),
            "kind": "configuration",
            "schema_version": "1",
        },
        "lineage_schema_version": "1",
    }


def test_source_document_graph_projection_one_version(tmp_path: Path) -> None:
    """A Source document version becomes an RDF resource with its source assertions.

    Guards the document projection from retaining source evidence only outside the
    RDF dataset.
    """
    source: dict[str, object] = {
        "base_path": "/guidance/example",
        "content_id": "document-id",
        "document_type": "detailed_guide",
        "locale": "en",
        "schema_name": "guide",
        "title": "Example guidance",
        "updated_at": "2026-08-16T10:00:00.000+00:00",
    }
    corpus_directory = _write_corpus(tmp_path, source)
    dataset_path = tmp_path / "source-documents.trig"

    result = CliRunner().invoke(
        app,
        [
            "documents",
            "run",
            "--corpus-directory",
            str(corpus_directory),
            "--dataset-path",
            str(dataset_path),
        ],
    )

    assert result.exit_code == 0, result.output
    store = Store()
    store.load(path=dataset_path, format=RdfFormat.TRIG)
    document = NamedNode(
        "https://w3id.org/aigg/source-document/document-id/"
        f"{sha256(_source_bytes(source)).hexdigest()}"
    )
    source_assertions = store.query(
        """
        PREFIX aigg: <https://w3id.org/aigg/>
        SELECT ?key ?value WHERE {
            GRAPH ?graph {
                ?assertion aigg:about <"""
        + document.value
        + """> ;
                    aigg:sourceKey ?key ;
                    aigg:sourceValue ?value .
            }
        }
        """
    )
    assert isinstance(source_assertions, QuerySolutions)
    rows = list(source_assertions)
    documents = store.query(
        """
        PREFIX aigg: <https://w3id.org/aigg/>
        SELECT ?document WHERE {
            GRAPH ?graph { ?document a aigg:SourceDocument . }
        }
        """
    )
    assert isinstance(documents, QuerySolutions)
    assert document in [row["document"] for row in documents]
    assert {(row["key"].value, row["value"].value) for row in rows} >= {
        ("document_type", '"detailed_guide"'),
        ("schema_name", '"guide"'),
    }


def test_source_document_graph_projection_source_classifications(
    tmp_path: Path,
) -> None:
    """Source classifications are retained as values rather than semantic types.

    Guards the source vocabulary boundary from treating GOV.UK classifications as
    emergent ontology classes.
    """
    source: dict[str, object] = {
        "base_path": "/guidance/example",
        "content_id": "document-id",
        "document_type": "detailed_guide",
        "links": {"organisations": [{"content_id": "organisation-id"}]},
        "locale": "en",
        "schema_name": "guide",
        "taxons": [{"base_path": "/topic/example"}],
        "updated_at": "2026-08-16T10:00:00.000+00:00",
    }
    corpus_directory = _write_corpus(tmp_path, source)
    dataset_path = tmp_path / "source-documents.trig"

    result = CliRunner().invoke(
        app,
        [
            "documents",
            "run",
            "--corpus-directory",
            str(corpus_directory),
            "--dataset-path",
            str(dataset_path),
        ],
    )

    assert result.exit_code == 0, result.output
    store = Store()
    store.load(path=dataset_path, format=RdfFormat.TRIG)
    assert not bool(
        store.query(
            """
            PREFIX aigg: <https://w3id.org/aigg/>
            PREFIX source: <https://w3id.org/aigg/source-classification/>
            ASK {
                GRAPH ?graph {
                    ?document a source:detailed_guide .
                }
            }
            """
        )
    )
    source_assertions = store.query(
        """
        PREFIX aigg: <https://w3id.org/aigg/>
        SELECT ?key ?value WHERE {
            GRAPH ?graph {
                ?assertion aigg:sourceKey ?key ; aigg:sourceValue ?value .
            }
        }
        """
    )
    assert isinstance(source_assertions, QuerySolutions)
    values = list(source_assertions)
    assert {(row["key"].value, row["value"].value) for row in values} >= {
        ("document_type", '"detailed_guide"'),
        ("links", '{"organisations":[{"content_id":"organisation-id"}]}'),
        ("schema_name", '"guide"'),
        ("taxons", '[{"base_path":"/topic/example"}]'),
    }


def _write_corpus(tmp_path: Path, source: dict[str, object]) -> Path:
    """Create one complete, manifest-backed source corpus fixture."""
    corpus_directory = tmp_path / "corpus"
    source_bytes = _source_bytes(source)
    source_hash = sha256(source_bytes).hexdigest()
    source_path = corpus_directory / "source-documents" / f"{source_hash}.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_bytes)
    manifest = {
        "schema_version": "1",
        "source_documents": [
            {
                "base_path": source["base_path"],
                "canonical_text": "canonical-documents/example.txt",
                "canonical_text_sha256": "canonical-hash",
                "canonicaliser_version": "1",
                "content_api_url": "https://www.gov.uk/api/content/guidance/example",
                "content_id": source["content_id"],
                "locale": source["locale"],
                "sequence": 0,
                "source_json": f"source-documents/{source_hash}.json",
                "source_json_sha256": source_hash,
                "source_version": (
                    f"content_id:{source['content_id']}:sha256:{source_hash}"
                ),
                "updated_at": source["updated_at"],
            }
        ],
        "status": "complete",
    }
    (corpus_directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return corpus_directory


def _source_bytes(source: dict[str, object]) -> bytes:
    """Serialise a fixture source document as its retained bytes."""
    return json.dumps(source, separators=(",", ":")).encode()
