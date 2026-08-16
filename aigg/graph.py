"""Commands for constructing a graph from a local evidence corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

import typer
from pyoxigraph import Literal, NamedNode, Quad, RdfFormat, Store

from aigg.artefacts import (
    ARTEFACT_SCHEMA_VERSION,
    ArtefactReference,
    ArtefactStore,
    JsonValue,
)

LINEAGE_SCHEMA_VERSION = "1"
RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDF_JSON = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON")
AIGG = "https://w3id.org/aigg/"
DOCUMENTS_GRAPH = NamedNode(f"{AIGG}graph/source-documents")
SOURCE_DOCUMENT = NamedNode(f"{AIGG}SourceDocument")
SOURCE_ASSERTION = NamedNode(f"{AIGG}SourceAssertion")
ABOUT = NamedNode(f"{AIGG}about")
SOURCE_KEY = NamedNode(f"{AIGG}sourceKey")
SOURCE_VALUE = NamedNode(f"{AIGG}sourceValue")
SOURCE_JSON = NamedNode(f"{AIGG}sourceJson")
SOURCE_VERSION = NamedNode(f"{AIGG}sourceVersion")


@dataclass(frozen=True)
class RetainedSourceDocument:
    """One manifest-listed source version and its unchanged JSON bytes."""

    content_id: str
    source: dict[str, object]
    source_bytes: bytes
    source_json_sha256: str
    source_version: str


app = typer.Typer(
    help="Run graph construction against locally acquired evidence.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(help="Create and inspect experiment lineages.")
app.add_typer(experiment_app, name="experiment")
documents_app = typer.Typer(help="Run graph construction against source documents.")
app.add_typer(documents_app, name="documents")


def initialise_lineage(
    lineage_directory: Path, configuration: dict[str, JsonValue]
) -> Path:
    """Create one lineage with a content-addressed configuration artefact."""
    if lineage_directory.exists() and any(lineage_directory.iterdir()):
        msg = f"Lineage directory is not empty: {lineage_directory}"
        raise ValueError(msg)

    store = ArtefactStore(lineage_directory / "artefacts")
    reference = store.write_json("configuration", configuration)
    manifest = {
        "artefact_schema_version": ARTEFACT_SCHEMA_VERSION,
        "configuration": _reference_data(reference),
        "lineage_schema_version": LINEAGE_SCHEMA_VERSION,
    }
    manifest_path = lineage_directory / "lineage.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


@experiment_app.command("initialise")
def initialise(
    lineage_directory: Annotated[
        Path,
        typer.Option(help="New directory that will hold the experiment lineage."),
    ],
    configuration: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON configuration recorded as the lineage's first artefact.",
        ),
    ],
) -> None:
    """Initialise a durable experiment lineage from a JSON configuration."""
    try:
        content = configuration.read_text(encoding="utf-8")
        parsed_configuration = json.loads(content)
    except json.JSONDecodeError as error:
        raise typer.BadParameter("Configuration must contain valid JSON.") from error
    if not isinstance(parsed_configuration, dict):
        raise typer.BadParameter("Configuration must be a JSON object.")

    try:
        manifest_path = initialise_lineage(lineage_directory, parsed_configuration)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Initialised experiment lineage: {manifest_path}")


@documents_app.command()
def run(
    corpus_directory: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            readable=True,
            help="Complete manifest-backed evidence corpus to project.",
        ),
    ],
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="TriG dataset file written from the retained source documents."
        ),
    ],
) -> None:
    """Project retained Source documents into a named RDF graph."""
    if dataset_path.resolve().is_relative_to(corpus_directory.resolve()):
        msg = "Dataset path must be outside the immutable evidence corpus."
        raise typer.BadParameter(msg, param_hint="--dataset-path")
    try:
        source_documents = _read_source_documents(corpus_directory)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--corpus-directory") from error

    store = Store()
    store.extend(_source_document_quads(source_documents))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(output=dataset_path, format=RdfFormat.TRIG)
    typer.echo(f"Projected source documents: {dataset_path}")


def _read_source_documents(
    corpus_directory: Path,
) -> list[RetainedSourceDocument]:
    """Read the retained source versions listed in one complete corpus manifest."""
    manifest_path = corpus_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except FileNotFoundError as error:
        msg = f"Corpus manifest is missing: {manifest_path}"
        raise ValueError(msg) from error
    except json.JSONDecodeError as error:
        msg = f"Corpus manifest is not valid JSON: {manifest_path}"
        raise ValueError(msg) from error

    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        msg = f"Corpus manifest is not complete: {manifest_path}"
        raise ValueError(msg)
    entries = manifest.get("source_documents")
    if not isinstance(entries, list):
        msg = f"Corpus manifest has no source documents: {manifest_path}"
        raise ValueError(msg)

    source_documents: list[RetainedSourceDocument] = []
    for entry in entries:
        if not isinstance(entry, dict):
            msg = "Corpus manifest contains an invalid source document entry."
            raise ValueError(msg)
        source_documents.append(_read_source_document(corpus_directory, entry))
    return source_documents


def _read_source_document(
    corpus_directory: Path,
    entry: dict[str, object],
) -> RetainedSourceDocument:
    """Read and verify one immutable source JSON document from a manifest entry."""
    source_json = entry.get("source_json")
    source_json_sha256 = entry.get("source_json_sha256")
    content_id = entry.get("content_id")
    source_version = entry.get("source_version")
    if (
        not isinstance(source_json, str)
        or not isinstance(source_json_sha256, str)
        or not isinstance(content_id, str)
        or not isinstance(source_version, str)
    ):
        msg = "Corpus manifest source document lacks its JSON identity."
        raise ValueError(msg)
    path = corpus_directory / source_json
    if not path.resolve().is_relative_to(corpus_directory.resolve()):
        msg = "Corpus manifest source document escapes the corpus directory."
        raise ValueError(msg)
    try:
        source = path.read_bytes()
    except FileNotFoundError as error:
        msg = f"Retained source document is missing: {path}"
        raise ValueError(msg) from error
    if sha256(source).hexdigest() != source_json_sha256:
        msg = f"Retained source document does not match its manifest identity: {path}"
        raise ValueError(msg)
    source_document = _parse_source_document(source)
    if source_document.get("content_id") != content_id:
        msg = (
            "Retained source document content ID does not match its manifest identity."
        )
        raise ValueError(msg)
    expected_source_version = f"content_id:{content_id}:sha256:{source_json_sha256}"
    if source_version != expected_source_version:
        msg = "Retained source document version does not match its manifest identity."
        raise ValueError(msg)
    return RetainedSourceDocument(
        content_id=content_id,
        source=source_document,
        source_bytes=source,
        source_json_sha256=source_json_sha256,
        source_version=source_version,
    )


def _source_document_quads(
    source_documents: list[RetainedSourceDocument],
) -> list[Quad]:
    """Return the source assertions for every retained Source document version."""
    quads: list[Quad] = []
    for source_document in source_documents:
        document = _document_node(source_document)
        quads.extend(
            [
                Quad(document, RDF_TYPE, SOURCE_DOCUMENT, DOCUMENTS_GRAPH),
                Quad(
                    document,
                    SOURCE_JSON,
                    Literal(
                        source_document.source_bytes.decode("utf-8"), datatype=RDF_JSON
                    ),
                    DOCUMENTS_GRAPH,
                ),
                Quad(
                    document,
                    SOURCE_VERSION,
                    Literal(source_document.source_version),
                    DOCUMENTS_GRAPH,
                ),
            ]
        )
        for key, value in source_document.source.items():
            assertion = NamedNode(f"{document.value}/assertion/{quote(key, safe='')}")
            quads.extend(
                [
                    Quad(assertion, RDF_TYPE, SOURCE_ASSERTION, DOCUMENTS_GRAPH),
                    Quad(assertion, ABOUT, document, DOCUMENTS_GRAPH),
                    Quad(assertion, SOURCE_KEY, Literal(key), DOCUMENTS_GRAPH),
                    Quad(
                        assertion,
                        SOURCE_VALUE,
                        Literal(
                            json.dumps(
                                value, ensure_ascii=False, separators=(",", ":")
                            ),
                            datatype=RDF_JSON,
                        ),
                        DOCUMENTS_GRAPH,
                    ),
                ]
            )
    return quads


def _parse_source_document(source_bytes: bytes) -> dict[str, object]:
    """Decode one retained GOV.UK source document without replacing its bytes."""
    try:
        source = json.loads(source_bytes)
    except json.JSONDecodeError as error:
        msg = "Retained source document is not valid JSON."
        raise ValueError(msg) from error
    if not isinstance(source, dict):
        msg = "Retained source document must be a JSON object."
        raise ValueError(msg)
    return source


def _document_node(source_document: RetainedSourceDocument) -> NamedNode:
    """Return the stable RDF resource for one immutable Source document version."""
    return NamedNode(
        f"{AIGG}source-document/{quote(source_document.content_id, safe='')}/"
        f"{source_document.source_json_sha256}"
    )


def _reference_data(reference: ArtefactReference) -> dict[str, str]:
    """Return the durable representation of an artefact reference."""
    return {
        "identity": reference.identity,
        "kind": reference.kind,
        "schema_version": reference.schema_version,
    }


if __name__ == "__main__":
    app()
