"""Commands for acquiring GOV.UK source documents into an evidence corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import urlopen

import typer

ACQUISITION_SCHEMA_VERSION = "1"
CONTENT_API_ROOT = "https://www.gov.uk/api/content"
SEARCH_API_URL = "https://www.gov.uk/api/search.json"


@dataclass(frozen=True)
class AcquisitionFailure:
    """One visible failure that prevents a complete evidence corpus."""

    stage: str
    message: str
    source_path: str | None = None

    def as_json(self) -> dict[str, str]:
        """Return the durable failure representation."""
        failure = {"message": self.message, "stage": self.stage}
        if self.source_path is not None:
            failure["source_path"] = self.source_path
        return failure


class AcquisitionError(ValueError):
    """Raised when acquisition cannot produce a complete evidence corpus."""

    def __init__(self, failure: AcquisitionFailure) -> None:
        """Record the durable failure information."""
        super().__init__(failure.message)
        self.failure = failure


@dataclass(frozen=True)
class SourceDocument:
    """One immutable source document version in a corpus manifest."""

    base_path: str
    content_api_url: str
    content_id: str
    locale: str
    sequence: int
    source_json: str
    source_json_sha256: str
    updated_at: str

    def as_json(self) -> dict[str, int | str]:
        """Return the durable source-document representation."""
        return {
            "base_path": self.base_path,
            "content_api_url": self.content_api_url,
            "content_id": self.content_id,
            "locale": self.locale,
            "sequence": self.sequence,
            "source_json": self.source_json,
            "source_json_sha256": self.source_json_sha256,
            "source_version": (
                f"content_id:{self.content_id}:sha256:{self.source_json_sha256}"
            ),
            "updated_at": self.updated_at,
        }


app = typer.Typer(
    help="Acquire GOV.UK source documents into a local evidence corpus.",
    no_args_is_help=True,
)
documents_app = typer.Typer(help="Acquire source documents into an evidence corpus.")
app.add_typer(documents_app, name="documents")


@documents_app.command()
def fetch(
    corpus_directory: Annotated[
        Path,
        typer.Option(help="New directory that will contain the evidence corpus."),
    ],
    organisation: Annotated[
        str,
        typer.Option(help="GOV.UK organisation slug used to enumerate documents."),
    ],
    maximum: Annotated[
        int,
        typer.Option(min=1, help="Maximum number of source documents to acquire."),
    ] = 100,
) -> None:
    """Acquire an immutable, manifest-backed GOV.UK evidence corpus."""
    _prepare_corpus_directory(corpus_directory)
    try:
        source_paths = _enumerate_source_paths(organisation, maximum)
    except AcquisitionError as error:
        _write_failure_report(corpus_directory, [error.failure])
        _report_incomplete(corpus_directory)

    failures: list[AcquisitionFailure] = []
    documents: list[SourceDocument] = []
    for sequence, source_path in enumerate(source_paths):
        try:
            documents.append(
                _acquire_source_document(corpus_directory, source_path, sequence)
            )
        except AcquisitionError as error:
            failures.append(error.failure)

    if failures:
        _write_failure_report(corpus_directory, failures)
        _report_incomplete(corpus_directory)

    manifest = {
        "acquisition": {
            "maximum": maximum,
            "ordering": "base_path",
            "organisation": organisation,
        },
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "source_documents": [document.as_json() for document in documents],
        "status": "complete",
    }
    _write_json(corpus_directory / "manifest.json", manifest)
    typer.echo(
        f"Acquired complete evidence corpus: {corpus_directory / 'manifest.json'}"
    )


def _enumerate_source_paths(organisation: str, maximum: int) -> list[str]:
    """Enumerate a bounded, deterministic set of GOV.UK source paths."""
    source_paths: list[str] = []
    start = 0
    total: int | None = None
    page_size = min(maximum, 100)

    while len(source_paths) < maximum and (total is None or start < total):
        search_query = [
            ("filter_organisations", organisation),
            ("count", str(page_size)),
            ("start", str(start)),
            ("fields", "link"),
        ]
        search_url = f"{SEARCH_API_URL}?{urlencode(search_query)}"
        response = _read_json(search_url, "enumeration")
        if not isinstance(response, dict):
            _raise_failure("enumeration", "Search API response must be a JSON object.")
        results = response.get("results")
        returned_total = response.get("total")
        if not isinstance(results, list) or not isinstance(returned_total, int):
            _raise_failure(
                "enumeration",
                "Search API response must contain results and total.",
            )
        if total is None:
            total = returned_total
        elif total != returned_total:
            _raise_failure(
                "enumeration", "Search API total changed during enumeration."
            )
        if not results and start < total:
            _raise_failure("enumeration", "Search API ended before the reported total.")

        for result in results:
            if len(source_paths) == maximum:
                break
            if not isinstance(result, dict) or not isinstance(result.get("link"), str):
                _raise_failure(
                    "enumeration", "Search API result is missing its document link."
                )
            source_paths.append(_source_path(result["link"]))
        start += len(results)

    if len(set(source_paths)) != len(source_paths):
        _raise_failure(
            "enumeration", "Search API returned a duplicate source document."
        )
    return sorted(source_paths)


def _acquire_source_document(
    corpus_directory: Path, source_path: str, sequence: int
) -> SourceDocument:
    """Download and describe one immutable source document version."""
    content_api_url = f"{CONTENT_API_ROOT}{quote(source_path, safe='/')}"
    raw_json = _read_bytes(content_api_url, "download", source_path)
    try:
        document = json.loads(raw_json)
    except json.JSONDecodeError:
        _raise_failure(
            "validation", "Content API response is not valid JSON.", source_path
        )
    if not isinstance(document, dict):
        _raise_failure(
            "validation", "Content API response must be a JSON object.", source_path
        )

    content_id = document.get("content_id")
    base_path = document.get("base_path")
    locale = document.get("locale")
    updated_at = document.get("updated_at")
    if not isinstance(content_id, str) or not content_id:
        _raise_failure(
            "validation", "Content API response is missing content_id.", source_path
        )
    if base_path != source_path:
        _raise_failure(
            "validation",
            "Content API base_path does not match its request.",
            source_path,
        )
    if not isinstance(locale, str) or not locale:
        _raise_failure(
            "validation", "Content API response is missing locale.", source_path
        )
    if not isinstance(updated_at, str) or not updated_at:
        _raise_failure(
            "validation", "Content API response is missing updated_at.", source_path
        )

    source_json_sha256 = sha256(raw_json).hexdigest()
    relative_source_json = Path("source-documents") / f"{source_json_sha256}.json"
    _write_bytes_immutable(corpus_directory / relative_source_json, raw_json)
    return SourceDocument(
        base_path=source_path,
        content_api_url=content_api_url,
        content_id=content_id,
        locale=locale,
        sequence=sequence,
        source_json=relative_source_json.as_posix(),
        source_json_sha256=source_json_sha256,
        updated_at=updated_at,
    )


def _source_path(link: str) -> str:
    """Return one GOV.UK source path from a Search API result link."""
    parsed = urlparse(link)
    if parsed.scheme or parsed.netloc:
        _raise_failure(
            "enumeration", "Search API returned a non-relative document link."
        )
    if not parsed.path.startswith("/"):
        _raise_failure("enumeration", "Search API returned an invalid document link.")
    return parsed.path


def _read_json(url: str, stage: str) -> object:
    """Read one JSON response from the external GOV.UK API boundary."""
    raw_json = _read_bytes(url, stage)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        _raise_failure(stage, "GOV.UK API response is not valid JSON.")


def _read_bytes(url: str, stage: str, source_path: str | None = None) -> bytes:
    """Read bytes from the external GOV.UK API boundary."""
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310
            return response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        _raise_failure(stage, str(error), source_path)


def _prepare_corpus_directory(corpus_directory: Path) -> None:
    """Create a new corpus directory without replacing a prior acquisition."""
    if corpus_directory.exists() and any(corpus_directory.iterdir()):
        msg = f"Corpus directory is not empty: {corpus_directory}"
        raise typer.BadParameter(msg)
    corpus_directory.mkdir(parents=True, exist_ok=True)


def _write_bytes_immutable(path: Path, content: bytes) -> None:
    """Write exact source bytes once, checking an existing content address."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            msg = f"Source artefact identity collision at {path}."
            raise AcquisitionError(AcquisitionFailure("storage", msg)) from None


def _write_failure_report(
    corpus_directory: Path, failures: list[AcquisitionFailure]
) -> None:
    """Persist the errors that make this acquisition incomplete."""
    report = {
        "failures": [failure.as_json() for failure in failures],
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "status": "incomplete",
    }
    _write_json(corpus_directory / "acquisition-failure.json", report)


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic JSON control artefact."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _raise_failure(
    stage: str, message: str, source_path: str | None = None
) -> NoReturn:
    """Raise one acquisition error with enough context for operators."""
    raise AcquisitionError(AcquisitionFailure(stage, message, source_path))


def _report_incomplete(corpus_directory: Path) -> NoReturn:
    """Make the absence of a usable manifest explicit to the operator."""
    typer.echo(
        "Acquisition incomplete. No corpus manifest was written. "
        f"Failure record: {corpus_directory / 'acquisition-failure.json'}",
        err=True,
    )
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
