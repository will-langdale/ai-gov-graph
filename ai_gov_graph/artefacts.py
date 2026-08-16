"""Content-addressed durable artefacts for experiment lineages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypeAlias

ARTEFACT_SCHEMA_VERSION = "1"
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class ArtefactIntegrityError(ValueError):
    """Raised when a durable artefact is absent, malformed, or has changed."""


@dataclass(frozen=True)
class ArtefactReference:
    """The stable identity and schema version of one durable artefact."""

    kind: str
    sha256: str
    schema_version: str

    @property
    def identity(self) -> str:
        """Return the content-addressed artefact identity."""
        return f"sha256:{self.sha256}"


class ArtefactStore:
    """Read and write immutable JSON artefacts, verifying every read."""

    def __init__(self, root: Path) -> None:
        """Create a store rooted at ``root``."""
        self.root = root

    def write_json(self, kind: str, payload: JsonValue) -> ArtefactReference:
        """Store a schema-versioned JSON payload under its content hash."""
        if not kind or "/" in kind or "\\" in kind:
            msg = "Artefact kind must be a single path component."
            raise ValueError(msg)

        artefact = {"schema_version": ARTEFACT_SCHEMA_VERSION, "payload": payload}
        content = self._canonical_json(artefact)
        digest = sha256(content).hexdigest()
        reference = ArtefactReference(kind, digest, ARTEFACT_SCHEMA_VERSION)
        path = self.path_for(reference)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and path.read_bytes() != content:
            msg = f"Artefact identity collision at {reference.identity}."
            raise ArtefactIntegrityError(msg)
        path.write_bytes(content)
        return reference

    def read_json(self, reference: ArtefactReference) -> JsonValue:
        """Verify and return the payload named by ``reference``."""
        path = self.path_for(reference)
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            msg = f"Artefact {reference.identity} is missing at {path}."
            raise ArtefactIntegrityError(msg) from error

        actual_digest = sha256(content).hexdigest()
        if actual_digest != reference.sha256:
            msg = (
                f"Artefact {reference.identity} at {path} does not match its "
                "recorded content hash."
            )
            raise ArtefactIntegrityError(msg)

        try:
            artefact = json.loads(content)
        except json.JSONDecodeError as error:
            msg = f"Artefact {reference.identity} is not valid JSON."
            raise ArtefactIntegrityError(msg) from error
        if not isinstance(artefact, dict) or set(artefact) != {
            "schema_version",
            "payload",
        }:
            msg = f"Artefact {reference.identity} does not match the artefact schema."
            raise ArtefactIntegrityError(msg)
        if artefact["schema_version"] != reference.schema_version:
            msg = f"Artefact {reference.identity} has an unexpected schema version."
            raise ArtefactIntegrityError(msg)
        return artefact["payload"]

    def path_for(self, reference: ArtefactReference) -> Path:
        """Return the deterministic path for an artefact reference."""
        return self.root / reference.kind / f"{reference.sha256}.json"

    @staticmethod
    def _canonical_json(value: JsonValue) -> bytes:
        """Encode JSON deterministically for repeatable content hashes."""
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
