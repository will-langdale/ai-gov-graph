"""Create deterministic source text and validate Evidence anchors."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser

CANONICALISER_VERSION = "1"
_HTML_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


class CanonicalisationError(ValueError):
    """Raised when retained source JSON cannot become canonical text."""


class EvidenceValidationError(ValueError):
    """Raised when an Evidence anchor cannot be verified."""

    def __init__(self, field: str, message: str) -> None:
        """Identify the invalid Evidence field in the diagnostic."""
        super().__init__(f"Evidence field '{field}': {message}")
        self.field = field


@dataclass(frozen=True)
class CanonicalDocument:
    """The deterministic text derived from one retained source version."""

    canonicaliser_version: str
    sha256: str
    text: str


@dataclass(frozen=True)
class EvidenceAnchor:
    """A serialisable, exact passage reference in canonical source text.

    Offsets are zero-based character offsets into the canonical text. The end
    offset is exclusive. Line numbers are also zero-based and provide a
    human-friendly independent check on the selected passage.
    """

    canonical_text_sha256: str
    canonicaliser_version: str
    content_id: str
    end_line: int
    end_offset: int
    prefix: str
    selected_text: str
    source_json_sha256: str
    source_url: str
    start_line: int
    start_offset: int
    suffix: str

    def as_json(self) -> dict[str, int | str]:
        """Return the durable Evidence representation."""
        return {
            "canonical_text_sha256": self.canonical_text_sha256,
            "canonicaliser_version": self.canonicaliser_version,
            "content_id": self.content_id,
            "end_line": self.end_line,
            "end_offset": self.end_offset,
            "prefix": self.prefix,
            "selected_text": self.selected_text,
            "source_json_sha256": self.source_json_sha256,
            "source_url": self.source_url,
            "start_line": self.start_line,
            "start_offset": self.start_offset,
            "suffix": self.suffix,
        }

    @classmethod
    def from_json(cls, value: object) -> EvidenceAnchor:
        """Parse a durable Evidence representation before processing claims."""
        if not isinstance(value, dict):
            raise EvidenceValidationError("anchor", "must be a JSON object.")
        return cls(
            canonical_text_sha256=_string_field(value, "canonical_text_sha256"),
            canonicaliser_version=_string_field(value, "canonicaliser_version"),
            content_id=_string_field(value, "content_id"),
            end_line=_integer_field(value, "end_line"),
            end_offset=_integer_field(value, "end_offset"),
            prefix=_string_field(value, "prefix"),
            selected_text=_string_field(value, "selected_text"),
            source_json_sha256=_string_field(value, "source_json_sha256"),
            source_url=_string_field(value, "source_url"),
            start_line=_integer_field(value, "start_line"),
            start_offset=_integer_field(value, "start_offset"),
            suffix=_string_field(value, "suffix"),
        )


def canonicalise_source_document(
    source_json: bytes, canonicaliser_version: str = CANONICALISER_VERSION
) -> CanonicalDocument:
    """Return versioned canonical text for one retained GOV.UK JSON response."""
    _validate_canonicaliser_version(canonicaliser_version)
    try:
        source = json.loads(source_json)
    except json.JSONDecodeError as error:
        msg = "Source JSON must be valid before canonicalisation."
        raise CanonicalisationError(msg) from error
    if not isinstance(source, dict):
        msg = "Source JSON must be an object before canonicalisation."
        raise CanonicalisationError(msg)

    passages = list(_string_passages(source))
    text = "\n\n".join(f"{path}:\n{passage}" for path, passage in passages)
    if text:
        text += "\n"
    return CanonicalDocument(
        canonicaliser_version=canonicaliser_version,
        sha256=sha256(text.encode("utf-8")).hexdigest(),
        text=text,
    )


def validate_evidence_anchor(
    anchor: EvidenceAnchor, source_json: bytes
) -> CanonicalDocument:
    """Verify all immutable identity and passage fields in an Evidence anchor."""
    _validate_canonicaliser_version(anchor.canonicaliser_version)
    source_sha256 = sha256(source_json).hexdigest()
    if anchor.source_json_sha256 != source_sha256:
        raise EvidenceValidationError(
            "source_json_sha256", "does not match the retained source JSON."
        )
    _validate_source_identity(anchor, source_json)

    canonical = canonicalise_source_document(
        source_json, canonicaliser_version=anchor.canonicaliser_version
    )
    if anchor.canonical_text_sha256 != canonical.sha256:
        raise EvidenceValidationError(
            "canonical_text_sha256", "does not match the canonical text."
        )
    _validate_offsets(anchor, canonical.text)
    return canonical


def validate_canonical_text(
    source_json: bytes,
    canonical_text: bytes,
    canonical_text_sha256: str,
    canonicaliser_version: str,
) -> CanonicalDocument:
    """Verify a retained canonical artefact against its source and manifest entry."""
    _validate_canonicaliser_version(canonicaliser_version)
    if sha256(canonical_text).hexdigest() != canonical_text_sha256:
        raise EvidenceValidationError(
            "canonical_text_sha256", "does not match the retained canonical text."
        )
    try:
        text = canonical_text.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceValidationError(
            "canonical_text", "must be valid UTF-8."
        ) from error
    canonical = canonicalise_source_document(source_json, canonicaliser_version)
    if text != canonical.text:
        raise EvidenceValidationError(
            "canonical_text", "does not match the declared source document."
        )
    return canonical


def _validate_canonicaliser_version(canonicaliser_version: str) -> None:
    """Reject Evidence that was produced by an unavailable canonicaliser."""
    if canonicaliser_version != CANONICALISER_VERSION:
        raise EvidenceValidationError(
            "canonicaliser_version",
            f"unsupported version {canonicaliser_version!r}; expected "
            f"{CANONICALISER_VERSION!r}.",
        )


def _string_field(value: dict[object, object], field: str) -> str:
    """Read a required string field from a durable Evidence object."""
    field_value = value.get(field)
    if not isinstance(field_value, str):
        raise EvidenceValidationError(field, "must be a string.")
    return field_value


def _integer_field(value: dict[object, object], field: str) -> int:
    """Read a required integer field from a durable Evidence object."""
    field_value = value.get(field)
    if isinstance(field_value, bool) or not isinstance(field_value, int):
        raise EvidenceValidationError(field, "must be an integer.")
    return field_value


def _validate_offsets(anchor: EvidenceAnchor, text: str) -> None:
    """Verify an exact selected passage, context and its declared positions."""
    if not anchor.selected_text:
        raise EvidenceValidationError("selected_text", "must not be empty.")
    if anchor.start_offset < 0 or anchor.end_offset > len(text):
        raise EvidenceValidationError("offset", "is outside the canonical text.")
    if anchor.start_offset >= anchor.end_offset:
        raise EvidenceValidationError("offset", "must describe a non-empty range.")

    selected_start = text.find(anchor.selected_text)
    if (
        selected_start != -1
        and text.find(anchor.selected_text, selected_start + 1) == -1
    ):
        if anchor.start_offset != selected_start:
            raise EvidenceValidationError(
                "start_offset", "does not point to the selected text."
            )
        if anchor.end_offset != selected_start + len(anchor.selected_text):
            raise EvidenceValidationError(
                "end_offset", "does not end after the selected text."
            )
    selected_text = text[anchor.start_offset : anchor.end_offset]
    if selected_text != anchor.selected_text:
        raise EvidenceValidationError(
            "selected_text", "does not match the text at the declared offsets."
        )
    prefix_start = anchor.start_offset - len(anchor.prefix)
    if prefix_start < 0 or text[prefix_start : anchor.start_offset] != anchor.prefix:
        raise EvidenceValidationError(
            "prefix", "does not match the context before the selected text."
        )
    suffix_end = anchor.end_offset + len(anchor.suffix)
    if suffix_end > len(text) or text[anchor.end_offset : suffix_end] != anchor.suffix:
        raise EvidenceValidationError(
            "suffix", "does not match the context after the selected text."
        )

    start_line = text.count("\n", 0, anchor.start_offset)
    end_line = text.count("\n", 0, anchor.end_offset - 1)
    if anchor.start_line != start_line:
        raise EvidenceValidationError(
            "start_line", "does not match the selected text offset."
        )
    if anchor.end_line != end_line:
        raise EvidenceValidationError(
            "end_line", "does not match the selected text offset."
        )


def _validate_source_identity(anchor: EvidenceAnchor, source_json: bytes) -> None:
    """Bind an Evidence anchor to the identity declared by its Source document."""
    try:
        source = json.loads(source_json)
    except json.JSONDecodeError as error:
        raise EvidenceValidationError("source_json", "must be valid JSON.") from error
    if not isinstance(source, dict):
        raise EvidenceValidationError("source_json", "must be a JSON object.")
    content_id = source.get("content_id")
    if not isinstance(content_id, str) or not content_id:
        raise EvidenceValidationError(
            "content_id", "is missing from the retained source JSON."
        )
    if anchor.content_id != content_id:
        raise EvidenceValidationError(
            "content_id", "does not match the retained source JSON."
        )
    base_path = source.get("base_path")
    if not isinstance(base_path, str) or not base_path.startswith("/"):
        raise EvidenceValidationError(
            "source_url", "cannot be derived from the retained source JSON."
        )
    source_url = f"https://www.gov.uk{base_path}"
    if anchor.source_url != source_url:
        raise EvidenceValidationError(
            "source_url", "does not match the retained source JSON."
        )


def _string_passages(value: object, path: str = "") -> list[tuple[str, str]]:
    """Return normalised string leaves ordered by JSON pointer."""
    if isinstance(value, dict):
        passages: list[tuple[str, str]] = []
        for key in sorted(value):
            escaped_key = key.replace("~", "~0").replace("/", "~1")
            passages.extend(_string_passages(value[key], f"{path}/{escaped_key}"))
        return passages
    if isinstance(value, list):
        passages = []
        for index, item in enumerate(value):
            passages.extend(_string_passages(item, f"{path}/{index}"))
        return passages
    if isinstance(value, str):
        return [(path or "/", _canonicalise_string(value))]
    return []


def _canonicalise_string(value: str) -> str:
    """Normalise plain text or render simple GOV.UK HTML as stable text."""
    normalised = unicodedata.normalize("NFC", value).replace("\r\n", "\n")
    normalised = normalised.replace("\r", "\n")
    if not _HTML_PATTERN.search(normalised):
        return "\n".join(
            _WHITESPACE_PATTERN.sub(" ", line).strip()
            for line in normalised.split("\n")
        ).strip()
    parser = _CanonicalHtmlParser()
    parser.feed(normalised)
    parser.close()
    return parser.text


class _CanonicalHtmlParser(HTMLParser):
    """Render a small, deliberately stable text projection of HTML content."""

    _BLOCK_TAGS = frozenset({"div", "p", "section", "table", "tbody", "thead"})
    _HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        """Start with an empty rendered document."""
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._links: list[str | None] = []
        self._table_cell_seen = False

    @property
    def text(self) -> str:
        """Return the completed canonical text."""
        return "".join(self._parts).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Render structural HTML elements with deterministic delimiters."""
        if tag in self._HEADING_TAGS:
            self._break()
            self._append("#" * int(tag[1]) + " ")
        elif tag in self._BLOCK_TAGS:
            self._break()
        elif tag == "tr":
            self._break()
            self._table_cell_seen = False
        elif tag == "li":
            self._break()
            self._append("- ")
        elif tag == "br":
            self._break()
        elif tag in {"td", "th"}:
            if self._table_cell_seen:
                self._append(" | ")
            self._table_cell_seen = True
        elif tag == "a":
            self._links.append(dict(attrs).get("href"))

    def handle_endtag(self, tag: str) -> None:
        """Finish structural HTML elements and retain link targets."""
        if tag == "a" and self._links:
            href = self._links.pop()
            if href:
                self._append(f" <{href}>")
        elif (
            tag in self._HEADING_TAGS or tag in self._BLOCK_TAGS or tag in {"li", "tr"}
        ):
            self._break()

    def handle_data(self, data: str) -> None:
        """Add Unicode-normalised, collapsed visible text."""
        text = _WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", data))
        self._append(text)

    def _append(self, value: str) -> None:
        """Append text without allowing trailing whitespace before a newline."""
        if not value:
            return
        if self._parts and self._parts[-1].endswith("\n"):
            value = value.lstrip()
        self._parts.append(value)

    def _break(self) -> None:
        """Separate rendered blocks with exactly one blank line."""
        current = "".join(self._parts).rstrip()
        self._parts = [current] if current else []
        if self._parts:
            self._parts.append("\n\n")
