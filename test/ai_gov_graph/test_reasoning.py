"""Tests for recorded structured reasoning."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from ai_gov_graph.artefacts import ArtefactStore, JsonValue
from ai_gov_graph.reasoning import (
    ModelConfiguration,
    ReasoningRunner,
    ReasoningValidationError,
)


@dataclass
class StaticModel:
    """Return one predetermined structured model output."""

    output: JsonValue
    calls: list[JsonValue]
    configurations: list[ModelConfiguration]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the configured output and retain the observed input."""
        self.calls.append(structured_input)
        self.configurations.append(configuration)
        return self.output


@dataclass
class SequencedModel:
    """Return structured outputs in their configured order."""

    outputs: list[JsonValue]
    calls: list[JsonValue]
    configurations: list[ModelConfiguration]

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the next output and retain the observed input."""
        self.calls.append(structured_input)
        self.configurations.append(configuration)
        return self.outputs.pop(0)


def test_methodology_recorded_output(tmp_path: Path) -> None:
    """A reasoning stage retains its complete successful invocation.

    Guards later analysis from needing another model call to inspect the result.
    """
    model = StaticModel({"claims": ["The department has a minister."]}, [], [])
    store = ArtefactStore(tmp_path / "artefacts")
    runner = ReasoningRunner(
        store,
        model,
        ModelConfiguration(
            provider="example",
            model="example-1",
            parameters={"temperature": 0},
        ),
        maximum_attempts=2,
    )

    invocation = runner.run(
        stage="claim-extraction",
        structured_input={"canonical_text": "The department has a minister."},
        validate_output=_valid_claims,
    )

    assert store.read_json(invocation.reference) == {
        "stage": "claim-extraction",
        "provider": "example",
        "model": "example-1",
        "parameters": {"temperature": 0},
        "structured_input": {"canonical_text": "The department has a minister."},
        "structured_output": {"claims": ["The department has a minister."]},
        "retry_history": [],
    }


def test_methodology_configuration(tmp_path: Path) -> None:
    """A methodology passes its configured model identity to the model boundary.

    Guards a recorded model configuration from becoming audit metadata only.
    """
    model = StaticModel({"claims": []}, [], [])
    runner = ReasoningRunner(
        ArtefactStore(tmp_path / "artefacts"),
        model,
        ModelConfiguration("example", "example-1", {"temperature": 0}),
        maximum_attempts=1,
    )

    runner.run(
        stage="claim-extraction",
        structured_input={"canonical_text": ""},
        validate_output=_valid_claims,
    )

    assert model.configurations == [
        ModelConfiguration("example", "example-1", {"temperature": 0})
    ]


def test_methodology_invalid_then_valid(tmp_path: Path) -> None:
    """An invalid output is retained before the bounded retry succeeds.

    Guards analytical recovery of the failed validation that led to an output.
    """
    model = SequencedModel(
        [{"not_claims": []}, {"claims": ["The department has a minister."]}],
        [],
        [],
    )
    store = ArtefactStore(tmp_path / "artefacts")
    runner = ReasoningRunner(
        store,
        model,
        ModelConfiguration("example", "example-1", {"temperature": 0}),
        maximum_attempts=2,
    )

    invocation = runner.run(
        stage="claim-extraction",
        structured_input={"canonical_text": "The department has a minister."},
        validate_output=_valid_claims,
    )

    assert model.calls == [
        {"canonical_text": "The department has a minister."},
        {"canonical_text": "The department has a minister."},
    ]
    record = store.read_json(invocation.reference)
    assert isinstance(record, dict)
    assert record["retry_history"] == [
        {
            "attempt": 1,
            "structured_output": {"not_claims": []},
            "validation_error": "Output must contain a claims list.",
        }
    ]


def test_methodology_invalid_output_exhausted(tmp_path: Path) -> None:
    """An exhausted validation policy records failure then reports it loudly.

    Guards an invalid structured output from being silently accepted or lost.
    """
    model = SequencedModel([{"not_claims": []}, {"not_claims": []}], [], [])
    store = ArtefactStore(tmp_path / "artefacts")
    runner = ReasoningRunner(
        store,
        model,
        ModelConfiguration("example", "example-1", {"temperature": 0}),
        maximum_attempts=2,
    )

    with pytest.raises(ReasoningValidationError) as error:
        runner.run(
            stage="claim-extraction",
            structured_input={"canonical_text": "The department has a minister."},
            validate_output=_valid_claims,
        )

    assert model.calls == [
        {"canonical_text": "The department has a minister."},
        {"canonical_text": "The department has a minister."},
    ]
    record = store.read_json(error.value.reference)
    assert isinstance(record, dict)
    assert record["structured_output"] is None
    assert record["retry_history"] == [
        {
            "attempt": 1,
            "structured_output": {"not_claims": []},
            "validation_error": "Output must contain a claims list.",
        },
        {
            "attempt": 2,
            "structured_output": {"not_claims": []},
            "validation_error": "Output must contain a claims list.",
        },
    ]


def _valid_claims(value: JsonValue) -> JsonValue:
    """Validate the smallest claim-extraction output shape used in this test."""
    if not isinstance(value, dict) or not isinstance(value.get("claims"), list):
        raise ValueError("Output must contain a claims list.")
    return value
