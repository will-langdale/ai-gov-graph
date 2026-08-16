"""Tests for recorded structured reasoning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ai_gov_graph.artefacts import ArtefactStore, JsonValue
from ai_gov_graph.reasoning import (
    ModelConfiguration,
    OpenRouterInvocationError,
    OpenRouterStructuredModel,
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


@dataclass
class StaticChatModel:
    """Return one configured OpenRouter chat response."""

    output: object
    messages: list[object]
    schemas: list[tuple[dict[str, Any], str]]

    def with_structured_output(
        self, schema: dict[str, Any], *, method: str
    ) -> "StaticChatModel":
        """Retain the requested JSON schema and return this invoker."""
        self.schemas.append((schema, method))
        return self

    def invoke(self, messages: object) -> object:
        """Retain the OpenRouter request and return JSON content."""
        self.messages.append(messages)
        return self.output


def test_methodology_configured_output() -> None:
    """An OpenRouter model returns JSON-compatible structured output.

    Guards the provider adapter from leaking chat-message response handling into
    reasoning stages. Uses a static chat model because OpenRouter is an
    external, non-deterministic boundary.
    """
    chat_model = StaticChatModel({"claims": []}, [], [])
    created_models: list[tuple[str, dict[str, Any]]] = []

    def create_model(model: str, parameters: dict[str, Any]) -> StaticChatModel:
        """Retain the configured OpenRouter model identity and parameters."""
        created_models.append((model, parameters))
        return chat_model

    output = OpenRouterStructuredModel(create_model).invoke(
        configuration=ModelConfiguration(
            "openrouter", "openai/gpt-5-nano", {"temperature": 0}
        ),
        structured_input={"canonical_text": "The department has a minister."},
    )

    assert output == {"claims": []}
    assert created_models == [("openai/gpt-5-nano", {"temperature": 0})]
    assert chat_model.schemas == [
        (
            {
                "title": "StructuredModelOutput",
                "type": "object",
                "additionalProperties": True,
            },
            "json_schema",
        )
    ]
    assert chat_model.messages == [
        [
            ("system", "Return only a JSON value."),
            ("human", '{"canonical_text": "The department has a minister."}'),
        ]
    ]


@pytest.mark.parametrize(
    ("output", "diagnostic"),
    (
        pytest.param("not JSON", "non-object JSON", id="response"),
        pytest.param({"claims": [object()]}, "non-JSON values", id="nested-value"),
    ),
)
def test_methodology_invalid_output(output: object, diagnostic: str) -> None:
    """A malformed OpenRouter response reports its configured model.

    Guards the reasoning stage from accepting provider text as structured output.
    Uses a static chat model because OpenRouter is an external,
    non-deterministic boundary.
    """
    chat_model = StaticChatModel(output, [], [])
    model = OpenRouterStructuredModel(lambda _model, _parameters: chat_model)

    with pytest.raises(OpenRouterInvocationError, match=f"gpt-5-nano.*{diagnostic}"):
        model.invoke(
            configuration=ModelConfiguration(
                "openrouter", "openai/gpt-5-nano", {"temperature": 0}
            ),
            structured_input={"canonical_text": "The department has a minister."},
        )


def test_methodology_invocation_failure() -> None:
    """An OpenRouter provider failure reports its configured model.

    Guards operators from receiving a transport exception without a model
    diagnostic. Uses a static factory because OpenRouter is an external,
    non-deterministic boundary.
    """

    def fail_to_create(_model: str, _parameters: dict[str, Any]) -> StaticChatModel:
        """Represent a provider authentication failure."""
        msg = "unauthorised"
        raise RuntimeError(msg)

    model = OpenRouterStructuredModel(fail_to_create)

    with pytest.raises(
        OpenRouterInvocationError, match="invocation failed.*gpt-5-nano"
    ):
        model.invoke(
            configuration=ModelConfiguration(
                "openrouter", "openai/gpt-5-nano", {"temperature": 0}
            ),
            structured_input={"canonical_text": "The department has a minister."},
        )


@pytest.mark.parametrize(
    ("parameters",),
    (
        pytest.param({"openrouter_api_key": "secret"}, id="api-key"),
        pytest.param(
            {"default_headers": {"X-OpenRouter-API-Key": "secret"}},
            id="api-key-header",
        ),
    ),
)
def test_methodology_configuration_credential(parameters: dict[str, JsonValue]) -> None:
    """An OpenRouter credential cannot become a recorded model parameter.

    Guards reasoning-invocation artefacts from retaining a provider credential.
    """
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        ModelConfiguration("openrouter", "openai/gpt-5-nano", parameters)


def test_methodology_configuration_mutated_credential(tmp_path: Path) -> None:
    """A later credential mutation cannot create a reasoning invocation artefact.

    Guards a frozen model configuration's mutable parameters from bypassing the
    durable credential check.
    """
    configuration = ModelConfiguration(
        "openrouter", "openai/gpt-5-nano", {"temperature": 0}
    )
    configuration.parameters["api_key"] = "secret"
    store = ArtefactStore(tmp_path / "artefacts")
    runner = ReasoningRunner(
        store,
        StaticModel({"claims": []}, [], []),
        configuration,
        maximum_attempts=1,
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        runner.run(
            stage="claim-extraction",
            structured_input={"canonical_text": ""},
            validate_output=_valid_claims,
        )

    assert not store.root.exists()


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
