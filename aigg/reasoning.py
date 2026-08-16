"""Structured model reasoning with durable invocation records."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

from langchain_openrouter import ChatOpenRouter

from aigg.artefacts import ArtefactReference, ArtefactStore, JsonValue


@dataclass(frozen=True)
class ModelConfiguration:
    """The configured identity and parameters of a reasoning model."""

    provider: str
    model: str
    parameters: dict[str, JsonValue]

    def __post_init__(self) -> None:
        """Keep OpenRouter credentials out of durable invocation records."""
        object.__setattr__(
            self,
            "parameters",
            cast(dict[str, JsonValue], json.loads(json.dumps(self.parameters))),
        )
        self.validate()

    def validate(self) -> None:
        """Confirm the current configuration remains safe to record."""
        if self.provider != "openrouter":
            return
        if _contains_openrouter_credential(self.parameters):
            msg = (
                "OpenRouter credentials must be supplied through "
                "OPENROUTER_API_KEY, not model parameters."
            )
            raise ValueError(msg)
        unsupported_parameters = set(self.parameters) - _OPENROUTER_MODEL_PARAMETERS
        if unsupported_parameters:
            parameter = min(unsupported_parameters)
            msg = f"OpenRouter model parameter {parameter!r} is not supported."
            raise ValueError(msg)


_OPENROUTER_MODEL_PARAMETERS = frozenset(
    {
        "frequency_penalty",
        "max_completion_tokens",
        "max_retries",
        "max_tokens",
        "n",
        "openrouter_provider",
        "plugins",
        "presence_penalty",
        "reasoning",
        "route",
        "seed",
        "stop",
        "temperature",
        "timeout",
        "top_p",
    }
)


def _contains_openrouter_credential(value: JsonValue) -> bool:
    """Return whether a JSON parameter object contains an OpenRouter credential."""
    credential_fields = {
        "api_key",
        "apikey",
        "authorization",
        "openrouter_api_key",
        "x_api_key",
        "x_openrouter_api_key",
    }
    if isinstance(value, dict):
        for field, nested_value in value.items():
            normalised_field = field.lower().replace("-", "_")
            if normalised_field in credential_fields:
                return True
            if _contains_openrouter_credential(nested_value):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_openrouter_credential(item) for item in value)
    return False


class StructuredModel(Protocol):
    """Produce structured output using the supplied model configuration."""

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the configured model output for ``structured_input``."""


class OpenRouterInvocationError(ValueError):
    """Raised when OpenRouter cannot produce a structured model response."""


class OpenRouterStructuredInvoker(Protocol):
    """The structured-output invoker returned by the LangChain chat model."""

    def invoke(self, messages: list[tuple[str, str]]) -> object:
        """Return one chat response for the supplied messages."""


class OpenRouterChatModel(Protocol):
    """The small LangChain chat-model surface used by the provider adapter."""

    def with_structured_output(
        self, schema: dict[str, Any], *, method: str
    ) -> OpenRouterStructuredInvoker:
        """Configure the model to return values matching a JSON schema."""


OpenRouterModelFactory: TypeAlias = Callable[[str, dict[str, Any]], OpenRouterChatModel]

_JSON_OBJECT_SCHEMA: dict[str, Any] = {
    "title": "StructuredModelOutput",
    "type": "object",
    "additionalProperties": True,
}


class OpenRouterStructuredModel:
    """Invoke OpenRouter through LangChain at the structured-model boundary."""

    def __init__(self, model_factory: OpenRouterModelFactory | None = None) -> None:
        """Create an adapter with the real LangChain factory by default."""
        self._model_factory = model_factory or _create_openrouter_chat_model

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the JSON response produced by the configured OpenRouter model."""
        if configuration.provider != "openrouter":
            msg = f"OpenRouter model received provider {configuration.provider!r}."
            raise OpenRouterInvocationError(msg)

        try:
            output = (
                self._model_factory(configuration.model, dict(configuration.parameters))
                .with_structured_output(_JSON_OBJECT_SCHEMA, method="json_schema")
                .invoke(
                    [
                        ("system", "Return only a JSON value."),
                        ("human", _structured_input_message(structured_input)),
                    ]
                )
            )
        except Exception as error:
            msg = (
                "OpenRouter structured invocation failed for model "
                f"{configuration.model!r}."
            )
            raise OpenRouterInvocationError(msg) from error

        if not isinstance(output, dict):
            msg = f"OpenRouter model {configuration.model!r} returned non-object JSON."
            raise OpenRouterInvocationError(msg)
        try:
            return cast(JsonValue, json.loads(json.dumps(output, allow_nan=False)))
        except (TypeError, ValueError) as error:
            msg = f"OpenRouter model {configuration.model!r} returned non-JSON values."
            raise OpenRouterInvocationError(msg) from error


def _create_openrouter_chat_model(
    model: str, parameters: dict[str, Any]
) -> OpenRouterChatModel:
    """Create the LangChain OpenRouter chat model without handling credentials."""
    return cast(OpenRouterChatModel, ChatOpenRouter(model=model, **parameters))


def _structured_input_message(structured_input: JsonValue) -> str:
    """Serialise a JSON-compatible reasoning input for the chat model."""
    return json.dumps(structured_input)


OutputValidator: TypeAlias = Callable[[JsonValue], JsonValue]


@dataclass(frozen=True)
class ReasoningInvocation:
    """One successful reasoning result and its durable record."""

    reference: ArtefactReference
    output: JsonValue


class ReasoningValidationError(ValueError):
    """Raised when every bounded structured-output validation attempt fails."""

    def __init__(self, reference: ArtefactReference) -> None:
        """Describe the durable record of the failed invocation."""
        super().__init__(f"Reasoning output failed validation: {reference.identity}.")
        self.reference = reference


class ReasoningRunner:
    """Run structured reasoning and retain every invocation as an artefact."""

    def __init__(
        self,
        store: ArtefactStore,
        model: StructuredModel,
        configuration: ModelConfiguration,
        *,
        maximum_attempts: int,
    ) -> None:
        """Create a runner with an explicit bounded validation retry policy."""
        if maximum_attempts < 1:
            msg = "Reasoning must allow at least one validation attempt."
            raise ValueError(msg)
        self._store = store
        self._model = model
        self._configuration = configuration
        self._maximum_attempts = maximum_attempts

    def run(
        self,
        *,
        stage: str,
        structured_input: JsonValue,
        validate_output: OutputValidator,
    ) -> ReasoningInvocation:
        """Run one stage, validate its output, and retain the complete invocation."""
        self._configuration.validate()
        retry_history: list[JsonValue] = []
        for attempt in range(1, self._maximum_attempts + 1):
            output = self._model.invoke(
                configuration=self._configuration,
                structured_input=structured_input,
            )
            try:
                validated_output = validate_output(output)
            except ValueError as error:
                retry_history.append(
                    {
                        "attempt": attempt,
                        "structured_output": output,
                        "validation_error": str(error),
                    }
                )
                continue

            reference = self._write_invocation(
                stage=stage,
                structured_input=structured_input,
                structured_output=validated_output,
                retry_history=retry_history,
            )
            return ReasoningInvocation(reference, validated_output)

        reference = self._write_invocation(
            stage=stage,
            structured_input=structured_input,
            structured_output=None,
            retry_history=retry_history,
        )
        raise ReasoningValidationError(reference)

    def _write_invocation(
        self,
        *,
        stage: str,
        structured_input: JsonValue,
        structured_output: JsonValue,
        retry_history: list[JsonValue],
    ) -> ArtefactReference:
        """Persist one invocation in its analysis-friendly representation."""
        self._configuration.validate()
        return self._store.write_json(
            "reasoning-invocation",
            {
                "stage": stage,
                "provider": self._configuration.provider,
                "model": self._configuration.model,
                "parameters": self._configuration.parameters,
                "structured_input": structured_input,
                "structured_output": structured_output,
                "retry_history": retry_history,
            },
        )
