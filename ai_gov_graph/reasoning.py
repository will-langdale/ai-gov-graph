"""Structured model reasoning with durable invocation records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from ai_gov_graph.artefacts import ArtefactReference, ArtefactStore, JsonValue


@dataclass(frozen=True)
class ModelConfiguration:
    """The configured identity and parameters of a reasoning model."""

    provider: str
    model: str
    parameters: dict[str, JsonValue]


class StructuredModel(Protocol):
    """Produce structured output using the supplied model configuration."""

    def invoke(
        self,
        *,
        configuration: ModelConfiguration,
        structured_input: JsonValue,
    ) -> JsonValue:
        """Return the configured model output for ``structured_input``."""


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
