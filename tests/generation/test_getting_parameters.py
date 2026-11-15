"""Tests for parameter extraction (``LLM_Function._get_parameters``)."""

from typing import Any

import pytest

from src.llm_function import LLM_Function, PromptBuilder
from src.parser import Function, Prompt
from tests.generation.fakes import (
    SPECIAL_TOKENS,
    ScriptedLLM,
    scripted_llm_answers,
    scripted_llm_steps,
)


class RecordingGenerator:
    """Generator double that records which typed method was called."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.calls: list[tuple[str, str]] = []

    def _record(self, kind: str, context: str) -> Any:
        self.calls.append((kind, context))
        return self.values[kind]

    def string(self, context: str) -> str:
        return self._record("string", context)

    def number(self, context: str) -> float:
        return self._record("number", context)

    def integer(self, context: str) -> int:
        return self._record("integer", context)

    def bool(self, context: str) -> bool:
        return self._record("boolean", context)


def make_llm_function(
    llm: ScriptedLLM, functions: list[Function]
) -> LLM_Function:
    return LLM_Function(llm, functions, SPECIAL_TOKENS)  # type: ignore[arg-type]


def test_get_parameters_dispatches_on_each_parameter_type(
    all_types_function: Function,
) -> None:
    llm = scripted_llm_steps([])
    llm_function = make_llm_function(llm, [all_types_function])
    recording = RecordingGenerator(
        {"string": "hello", "number": 2.5, "integer": 7, "boolean": True}
    )
    llm_function.constrained_generator = recording  # type: ignore[assignment]
    builder = PromptBuilder(llm, [all_types_function], "extract everything")

    params = llm_function._get_parameters(all_types_function, builder)

    assert params == {"s": "hello", "n": 2.5, "i": 7, "b": True}
    assert [kind for kind, _ in recording.calls] == [
        "string",
        "number",
        "integer",
        "boolean",
    ]
    assert [context for _, context in recording.calls] == [
        builder.parameter(all_types_function, name)
        for name in all_types_function.parameters
    ]


def test_get_parameters_returns_python_values_with_expected_types(
    all_types_function: Function,
) -> None:
    llm = scripted_llm_answers("hello", "2.5", "4", "false")
    llm_function = make_llm_function(llm, [all_types_function])
    builder = PromptBuilder(llm, [all_types_function], "extract everything")

    params = llm_function._get_parameters(all_types_function, builder)

    assert params == {"s": "hello", "n": 2.5, "i": 4, "b": False}
    assert isinstance(params["s"], str)
    assert isinstance(params["n"], float)
    assert isinstance(params["i"], int)
    assert isinstance(params["b"], bool)


def test_get_parameters_sends_a_prompt_for_every_parameter(
    all_types_function: Function,
) -> None:
    llm = scripted_llm_answers("hello", "2.5", "4", "false")
    llm_function = make_llm_function(llm, [all_types_function])
    builder = PromptBuilder(llm, [all_types_function], "extract everything")

    llm_function._get_parameters(all_types_function, builder)

    # Every parameter produces two model calls: its value and the end token.
    assert len(llm.contexts) == 2 * len(all_types_function.parameters)
    assert "Extract the following parameter:\ns (string)" in llm.contexts[0]
    assert "Extract the following parameter:\nn (number)" in llm.contexts[2]
    assert "Extract the following parameter:\ni (integer)" in llm.contexts[4]
    assert "Extract the following parameter:\nb (boolean)" in llm.contexts[6]


def test_get_parameters_rejects_an_unsupported_type() -> None:
    function = Function(
        name="fn_bad",
        description="A function with a type the generator cannot handle.",
        parameters={"value": {"type": "string"}},
        returns={"type": "string"},
    )
    function.parameters["value"]["type"] = "array"  # type: ignore[typeddict-item]

    llm = scripted_llm_steps([])
    llm_function = make_llm_function(llm, [function])
    recording = RecordingGenerator(
        {"string": "hello", "number": 2.5, "integer": 7, "boolean": True}
    )
    llm_function.constrained_generator = recording  # type: ignore[assignment]
    builder = PromptBuilder(llm, [function], "extract it")

    with pytest.raises(ValueError, match="Unsupported parameter type: array"):
        llm_function._get_parameters(function, builder)

    assert recording.calls == []


def test_get_parameters_preserves_parameter_order() -> None:
    function = Function(
        name="fn_ordered",
        description="Parameters in a deliberately non-alphabetical order.",
        parameters={
            "z": {"type": "string"},
            "a": {"type": "string"},
            "m": {"type": "string"},
        },
        returns={"type": "string"},
    )
    llm = scripted_llm_answers("first", "second", "third")
    llm_function = make_llm_function(llm, [function])
    builder = PromptBuilder(llm, [function], "extract")

    params = llm_function._get_parameters(function, builder)

    assert list(params) == ["z", "a", "m"]
    assert params == {"z": "first", "a": "second", "m": "third"}


@pytest.mark.slow
def test_adder_parameters_real_model(
    function_llm_adder, adder_function: Function
) -> None:
    prompt = "What is the addition of 5 + 2?"
    builder = PromptBuilder(
        function_llm_adder.llm, function_llm_adder.functions, prompt
    )

    params = function_llm_adder._get_parameters(adder_function, builder)

    assert params == {"a": 5.0, "b": 2.0}
