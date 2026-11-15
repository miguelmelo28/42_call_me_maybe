"""Tests for the ``LLM_Function`` orchestration layer."""

import json

import pytest
from pydantic import ValidationError

from src.llm_function import LLM_Function
from src.parser import Function, Prompt
from tests.generation.fakes import (
    SPECIAL_TOKENS,
    ScriptedLLM,
    scripted_llm_answers,
    scripted_llm_steps,
)


def make_llm_function(
    llm: ScriptedLLM,
    functions: list[Function],
    max_tries: int = 1000,
) -> LLM_Function:
    return LLM_Function(  # type: ignore[arg-type]
        llm, functions, SPECIAL_TOKENS, max_tries
    )


# ---------------------------------------------------------------------------
# Construction and parsing
# ---------------------------------------------------------------------------


def test_init_stores_configuration(adder_function: Function) -> None:
    llm = scripted_llm_steps([])

    llm_function = make_llm_function(llm, [adder_function], max_tries=5)

    assert llm_function.llm is llm
    assert llm_function.functions == [adder_function]
    assert llm_function.max_tries == 5
    assert llm_function.constrained_generator.llm is llm


def test_init_propagates_max_tries_to_the_generator(
    adder_function: Function,
) -> None:
    llm = scripted_llm_steps([])

    llm_function = make_llm_function(llm, [adder_function], max_tries=5)

    assert llm_function.constrained_generator.max_tries == 5


def test_from_json_parses_the_function_definitions(functions_json: str) -> None:
    llm = scripted_llm_steps([])

    llm_function = LLM_Function.from_json(llm, functions_json, SPECIAL_TOKENS)

    assert len(llm_function.functions) == 5
    assert [function.name for function in llm_function.functions] == [
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    ]


def test_from_json_preserves_function_metadata(functions_json: str) -> None:
    llm = scripted_llm_steps([])

    llm_function = LLM_Function.from_json(llm, functions_json, SPECIAL_TOKENS)

    adder = llm_function.functions[0]
    assert adder.description == "Add two numbers together and return their sum."
    assert adder.parameters == {
        "a": {"type": "number"},
        "b": {"type": "number"},
    }
    assert adder.returns == {"type": "number"}


def test_from_json_accepts_an_empty_list() -> None:
    llm = scripted_llm_steps([])

    llm_function = LLM_Function.from_json(llm, "[]", SPECIAL_TOKENS)

    assert llm_function.functions == []


def test_from_json_rejects_invalid_json() -> None:
    llm = scripted_llm_steps([])

    with pytest.raises(ValidationError):
        LLM_Function.from_json(llm, "{not valid json", SPECIAL_TOKENS)


@pytest.mark.parametrize(
    "bad_definition",
    [
        {
            "name": "add_numbers",
            "description": "Missing the fn_ prefix.",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"},
        },
        {
            "name": "fn_bad",
            "description": "Unsupported parameter type.",
            "parameters": {"a": {"type": "array"}},
            "returns": {"type": "number"},
        },
        {
            "name": "fn_bad",
            "description": "Missing parameters.",
            "parameters": {},
            "returns": {"type": "number"},
        },
        {
            "name": "fn_bad",
            "description": "Missing the returns field.",
            "parameters": {"a": {"type": "number"}},
        },
        {
            "name": "fn_bad",
            "description": "Extra field is forbidden.",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"},
            "extra": True,
        },
    ],
)
def test_from_json_rejects_invalid_definitions(bad_definition: dict) -> None:
    llm = scripted_llm_steps([])

    with pytest.raises(ValidationError):
        LLM_Function.from_json(
            llm, json.dumps([bad_definition]), SPECIAL_TOKENS
        )


# ---------------------------------------------------------------------------
# get_response
# ---------------------------------------------------------------------------


def test_get_response_builds_a_complete_response(
    mixed_function: Function,
) -> None:
    llm = scripted_llm_answers("fn_mixed", "hello", "3.5", "true")
    llm_function = make_llm_function(llm, [mixed_function])
    prompt = Prompt(prompt="Do the mixed thing")

    response = llm_function.get_response(prompt)

    assert response.prompt == prompt
    assert response.function == mixed_function
    assert response.parameters == {"s": "hello", "n": 3.5, "b": True}
    assert isinstance(response.parameters["n"], float)
    assert isinstance(response.parameters["b"], bool)


def test_get_response_serializes_to_the_expected_output_format(
    mixed_function: Function,
) -> None:
    llm = scripted_llm_answers("fn_mixed", "hello", "3.5", "true")
    llm_function = make_llm_function(llm, [mixed_function])

    response = llm_function.get_response(Prompt(prompt="Do the mixed thing"))
    serialized = response.model_dump(mode="json")

    assert serialized == {
        "prompt": "Do the mixed thing",
        "name": "fn_mixed",
        "parameters": {"s": "hello", "n": 3.5, "b": True},
    }
    assert set(serialized) == {"prompt", "name", "parameters"}


def test_get_response_uses_the_selection_prompt_first(
    adder_function: Function, greet_function: Function
) -> None:
    llm = scripted_llm_answers("fn_greet", "shrek")
    llm_function = make_llm_function(llm, [adder_function, greet_function])

    llm_function.get_response(Prompt(prompt="Greet shrek"))

    selection_prompt = llm.contexts[0]
    assert "- fn_add_numbers" in selection_prompt
    assert "- fn_greet" in selection_prompt
    assert "Greet shrek" in selection_prompt


def test_get_response_uses_one_parameter_prompt_per_parameter(
    mixed_function: Function,
) -> None:
    llm = scripted_llm_answers("fn_mixed", "hello", "3.5", "true")
    llm_function = make_llm_function(llm, [mixed_function])

    llm_function.get_response(Prompt(prompt="Do the mixed thing"))

    assert "Extract the following parameter:\ns (string)" in llm.contexts[2]
    assert "Extract the following parameter:\nn (number)" in llm.contexts[4]
    assert "Extract the following parameter:\nb (boolean)" in llm.contexts[6]


def test_get_response_works_with_a_single_string_parameter(
    greet_function: Function,
) -> None:
    llm = scripted_llm_answers("fn_greet", "shrek")
    llm_function = make_llm_function(llm, [greet_function])

    response = llm_function.get_response(Prompt(prompt="Greet shrek"))

    assert response.function == greet_function
    assert response.parameters == {"name": "shrek"}


def test_get_response_raises_for_an_invalid_function(
    adder_function: Function,
) -> None:
    llm = scripted_llm_answers("fn_")
    llm_function = make_llm_function(llm, [adder_function])

    with pytest.raises(ValueError, match="invalid function: fn_"):
        llm_function.get_response(Prompt(prompt="add 2 and 3"))
