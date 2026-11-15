"""Tests for function selection (``LLM_Function._get_function``)."""

import pytest

from src.llm_function import LLM_Function, PromptBuilder
from src.parser import Function, Prompt
from tests.generation.fakes import (
    SPECIAL_TOKENS,
    ScriptedLLM,
    scripted_llm,
    scripted_llm_answers,
)


def make_llm_function(
    llm: ScriptedLLM, functions: list[Function]
) -> LLM_Function:
    return LLM_Function(llm, functions, SPECIAL_TOKENS)  # type: ignore[arg-type]


def make_builder(
    llm: ScriptedLLM, functions: list[Function], question: str
) -> PromptBuilder:
    return PromptBuilder(llm, functions, question)  # type: ignore[arg-type]


def test_get_function_selects_the_generated_function(
    adder_function: Function,
) -> None:
    llm = scripted_llm_answers("fn_add_numbers")
    llm_function = make_llm_function(llm, [adder_function])
    builder = make_builder(llm, [adder_function], "add 2 and 3")

    result = llm_function._get_function(builder)

    assert result == adder_function


def test_get_function_works_across_several_tokens(
    adder_function: Function,
) -> None:
    llm = scripted_llm("fn_add", "_numbers")
    llm_function = make_llm_function(llm, [adder_function])
    builder = make_builder(llm, [adder_function], "add 2 and 3")

    result = llm_function._get_function(builder)

    assert result == adder_function


def test_get_function_chooses_between_available_functions(
    adder_function: Function, greet_function: Function
) -> None:
    llm = scripted_llm_answers("fn_greet")
    llm_function = make_llm_function(llm, [adder_function, greet_function])
    builder = make_builder(
        llm, [adder_function, greet_function], "say hello to shrek"
    )

    result = llm_function._get_function(builder)

    assert result == greet_function


def test_get_function_sends_the_function_prompt_to_the_generator(
    adder_function: Function,
) -> None:
    llm = scripted_llm_answers("fn_add_numbers")
    llm_function = make_llm_function(llm, [adder_function])
    builder = make_builder(llm, [adder_function], "add 2 and 3")

    llm_function._get_function(builder)

    assert llm.contexts[0] == builder.function()
    assert "fn_add_numbers" in llm.contexts[0]


def test_get_function_rejects_an_incomplete_name(
    adder_function: Function,
) -> None:
    llm = scripted_llm("fn_add")
    llm_function = make_llm_function(llm, [adder_function])
    builder = make_builder(llm, [adder_function], "add 2 and 3")

    with pytest.raises(ValueError, match="invalid function: fn_add"):
        llm_function._get_function(builder)


@pytest.mark.slow
@pytest.mark.parametrize(
    "prompt",
    [
        "What is the addition of 5 + 2?",
        "Please add 5 and 2",
    ],
)
def test_get_function_real_model(function_llm_adder, prompt: str) -> None:
    builder = PromptBuilder(
        function_llm_adder.llm, function_llm_adder.functions, prompt
    )

    result = function_llm_adder._get_function(builder)

    assert result.name == "fn_add_numbers"


@pytest.mark.slow
def test_get_function_real_model_uses_the_prompt(
    function_llm_from_json,
) -> None:
    prompt = Prompt(prompt="Reverse the string 'hello'")
    builder = PromptBuilder(
        function_llm_from_json.llm, function_llm_from_json.functions, prompt.prompt
    )

    result = function_llm_from_json._get_function(builder)

    assert result.name == "fn_reverse_string"
