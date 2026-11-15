"""Tests for the prompt construction logic."""

import pytest

from src.llm_function import PromptBuilder, SpecialTokens
from src.parser import Function
from tests.generation.fakes import SPECIAL_TOKENS, scripted_llm_steps


def make_builder(
    functions: list[Function],
    question: str = "What is the sum of 2 and 3?",
    think: bool = False,
) -> PromptBuilder:
    return PromptBuilder(scripted_llm_steps([]), functions, question, think)


@pytest.mark.parametrize("profile", ["system", "user"])
def test_context_block_formats_profile_and_message(profile: str) -> None:
    builder = make_builder([])

    block = builder.context_block(profile, "hello")  # type: ignore[arg-type]

    assert block == f"<|im_start|>{profile}\nhello\n<|im_end|>"


def test_context_block_preserves_multiline_messages() -> None:
    builder = make_builder([])

    block = builder.context_block("system", "line one\nline two")

    assert "line one\nline two" in block
    assert block.startswith("<|im_start|>system\n")
    assert block.endswith("\n<|im_end|>")


def test_context_block_uses_the_configured_special_tokens() -> None:
    tokens = SpecialTokens("ST", "ET", "SC", "EC")
    builder = PromptBuilder(scripted_llm_steps([], special_tokens=tokens), [], "q")

    assert builder.context_block("user", "hi") == "SCuser\nhi\nEC"


def test_assistant_block_without_thinking_contains_empty_think_block() -> None:
    builder = make_builder([], think=False)

    block = builder.assistant_block()

    assert block == "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_assistant_block_with_thinking_does_not_force_an_empty_think_block() -> None:
    builder = make_builder([], think=True)

    block = builder.assistant_block()

    assert block == "<|im_start|>assistant\n"
    assert "<think>" not in block
    assert "</think>" not in block


def test_function_prompt_lists_every_function_name(
    prompt_builder: PromptBuilder,
) -> None:
    prompt = prompt_builder.function()

    assert "- fn_add_numbers" in prompt
    assert "- fn_greet" in prompt
    assert prompt_builder.question in prompt


def test_function_prompt_has_expected_blocks(prompt_builder: PromptBuilder) -> None:
    prompt = prompt_builder.function()

    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\n" in prompt
    assert "<|im_start|>assistant\n" in prompt
    assert prompt.count("<|im_start|>system") == 1
    assert prompt.count("<|im_start|>user") == 1
    assert prompt.count("<|im_start|>assistant") == 1
    assert prompt.endswith("<think>\n\n</think>\n\n")


def test_function_prompt_uses_the_user_question_as_the_user_block() -> None:
    builder = make_builder([], question="Pick the right one")

    prompt = builder.function()

    assert "<|im_start|>user\nPick the right one\n<|im_end|>" in prompt


@pytest.mark.parametrize(
    ("function_fixture", "param", "param_type"),
    [
        ("adder_function", "a", "number"),
        ("adder_function", "b", "number"),
        ("greet_function", "name", "string"),
        ("all_types_function", "i", "integer"),
        ("all_types_function", "b", "boolean"),
    ],
)
def test_parameter_prompt_contains_function_metadata(
    request: pytest.FixtureRequest,
    function_fixture: str,
    param: str,
    param_type: str,
) -> None:
    function: Function = request.getfixturevalue(function_fixture)
    builder = make_builder([function], question="Extract this")

    prompt = builder.parameter(function, param)

    assert function.get_formated() in prompt
    assert f"Extract the following parameter:\n{param} ({param_type})" in prompt
    assert f"Return only the value of {param}." in prompt
    assert "<|im_start|>user\nExtract this\n<|im_end|>" in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")
    assert prompt.count("<|im_start|>assistant") == 1


def test_parameter_prompt_does_not_mutate_the_function(
    adder_function: Function,
) -> None:
    builder = make_builder([adder_function])
    before = adder_function.model_dump()

    builder.parameter(adder_function, "a")

    assert adder_function.model_dump() == before


def test_parameter_prompt_works_for_every_parameter_of_a_function(
    all_types_function: Function,
) -> None:
    builder = make_builder([all_types_function])

    for name, metadata in all_types_function.parameters.items():
        prompt = builder.parameter(all_types_function, name)
        assert f"{name} ({metadata['type']})" in prompt
        assert f"Return only the value of {name}." in prompt


def test_prompt_builder_keeps_the_given_question() -> None:
    builder = make_builder([], question="question text")

    assert builder.question == "question text"
    assert builder.question in builder.function()
