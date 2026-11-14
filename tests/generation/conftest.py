from src.llm_function import LLM_Function, LLM
from llm_sdk import Small_LLM_Model
from src.parser import Function, Prompt
from pytest import fixture
from src.prompt_builder import SpecialTokens
import pytest
from pydantic import TypeAdapter

SPECIAL_TOKENS = SpecialTokens(start_think="<think>",end_think="</think>",start_ctx_block="<|im_start|>", end_ctx_block="<|im_end|>")

@pytest.fixture(scope="session")
def model() -> LLM:
    return LLM(Small_LLM_Model(), SPECIAL_TOKENS)

@fixture(scope="session")
def function_llm_from_json(model) -> LLM_Function:
    with open("tests/input/functions_definition.json") as f:
        return LLM_Function.from_json(model, f.read(), SPECIAL_TOKENS)

@fixture(scope="session")
def empty_function_llm(model) -> LLM_Function:
    return LLM_Function(model, [], SPECIAL_TOKENS)

@fixture(scope="session")
def adder_function() -> Function:
    return Function(name="fn_add_numbers", description="adds two numbers together",
                    parameters={"a": {"type": "number"}, "b": {"type": "number"}}, returns={"type": "number"})

@fixture(scope="session")
def reverse_function() -> Function:
    return Function(name="fn_reverse_str", description="reverses a string and returns the reversed result",
                    parameters={"s": {"type": "string"}}, returns={"type": "string"})


@fixture(scope="session")
def adder_prompt() -> Prompt:
    return Prompt(prompt="What is the adition of 5 + 2?")

@fixture(scope="session")
def function_llm_adder(adder_function, model):
    return LLM_Function(model, [adder_function], SPECIAL_TOKENS)

@fixture(scope = "session")
def prompt_list_from_json() -> list[Prompt]:
    with open("tests/input/function_calling_tests.json") as f:
        return TypeAdapter(list[Prompt]).validate_json(f.read())
