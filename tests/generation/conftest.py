from llm_function import LLM_Function
from llm_sdk import Small_LLM_Model
from parser import Function, Prompt
from pytest import fixture
import pytest
from pydantic import TypeAdapter

@pytest.fixture(scope="session")
def model() -> Small_LLM_Model:
    return Small_LLM_Model()

@fixture(scope="session")
def function_llm_from_json(model) -> LLM_Function:
    with open("tests/input/functions_definition.json") as f:
        return LLM_Function.from_json(model, f.read())

@fixture(scope="session")
def empty_function_llm(model) -> LLM_Function:
    return LLM_Function(model, [])

@fixture(scope="session")
def adder_function() -> Function:
    return Function(name="fn_add_numbers", description="adds two numbers together",
                    parameters={"a": {"type": "number"}, "b": {"type": "number"}}, returns={"type": "number"})

@fixture(scope="session")
def adder_prompt() -> Function:
    return Prompt(prompt="What is the adition of 5 + 2?")

@fixture(scope="session")
def function_llm_adder(adder_function, model):
    return LLM_Function(model, [adder_function])