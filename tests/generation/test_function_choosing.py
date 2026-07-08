from llm_function import LLM_Function
from llm_sdk import Small_LLM_Model
from parser import Function, Prompt
from pytest import fixture
import pytest

@fixture(scope="module")
def empty_function_llm() -> LLM_Function:
    return LLM_Function(Small_LLM_Model(), [])

@fixture(scope="module")
def adder_function() -> Function:
    return Function(name="fn_add_numbers", description="adds two numbers together",
                    parameters={"a": {"type": "number"}, "b": {"type": "number"}}, returns={"type": "number"})

@fixture(scope="module")
def adder_prompt() -> Function:
    return Prompt(prompt="What is the adition of 5 + 2?")

@fixture(scope="module")
def function_llm_adder(adder_function):
    return LLM_Function(Small_LLM_Model(), [adder_function])

@pytest.mark.parametrize("phrase", ["coding is my", "the yellow fruit that monkeys eat is called", "queer people are", "the correct function to cerate a "])
def test_get_next_word(empty_function_llm: LLM_Function, phrase: str):
    wd = empty_function_llm._get_next_word(phrase)
    print(wd)
    assert len(wd.split()) == 1


@pytest.mark.parametrize("condition", [lambda k: 'a' not in k, lambda k: 'pear' in k])
def test_get_next_word_condition(empty_function_llm: LLM_Function, condition):
    phrase = "the yellow fruit that monkeys eat is called a"
    wd = empty_function_llm._get_next_word(phrase)
    wd_condition = empty_function_llm._get_next_word(phrase, condition)
    print(f"\n{wd} != {wd_condition}")
    assert condition(wd_condition)

@pytest.mark.parametrize("prompt", ["What is the adition of 5 + 2?", "Please add 5 and 2", "Add 2 and 5", "Sum of both 5 and 2"])
def test_adder_example(function_llm_adder: LLM_Function, prompt: str):
    print(function_llm_adder.get_response(Prompt(prompt=prompt)))