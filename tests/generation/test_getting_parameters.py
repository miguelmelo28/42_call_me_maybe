from llm_function import LLM_Function
from llm_sdk import Small_LLM_Model
from parser import Function, Prompt
from pytest import fixture
import pytest
from pydantic import TypeAdapter

@pytest.mark.parametrize("prompt", ["What is the adition of 5 + 2?", "Please add 5 and 2", "Add 2 and 5", "Sum of both 5 and 2"])
def test_adding_numbers(empty_function_llm: LLM_Function, adder_function: Function, prompt):
    parameters = empty_function_llm._get_parameters(adder_function, Prompt(prompt=prompt))
    print(parameters)

@pytest.mark.parametrize("prompt", ["reverse the name 'miguel'", "put the phrase 'hello i am good how are u' in the oposite order", "'stay away' reverse the order of that"])
def test_reverse_string(empty_function_llm: LLM_Function, reverse_function: Function, prompt):
    parameters = empty_function_llm._get_parameters(reverse_function, Prompt(prompt=prompt))
    print(parameters)