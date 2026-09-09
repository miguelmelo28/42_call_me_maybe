from src.llm_function import LLM_Function
from llm_sdk import Small_LLM_Model
from src.parser import Function, Prompt
from pytest import fixture
import pytest
from pydantic import TypeAdapter

@pytest.mark.parametrize("prompt", ["What is the adition of 5 + 2?", "Please add 5 and 2", "Add 2 and 5", "Sum of both 5 and 2"])
def test_adding_numbers(empty_function_llm: LLM_Function, adder_function: Function, prompt):
    parameters = empty_function_llm._get_parameters(adder_function, Prompt(prompt=prompt))
    print(parameters)

@pytest.mark.parametrize("prompt", ["reverse the name 'miguel'", "put the phrase 'hello i am good how are u' in the oposite order", "'stay away' reverse the order of that"])
def test_reverse_string(function_llm_from_json, prompt):
    parameters = function_llm_from_json._get_parameters(function_llm_from_json.functions[2], Prompt(prompt=prompt))
    print(parameters)

@pytest.mark.parametrize("prompt", ["Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS", "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat", "Replace all vowels in 'Programming is fun' with asterisks"])
def test_regex_str(function_llm_from_json, prompt):
    parameters = function_llm_from_json._get_parameters(function_llm_from_json.functions[4], Prompt(prompt=prompt))
    print(parameters)