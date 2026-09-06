from pytest import fixture, mark
from output import Response
from parser import Function, Prompt
from pydantic import TypeAdapter

responses = [
    ('fn_add_numbers', {'a' : 2.0, 'b': 3.0}),
    ('fn_add_numbers', {'a' : 265.0, 'b': 345.0}),
    ('fn_greet', {'name': 'shrek'}),
    ('fn_greet', {'name': 'john'}),
    ('fn_reverse_string', {'s':'hello'}),
    ('fn_reverse_string', {'s':'world'}),
    ('fn_get_square_root', {'a': 16.0}),
    ('fn_get_square_root', {'a': 144.0}),
    ('fn_substitute_string_with_regex', {'source_string': 'Hello 34 I\'m 233 years old', 'regex': '([0-9]+)', 'replacement': 'NUMBERS'}),
    ('fn_substitute_string_with_regex', {'source_string': 'Programming is fun', 'regex': 'a|e|i|o|u', 'replacement': '*'}),
    ('fn_substitute_string_with_regex', {'source_string': 'The cat sat on the mat with another cat', 'regex': 'cat', 'replacement': 'dog'})
]

def prompt_list_from_json() -> list[Prompt]:
    with open("tests/input/function_calling_tests.json") as f:
        return TypeAdapter(list[Prompt]).validate_json(f.read())

@mark.parametrize('prompt, responses', list(zip(prompt_list_from_json(), responses)))
def test_ex_inputs(function_llm_from_json, prompt, responses):
    response = function_llm_from_json.get_response(prompt)
    print(response)
    assert response.function.name == responses[0]
    assert response.parameters == responses[1]
