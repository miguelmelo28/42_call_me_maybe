"""End-to-end tests against the real local model.

These are marked ``slow`` because they load Qwen3-0.6B and run a full
constrained generation for every prompt.  Run them with::

    uv run pytest -m slow

or run everything with::

    uv run pytest
"""

from pathlib import Path

import pytest

from src.parser import Prompt

PROMPTS_FILE = Path(__file__).resolve().parents[1] / "input" / "function_calling_tests.json"

PROMPTS = Prompt.parse_list_json(PROMPTS_FILE.read_text())

EXPECTED_RESPONSES = [
    ("fn_add_numbers", {"a": 2.0, "b": 3.0}),
    ("fn_add_numbers", {"a": 265.0, "b": 345.0}),
    ("fn_greet", {"name": "shrek"}),
    ("fn_greet", {"name": "john"}),
    ("fn_reverse_string", {"s": "hello"}),
    ("fn_reverse_string", {"s": "world"}),
    ("fn_get_square_root", {"a": 16.0}),
    ("fn_get_square_root", {"a": 144.0}),
    (
        "fn_substitute_string_with_regex",
        {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": "([0-9]+)",
            "replacement": "NUMBERS",
        },
    ),
    (
        "fn_substitute_string_with_regex",
        {"source_string": "Programming is fun", "regex": "a|e|i|o|u", "replacement": "*"},
    ),
    (
        "fn_substitute_string_with_regex",
        {
            "source_string": "The cat sat on the mat with another cat",
            "regex": "cat",
            "replacement": "dog",
        },
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("prompt, responses", list(zip(PROMPTS, EXPECTED_RESPONSES)))
def test_ex_inputs(function_llm_from_json, prompt: Prompt, responses) -> None:
    expected_name, expected_parameters = responses

    response = function_llm_from_json.get_response(prompt)

    assert response.prompt == prompt
    assert response.function.name == expected_name
    assert response.parameters == expected_parameters

    serialized = response.model_dump(mode="json")
    assert serialized == {
        "prompt": prompt.prompt,
        "name": expected_name,
        "parameters": expected_parameters,
    }
    assert set(serialized) == {"prompt", "name", "parameters"}
