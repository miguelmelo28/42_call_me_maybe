"""Slow, real-model tests using the custom function set.

The data lives in ``tests/input/custom_*.json``.  These tests load Qwen and are
therefore marked ``slow``.  They intentionally cover every parameter type,
multi-parameter functions, negative numbers, empty strings and punctuation.
"""

import json
from pathlib import Path

import pytest

from src.llm_function import LLM_Function, PromptBuilder
from src.parser import Function, Prompt

INPUT_DIR = Path(__file__).resolve().parents[1] / "input"
FUNCTIONS_FILE = INPUT_DIR / "custom_functions_definition.json"
PROMPTS_FILE = INPUT_DIR / "custom_function_calling_tests.json"
EXPECTED_FILE = INPUT_DIR / "custom_expected_responses.json"

CUSTOM_FUNCTIONS = Function.parse_list_json(FUNCTIONS_FILE.read_text())
CUSTOM_PROMPTS = Prompt.parse_list_json(PROMPTS_FILE.read_text())
EXPECTED_RESPONSES = json.loads(EXPECTED_FILE.read_text())
FUNCTIONS_BY_NAME = {function.name: function for function in CUSTOM_FUNCTIONS}

CASES = list(zip(CUSTOM_PROMPTS, EXPECTED_RESPONSES))
CASE_IDS = [
    f"{index}-{expected['name']}"
    for index, expected in enumerate(EXPECTED_RESPONSES)
]

# One selection-only case per custom function.
SELECTION_CASES: list[tuple[Prompt, dict]] = []
_seen_functions: set[str] = set()
for prompt, expected in CASES:
    if expected["name"] not in _seen_functions:
        _seen_functions.add(expected["name"])
        SELECTION_CASES.append((prompt, expected))
SELECTION_IDS = [
    f"{expected['name']}" for _, expected in SELECTION_CASES
]

# Parameter-extraction subset: all four types, multi-parameter functions,
# negative numbers, an empty string and punctuation.
PARAMETER_INDICES = [0, 3, 5, 7, 8, 11, 13, 14, 15, 16, 17, 18]
PARAMETER_CASES = [(CUSTOM_PROMPTS[index], EXPECTED_RESPONSES[index])
                   for index in PARAMETER_INDICES]
PARAMETER_IDS = [
    f"{index}-{EXPECTED_RESPONSES[index]['name']}"
    for index in PARAMETER_INDICES
]

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("prompt, expected", CASES, ids=CASE_IDS)
def test_custom_full_response_real_model(
    custom_function_llm: LLM_Function, prompt: Prompt, expected: dict
) -> None:
    response = custom_function_llm.get_response(prompt)

    assert response.prompt == prompt
    assert response.function.name == expected["name"]
    assert response.parameters == expected["parameters"]

    serialized = response.model_dump(mode="json")
    assert serialized == {
        "prompt": prompt.prompt,
        "name": expected["name"],
        "parameters": expected["parameters"],
    }
    assert set(serialized) == {"prompt", "name", "parameters"}


@pytest.mark.parametrize(
    "prompt, expected", SELECTION_CASES, ids=SELECTION_IDS
)
def test_custom_function_selection_real_model(
    custom_function_llm: LLM_Function, prompt: Prompt, expected: dict
) -> None:
    builder = PromptBuilder(
        custom_function_llm.llm,
        custom_function_llm.functions,
        prompt.prompt,
    )

    selected = custom_function_llm._get_function(builder)

    assert selected.name == expected["name"]
    assert selected == FUNCTIONS_BY_NAME[expected["name"]]


@pytest.mark.parametrize(
    "prompt, expected", PARAMETER_CASES, ids=PARAMETER_IDS
)
def test_custom_parameter_extraction_real_model(
    custom_function_llm: LLM_Function, prompt: Prompt, expected: dict
) -> None:
    function = FUNCTIONS_BY_NAME[expected["name"]]
    builder = PromptBuilder(
        custom_function_llm.llm,
        custom_function_llm.functions,
        prompt.prompt,
    )

    parameters = custom_function_llm._get_parameters(function, builder)

    assert parameters == expected["parameters"]
    for name, value in parameters.items():
        declared_type = function.parameters[name]["type"]
        if declared_type == "number":
            assert isinstance(value, float)
        elif declared_type == "string":
            assert isinstance(value, str)
        elif declared_type == "integer":
            assert isinstance(value, int)
            assert not isinstance(value, bool)
        elif declared_type == "boolean":
            assert isinstance(value, bool)
        else:
            pytest.fail(f"Unknown declared parameter type: {declared_type}")
