"""Validation for the custom real-model test data.

These tests are fast and do not touch the model.  They make sure the custom
JSON fixtures are internally consistent before the slow generation tests run.
"""

import json
from pathlib import Path

import pytest

from src.parser import Function, Prompt

INPUT_DIR = Path(__file__).parent
EXPECTED_FILE = INPUT_DIR / "custom_expected_responses.json"

EXPECTED = json.loads(EXPECTED_FILE.read_text())


def test_custom_functions_parse_and_have_unique_names(
    custom_functions: list[Function],
) -> None:
    assert len(custom_functions) == 12
    assert len({function.name for function in custom_functions}) == len(
        custom_functions
    )
    assert all(function.name.startswith("fn_") for function in custom_functions)


def test_custom_functions_cover_every_parameter_type(
    custom_functions: list[Function],
) -> None:
    parameter_types = {
        metadata["type"]
        for function in custom_functions
        for metadata in function.parameters.values()
    }

    assert parameter_types == {"number", "string", "integer", "boolean"}


def test_custom_functions_include_multi_parameter_examples(
    custom_functions: list[Function],
) -> None:
    assert any(len(function.parameters) >= 3 for function in custom_functions)


def test_custom_prompts_parse_and_are_unique(
    custom_prompts: list[Prompt],
) -> None:
    assert len(custom_prompts) == 20
    assert len({prompt.prompt for prompt in custom_prompts}) == len(
        custom_prompts
    )


def test_expected_responses_align_with_prompts(
    custom_prompts: list[Prompt], custom_expected: list[dict]
) -> None:
    assert len(custom_expected) == len(custom_prompts)
    assert [expected["prompt"] for expected in custom_expected] == [
        prompt.prompt for prompt in custom_prompts
    ]


def test_expected_responses_use_every_defined_function(
    custom_functions: list[Function], custom_expected: list[dict]
) -> None:
    expected_names = {expected["name"] for expected in custom_expected}

    assert expected_names == {function.name for function in custom_functions}


@pytest.mark.parametrize(
    "expected",
    EXPECTED,
    ids=[f"{index}-{expected['name']}" for index, expected in enumerate(EXPECTED)],
)
def test_expected_parameters_match_the_function_schema(
    custom_functions: list[Function], expected: dict
) -> None:
    functions_by_name = {function.name: function for function in custom_functions}
    function = functions_by_name[expected["name"]]

    assert set(expected["parameters"]) == set(function.parameters)
    for name, value in expected["parameters"].items():
        declared_type = function.parameters[name]["type"]
        if declared_type == "number":
            assert isinstance(value, (int, float))
            assert not isinstance(value, bool)
        elif declared_type == "string":
            assert isinstance(value, str)
        elif declared_type == "integer":
            assert isinstance(value, int)
            assert not isinstance(value, bool)
        elif declared_type == "boolean":
            assert isinstance(value, bool)
        else:
            pytest.fail(f"Unknown declared parameter type: {declared_type}")
