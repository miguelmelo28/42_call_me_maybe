"""Shared paths and fixtures for the whole test suite.

Keeping the input data and the small, immutable ``Function`` fixtures here
means the parser, output, generation and CLI tests all use exactly the same
definitions.
"""

import json
from pathlib import Path

import pytest

from src.parser import Function, Prompt

INPUT_DIR = Path(__file__).parent / "input"
FUNCTIONS_FILE = INPUT_DIR / "functions_definition.json"
PROMPTS_FILE = INPUT_DIR / "function_calling_tests.json"
CUSTOM_FUNCTIONS_FILE = INPUT_DIR / "custom_functions_definition.json"
CUSTOM_PROMPTS_FILE = INPUT_DIR / "custom_function_calling_tests.json"
CUSTOM_EXPECTED_FILE = INPUT_DIR / "custom_expected_responses.json"


# ---------------------------------------------------------------------------
# Raw JSON fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def functions_json() -> str:
    return FUNCTIONS_FILE.read_text()


@pytest.fixture(scope="session")
def prompts_json() -> str:
    return PROMPTS_FILE.read_text()


@pytest.fixture(scope="session")
def custom_functions_json() -> str:
    return CUSTOM_FUNCTIONS_FILE.read_text()


@pytest.fixture(scope="session")
def custom_prompts_json() -> str:
    return CUSTOM_PROMPTS_FILE.read_text()


@pytest.fixture(scope="session")
def custom_expected_json() -> str:
    return CUSTOM_EXPECTED_FILE.read_text()


# ---------------------------------------------------------------------------
# Parsed data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def functions(functions_json: str) -> list[Function]:
    return Function.parse_list_json(functions_json)


@pytest.fixture(scope="session")
def prompts(prompts_json: str) -> list[Prompt]:
    return Prompt.parse_list_json(prompts_json)


@pytest.fixture(scope="session")
def custom_functions(custom_functions_json: str) -> list[Function]:
    return Function.parse_list_json(custom_functions_json)


@pytest.fixture(scope="session")
def custom_prompts(custom_prompts_json: str) -> list[Prompt]:
    return Prompt.parse_list_json(custom_prompts_json)


@pytest.fixture(scope="session")
def custom_expected(custom_expected_json: str) -> list[dict]:
    return json.loads(custom_expected_json)


# ---------------------------------------------------------------------------
# Small reusable Function definitions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def adder_function() -> Function:
    return Function(
        name="fn_add_numbers",
        description="Add two numbers together and return their sum.",
        parameters={"a": {"type": "number"}, "b": {"type": "number"}},
        returns={"type": "number"},
    )


@pytest.fixture(scope="session")
def greet_function() -> Function:
    return Function(
        name="fn_greet",
        description="Generate a greeting message for a person by name.",
        parameters={"name": {"type": "string"}},
        returns={"type": "string"},
    )


@pytest.fixture(scope="session")
def reverse_function() -> Function:
    return Function(
        name="fn_reverse_string",
        description="Reverse a string and return the reversed result.",
        parameters={"s": {"type": "string"}},
        returns={"type": "string"},
    )


@pytest.fixture(scope="session")
def square_root_function() -> Function:
    return Function(
        name="fn_get_square_root",
        description="Calculate the square root of a number.",
        parameters={"a": {"type": "number"}},
        returns={"type": "number"},
    )


@pytest.fixture(scope="session")
def regex_function() -> Function:
    return Function(
        name="fn_substitute_string_with_regex",
        description="Replace all occurrences matching a regex pattern in a string.",
        parameters={
            "source_string": {"type": "string"},
            "regex": {"type": "string"},
            "replacement": {"type": "string"},
        },
        returns={"type": "string"},
    )


@pytest.fixture(scope="session")
def mixed_function() -> Function:
    """Function with the string / number / boolean types used by the pipeline."""
    return Function(
        name="fn_mixed",
        description="Exercise string, number and boolean parameters.",
        parameters={
            "s": {"type": "string"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
        },
        returns={"type": "string"},
    )


@pytest.fixture(scope="session")
def all_types_function() -> Function:
    """Function covering every parameter type accepted by ``TypeDict``."""
    return Function(
        name="fn_all_types",
        description="Exercise every supported parameter type.",
        parameters={
            "s": {"type": "string"},
            "n": {"type": "number"},
            "i": {"type": "integer"},
            "b": {"type": "boolean"},
        },
        returns={"type": "string"},
    )
