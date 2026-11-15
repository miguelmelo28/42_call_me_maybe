"""Tests for the ``Function`` input model and its formatting helpers."""

import json

import pytest
from pydantic import ValidationError

from src.parser import Function

VALID_PARAMETER_TYPES = ["number", "string", "integer", "boolean"]


def make_payload(
    *,
    name: str = "fn_test",
    description: str = "Test function.",
    parameters: object = None,
    returns: object = None,
) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": (
            {"a": {"type": "number"}} if parameters is None else parameters
        ),
        "returns": {"type": "number"} if returns is None else returns,
    }


def test_parse_list_json_reads_the_provided_definitions(
    functions_json: str,
) -> None:
    parsed = Function.parse_list_json(functions_json)

    assert len(parsed) == 5
    assert all(isinstance(function, Function) for function in parsed)
    assert [function.name for function in parsed] == [
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    ]


def test_functions_fixture_matches_the_json(functions: list[Function]) -> None:
    assert len(functions) == 5
    assert {function.name for function in functions} == {
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    }


@pytest.mark.parametrize("parameter_type", VALID_PARAMETER_TYPES)
def test_valid_function_accepts_every_parameter_type(
    parameter_type: str,
) -> None:
    function = Function.model_validate(
        make_payload(parameters={"value": {"type": parameter_type}})
    )

    assert function.parameters == {"value": {"type": parameter_type}}


def test_valid_function_accepts_multiple_parameters(
    regex_function: Function,
) -> None:
    assert list(regex_function.parameters) == [
        "source_string",
        "regex",
        "replacement",
    ]


def test_empty_description_is_allowed() -> None:
    function = Function.model_validate(make_payload(description=""))

    assert function.description == ""


def test_function_description_formats_name_and_description(
    adder_function: Function,
) -> None:
    assert (
        adder_function.function_description()
        == "fn_add_numbers : Add two numbers together and return their sum."
    )


def test_get_formated_lists_every_parameter(adder_function: Function) -> None:
    assert adder_function.get_formated() == (
        "Function: fn_add_numbers\n\n"
        "Description: Add two numbers together and return their sum.\n\n"
        "Parameters:\n"
        "- a: number\n"
        "- b: number"
    )


@pytest.mark.parametrize("parameter_type", VALID_PARAMETER_TYPES)
def test_get_formated_supports_every_parameter_type(
    parameter_type: str,
) -> None:
    function = Function.model_validate(
        make_payload(parameters={"value": {"type": parameter_type}})
    )

    assert f"- value: {parameter_type}" in function.get_formated()


def test_get_formated_keeps_parameter_insertion_order() -> None:
    function = Function(
        name="fn_ordered",
        description="Ordered parameters.",
        parameters={
            "z": {"type": "string"},
            "a": {"type": "number"},
            "m": {"type": "boolean"},
        },
        returns={"type": "string"},
    )

    formatted = function.get_formated()

    assert formatted.index("- z: string") < formatted.index("- a: number")
    assert formatted.index("- a: number") < formatted.index("- m: boolean")


def test_model_dump_preserves_the_definition(adder_function: Function) -> None:
    assert adder_function.model_dump() == {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    }


def test_model_validate_json_round_trips(adder_function: Function) -> None:
    parsed = Function.model_validate_json(adder_function.model_dump_json())

    assert parsed == adder_function


def test_parse_list_json_accepts_an_empty_list() -> None:
    assert Function.parse_list_json("[]") == []


def test_parse_list_json_accepts_multiple_functions(
    adder_function: Function, greet_function: Function
) -> None:
    payload = json.dumps(
        [adder_function.model_dump(), greet_function.model_dump()]
    )

    parsed = Function.parse_list_json(payload)

    assert parsed == [adder_function, greet_function]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            make_payload(name="add_numbers"),
            id="missing-fn-prefix",
        ),
        pytest.param(make_payload(name=""), id="empty-name"),
        pytest.param(make_payload(name="FN_test"), id="uppercase-name"),
        pytest.param(make_payload(name=" fn_test"), id="leading-space-name"),
        pytest.param(
            make_payload(description=None),
            id="missing-description",
        ),
        pytest.param(make_payload(parameters={}), id="empty-parameters"),
        pytest.param(
            make_payload(parameters={"a": {"type": "array"}}),
            id="unsupported-parameter-type",
        ),
        pytest.param(
            make_payload(parameters={"a": {"typ": "number"}}),
            id="misspelled-parameter-type-key",
        ),
        pytest.param(
            make_payload(returns={"type": "array"}),
            id="invalid-return-type",
        ),
        pytest.param(make_payload(returns={}), id="empty-returns"),
        pytest.param(
            make_payload(returns={"type": "number", "extra": 1}),
            id="two-returns-keys",
        ),
        pytest.param(
            make_payload(parameters={"a": "number"}),
            id="non-object-parameter",
        ),
        pytest.param(
            make_payload(parameters={"a": {"type": "number", "extra": 1}}),
            id="extra-parameter-key",
        ),
        pytest.param(
            make_payload(parameters=[]),
            id="parameters-is-a-list",
        ),
        pytest.param(
            {**make_payload(), "extra": True},
            id="extra-top-level-field",
        ),
    ],
)
def test_invalid_function_is_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        Function.model_validate(payload)


def test_parameter_definition_requires_a_type_key() -> None:
    """Every parameter must have a type; an empty object is not enough."""
    payload = make_payload(parameters={"a": {}})

    with pytest.raises(ValidationError):
        Function.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "{not valid json",
        "null",
        "123",
        '"just a string"',
    ],
)
def test_invalid_json_is_rejected(payload: str) -> None:
    with pytest.raises(ValidationError):
        Function.parse_list_json(payload)


def test_parse_list_json_rejects_an_object_instead_of_a_list() -> None:
    payload = (
        '{"name": "fn_add_numbers", "description": "d",'
        ' "parameters": {"a": {"type": "number"}},'
        ' "returns": {"type": "number"}}'
    )

    with pytest.raises(ValidationError):
        Function.parse_list_json(payload)


def test_function_is_frozen(adder_function: Function) -> None:
    with pytest.raises(ValidationError):
        adder_function.name = "fn_other"  # type: ignore[misc]
