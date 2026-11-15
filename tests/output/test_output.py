"""Tests for the output ``Response`` model."""

import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from src.output import Response
from src.parser import Function, Prompt


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("144", 144.0),
        (144, 144.0),
        (144.5, 144.5),
        ("-12.5", -12.5),
        ("1e3", 1000.0),
        ("+3", 3.0),
        (" 2.5 ", 2.5),
        (0, 0.0),
        (-0.0, -0.0),
    ],
    ids=[
        "numeric-string",
        "int",
        "float",
        "negative-string",
        "scientific-string",
        "plus-string",
        "whitespace-string",
        "zero-int",
        "negative-zero",
    ],
)
def test_valid_number_parameters_are_converted_to_float(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
    value,
    expected: float,
) -> None:
    function = function_factory(parameters={"x": {"type": "number"}})

    response = response_factory(function, {"x": value})

    assert response.parameters["x"] == expected
    assert isinstance(response.parameters["x"], float)


@pytest.mark.parametrize(
    "value",
    ["", "hello", "Hello, world!", "café ✓", "line one\nline two", "a" * 1000],
    ids=["empty", "simple", "punctuation", "unicode", "multiline", "long"],
)
def test_valid_string_parameters_are_preserved(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
    value: str,
) -> None:
    function = function_factory(parameters={"x": {"type": "string"}})

    response = response_factory(function, {"x": value})

    assert response.parameters["x"] == value
    assert isinstance(response.parameters["x"], str)


@pytest.mark.parametrize("value", [True, False])
def test_valid_boolean_parameters_are_preserved(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
    value: bool,
) -> None:
    function = function_factory(parameters={"x": {"type": "boolean"}})

    response = response_factory(function, {"x": value})

    assert response.parameters["x"] is value


def test_integer_parameter_is_supported(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    """``TypeDict`` allows integers, so ``Response`` must validate them too."""
    function = function_factory(parameters={"x": {"type": "integer"}})

    response = response_factory(function, {"x": 7})

    assert response.parameters["x"] == 7
    assert isinstance(response.parameters["x"], int)


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param({}, id="missing"),
        pytest.param({"x": "hello", "extra": "value"}, id="extra"),
        pytest.param({"extra": "value"}, id="only-extra"),
        pytest.param({"y": "hello"}, id="wrong-name"),
    ],
)
def test_parameter_keys_must_match_the_function(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
    parameters: dict,
) -> None:
    function = function_factory(parameters={"x": {"type": "string"}})

    with pytest.raises(ValidationError):
        response_factory(function, parameters)


@pytest.mark.parametrize(
    ("parameter_type", "value"),
    [
        ("number", "hello"),
        ("number", {"hello": "world"}),
        ("number", []),
        ("number", None),
        ("string", 12),
        ("string", 12.5),
        ("string", True),
        ("string", None),
        ("boolean", 1),
        ("boolean", 0),
        ("boolean", 1.0),
        ("boolean", "true"),
        ("boolean", "false"),
        ("boolean", None),
    ],
)
def test_wrong_parameter_type_is_rejected(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
    parameter_type: str,
    value,
) -> None:
    function = function_factory(parameters={"x": {"type": parameter_type}})

    with pytest.raises(ValidationError):
        response_factory(function, {"x": value})


@pytest.mark.parametrize("value", [True, False])
def test_boolean_is_rejected_for_a_number_parameter(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
    value: bool,
) -> None:
    """A JSON boolean is not a JSON number."""
    function = function_factory(parameters={"x": {"type": "number"}})

    with pytest.raises(ValidationError):
        response_factory(function, {"x": value})


def test_multiple_parameters_are_validated(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory(
        parameters={
            "s": {"type": "string"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
        }
    )

    response = response_factory(
        function, {"s": "hello", "n": "2.5", "b": False}
    )

    assert response.parameters == {"s": "hello", "n": 2.5, "b": False}
    assert isinstance(response.parameters["n"], float)


def test_multiple_parameters_reject_one_wrong_value(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory(
        parameters={
            "s": {"type": "string"},
            "n": {"type": "number"},
        }
    )

    with pytest.raises(ValidationError):
        response_factory(function, {"s": 12, "n": 2.5})


def test_extra_top_level_key_is_rejected(
    function_factory: Callable[..., Function],
) -> None:
    function = function_factory()
    payload = {
        "prompt": {"prompt": "hello"},
        "function": function.model_dump(),
        "parameters": {"x": "hello"},
        "extra": True,
    }

    with pytest.raises(ValidationError):
        Response.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"function": {"name": "fn_test"}, "parameters": {"x": "hello"}},
            id="missing-prompt",
        ),
        pytest.param(
            {"prompt": {"prompt": "hello"}, "parameters": {"x": "hello"}},
            id="missing-function",
        ),
        pytest.param(
            {"prompt": {"prompt": "hello"}, "function": {"name": "fn_test"}},
            id="missing-parameters",
        ),
    ],
)
def test_missing_top_level_key_is_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "prompt": "hello",
                "function": {"name": "fn_test"},
                "parameters": {},
            },
            id="prompt-is-a-string",
        ),
        pytest.param(
            {
                "prompt": {"prompt": "hello"},
                "function": "fn_test",
                "parameters": {},
            },
            id="function-is-a-string",
        ),
        pytest.param(
            {
                "prompt": {"prompt": "hello"},
                "function": {"name": "fn_test"},
                "parameters": [],
            },
            id="parameters-is-a-list",
        ),
    ],
)
def test_wrong_top_level_types_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(payload)


def test_json_serialization_uses_the_required_output_schema(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory(
        name="fn_mixed",
        parameters={
            "s": {"type": "string"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
        },
    )
    response = response_factory(
        function,
        {"s": "hello", "n": 2.0, "b": True},
        prompt="Do the mixed thing",
    )

    serialized = response.model_dump(mode="json")

    assert serialized == {
        "prompt": "Do the mixed thing",
        "name": "fn_mixed",
        "parameters": {"s": "hello", "n": 2.0, "b": True},
    }
    assert set(serialized) == {"prompt", "name", "parameters"}
    assert isinstance(serialized["prompt"], str)
    assert isinstance(serialized["name"], str)
    assert isinstance(serialized["parameters"], dict)


def test_serialized_json_is_parseable_and_uses_aliases(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory(name="fn_greet")
    response = response_factory(
        function, {"x": "hello"}, prompt="Greet hello"
    )

    serialized = json.loads(response.model_dump_json())

    assert serialized == {
        "prompt": "Greet hello",
        "name": "fn_greet",
        "parameters": {"x": "hello"},
    }
    assert "function" not in serialized


def test_serialization_preserves_unicode_and_special_characters(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory(parameters={"x": {"type": "string"}})
    response = response_factory(
        function,
        {"x": "café ✓ \"quoted\""},
        prompt="Prompt with café ✓ and 'quotes'",
    )

    serialized = json.loads(response.model_dump_json())

    assert serialized["prompt"] == "Prompt with café ✓ and 'quotes'"
    assert serialized["parameters"]["x"] == "café ✓ \"quoted\""


def test_serialization_does_not_include_nested_models(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    response = response_factory(
        function_factory(), {"x": "hello"}, prompt="hello"
    )

    serialized = response.model_dump(mode="json")

    assert serialized["prompt"] == "hello"
    assert serialized["name"] == "fn_test"
    assert serialized["parameters"] == {"x": "hello"}
    assert "description" not in serialized
    assert "returns" not in serialized


def test_model_validate_json_accepts_nested_input(
    function_factory: Callable[..., Function],
) -> None:
    function = function_factory()
    payload = json.dumps(
        {
            "prompt": {"prompt": "hello"},
            "function": function.model_dump(),
            "parameters": {"x": "hello"},
        }
    )

    response = Response.model_validate_json(payload)

    assert response.prompt == Prompt(prompt="hello")
    assert response.function == function
    assert response.parameters == {"x": "hello"}


@pytest.mark.parametrize(
    "payload",
    [
        "{not valid json",
        "[]",
        "null",
        '"just a string"',
    ],
)
def test_invalid_json_is_rejected(payload: str) -> None:
    with pytest.raises(ValidationError):
        Response.model_validate_json(payload)


def test_response_keeps_nested_prompt_and_function_models(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory()
    response = response_factory(function, {"x": "hello"}, prompt="hello")

    assert isinstance(response.prompt, Prompt)
    assert isinstance(response.function, Function)
    assert response.prompt.prompt == "hello"
    assert response.function == function


def test_parameter_order_does_not_matter(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory(
        name="fn_two",
        parameters={
            "a": {"type": "number"},
            "b": {"type": "string"},
        },
    )

    response = response_factory(function, {"b": "hello", "a": "2"})

    assert response.parameters == {"b": "hello", "a": 2.0}


def test_empty_prompt_and_empty_string_parameter_are_allowed(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    function = function_factory(parameters={"x": {"type": "string"}})

    response = response_factory(function, {"x": ""}, prompt="")

    assert response.prompt.prompt == ""
    assert response.parameters["x"] == ""


def test_response_is_frozen(
    function_factory: Callable[..., Function],
    response_factory: Callable[..., Response],
) -> None:
    response = response_factory(function_factory(), {"x": "hello"})

    with pytest.raises(ValidationError):
        response.prompt = Prompt(prompt="other")  # type: ignore[misc]
