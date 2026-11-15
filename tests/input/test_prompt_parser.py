"""Tests for the ``Prompt`` input model."""

import json

import pytest
from pydantic import ValidationError

from src.parser import Prompt


def test_parse_list_json_reads_the_provided_prompts(prompts_json: str) -> None:
    parsed = Prompt.parse_list_json(prompts_json)

    assert len(parsed) == 11
    assert all(isinstance(prompt, Prompt) for prompt in parsed)
    assert parsed[0].prompt == "What is the sum of 2 and 3?"
    assert parsed[-1].prompt == (
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'"
    )


def test_prompts_fixture_matches_the_json(prompts: list[Prompt]) -> None:
    assert len(prompts) == 11
    assert prompts[0] == Prompt(prompt="What is the sum of 2 and 3?")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "hello",
        "Hello, world!",
        "unicode: café ✓",
        "a" * 5000,
        "line one\nline two",
    ],
    ids=[
        "empty",
        "whitespace",
        "simple",
        "punctuation",
        "unicode",
        "long",
        "multiline",
    ],
)
def test_valid_prompt_values_are_preserved(value: str) -> None:
    prompt = Prompt(prompt=value)

    assert prompt.prompt == value


def test_model_dump_only_contains_the_prompt() -> None:
    prompt = Prompt(prompt="Greet john")

    assert prompt.model_dump() == {"prompt": "Greet john"}
    assert prompt.model_dump(mode="json") == {"prompt": "Greet john"}


def test_model_validate_json_round_trips() -> None:
    prompt = Prompt(prompt="Greet john")

    assert Prompt.model_validate_json(prompt.model_dump_json()) == prompt


def test_parse_list_json_accepts_an_empty_list() -> None:
    assert Prompt.parse_list_json("[]") == []


def test_parse_list_json_accepts_multiple_prompts(
    prompts: list[Prompt],
) -> None:
    payload = json.dumps(
        [prompt.model_dump() for prompt in prompts[:3]]
    )

    assert Prompt.parse_list_json(payload) == prompts[:3]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing-prompt"),
        pytest.param({"promp": "Greet john"}, id="misspelled-key"),
        pytest.param(
            {"prompt": "Greet john", "extra": True},
            id="extra-field",
        ),
        pytest.param({"prompt": 123}, id="integer-prompt"),
        pytest.param({"prompt": None}, id="null-prompt"),
        pytest.param({"prompt": ["Greet", "john"]}, id="list-prompt"),
    ],
)
def test_invalid_prompt_is_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        Prompt.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        '"just a string"',
        "123",
        "null",
        '{"prompt": "Greet john", "extra": true}',
        "{not valid json",
    ],
)
def test_invalid_json_is_rejected(payload: str) -> None:
    with pytest.raises(ValidationError):
        Prompt.model_validate_json(payload)


def test_parse_list_json_rejects_an_object_instead_of_a_list() -> None:
    with pytest.raises(ValidationError):
        Prompt.parse_list_json('{"prompt": "Greet john"}')


def test_prompt_is_frozen() -> None:
    prompt = Prompt(prompt="hello")

    with pytest.raises(ValidationError):
        prompt.prompt = "other"  # type: ignore[misc]
