"""Tests for ``ConstrainedGenerator``.

These are the most important unit tests in the project: each generator method
is exercised with deterministic token scripts, including malformed and
unexpected model outputs.
"""

import pytest

from src.llm_function import ConstrainedGenerator, SpecialTokens
from src.parser import Function
from tests.generation.fakes import (
    END_TOKEN_ID,
    ScriptedLLM,
    scripted_llm,
    scripted_llm_steps,
)


def make_generator(llm: ScriptedLLM, max_tries: int = 1000) -> ConstrainedGenerator:
    return ConstrainedGenerator(llm, max_tries)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_finds_the_end_token_and_keeps_configuration() -> None:
    llm = scripted_llm_steps([])
    generator = make_generator(llm, max_tries=7)

    assert generator.llm is llm
    assert generator.max_tries == 7
    assert generator.end_token == END_TOKEN_ID


def test_init_uses_the_special_tokens_from_the_llm() -> None:
    tokens = SpecialTokens("ST", "ET", "SC", "EC")
    llm = scripted_llm_steps([], special_tokens=tokens)

    generator = make_generator(llm)

    assert llm.model.encoded_texts == ["EC"]
    assert generator.end_token == END_TOKEN_ID


# ---------------------------------------------------------------------------
# unconstrained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("hello",), "hello"),
        (("he", "llo"), "hello"),
        (("hello", " world"), "hello world"),
        (("a", "b", "c", "d"), "abcd"),
    ],
)
def test_unconstrained_collects_tokens_until_the_end_token(
    parts: tuple[str, ...], expected: str
) -> None:
    llm = scripted_llm(*parts)
    generator = make_generator(llm)

    assert generator.unconstrained("ctx") == expected


def test_unconstrained_passes_the_growing_context_to_the_model() -> None:
    llm = scripted_llm("he", "llo")
    generator = make_generator(llm)

    generator.unconstrained("ctx")

    assert llm.contexts == ["ctx", "ctxhe", "ctxhello"]


def test_unconstrained_returns_empty_string_for_immediate_end_token() -> None:
    llm = scripted_llm()
    generator = make_generator(llm)

    assert generator.unconstrained("ctx") == ""


def test_unconstrained_stops_at_the_first_end_token() -> None:
    llm = scripted_llm_steps(
        [
            [(1, "a"), (2, "b")],
            [(END_TOKEN_ID, ""), (3, "c")],
        ]
    )
    generator = make_generator(llm)

    assert generator.unconstrained("ctx") == "a"
    assert llm.contexts == ["ctx", "ctxa"]


# ---------------------------------------------------------------------------
# word
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "expected", "contexts"),
    [
        (("hello",), "hello", ["ctx", "ctxhello"]),
        (("hel", "lo"), "hello", ["ctx", "ctxhel", "ctxhello"]),
        (("hello", " world"), "hello", ["ctx", "ctxhello"]),
        (("hello ",), "hello", ["ctx"]),
    ],
)
def test_word_stops_at_one_word(
    parts: tuple[str, ...], expected: str, contexts: list[str]
) -> None:
    llm = scripted_llm(*parts)
    generator = make_generator(llm)

    assert generator.word("ctx") == expected
    assert llm.contexts == contexts


def test_word_accepts_the_first_candidate_satisfying_the_condition() -> None:
    llm = scripted_llm_steps(
        [
            [(1, "x"), (2, "2")],
            [(END_TOKEN_ID, "")],
        ]
    )
    generator = make_generator(llm)

    assert generator.word("ctx", lambda value: value.isdigit()) == "2"


def test_word_applies_the_condition_to_the_stripped_candidate() -> None:
    llm = scripted_llm_steps(
        [
            [(1, " 2")],
            [(END_TOKEN_ID, "")],
        ]
    )
    generator = make_generator(llm)

    assert generator.word("ctx", lambda value: value == "2") == "2"


def test_word_returns_first_word_when_several_words_are_generated() -> None:
    llm = scripted_llm("hello", " beautiful", " world")
    generator = make_generator(llm)

    assert generator.word("ctx") == "hello"


def test_word_raises_when_no_candidate_matches_within_max_tries() -> None:
    llm = scripted_llm_steps([[(1, "x"), (2, "y")]])
    generator = make_generator(llm, max_tries=2)

    with pytest.raises(ValueError, match="could not find the answer"):
        generator.word("ctx", lambda value: False)


def test_word_raises_gracefully_when_candidates_are_exhausted() -> None:
    """A short candidate list must not cause an IndexError."""
    llm = scripted_llm_steps([[(1, "x")]])
    generator = make_generator(llm, max_tries=1000)

    with pytest.raises(ValueError, match="could not find the answer"):
        generator.word("ctx", lambda value: False)


def test_word_immediate_end_token_does_not_crash() -> None:
    llm = scripted_llm()
    generator = make_generator(llm)

    assert generator.word("ctx") == ""


# ---------------------------------------------------------------------------
# function
# ---------------------------------------------------------------------------


def test_function_selects_an_exact_function(
    adder_function: Function, greet_function: Function
) -> None:
    llm = scripted_llm("fn_add_numbers")
    generator = make_generator(llm)

    result = generator.function("ctx", [adder_function, greet_function])

    assert result == adder_function


def test_function_can_be_generated_across_several_tokens(
    adder_function: Function,
) -> None:
    llm = scripted_llm("fn_add", "_numbers")
    generator = make_generator(llm)

    result = generator.function("ctx", [adder_function])

    assert result == adder_function


def test_function_selects_between_several_functions(
    adder_function: Function, greet_function: Function, reverse_function: Function
) -> None:
    llm = scripted_llm("fn_greet")
    generator = make_generator(llm)

    result = generator.function(
        "ctx", [adder_function, greet_function, reverse_function]
    )

    assert result == greet_function


def test_function_rejects_a_prefix_that_is_not_a_full_name(
    adder_function: Function,
) -> None:
    llm = scripted_llm("fn_add")
    generator = make_generator(llm)

    with pytest.raises(ValueError, match="invalid function: fn_add"):
        generator.function("ctx", [adder_function])


def test_function_rejects_an_unknown_name(adder_function: Function) -> None:
    llm = scripted_llm("fn_")
    generator = make_generator(llm)

    with pytest.raises(ValueError, match="invalid function: fn_"):
        generator.function("ctx", [adder_function])


# ---------------------------------------------------------------------------
# string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("hello",), "hello"),
        (("'hello'",), "hello"),
        (('"hello"',), "hello"),
        (("'hello world'",), "hello world"),
        (("''hello''",), "hello"),
        (("''",), ""),
        (("'he", "llo'"), "hello"),
    ],
)
def test_string_strips_surrounding_quotes(
    parts: tuple[str, ...], expected: str
) -> None:
    llm = scripted_llm(*parts)
    generator = make_generator(llm)

    assert generator.string("ctx") == expected


def test_string_only_strips_quotes_not_whitespace() -> None:
    llm = scripted_llm("  hello  ")
    generator = make_generator(llm)

    assert generator.string("ctx") == "  hello  "


# ---------------------------------------------------------------------------
# number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", 42.0),
        ("3.14", 3.14),
        ("-7", -7.0),
        ("0", 0.0),
        ("-0.5", -0.5),
        ("123456789", 123456789.0),
    ],
)
def test_number_parses_valid_numbers(text: str, expected: float) -> None:
    llm = scripted_llm(text)
    generator = make_generator(llm)

    result = generator.number("ctx")

    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_number_strips_surrounding_whitespace() -> None:
    llm = scripted_llm("  12  ")
    generator = make_generator(llm)

    assert generator.number("ctx") == 12.0


def test_number_rejects_a_second_decimal_point() -> None:
    llm = scripted_llm_steps(
        [
            [(1, "1.2")],
            [(2, "."), (END_TOKEN_ID, "")],
        ]
    )
    generator = make_generator(llm)

    assert generator.number("ctx") == 1.2


def test_number_rejects_a_minus_sign_after_the_first_character() -> None:
    llm = scripted_llm_steps(
        [
            [(1, "1")],
            [(2, "-2"), (END_TOKEN_ID, "")],
        ]
    )
    generator = make_generator(llm)

    assert generator.number("ctx") == 1.0


def test_number_raises_for_a_bare_minus_sign() -> None:
    llm = scripted_llm("-")
    generator = make_generator(llm)

    with pytest.raises(ValueError, match="Could not generate a valid number"):
        generator.number("ctx")


# ---------------------------------------------------------------------------
# integer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", 42),
        ("-7", -7),
        ("0", 0),
        ("123456789", 123456789),
    ],
)
def test_integer_parses_valid_integers(text: str, expected: int) -> None:
    llm = scripted_llm(text)
    generator = make_generator(llm)

    result = generator.integer("ctx")

    assert result == expected
    assert isinstance(result, int)


def test_integer_strips_surrounding_whitespace() -> None:
    llm = scripted_llm("  12  ")
    generator = make_generator(llm)

    assert generator.integer("ctx") == 12


def test_integer_rejects_decimal_points() -> None:
    llm = scripted_llm_steps(
        [
            [(1, "3")],
            [(2, ".14"), (END_TOKEN_ID, "")],
        ]
    )
    generator = make_generator(llm)

    assert generator.integer("ctx") == 3


def test_integer_rejects_a_minus_sign_after_the_first_character() -> None:
    llm = scripted_llm_steps(
        [
            [(1, "1")],
            [(2, "-2"), (END_TOKEN_ID, "")],
        ]
    )
    generator = make_generator(llm)

    assert generator.integer("ctx") == 1


def test_integer_raises_for_a_bare_minus_sign() -> None:
    llm = scripted_llm("-")
    generator = make_generator(llm)

    with pytest.raises(ValueError, match="Could not generate a valid number"):
        generator.integer("ctx")


# ---------------------------------------------------------------------------
# bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True),
        ("false", False),
        ("True", True),
        ("False", False),
        ("TRUE", True),
        ("FALSE", False),
    ],
)
def test_bool_parses_boolean_spellings(text: str, expected: bool) -> None:
    llm = scripted_llm(text)
    generator = make_generator(llm)

    result = generator.bool("ctx")

    assert result is expected
    assert isinstance(result, bool)


def test_bool_accepts_a_prefix_while_generating() -> None:
    llm = scripted_llm_steps(
        [
            [(1, "tr"), (2, "maybe")],
            [(3, "ue")],
            [(END_TOKEN_ID, "")],
        ]
    )
    generator = make_generator(llm)

    assert generator.bool("ctx") is True


def test_bool_rejects_a_prefix_that_never_becomes_a_boolean() -> None:
    llm = scripted_llm("tru")
    generator = make_generator(llm)

    with pytest.raises(ValueError, match="Invalid booleal: tru"):
        generator.bool("ctx")
