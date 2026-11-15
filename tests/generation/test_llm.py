"""Tests for the thin ``LLM`` wrapper and ``SpecialTokens``."""

import pytest

from src.llm_function import LLM, SpecialTokens
from tests.generation.fakes import (
    SPECIAL_TOKENS,
    FakeModel,
    scripted_llm_steps,
)


def test_special_tokens_stores_all_fields() -> None:
    tokens = SpecialTokens(
        start_think="<t>",
        end_think="</t>",
        start_ctx_block="<s>",
        end_ctx_block="</s>",
    )
    assert tokens.start_think == "<t>"
    assert tokens.end_think == "</t>"
    assert tokens.start_ctx_block == "<s>"
    assert tokens.end_ctx_block == "</s>"


def test_special_tokens_uses_slots() -> None:
    tokens = SPECIAL_TOKENS
    with pytest.raises(AttributeError):
        tokens.unknown = "value"  # type: ignore[attr-defined]


def test_llm_stores_model_and_special_tokens() -> None:
    model = FakeModel()
    llm = LLM(model, SPECIAL_TOKENS)  # type: ignore[arg-type]
    assert llm.model is model
    assert llm.special_tokens is SPECIAL_TOKENS


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        ([0.1, 0.9, 0.5], [(1, 0.9), (2, 0.5), (0, 0.1)]),
        ([3.0, 1.0, 2.0], [(0, 3.0), (2, 2.0), (1, 1.0)]),
        ([5.0], [(0, 5.0)]),
    ],
)
def test_next_token_sorts_by_logit_descending(
    logits: list[float], expected: list[tuple[int, float]]
) -> None:
    fake_model = FakeModel(encoded_ids=[11, 22, 33], logits=logits)
    llm = LLM(fake_model, SPECIAL_TOKENS)  # type: ignore[arg-type]

    result = llm.next_token("hello")

    assert [token_id for token_id, _ in result] == [
        token_id for token_id, _ in expected
    ]
    assert [score for _, score in result] == pytest.approx(
        [score for _, score in expected]
    )


def test_next_token_encodes_the_context_and_uses_every_logit() -> None:
    fake_model = FakeModel(encoded_ids=[11, 22, 33], logits=[0.1, 0.9, 0.5])
    llm = LLM(fake_model, SPECIAL_TOKENS)  # type: ignore[arg-type]

    result = llm.next_token("some context")

    assert fake_model.encoded_texts == ["some context"]
    assert fake_model.logit_inputs == [[11, 22, 33]]
    assert len(result) == 3
    assert all(isinstance(token_id, int) for token_id, _ in result)
    assert all(isinstance(score, float) for _, score in result)


@pytest.mark.parametrize(
    ("token_id", "decoded"),
    [(1, "hello"), (2, " world"), (3, "")],
)
def test_decode_delegates_to_the_model(token_id: int, decoded: str) -> None:
    fake_model = FakeModel(decoded={(token_id,): decoded})
    llm = LLM(fake_model, SPECIAL_TOKENS)  # type: ignore[arg-type]

    assert llm.decode(token_id) == decoded
    assert fake_model.decode_calls == [[token_id]]


@pytest.mark.slow
def test_next_token_real_model_returns_ranked_candidates(model: LLM) -> None:
    candidates = model.next_token("The capital of France is")

    assert candidates
    assert all(isinstance(token_id, int) for token_id, _ in candidates)
    assert all(isinstance(score, float) for _, score in candidates)
    scores = [score for _, score in candidates]
    assert scores == sorted(scores, reverse=True)


def test_scripted_llm_steps_are_consumed_in_order() -> None:
    llm = scripted_llm_steps([[(1, "a")], [(2, "b")]])

    assert llm.next_token("ctx") == [(1, 1.0)]
    assert llm.next_token("ctx") == [(2, 1.0)]
    assert llm.contexts == ["ctx", "ctx"]
