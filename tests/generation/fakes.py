"""Deterministic test doubles for the LLM layer.

The real model is slow to load and its output is not guaranteed to be stable
between runs, so most unit tests use these fakes instead.  Tests that really
need the model are marked with ``pytest.mark.slow``.
"""

from collections import deque
from collections.abc import Iterable, Sequence

from src.llm_function import SpecialTokens

SPECIAL_TOKENS = SpecialTokens(
    start_think="<think>",
    end_think="</think>",
    start_ctx_block="<|im_start|>",
    end_ctx_block="<|im_end|>",
)

END_TOKEN_ID = 0


class ArrayLike:
    """Small stand-in for the tensor returned by ``Small_LLM_Model.encode``."""

    def __init__(self, values: list[list[int]]) -> None:
        self.values = values

    def tolist(self) -> list[list[int]]:
        return self.values

    def __getitem__(self, index: int) -> list[int]:
        return self.values[index]


class FakeModel:
    """Minimal model exposing the small subset of the SDK used by ``LLM``."""

    def __init__(
        self,
        encoded_ids: Sequence[int] = (END_TOKEN_ID,),
        logits: Sequence[float] | None = None,
        decoded: dict[tuple[int, ...], str] | None = None,
    ) -> None:
        self.encoded_ids = list(encoded_ids)
        self.logits = list(logits) if logits is not None else []
        self.decoded = dict(decoded or {})
        self.encoded_texts: list[str] = []
        self.logit_inputs: list[list[int]] = []
        self.decode_calls: list[list[int]] = []

    def encode(self, text: str) -> ArrayLike:
        self.encoded_texts.append(text)
        return ArrayLike([list(self.encoded_ids)])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        self.logit_inputs.append(list(input_ids))
        return list(self.logits)

    def decode(self, ids: list[int]) -> str:
        values = list(ids)
        self.decode_calls.append(values)
        return self.decoded.get(tuple(values), "")


class ScriptedLLM:
    """LLM double that returns pre-scripted, already ranked token candidates.

    Each call to :meth:`next_token` pops one batch.  A batch is a list of
    ``(token_id, score)`` pairs, ordered from most to least likely.
    """

    def __init__(
        self,
        steps: Iterable[Iterable[tuple[int, float]]],
        vocab: dict[int, str] | None = None,
        end_token_id: int = END_TOKEN_ID,
        special_tokens: SpecialTokens = SPECIAL_TOKENS,
    ) -> None:
        self.steps: deque[list[tuple[int, float]]] = deque(
            [list(step) for step in steps]
        )
        self.vocab: dict[int, str] = dict(vocab or {})
        self.vocab.setdefault(end_token_id, "")
        self.end_token_id = end_token_id
        self.special_tokens = special_tokens
        self.model = FakeModel(encoded_ids=[end_token_id])
        self.contexts: list[str] = []
        self.decode_calls: list[int] = []

    def next_token(self, context: str) -> list[tuple[int, float]]:
        self.contexts.append(context)
        if not self.steps:
            raise AssertionError(
                "ScriptedLLM ran out of steps: the code under test generated "
                "more tokens than the test scripted"
            )
        return list(self.steps.popleft())

    def decode(self, token_id: int) -> str:
        self.decode_calls.append(token_id)
        return self.vocab.get(token_id, "")


def scripted_llm_steps(
    steps: Iterable[Sequence[tuple[int, str]]],
    special_tokens: SpecialTokens = SPECIAL_TOKENS,
) -> ScriptedLLM:
    """Build a :class:`ScriptedLLM` from ``(token_id, text)`` batches."""
    vocab: dict[int, str] = {END_TOKEN_ID: ""}
    batches: list[list[tuple[int, float]]] = []
    for step in steps:
        batch: list[tuple[int, float]] = []
        for rank, (token_id, text) in enumerate(step):
            vocab[token_id] = text
            batch.append((token_id, float(len(step) - rank)))
        batches.append(batch)
    return ScriptedLLM(batches, vocab, END_TOKEN_ID, special_tokens)


def scripted_llm(
    *parts: str,
    special_tokens: SpecialTokens = SPECIAL_TOKENS,
) -> ScriptedLLM:
    """Emit each part as one token, then emit the end-of-context token."""
    steps: list[list[tuple[int, str]]] = [
        [(index + 1, part)] for index, part in enumerate(parts)
    ]
    steps.append([(END_TOKEN_ID, "")])
    return scripted_llm_steps(steps, special_tokens)


def scripted_llm_answers(
    *answers: str,
    special_tokens: SpecialTokens = SPECIAL_TOKENS,
) -> ScriptedLLM:
    """Emit each answer followed by the end token.

    This models one complete constrained generation per answer, which is handy
    for tests that exercise a full ``get_response`` call.
    """
    steps: list[list[tuple[int, str]]] = []
    for index, answer in enumerate(answers):
        steps.append([(index + 1, answer)])
        steps.append([(END_TOKEN_ID, "")])
    return scripted_llm_steps(steps, special_tokens)
