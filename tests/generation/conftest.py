"""Fixtures specific to the generation tests.

The input data and small ``Function`` fixtures live in ``tests/conftest.py``.
The expensive ``model`` fixture here loads the real local model, so tests that
use it are marked ``slow``.
"""

import pytest

from llm_sdk import Small_LLM_Model
from src.llm_function import LLM, LLM_Function, PromptBuilder
from src.parser import Function
from tests.generation.fakes import SPECIAL_TOKENS, ScriptedLLM, scripted_llm_steps


@pytest.fixture
def llm_stub() -> ScriptedLLM:
    """An LLM double that only provides ``special_tokens``.

    ``PromptBuilder`` never calls the model, so this is enough for prompt tests.
    """
    return scripted_llm_steps([])


@pytest.fixture
def prompt_builder(
    llm_stub: ScriptedLLM, adder_function: Function, greet_function: Function
) -> PromptBuilder:
    return PromptBuilder(
        llm_stub,
        [adder_function, greet_function],
        "What is the sum of 2 and 3?",
    )


# ---------------------------------------------------------------------------
# Expensive fixtures: these load the real model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model() -> LLM:
    return LLM(Small_LLM_Model(), SPECIAL_TOKENS)


@pytest.fixture(scope="session")
def function_llm_from_json(model: LLM, functions_json: str) -> LLM_Function:
    return LLM_Function.from_json(model, functions_json, SPECIAL_TOKENS)


@pytest.fixture(scope="session")
def function_llm_adder(model: LLM, adder_function: Function) -> LLM_Function:
    return LLM_Function(model, [adder_function], SPECIAL_TOKENS)


@pytest.fixture(scope="session")
def custom_function_llm(
    model: LLM, custom_functions_json: str
) -> LLM_Function:
    return LLM_Function.from_json(
        model, custom_functions_json, SPECIAL_TOKENS
    )
