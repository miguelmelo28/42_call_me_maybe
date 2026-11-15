"""Fixtures for the output-model tests."""

from collections.abc import Callable

import pytest

from src.output import Response
from src.parser import Function, Prompt


@pytest.fixture
def function_factory() -> Callable[..., Function]:
    """Build a small ``Function`` without repeating the boilerplate."""

    def make(
        *,
        name: str = "fn_test",
        parameters: dict | None = None,
        returns: str = "string",
        description: str = "Test function.",
    ) -> Function:
        if parameters is None:
            parameters = {"x": {"type": "string"}}
        return Function(
            name=name,
            description=description,
            parameters=parameters,
            returns={"type": returns},
        )

    return make


@pytest.fixture
def response_factory() -> Callable[..., Response]:
    """Build a ``Response`` from a function and a parameter mapping."""

    def make(
        function: Function,
        parameters: dict,
        prompt: str = "test prompt",
    ) -> Response:
        return Response(
            prompt=Prompt(prompt=prompt),
            function=function,
            parameters=parameters,
        )

    return make
