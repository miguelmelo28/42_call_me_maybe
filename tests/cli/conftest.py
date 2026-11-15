"""Fixtures for the CLI tests."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

import src.__main__ as main_module
from src.output import Response

GREET_FUNCTION = {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"},
}


@pytest.fixture
def run_main(monkeypatch, tmp_path: Path):
    """Run ``src.__main__.main`` with fake model/LLM_Function objects.

    Returns ``(output_path, prompts_seen_by_fake)``.
    """

    def run(prompts: list[str]) -> tuple[Path, list]:
        functions_path = tmp_path / "functions.json"
        prompts_path = tmp_path / "prompts.json"
        output_path = tmp_path / "output.json"
        functions_path.write_text(json.dumps([GREET_FUNCTION]))
        prompts_path.write_text(
            json.dumps([{"prompt": prompt} for prompt in prompts])
        )
        prompts_seen: list = []

        class FakeLLMFunction:
            def __init__(self, llm, functions, special_tokens) -> None:
                self.llm = llm
                self.functions = functions
                self.special_tokens = special_tokens

            def get_response(self, prompt) -> Response:
                prompts_seen.append(prompt)
                return Response(
                    prompt=prompt,
                    function=self.functions[0],
                    parameters={"name": "shrek"},
                )

        monkeypatch.setattr(
            main_module,
            "parse",
            lambda: Namespace(
                functions_definition=functions_path,
                input=prompts_path,
                output=output_path,
            ),
        )
        monkeypatch.setattr(main_module, "Small_LLM_Model", lambda: object())
        monkeypatch.setattr(
            main_module, "LLM", lambda model, special_tokens: object()
        )
        monkeypatch.setattr(main_module, "LLM_Function", FakeLLMFunction)

        main_module.main()

        return output_path, prompts_seen

    return run
