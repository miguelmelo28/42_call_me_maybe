"""Tests for ``src.__main__.main`` with fakes instead of the real model."""

import json

from src.parser import Prompt


def test_main_writes_one_response_per_prompt(run_main) -> None:
    output, prompts_seen = run_main(["Greet shrek", "Greet john"])

    written = json.loads(output.read_text())

    assert written == [
        {
            "prompt": "Greet shrek",
            "name": "fn_greet",
            "parameters": {"name": "shrek"},
        },
        {
            "prompt": "Greet john",
            "name": "fn_greet",
            "parameters": {"name": "shrek"},
        },
    ]
    assert [prompt.prompt for prompt in prompts_seen] == [
        "Greet shrek",
        "Greet john",
    ]
    assert all(isinstance(prompt, Prompt) for prompt in prompts_seen)


def test_main_output_has_only_the_required_keys(run_main) -> None:
    output, _ = run_main(["Greet shrek"])

    written = json.loads(output.read_text())

    assert len(written) == 1
    assert set(written[0]) == {"prompt", "name", "parameters"}
    assert written[0]["name"] == "fn_greet"
    assert written[0]["parameters"] == {"name": "shrek"}


def test_main_handles_an_empty_prompt_list(run_main) -> None:
    output, prompts_seen = run_main([])

    assert json.loads(output.read_text()) == []
    assert prompts_seen == []


def test_main_prints_the_responses(run_main, capsys) -> None:
    run_main(["Greet shrek"])

    captured = capsys.readouterr()

    assert "Response" in captured.out
    assert "fn_greet" in captured.out


def test_main_overwrites_an_existing_output_file(run_main) -> None:
    output, _ = run_main(["Greet shrek"])
    output.write_text("old content")

    output_again, _ = run_main(["Greet john"])

    assert output_again == output
    assert json.loads(output.read_text()) == [
        {
            "prompt": "Greet john",
            "name": "fn_greet",
            "parameters": {"name": "shrek"},
        }
    ]
