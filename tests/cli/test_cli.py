"""Tests for ``src.cli.parse``."""

from pathlib import Path

import pytest

from src.cli import parse


def test_default_paths(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["call_me_maybe"])

    args = parse()

    assert args.functions_definition == Path(
        "data/input/functions_definition.json"
    )
    assert args.input == Path("data/input/function_calling_tests.json")
    assert args.output == Path("data/output/hello.json")


def test_custom_paths(monkeypatch, tmp_path: Path) -> None:
    functions = tmp_path / "functions.json"
    prompts = tmp_path / "prompts.json"
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "call_me_maybe",
            "--functions_definition",
            str(functions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ],
    )

    args = parse()

    assert args.functions_definition == functions
    assert args.input == prompts
    assert args.output == output


def test_parse_prints_the_selected_paths(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "call_me_maybe",
            "--functions_definition",
            "functions.json",
            "--input",
            "prompts.json",
            "--output",
            "output.json",
        ],
    )

    parse()

    captured = capsys.readouterr()
    assert "functions.json" in captured.out
    assert "prompts.json" in captured.out
    assert "output.json" in captured.out


def test_help_exits_successfully(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["call_me_maybe", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        parse()

    assert exc_info.value.code == 0


def test_unknown_argument_exits_with_error(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["call_me_maybe", "--unknown"])

    with pytest.raises(SystemExit) as exc_info:
        parse()

    assert exc_info.value.code == 2
