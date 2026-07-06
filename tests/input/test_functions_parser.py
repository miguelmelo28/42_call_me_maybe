from functions_parser import Function
from pathlib import Path
from typing import Any
from pytest import fixture
from pydantic import TypeAdapter

# @fixture(scope="module")
# def function_definitions() -> Any:
#     with open(Path("functions_definition.json")) as f:
#         data = json.load(f)
#     return data

def test_function_model() -> None:
    with open("tests/input/functions_definition.json") as f:
        data = TypeAdapter(list[Function]).validate_json(f.read())
    print(data)

