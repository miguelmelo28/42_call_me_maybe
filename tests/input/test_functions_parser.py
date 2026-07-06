from parser import Function
from pytest import fixture
import pytest
from pydantic import TypeAdapter

@fixture(scope="module")
def given_json() -> str:
    with open("tests/input/functions_definition.json") as f:
        return f.read()

def test_given_json(given_json) -> None:
    data = TypeAdapter(list[Function]).validate_json(given_json)
    print(data)

@pytest.mark.xfail
def test_bad_json(given_json):
    bad_json = """  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "typ": "number"
    }
  }"""
    print(Function.model_validate_json(bad_json))