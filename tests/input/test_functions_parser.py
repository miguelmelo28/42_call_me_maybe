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

def test_list_parsing(given_json) -> None:
    print(Function.parse_list_json(given_json))

BAD_FUNCTIONS = [
  { "name": "fn_add_numbers",
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
  },
  { "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "str"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  { "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "string"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "str"
    }
  },
  { "name": "fn_add_numbers",
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
      "type": "number"
    }
  }
]

@pytest.mark.parametrize("bad_function", BAD_FUNCTIONS)
@pytest.mark.xfail(strict=True)
def test_bad_json(bad_function):
    print(Function.model_validate_json(bad_function))

