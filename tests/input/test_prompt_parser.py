from parser import Prompt
from pytest import fixture
import pytest
from pydantic import TypeAdapter

@fixture
def given_json() -> str:
    with open("tests/input/function_calling_tests.json") as f:
        return f.read()

def test_function_model(given_json) -> None:
    data = TypeAdapter(list[Prompt]).validate_json(given_json)
    print(data)


def test_list_parsing(given_json) -> None:
    print(Prompt.parse_list_json(given_json))

@pytest.mark.parametrize("bad_data", ['"prompt": "Greet john"',
                          r'{"promp": "Greet john"}'])
@pytest.mark.xfail(strict=True)
def test_bad_json(bad_data) -> None:
    Prompt.model_validate_json(bad_data)