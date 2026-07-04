from pytest import fixture
from llm_sdk import Small_LLM_Model
from torch import Tensor

@fixture
def small_llm_model() -> Small_LLM_Model:
    return Small_LLM_Model()

def test_is_function(small_llm_model: Small_LLM_Model):
    assert isinstance(small_llm_model.encode("hi"), Tensor)