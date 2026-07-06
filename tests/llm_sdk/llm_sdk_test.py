from pytest import fixture
from llm_sdk import Small_LLM_Model
from torch import Tensor

@fixture(scope="module")
def small_llm_model() -> Small_LLM_Model:
    return Small_LLM_Model()

def test_is_function(small_llm_model: Small_LLM_Model) -> None:
    assert isinstance(small_llm_model.encode("hi"), Tensor)

def test_get_logits_from_input_ids(small_llm_model: Small_LLM_Model) -> None:
    logits = small_llm_model.get_logits_from_input_ids([1,2,8])
    # print(f"{logits = }")
    assert isinstance(logits, list)
    # assert logits == [0.001, 0.002, 0.997]

def test_get_vocab(small_llm_model: Small_LLM_Model) -> None:
    vocab = small_llm_model.get_path_to_vocab_file()
    print(f"{vocab = }")


def test_model(small_llm_model: Small_LLM_Model):
    tokens = small_llm_model.encode("I'm testing this thing let's see")[0].tolist()
    for i in range(100):
        logits = small_llm_model.get_logits_from_input_ids(tokens)
        tokens.append(logits.index(max(logits)))
        print(f"{i}")
    print(small_llm_model.decode(tokens))
