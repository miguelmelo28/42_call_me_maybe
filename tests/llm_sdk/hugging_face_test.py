"""Optional smoke test for the Hugging Face Hub integration."""

import pytest


@pytest.mark.slow
def test_hugging_face_hub_lists_models() -> None:
    from huggingface_hub import list_models

    assert next(iter(list_models(limit=1))) is not None
