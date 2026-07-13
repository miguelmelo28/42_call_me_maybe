from huggingface_hub import list_models

print(next(iter(list_models())))