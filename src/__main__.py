# my_package/__main__.py
from .cli import parse
from .parser import Prompt, Function
from .llm_function import LLM_Function
from .output import Response
from pydantic import TypeAdapter
from llm_sdk import Small_LLM_Model

def main() -> None:
    args = parse()
    functions = TypeAdapter(list[Function]).validate_json(
        args.functions_definition.read_text())
    prompts = TypeAdapter(list[Prompt]).validate_json(
        args.input.read_text())
    llm = LLM_Function(Small_LLM_Model, functions=functions)
    responses = [print(llm.get_response(prompt)) for prompt in prompts]
    args.output.write_bytes(TypeAdapter(list[Response]).dump_json(responses,
                                                                   indent = 4))
    
if __name__ == "__main__":
    main()
