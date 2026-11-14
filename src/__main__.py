# my_package/__main__.py
from .cli import parse
from .parser import Prompt, Function
from .llm_function import LLM_Function
from .output import Response
from pydantic import TypeAdapter
from llm_sdk import Small_LLM_Model
from .llm_function import SpecialTokens, LLM

SPECIAL_TOKENS = SpecialTokens(start_think="<think>",end_think="</think>",start_ctx_block="<|im_start|>", end_ctx_block="<|im_end|>")

def main() -> None:
    args = parse()
    functions = TypeAdapter(list[Function]).validate_json(
        args.functions_definition.read_text())
    prompts = TypeAdapter(list[Prompt]).validate_json(
        args.input.read_text())
    llm = LLM(Small_LLM_Model(), SPECIAL_TOKENS)
    llm_functions = LLM_Function(llm, functions=functions, special_tokens=SPECIAL_TOKENS)
    responses = [llm_functions.get_response(prompt) for prompt in prompts]
    print(responses)
    args.output.write_bytes(TypeAdapter(list[Response]).dump_json(responses,
                                                                   indent = 4))
    
if __name__ == "__main__":
    main()
