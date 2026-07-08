from parser import Function, Prompt
from output import Response
from llm_sdk import Small_LLM_Model
from pydantic import TypeAdapter
from typing import Self
from collections.abc import Callable
from heapq import nlargest
from itertools import count

class LLM_Function:
    def __init__(self, llm_model: Small_LLM_Model,
                 functions: list[Function]) -> None:
        self.llm_model = llm_model
        self.functions = functions

    @classmethod
    def from_json(cls, llm_model, json_functions: str)-> Self:
        functions = TypeAdapter(list[Function]).validate_json(json_functions)
        return cls(llm_model, functions)
    
    def get_response(self, prompt: Prompt):
        context = self._build_context(prompt)
        context += self._get_next_word(context, lambda w: any("fn_" + w in f.name for f in self.functions))
        return self._get_parameters(context)

    def _build_context(self, prompt: Prompt) -> str:
        context = "You have these available functions:\n"
        context += TypeAdapter(list[Function]).dump_json(self.functions).decode()
        context += "\nUsing ONLY these functions solve this problem: " + prompt.prompt
        context += "\nFunction = fn_"
        return context


    def _get_next_word(self, context: str,
                       condition: Callable[[str], bool] = lambda w: True) -> str:
        word = ""
        while len(word_list := word.split()) < 2 and not word.endswith(" "):
            encoded_ctxt: list[int] = self.llm_model.encode(context + word).tolist()[0]
            next = self.llm_model.get_logits_from_input_ids(encoded_ctxt)
            chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
            for i in count():
                idx_chosen = chosens[i][0]
                # print(chosens[i])
                wordpart = word + self.llm_model.decode([idx_chosen])
                # print(f"\n{word=}\n{wordpart=}\n")
                if condition(wordpart.strip()):
                    word = wordpart
                    break
        return word_list[0]

    def _get_parameters(self, context: str):
        params = []
        function_str = context.split()[-1]
        for func in self.functions:
            if func.name == function_str:
                function = func
                params.append(func.name)
                break
        else:
            raise ValueError(f"invalid function: {function_str}")
        for param in function.parameters:
            context += "\n" + param + " = "
            params.append(self._get_next_word(context))
        return params
