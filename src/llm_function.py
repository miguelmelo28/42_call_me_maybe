from parser import Function, Prompt
from output import Response
from llm_sdk import Small_LLM_Model
from pydantic import TypeAdapter
from typing import Self
from collections.abc import Callable
from itertools import count
from time import sleep


class LLM_Function:
    def __init__(self, llm_model: Small_LLM_Model,
                 functions: list[Function]) -> None:
        self.llm_model = llm_model
        self.functions = functions

    @classmethod
    def from_json(cls, llm_model: Small_LLM_Model, json_functions: str) -> Self:
        functions = TypeAdapter(list[Function]).validate_json(json_functions)
        return cls(llm_model, functions)

    def get_response(self, prompt: Prompt) -> Response:
        context = self._build_context(prompt)
        function = self._get_function(context)
        parameters = self._get_parameters(context, function, prompt.prompt)
        return Response(prompt=prompt, function=function, parameters=parameters)

    # def _build_context(self, prompt: Prompt) -> str:
    #     context = "You have these available functions:\n"
    #     context += "\n".join(func.function_description() for func in self.functions)
    #     # context += TypeAdapter[list[Function]].dump_json(self.functions)
    #     context += "\nUsing ONLY these functions solve this problem: " + prompt.prompt
    #     context += "\nFunction = fn_"
    #     return context

    # def _build_context(self, prompt: Prompt) -> str:
    #     context = "Out of the functions "
    #     context += ", ".join([f.name for f in self.functions])
    #     context += " the one that solves this query: '"
    #     context += prompt.prompt + "' is "
    #     return context

    def _build_context(self, prompt: Prompt) -> str:
        # context += TypeAdapter[list[Function]].dump_json(self.functions)
        context = "\n".join(f.get_formated() for f in self.functions)
        context += "\nThis problem: " + prompt.prompt + ", can be solved with this function:"
        #context += "\nFunction = fn_"
        return context

    def _get_function_word(self, context: str) -> str:
        word = ""
        functions = [f.name for f in self.functions]
        while (word).strip() not in functions:
            encoded_ctxt: list[int] = self.llm_model.encode(context + word).tolist()[0]
            next = self.llm_model.get_logits_from_input_ids(encoded_ctxt)
            chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
            for i in range(1000):
                idx_chosen = chosens[i][0]
                # print(chosens[i])
                wordpart = self.llm_model.decode([idx_chosen]).strip()
                # print(f"\n{word=}\n{wordpart=}\n")
                # sleep(0.005)

                if any(f.startswith(word + wordpart) for f in functions) and wordpart:
                    word += wordpart
                    break
            else:
                functions = [func for func in functions if not word in func]
                word=""
                print(functions)
        print(word)
        return word.strip()

    def _get_next_string(self, context: str, prompt: str) -> str:
        string = ""
        divider = "'\""
        while not string.endswith(divider) or len(string) <= 1:
            encoded_ctxt: list[int] = self.llm_model.encode(context + string).tolist()[0]
            next = self.llm_model.get_logits_from_input_ids(encoded_ctxt)
            chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
            for i in count():
                idx_chosen = chosens[i][0]
                # print(chosens[i])
                wordpart = self.llm_model.decode([idx_chosen])
                if string + wordpart in prompt:
                    if string == "":
                        if any(wordpart.startswith(d) for d in divider):
                            divider = wordpart[0]
                        else:
                            continue
                    string += wordpart
                    break
        print(string)
        return string[1:-1]

    # def _get_number_word(self, context: str) -> str:
    #     number = ""
        

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
        print(word)
        return word_list[0]

    def _get_function(self, context: str) -> Function:
        function: str = self._get_function_word(context)
        # function: str = self._get_next_word(context, lambda w: any("fn_" + w in f.name for f in self.functions))
        for func in self.functions:
            if func.name == function:
                return func
        else:
            raise ValueError(f"invalid function: {function}")

    def _get_parameters(self, context: str, func: Function, prompt: str) -> dict[str, str | float]:
        params: dict[str, str | float] = {}
        context = context + func.name + ", given the parameters"
        for arg in func.parameters:
            arg_type = func.parameters[arg]["type"]
            context += f" {arg}({arg_type}): "
            match arg_type:
                case "string":
                    param = self._get_next_string(context, prompt)
                case "number":
                    param = self._get_next_word(context)
            params[arg] = param
            context += f"{param},"
        print(context)
        return params
