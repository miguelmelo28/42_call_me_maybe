from .parser import Function, Prompt
from .output import Response
from llm_sdk import Small_LLM_Model
from pydantic import TypeAdapter
from typing import Self
from collections.abc import Callable
from itertools import count


class LLM_Function:
    def __init__(self, llm_model: Small_LLM_Model,
                 functions: list[Function], max_tries: int = 1000) -> None:
        self.llm_model = llm_model
        self.functions = functions
        self.max_tries = max_tries

    @classmethod
    def from_json(cls, llm_model: Small_LLM_Model,
                  json_functions: str) -> Self:
        functions = TypeAdapter(list[Function]).validate_json(json_functions)
        return cls(llm_model, functions)

    def get_response(self, prompt: Prompt) -> Response:
        function: Function = self._get_function(prompt)
        parameters = self._get_parameters(function, prompt)
        return Response(prompt=prompt, function=function,
                        parameters=parameters)

    def _build_context_function(self, prompt: Prompt) -> str:
        context = "Out of the functions "
        context += " ".join([f.name for f in self.functions])
        context += " the one that solves this query: '"
        context += prompt.prompt + "' is "
        return context

    def _build_context_parameters(self, prompt: Prompt,
                                  function: Function) -> str:
        context = function.get_formated()
        context += f"\nIt can be used to solve this problem:" \
                   f"\"{prompt.prompt}\" using the following arguments:\n"
        return context

    def _get_function_word(self, context: str) -> str:
        word = ""
        functions = [f.name for f in self.functions]
        print(context)
        while (word).strip() not in functions:
            encoded_ctxt = self.llm_model.encode(context + word).tolist()[0]
            next = self.llm_model.get_logits_from_input_ids(encoded_ctxt)
            chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
            for i in range(self.max_tries):
                idx_chosen = chosens[i][0]
                # print(chosens[i])
                wordpart = self.llm_model.decode([idx_chosen]).strip()
                # print(f"\n{word=}\n{wordpart=}\n")
                # sleep(0.05)

                if (any(f.startswith(word + wordpart) for f in functions)
                   and wordpart):
                    word += wordpart
                    break
            else:
                functions = [func for func in functions if word not in func]
                word = ""
                # print(functions)
        print(word)
        return word.strip()

    def _get_next_string(self, context: str) -> str:
        string = "'"
        while (not (string.endswith("'") or string.endswith("\n"))
               or len(string) <= 1):
            encoded_ctxt = self.llm_model.encode(context + string).tolist()[0]
            next = self.llm_model.get_logits_from_input_ids(encoded_ctxt)
            chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
            idx_chosen = chosens[0][0]
            wordpart = self.llm_model.decode([idx_chosen])
            string += wordpart
            # print(string)
        return string[1:string.rfind("'")]

    def _get_number(self, context: str) -> float:
        number: str = self._get_next_word(context)
        allowed = set("0123456789.")
        number: str = "".join(c for c in number if c in allowed)
        if number[-1] == '.':
            number = number[:-1]
        return float(number)

    def _get_next_word(self, context: str,
                       condition: Callable[[str], bool] = lambda w: True) -> str:
        word = ""
        while len(word_list := word.split()) < 2 and not word.endswith(" "):
            encoded_ctxt = self.llm_model.encode(context + word).tolist()[0]
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

    def _get_function(self, prompt=Prompt) -> Function:
        context = self._build_context_function(prompt)
        function: str = self._get_function_word(context)
        for func in self.functions:
            if func.name == function:
                return func
        else:
            raise ValueError(f"invalid function: {function}")

    def _get_parameters(self, func: Function,
                        prompt: Prompt) -> dict[str, str | float]:
        params: dict[str, str | float] = {}
        context = self._build_context_parameters(prompt, func)
        for arg in func.parameters:
            arg_type = func.parameters[arg]["type"]
            context += f"{arg}({arg_type}) = "
            match arg_type:
                case "string":
                    param = self._get_next_string(context)
                case "number":
                    param = self._get_number(context)
            params[arg] = param
            context += f'{str(param)}\n'
        print(context)
        return params
