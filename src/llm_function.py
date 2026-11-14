from .parser import Function, Prompt, TypeDict
from .prompt_builder import PromptBuilder
from .output import Response
from llm_sdk import Small_LLM_Model
from pydantic import TypeAdapter
from typing import Self
from collections.abc import Callable
from itertools import count
from typing import Literal
from time import sleep
from typing import TypedDict
from dataclasses import dataclass

@dataclass(slots=True)
class SpecialTokens():
    start_think: str
    end_think: str
    start_ctx_block: str
    end_ctx_block: str

class LLM:
    def __init__(self, model: Small_LLM_Model, special_tokens: SpecialTokens):
        self.model = model
        self.special_tokens = special_tokens

    def next_token(self, context: str) -> list[tuple[int, float]]:
        encoded_ctx: list[int] = self.model.encode(context).tolist()[0]
        next: list[float] = self.model.get_logits_from_input_ids(encoded_ctx)
        chosen: list[tuple[int, float]] = sorted(
            enumerate(next), key=lambda x: x[1], reverse=True
        )
        return chosen

    def decode(self, token_id: int) -> str:
        return self.model.decode([token_id])

class ConstrainedGenerator:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.max_tries = 1000
        self.end_token = self.llm.model.encode(self.llm.special_tokens.end_ctx_block)[0][0]

    def unconstrained(self, context: str) -> str:
        word = ""
        while True:
            chosens = self.llm.next_token(context + word)
            idx_chosen = chosens[0][0]
            word += self.llm.decode(idx_chosen)
            # print(word, idx_chosen)
            if idx_chosen == self.end_token:
                break
        return word


    def word(
        self, context: str, condition: Callable[[str], bool] = lambda w: True
    ) -> str:
        word = ""
        while len(word_list := word.split()) < 2 and not word.endswith(" "):
            chosens = self.llm.next_token(context + word)
            for i in count():
                idx_chosen = chosens[i][0]
                # print(chosens[i])
                wordpart = word + self.llm.decode(idx_chosen)
                # print(f"\n{word=}\n{wordpart=}\n")
                print(word, idx_chosen)
                if condition(wordpart.strip()):
                    word = wordpart
                    break
        # print(word)
        return word_list[0]

    def function(self, context: str, functions_: list[Function]) -> str:
        word = ""
        functions = [f.name for f in functions_]
        # print(context)
        while (word).strip() not in functions:
            chosens = self.llm.next_token(context + word)
            for i in range(1000):
                idx_chosen = chosens[i][0]
                # print(chosens[i])
                wordpart = self.llm.decode(idx_chosen).strip()
                # print(f"\n{word=}\n{wordpart=}\n")
                # sleep(0.05)

                if any(f.startswith(word + wordpart) for f in functions) and wordpart:
                    word += wordpart
                    break
            else:
                functions = [func for func in functions if word not in func]
                word = ""
                # print(functions)
        # print(word)
        return word.strip()

    def string(self, context: str) -> str:
        string = "'"
        while not (string.endswith("'") or string.endswith("\n")) or len(string) <= 1:
            chosens = self.llm.next_token(context + string)
            idx_chosen = chosens[0][0]
            wordpart = self.llm.decode(idx_chosen)
            string += wordpart
            # print(string)
        return string[1 : string.rfind("'")]

    def number(self, context: str) -> float:
        number = self.word(context).strip()

        try:
            return float(number)
        except ValueError:
            raise ValueError(f"Could not generate a valid number: {number!r}")

    def integer(self, context: str) -> int:
        number: str = self.word(context).strip()

        try:
            return int(number)
        except ValueError:
            raise ValueError(f"Could not generate a valid number: {number!r}")

    def bool(self, context: str) -> bool:
        word = ""
        while True:
            chosens = self.llm.next_token(context)
            for i in range(self.max_tries):
                idx_chosen = chosens[i][0]
                token = self.llm.decode(idx_chosen).strip().lower()
                token = token.rstrip(".,;:!?\"'")
                if not token:
                    continue
                candidate = (word + token).strip().lower()
                if candidate in ("true", "false"):
                    return candidate == "true"
                if "true".startswith(candidate) or "false".startswith(candidate):
                    word = candidate
                    break
            else:
                raise ValueError(f"could not parse a boolean from {word!r}")

class LLM_Function:
    def __init__(
        self,
        llm_model: Small_LLM_Model,
        functions: list[Function],
        special_tokens: SpecialTokens,
        max_tries: int = 1000,
    ) -> None:
        self.llm_model = llm_model
        self.functions = functions
        self.max_tries = max_tries
        self.constrained_generator = ConstrainedGenerator(LLM(llm_model, special_tokens))

    @classmethod
    def from_json(cls, llm_model: Small_LLM_Model, json_functions: str, special_tokens: SpecialTokens) -> Self:
        functions = TypeAdapter(list[Function]).validate_json(json_functions)
        return cls(llm_model, functions, special_tokens)

    def get_response(self, prompt: Prompt) -> Response:
        prompt_builder = PromptBuilder(self.functions, prompt.prompt, think = False)
        function: Function = self._get_function(prompt_builder)
        parameters = self._get_parameters(function, prompt_builder)
        return Response(prompt=prompt, function=function, parameters=parameters)

    def _get_function(self, prompt_builder: PromptBuilder) -> Function:
        context: str = prompt_builder.function()
        print(context)
        # function: str = self.constrained_generator.function(context, self.functions)
        function: str = self.constrained_generator.function(context, self.functions)
        # print(function)
        for func in self.functions:
            if func.name == function:
                return func
        else:
            raise ValueError(f"invalid function: {function}")

    def _get_parameters(self, func: Function, prompt_builder: PromptBuilder) -> dict[str, str | float]:
        params: dict[str, str | float] = {}
        for arg in func.parameters:
            context = prompt_builder.parameter(func, arg)
            print(context)
            arg_type = func.parameters[arg]["type"]
            match arg_type:
                case "string":
                    param = self.constrained_generator.string(context)
                case "number":
                    param = self.constrained_generator.number(context)
                case "boolean":
                    param = self.constrained_generator.bool(context)
                case "integer":
                    param = self.constrained_generator.integer(context)
                case _:
                    raise ValueError(f"Unsupported parameter type: {arg_type}")
            params[arg] = param
            print(params)
        return params






