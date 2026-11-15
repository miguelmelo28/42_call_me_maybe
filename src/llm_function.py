from .parser import Function, Prompt, TypeDict
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

class PromptBuilder:
    def __init__(self, llm: LLM, functions: list[Function], question: str, think: bool = False):
        self.llm = llm
        self.functions = functions
        self.question = question
        self.think = think

    def function(self) -> str:
        functions: str = "- " + "\n- ".join(f.name for f in self.functions)
        system: str = f"""You are a function-selection assistant.

Your task is to select the single function from the available functions that is most appropriate for solving the user's problem.

Available functions:
{functions}

Choose exactly one function.

Respond with only the name of the selected function."""
        user: str = self.question
        return (
            f"{self.context_block('system', system)}"
            f"\n{self.context_block('user', user)}"
            f"\n{self.assistant_block()}"
        )

    def parameter(
        self, function: Function, param: str) -> str:
        system: str = f"""You are a function-parameter extraction assistant.

Your job is to identify the value of the requested parameter.

IMPORTANT:
- Output ONLY the parameter value.
- Never output the parameter name.
- Never output explanations.
- Never output JSON.
- Do not calculate the function's return value.

{function.get_formated()}

Extract the following parameter:
{param} ({function.parameters[param]['type']})

Return only the value of {param}."""
        user: str = self.question

        return (
            f"{self.context_block('system', system)}"
            f"\n{self.context_block('user', user)}"
            f"\n{self.assistant_block()}"
        )

    def context_block(self, profile: Literal["user", "system"], message: str) -> str:
        return f"{self.llm.special_tokens.start_ctx_block}{profile}\n{message}\n{self.llm.special_tokens.end_ctx_block}"

    def assistant_block(self) -> str:
        think_str: str = f"{self.llm.special_tokens.start_think}\n\n{self.llm.special_tokens.end_think}\n\n"
        # think_str: str = ""
        return f"{self.llm.special_tokens.start_ctx_block}assistant\n" + (think_str if not self.think else "")


class ConstrainedGenerator:
    def __init__(self, llm: LLM, max_tries: int = 1000):
        self.llm = llm
        self.max_tries = max_tries
        self.end_token = self.llm.model.encode(self.llm.special_tokens.end_ctx_block)[0][0]

    def unconstrained(self, context: str) -> str:
        word = ""
        while True:
            chosens = self.llm.next_token(context + word)
            idx_chosen = chosens[0][0]
            word += self.llm.decode(idx_chosen)
            print(word, idx_chosen)
            if idx_chosen == self.end_token:
                return word


    def word(
        self, context: str, condition: Callable[[str], bool] = lambda w: True
    ) -> str:
        word = ""
        while len(word_list := word.split()) < 2 and not word.endswith(" "):
            chosens = self.llm.next_token(context + word)
            for i in range(self.max_tries):
                idx_chosen = chosens[i][0]
                if idx_chosen == self.end_token:
                    return word_list[0]
                # print(chosens[i])
                wordpart = word + self.llm.decode(idx_chosen)
                # print(f"\n{word=}\n{wordpart=}\n")
                print(i, wordpart, idx_chosen)
                if condition(wordpart.strip()):
                    word = wordpart
                    break
            else:
                raise ValueError("could not find the answer")
        # print(word)
        return word_list[0]

    def function(self, context: str, functions: list[Function]) -> Function:
        function_str = self.word(context, lambda k: any(func.name.startswith(k) for func in functions))

        for func in functions:
            if func.name == function_str:
                return func
        else:
            raise ValueError(f"invalid function: {function_str}")

    def string(self, context: str) -> str:
        # string = ""
        # while True:
        #     chosens = self.llm.next_token(context + string)
        #     idx_chosen = chosens[0][0]
        #     wordpart = self.llm.decode(idx_chosen)
        #     string += wordpart
        #     if idx_chosen == self.end_token:
        #         break
        #     # print(string)

        return self.unconstrained(context).strip("\"'")

    def number(self, context: str) -> float:
        digits = set("0123456789-.")
        number = self.word(context, lambda k: all(c in digits for c in k) and k.count(".") <= 1 and "-" not in k[1:]).strip()

        try:
            return float(number)
        except ValueError:
            raise ValueError(f"Could not generate a valid number: {number!r}")

    def integer(self, context: str) -> int:
        digits = set("0123456789-")
        number: str = self.word(context, lambda k: all(c in digits for c in k) and "-" not in k[1:]).strip()

        try:
            return int(number)
        except ValueError:
            raise ValueError(f"Could not generate a valid number: {number!r}")

    def bool(self, context: str) -> bool:
        booleans = ["true", "false"]
        answer = self.word(context, lambda k: any(b.startswith(k.strip().lower()) for b in booleans)).lower()

        if answer not in booleans:
                raise ValueError(f"Invalid booleal: {answer}")
        return answer == "true"

class LLM_Function:
    def __init__(
        self,
        llm: LLM,
        functions: list[Function],
        special_tokens: SpecialTokens,
        max_tries: int = 1000,
    ) -> None:
        self.llm = llm
        self.functions = functions
        self.max_tries = max_tries
        self.constrained_generator = ConstrainedGenerator(llm)

    @classmethod
    def from_json(cls, llm: LLM, json_functions: str, special_tokens: SpecialTokens) -> Self:
        functions = TypeAdapter(list[Function]).validate_json(json_functions)
        return cls(llm, functions, special_tokens)

    def get_response(self, prompt: Prompt) -> Response:
        prompt_builder = PromptBuilder(self.llm,self.functions, prompt.prompt, think = False)
        function: Function = self._get_function(prompt_builder)
        parameters = self._get_parameters(function, prompt_builder)
        return Response(prompt=prompt, function=function, parameters=parameters)

    def _get_function(self, prompt_builder: PromptBuilder) -> Function:
        context: str = prompt_builder.function()
        # print(context)
        return self.constrained_generator.function(context, self.functions)

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
            # print(params)
        return params






