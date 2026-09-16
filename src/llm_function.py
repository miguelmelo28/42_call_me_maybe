from .parser import Function, Prompt
from .output import Response
from llm_sdk import Small_LLM_Model
from pydantic import TypeAdapter
from typing import Self
from collections.abc import Callable
from itertools import count
from typing import Literal
from time import sleep

class PromptBuilder:
    def __init__(self, functions: list[Function], question: str, think: bool = False):
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
        return f"<|im_start|>{profile}\n{message}\n<|im_end|>"

    def assistant_block(self) -> str:
        think_str: str = "<think>\n\n</think>\n\n"
        # think_str: str = ""
        return f"<|im_start|>assistant\n" + (think_str if not self.think else "")

class LLM_Function:
    def __init__(
        self,
        llm_model: Small_LLM_Model,
        functions: list[Function],
        max_tries: int = 1000,
    ) -> None:
        self.llm_model = llm_model
        self.functions = functions
        self.max_tries = max_tries
        self.constrained_generator = ConstrainedGenerator(LLM(llm_model))

    @classmethod
    def from_json(cls, llm_model: Small_LLM_Model, json_functions: str) -> Self:
        functions = TypeAdapter(list[Function]).validate_json(json_functions)
        return cls(llm_model, functions)

    def get_response(self, prompt: Prompt) -> Response:
        prompt_builder = PromptBuilder(self.functions, prompt.prompt, think = False)
        function: Function = self._get_function(prompt_builder)
        parameters = self._get_parameters(function, prompt_builder)
        return Response(prompt=prompt, function=function, parameters=parameters)

    def _get_function(self, prompt_builder: PromptBuilder) -> Function:
        context: str = prompt_builder.function()
        print(context)
        # function: str = self.constrained_generator.function(context, self.functions)
        function: str = self.constrained_generator.unconstrained(context)
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
            params[arg] = param
            print(params)
        return params


class LLM:
    def __init__(self, model: Small_LLM_Model):
        self.model = model

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

    def unconstrained(self, context: str) -> str:
        word = ""
        while True:
            chosens = self.llm.next_token(context + word)
            idx_chosen = chosens[0][0]
            word += self.llm.decode(idx_chosen)
            print(word, idx_chosen)
            if idx_chosen == 151645:
                break
        return word


    def word(
        self, context: str, condition: Callable[[str], bool] = lambda w: True
    ) -> str:
        word = ""
        while len(word_list := word.split()) < 2 and not word.endswith(" "):
        # while True:
            # encoded_ctxt = self.llm_model.encode(context + word).tolist()[0]
            # next = self.llm_model.get_logits_from_input_ids(encoded_ctxt)
            # chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
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

    def function(self, context: str, functions_ = list[Function]) -> str:
        word = ""
        functions = [f.name for f in functions_]
        # print(context)
        while (word).strip() not in functions:
            # encoded_ctxt = self.llm_model.encode(context + word).tolist()[0]
            # next = self.llm_model.get_logits_from_input_ids(encoded_ctxt) yuhg
            # chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
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
            # encoded_ctxt = self.llm_model.encode(context + string).tolist()[0]
            # next = self.llm_model.get_logits_from_input_ids(encoded_ctxt)
            # chosens = sorted(enumerate(next), key=lambda x: x[1], reverse=True)
            chosens = self.llm.next_token(context + string)
            idx_chosen = chosens[0][0]
            wordpart = self.llm.decode(idx_chosen)
            string += wordpart
            # print(string)
        return string[1 : string.rfind("'")]

    def number(self, context: str) -> float:
        number: str = self.word(context)
        print(number)
        allowed = set("0123456789.")
        number: str = "".join(c for c in number if c in allowed)
        if number[-1] == ".":
            number = number[:-1]
        return float(number)
