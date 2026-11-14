from .llm_function import LLM
from typing import Literal
from .parser import Function

class PromptBuilder:
    def __init__(self, llm: LLM, functions: list[Function], question: str, think: bool = False):
        self.functions = functions
        self.question = question
        self.think = think
        self.llm = llm

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

