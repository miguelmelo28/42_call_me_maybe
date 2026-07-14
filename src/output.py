from parser import Function, Prompt
from pydantic import BaseModel, model_validator, field_serializer, ConfigDict, Field
from typing import Self


class Response(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    prompt: Prompt
    function: Function = Field(serialization_alias="name")
    parameters: dict[str, str | float]

    @model_validator(mode="after")
    def valid_parameters(self) -> Self:
        if self.function.parameters.keys() != self.parameters.keys():
            raise ValueError("Given parameters don't match function parameter")
        return self

    @model_validator(mode="after")
    def turn_into_float(self) -> Self:
        for param in self.parameters:
            if self.function.parameters[param]["type"] == "number":
                self.parameters[param] = float(self.parameters[param])
        return self

    @model_validator(mode="after")
    def check_param_types(self) -> Self:
        for param in self.parameters:
            dictionary: dict[str, type] = {"number": float, "string": str}
            _type: str = self.function.parameters[param]["type"]
            if not isinstance(self.parameters[param], dictionary[_type]):
                raise ValueError(f"Argument {param} expects {_type}, got "
                                 f"{self.parameters[param]} (type "
                                 f"{type(self.parameters[param])}) instead")
        return self

    @field_serializer('function', mode="plain", when_used="json")
    def function_name(self, function: Function) -> str:
        return function.name

    @field_serializer('prompt', mode="plain", when_used="json")
    def prompt_name(self, prompt: Prompt) -> str:
        return prompt.prompt
