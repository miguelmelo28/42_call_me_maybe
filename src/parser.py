from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing import Literal, Self

type TypeDict = dict[Literal["type"], Literal["number", "string"]]


class Function(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern="^fn_")
    description: str
    parameters: dict[str, TypeDict] = Field(min_length=1)
    returns: TypeDict = Field(min_length=1, max_length=1)

    @classmethod
    def parse_list_json(cls, data: str) -> list[Self]:
        return TypeAdapter(list[cls]).validate_json(data)  # type: ignore

    def function_description(self) -> str:
        return self.name + ' : ' + self.description


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str

    @classmethod
    def parse_list_json(cls, data: str) -> list[Self]:
        return TypeAdapter(list[cls]).validate_json(data)  # type: ignore
