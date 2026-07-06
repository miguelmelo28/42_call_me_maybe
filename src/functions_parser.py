from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

type TypeDict = dict[Literal["type"], Literal["number", "string"]]

class Function(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern = "^fn_")
    description: str
    parameters: dict[str, TypeDict] = Field(min_length=1)
    returns: TypeDict = Field(min_length=1, max_length=1)
