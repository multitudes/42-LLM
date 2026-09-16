# src/schemas.py
from typing import Any
from pydantic import BaseModel


class ParameterSchema(BaseModel):
    """Schema for individual parameter details."""
    type: str


class ReturnSchema(BaseModel):
    """Schema for function return details."""
    type: str


class FunctionDefinition(BaseModel):
    """
    Schema for defining a function signature.

    This matches the updated format from input/functions_definition.json.

    Attributes:
        name (str): Name of the function.
        description (str): Description of what the function does.
        parameters (dict[str, ParameterSchema]): Mapping of argument names
        to their schema.
        returns (ReturnSchema): The return type schema.
    """
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ReturnSchema


class SelectedFunction(BaseModel):
    """
    Schema for the function selected by the model to call.

    This is what we expect the model to output after being prompted
    with a list of available functions and a user prompt.
    """
    prompt: str
    fn_name: str | None
    args: dict[str, Any]


class ToolParameter(BaseModel):
    """
    Represents the parameters for a tool function.
    """
    type: str = "object"
    properties: dict[str, dict[str, str]]
    required: list[str] = []


class ToolFunction(BaseModel):
    """
    Represents a function within a tool.
    """
    name: str
    description: str
    parameters: ToolParameter


class Tool(BaseModel):
    """
    Represents a tool schema for LLM tool selection.
    """
    type: str = "function"
    function: ToolFunction
