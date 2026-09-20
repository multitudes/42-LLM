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
    """Schema for defining a function signature.

    This matches the updated format from input/functions_definition.json.

    Attributes:
        name (str): Name of the function.
        description (str): Description of what the function does.
        parameters (dict[str, ParameterSchema]): Mapping of argument
            names to their schema.
        returns (ReturnSchema): The return type schema.

    """

    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ReturnSchema


class SelectedFunction(BaseModel):
    """Schema for the function selected by the model to call.

    Attributes:
        prompt (str): The user prompt triggering the selection.
        name (str | None): Name of the selected function, or None.
        parameters (dict[str, Any]): Dictionary of function parameters.

    """

    prompt: str
    name: str | None
    parameters: dict[str, Any]


class ToolParameter(BaseModel):
    """Represents the parameters for a tool function.

    Attributes:
        type (str): Parameter object type, defaults to "object".
        properties (dict[str, dict[str, str]]): Mapping of property names
            to property specs.
        required (list[str]): List of required parameter names.

    """

    type: str = "object"
    properties: dict[str, dict[str, str]]
    required: list[str] = []


class ToolFunction(BaseModel):
    """Represents a function within a tool.

    Attributes:
        name (str): Name of the tool function.
        description (str): Description of the tool function.
        parameters (ToolParameter): Parameter specifications.

    """

    name: str
    description: str
    parameters: ToolParameter


class Tool(BaseModel):
    """Represents a tool schema for LLM tool selection.

    Attributes:
        type (str): Type of tool, defaults to "function".
        function (ToolFunction): The tool function details.

    """

    type: str = "function"
    function: ToolFunction
