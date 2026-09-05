# src/schemas.py
from typing import Any

from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    """
    Schema for defining a function signature.

    This is what we get from input/functions_definition.json.
    This will be passed to the model in the prompt to let it know
    what functions are available to call.

    Example:
        {
            "fn_name": "fn_add_numbers",
            "args_names": ["a", "b"],
            "args_types": {
                "a": "float",
                "b": "float"
            },
            "return_type": "float"
        }

    Attributes:
        fn_name (str): Name of the function.
        args_names (List[str]): Ordered list of argument names.
        args_types (Dict[str, str]): Mapping of argument names to their types.
        return_type (str): The return type of the function.

    """

    fn_name: str
    args_names: list[str]
    args_types: dict[str, str]
    return_type: str


class SelectedFunction(BaseModel):
    """
    Schema for the function selected by the model to call.

    This is what we expect the model to output after being prompted
    with a list of available functions and a user prompt.

    Attributes:
        prompt (str): The original natural-language request.
        fn_name (str | None): The name of the function to call.
        args (Dict[str, Any]): All required arguments with the correct types.

    """

    prompt: str
    fn_name: str | None
    args: dict[str, Any]


class ToolParameter(BaseModel):
    """
    Represents the parameters for a tool function.

    Example:
        {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for"
                }
            },
            "required": ["city"]
        }

    Attributes:
        type (str): The type of the parameters object (default: "object").
        properties (Dict[str, Dict[str, str]]): Properties of the parameters.
        required (List[str]): List of required parameter names.

    """

    type: str = "object"
    properties: dict[str, dict[str, str]]
    required: list[str] = []


class ToolFunction(BaseModel):
    """
    Represents a function within a tool.

    Example:
        {
            "name": "get_weather",
            "description": "Get the weather in a given city",
            "parameters": { ... }
        }

    Attributes:
        name (str): The function name.
        description (str): The function description.
        parameters (ToolParameter): The function parameters.

    """

    name: str
    description: str
    parameters: ToolParameter


class Tool(BaseModel):
    """
    Represents a tool schema for LLM tool selection.

    Example:
        {
            "type": "function",
            "function": { ... }
        }

    Attributes:
        type (str): The type of the tool (default: "function").
        function (ToolFunction): The function associated with the tool.

    """

    type: str = "function"
    function: ToolFunction
