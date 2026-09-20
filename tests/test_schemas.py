from typing import cast

import pytest
from pydantic import ValidationError

from src.schemas import (
    FunctionDefinition,
    SelectedFunction,
    Tool,
    ToolParameter,
)


def test_function_definition_valid() -> None:
    """Verify FunctionDefinition parses correct payloads successfully."""
    data = {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    }
    model = FunctionDefinition.model_validate(data)
    assert model.name == "fn_add_numbers"
    assert model.description == "Add two numbers together and return their sum."
    assert model.parameters["a"].type == "number"
    assert model.returns.type == "number"


def test_function_definition_invalid() -> None:
    """Verify FunctionDefinition catches missing required fields."""
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(
            {
                "name": "fn_add_numbers",
                # Missing description, parameters, returns
            },
        )


def test_selected_function_valid() -> None:
    """
    Verify SelectedFunction works with valid parameters and optional
    name.
    """
    model = SelectedFunction(
        prompt="Add 2 and 3",
        name="fn_add_numbers",
        parameters={"a": 2, "b": 3},
    )
    assert model.prompt == "Add 2 and 3"
    assert model.name == "fn_add_numbers"
    assert model.parameters["a"] == 2

    # Test with None name (e.g. failed parse fallback)
    model_none = SelectedFunction(
        prompt="Hello",
        name=None,
        parameters={},
    )
    assert model_none.name is None


def test_tool_parameter_defaults() -> None:
    """Verify ToolParameter applies default values correctly."""
    param = ToolParameter(
        properties={"city": {"type": "string", "description": "Target city"}},
    )
    assert param.type == "object"
    assert param.required == []


def test_tool_structure_valid() -> None:
    """
    Verify nested Tool, ToolFunction, and ToolParameter schemas parse
    together.
    """
    tool_data = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                    },
                },
                "required": ["city"],
            },
        },
    }
    tool = Tool.model_validate(tool_data)
    assert tool.type == "function"
    assert tool.function.name == "get_weather"
    assert tool.function.parameters.required == ["city"]
    assert tool.function.parameters.properties["city"]["type"] == "string"


def test_tool_parameter_validation_error() -> None:
    """
    Verify malformed properties or types trigger Pydantic
    ValidationErrors.
    """
    with pytest.raises(ValidationError):
        # properties should be a dict, not a list
        ToolParameter(
            properties=cast(
                "dict[str, dict[str, str]]", ["invalid_property_list"],
            ),
        )
