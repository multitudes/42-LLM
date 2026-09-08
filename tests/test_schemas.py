import pytest
from pydantic import ValidationError
from src.schemas import (
    FunctionDefinition,
    SelectedFunction,
    ToolParameter,
    ToolFunction,
    Tool,
)


def test_function_definition_valid() -> None:
    """Verify FunctionDefinition parses correct payloads successfully."""
    data = {
        "fn_name": "fn_add_numbers",
        "args_names": ["a", "b"],
        "args_types": {"a": "float", "b": "float"},
        "return_type": "float",
    }
    model = FunctionDefinition.model_validate(data)
    assert model.fn_name == "fn_add_numbers"
    assert model.args_names == ["a", "b"]
    assert model.args_types["a"] == "float"
    assert model.return_type == "float"


def test_function_definition_invalid() -> None:
    """Verify FunctionDefinition catches missing required fields."""
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(
            {
                "fn_name": "fn_add_numbers",
                # Missing args_names, args_types, return_type
            }
        )


def test_selected_function_valid() -> None:
    """Verify SelectedFunction works with valid arguments and optional fn_name."""
    model = SelectedFunction(
        prompt="Add 2 and 3",
        fn_name="fn_add_numbers",
        args={"a": 2, "b": 3},
    )
    assert model.prompt == "Add 2 and 3"
    assert model.fn_name == "fn_add_numbers"
    assert model.args["a"] == 2

    # Test with None fn_name (e.g. failed parse fallback)
    model_none = SelectedFunction(
        prompt="Hello",
        fn_name=None,
        args={},
    )
    assert model_none.fn_name is None


def test_tool_parameter_defaults() -> None:
    """Verify ToolParameter applies default values correctly."""
    param = ToolParameter(
        properties={"city": {"type": "string", "description": "Target city"}}
    )
    assert param.type == "object"
    assert param.required == []


def test_tool_structure_valid() -> None:
    """Verify nested Tool, ToolFunction, and ToolParameter schemas parse together."""
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
                    }
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
    """Verify malformed properties or types trigger Pydantic ValidationErrors."""
    with pytest.raises(ValidationError):
        # properties should be a dict, not a list
        ToolParameter(properties=["invalid_property_list"])
