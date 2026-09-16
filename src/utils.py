# src/utils.py
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import (
    FunctionDefinition,
    SelectedFunction,
    Tool,
    ToolFunction,
    ToolParameter,
)

INPUT_FILE = "data/input/function_calling_tests.json"
OUTPUT_FILE = "data/output/function_calls.json"
TOOLS_DEFINITION_FILE = "data/input/functions_definition.json"
THINK_TAG = "</think>"


def get_functions(
    tools_file: str | Path = TOOLS_DEFINITION_FILE
) -> list[FunctionDefinition]:
    """Loads function definitions from the specified tools definition file."""
    tools_path = Path(tools_file)
    try:
        with tools_path.open("r", encoding="utf-8") as f:
            functions_raw = json.load(f)

    except (FileNotFoundError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            ) as e:
        msg = f"Error loading functions for tools: {e}"
        raise RuntimeError(msg) from e

    if not isinstance(functions_raw, list):
        actual_type = type(functions_raw).__name__
        msg = f"Error: Expected a list in {tools_path}, got {actual_type}."
        raise TypeError(msg)

    try:
        return [FunctionDefinition(**fn) for fn in functions_raw]
    except (TypeError, ValueError, ValidationError) as e:
        msg = f"Error loading functions for tools: {e}"
        raise RuntimeError(msg) from e


def get_input_prompts(file: str | Path = INPUT_FILE) -> list[str]:
    """Loads prompts from a JSON file."""
    file_path = Path(file)
    try:
        with file_path.open("r", encoding="utf-8") as f:
            prompts_raw = json.load(f)
        return [pr["prompt"] for pr in prompts_raw]

    except FileNotFoundError as e:
        msg = f"Error: {file_path} not found."
        raise RuntimeError(msg) from e
    except json.JSONDecodeError as e:
        msg = f"Error: JSON decode failed for {file_path}."
        raise RuntimeError(msg) from e
    except (KeyError, TypeError) as e:
        msg = f"Error: Invalid prompt format in {file_path}."
        raise RuntimeError(msg) from e


def get_tool_list(tools_file: str | Path = TOOLS_DEFINITION_FILE) -> str:
    """Converts a list of FunctionDefinition objects to a JSON string."""
    tools: list[dict[str, Any]] = []
    functions = get_functions(tools_file=tools_file)
    for fn in functions:
        properties: dict[str, dict[str, str]] = {}

        for arg_name, param_schema in fn.parameters.items():
            properties[arg_name] = {"type": param_schema.type}

        tool = Tool(
            function=ToolFunction(
                name=fn.name,
                description=fn.description,
                parameters=ToolParameter(
                    properties=properties,
                    required=list(fn.parameters.keys()),
                ),
            ),
        )
        tools.append(tool.model_dump())

    try:
        return json.dumps(tools, indent=2)
    except (TypeError, ValueError) as e:
        msg = f"Error converting tools to JSON: {e}"
        raise RuntimeError(msg) from e


def enforce_arg_types(
    fn_name: str,
    args: dict[str, Any],
    functions_def: list[dict[str, Any]],
) -> dict[str, Any]:
    """Converts argument values to their defined types based on function def."""
    fn_def = next((f for f in functions_def if f.get("name") == fn_name), None)

    if not fn_def or "parameters" not in fn_def:
        return args

    parameters = fn_def["parameters"]
    if not isinstance(parameters, dict):
        return args

    for arg_name, param_details in parameters.items():
        if arg_name in args:
            arg_type = param_details.get("type")
            try:
                if arg_type == "number":
                    args[arg_name] = float(args[arg_name])
                elif arg_type == "integer":
                    args[arg_name] = int(args[arg_name])
                elif arg_type == "string":
                    args[arg_name] = str(args[arg_name])
            except (ValueError, TypeError):
                pass

    return args


def extract_json_from_response(
    prompt: str,
    response: str,
    tools_file: str | Path = TOOLS_DEFINITION_FILE,
) -> SelectedFunction:
    """Extracts and parses a JSON object from the model's full output string."""
    if THINK_TAG in response:
        response = response.split(THINK_TAG, 1)[1].strip()

    pattern = r'\{\s*"fn_name":.*?\}\s*\}'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        print("No JSON object found in the response.")
        return SelectedFunction(prompt=prompt, fn_name="", args={})

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
        data["prompt"] = prompt
        parsed_fn = SelectedFunction.model_validate(data)

        fn_name = parsed_fn.fn_name
        if not fn_name:
            return SelectedFunction(prompt=prompt, fn_name="", args={})
        args = parsed_fn.args if isinstance(parsed_fn.args, dict) else {}

        functions_def = get_functions(tools_file=tools_file)
        functions_def_dicts = [fn.model_dump() for fn in functions_def]

        args = enforce_arg_types(fn_name, args, functions_def_dicts)

        return SelectedFunction(prompt=prompt, fn_name=fn_name, args=args)

    except (json.JSONDecodeError,
            ValidationError,
            TypeError,
            ValueError,
            AttributeError,
            ) as e:
        print(f"Error parsing JSON from response: {e}")
        return SelectedFunction(prompt=prompt, fn_name="", args={})


def write_output_to_file(
    output_to_write_to_file: list[Any],
    output_file: str | Path = OUTPUT_FILE,
) -> None:
    """Writes the processed output list to a JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([o.model_dump()
                  for o in output_to_write_to_file], f, indent=2)
    print(f"Output corrected and written to {output_path}")
