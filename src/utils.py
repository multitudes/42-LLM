# src/utils.py
import json
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

# defaults unless I specify a different path
INPUT_FILE = "data/input/function_calling_tests.json"
OUTPUT_FILE = "data/output/function_calls.json"
TOOLS_DEFINITION_FILE = "data/input/functions_definition.json"

THINK_TAG = "</think>"


def get_functions(
    tools_file: str | Path | None = None
) -> list[FunctionDefinition]:
    """Loads function definitions from the specified tools definition file."""

    if tools_file is None:
        tools_file = TOOLS_DEFINITION_FILE

    tools_path = Path(tools_file)
    if not tools_path.exists():
        raise RuntimeError(f"Tools definition file not found: {tools_path}")

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
    name: str,
    parameters: dict[str, Any],
    functions_def: list[dict[str, Any]],
) -> dict[str, Any]:
    """Converts argument values to their defined types based on function def."""
    fn_def = next((f for f in functions_def if f.get("name") == name), None)

    if not fn_def or "parameters" not in fn_def:
        return parameters

    params_def = fn_def["parameters"]
    if not isinstance(params_def, dict):
        return parameters

    for arg_name, param_details in params_def.items():
        if arg_name not in parameters:
            continue
        arg_type = param_details.get("type")
        val = parameters[arg_name]
        try:
            if arg_type == "number":
                parameters[arg_name] = float(val)

            elif arg_type == "integer":
                f_val = float(val)
                parameters[arg_name] = int(round(f_val))

            elif arg_type == "string":
                parameters[arg_name] = str(val)

            elif arg_type == "boolean" and not isinstance(val, bool):
                if isinstance(val, str):
                    parameters[arg_name] = val.strip(
                    ).lower() in ("true", "1", "yes")
                else:
                    parameters[arg_name] = bool(val)

        except (ValueError, TypeError):
            pass

    return parameters


# raw_decode reads token-by-token according to JSON spec.
def extract_first_json_string(text: str) -> str | None:
    """Extract the first valid balanced JSON object string"""
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return text[start: start + end]
        except json.JSONDecodeError:
            pass
        pos = start + 1
    return None


def extract_json_from_response(
    prompt: str,
    response: str,
    tools_file: str | Path = TOOLS_DEFINITION_FILE,
) -> SelectedFunction:
    """Extracts and parses a JSON object from the model's full output string."""

    # Strip structural control tags
    clean_output = (
        response.replace(
            "</tool_call>", "")
        .replace("<|im_end|>", "")
        .strip()
    )

    json_str = extract_first_json_string(clean_output)

    if not json_str:
        print("No valid JSON object found in the response.")
        return SelectedFunction(prompt=prompt, name="", parameters={})

    # Parse, validate, and enforce types
    try:
        data = json.loads(json_str)
        data["prompt"] = prompt

        parsed_fn = SelectedFunction.model_validate(data)

        name = parsed_fn.name
        if not name:
            return SelectedFunction(prompt=prompt, name="", parameters={})

        # Load available tool definitions and validate the tool name exists
        functions_def = get_functions(tools_file=tools_file)
        target_fn = next((fn for fn in functions_def if fn.name == name), None)
        if not target_fn:
            print(
                f"Tool name '{name}' not found in tool definitions.")
            return SelectedFunction(prompt=prompt, name="", parameters={})

        parameters = parsed_fn.parameters

        # Ensure all required keys defined in target_fn.parameters are present
        required_keys = set(parsed_fn.parameters.keys())
        provided_keys = set(parameters.keys())

        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            print(f"Tool call '{name}' missing required parameters: {missing}")
            return SelectedFunction(prompt=prompt, name="", parameters={})

        # validate against the function definitions and enforce types
        # example the model might return a string for a number, like "11.0"
        # we want to convert it to a float
        functions_def_dicts = [fn.model_dump() for fn in functions_def]
        parameters = enforce_arg_types(name, parameters, functions_def_dicts)

        return SelectedFunction(prompt=prompt, name=name, parameters=parameters)

    except (
        json.JSONDecodeError,
        ValidationError,
        TypeError,
        ValueError,
        AttributeError,
    ):
        print("Error parsing the JSON object.")
        return SelectedFunction(prompt=prompt, name="", parameters={})


def write_output_to_file(
    results: list[SelectedFunction],
    output_file: str | Path = OUTPUT_FILE,
) -> None:
    """Serializes the list of SelectedFunction models to a JSON file."""
    # Path() is idempotent so if the output_file is a Path it changes nothing
    path = Path(output_file)

    # Ensure parent directories exist before writing
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = [item.model_dump() for item in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing output to file: {e}")
        # Re-raise to ensure main() terminates with non-zero exit status
        raise
