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

# Defaults unless specified otherwise
INPUT_FILE = "data/input/function_calling_tests.json"
OUTPUT_FILE = "data/output/function_calls.json"
TOOLS_DEFINITION_FILE = "data/input/functions_definition.json"


def get_functions(
    tools_file: str | Path | None = None,
) -> list[FunctionDefinition]:
    """Loads function definitions from the specified tools definition file.

    Args:
        tools_file: Path to the JSON tools definition file.

    Returns:
        List of FunctionDefinition instances.

    Raises:
        RuntimeError: If the file is not found, cannot be parsed, or fails
            validation.
        TypeError: If the file content is not a list.

    """
    if tools_file is None:
        tools_file = TOOLS_DEFINITION_FILE

    tools_path = Path(tools_file)
    if not tools_path.exists():
        raise RuntimeError(f"Tools definition file not found: {tools_path}")

    try:
        with tools_path.open("r", encoding="utf-8") as f:
            functions_raw = json.load(f)

    except (
        FileNotFoundError,
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
    """Loads prompts from a JSON file.

    Args:
        file: Path to the JSON input prompts file.

    Returns:
        List of prompt strings.

    Raises:
        RuntimeError: If the file is missing, invalid JSON, or has an
            unexpected format.

    """
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
    """Converts function definitions to a JSON tool specification string.

    Args:
        tools_file: Path to the tools definition JSON file.

    Returns:
        Indented JSON string representing the tools schema.

    Raises:
        RuntimeError: If tools fails to convert to JSON.

    """
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
    """Converts argument values to their defined types based on schema.

    Args:
        name: Function name.
        parameters: Dictionary of parsed function arguments.
        functions_def: List of function definition dictionaries.

    Returns:
        Dictionary with type-coerced argument values.

    """
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
                    is_true = val.strip().lower() in ("true", "1", "yes")
                    parameters[arg_name] = is_true
                else:
                    parameters[arg_name] = bool(val)

        except (ValueError, TypeError):
            pass

    return parameters


def extract_first_json_string(text: str) -> str | None:
    """Extracts the first valid balanced JSON object string from text.

    Args:
        text: Input string potentially containing a JSON object.

    Returns:
        Extracted JSON string if found, otherwise None.

    """
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
    functions: list[FunctionDefinition],
) -> SelectedFunction:
    """Parses and validates JSON tool calls from raw LLM output.

    Args:
        prompt: Original user input string.
        response: Decoded raw response from the LLM.
        functions: Pre-loaded list of validated function definitions.

    Returns:
        SelectedFunction model containing parsed function call details.

    """
    # Strip structural control tags
    clean_output = (
        response.replace("</tool_call>", "")
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

        target_fn = next((fn for fn in functions if fn.name == name), None)
        if not target_fn:
            print(f"Function name '{name}' not found in tool definitions.")
            return SelectedFunction(prompt=prompt, name="", parameters={})

        parameters = parsed_fn.parameters

        # Compare target_fn schema expected keys against response provided keys
        required_keys = set(target_fn.parameters.keys())
        provided_keys = set(parameters.keys())

        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            print(f"Tool call '{name}' missing required parameters: {missing}")
            return SelectedFunction(prompt=prompt, name="", parameters={})

        # Validate against the function definitions and enforce types
        functions_def_dicts = [fn.model_dump() for fn in functions]
        parameters = enforce_arg_types(name, parameters, functions_def_dicts)

        return SelectedFunction(
            prompt=prompt,
            name=name,
            parameters=parameters,
        )

    except (
        json.JSONDecodeError,
        ValidationError,
        TypeError,
        ValueError,
        AttributeError,
    ) as e:
        print(f"Error parsing the JSON object: {e}")
        return SelectedFunction(prompt=prompt, name="", parameters={})


def write_output_to_file(
    results: list[SelectedFunction],
    output_file: str | Path = OUTPUT_FILE,
) -> None:
    """Serializes the list of SelectedFunction models to a JSON file.

    Args:
        results: List of SelectedFunction model instances to output.
        output_file: Path to the target output JSON file.

    Raises:
        Exception: Re-raises any exception encountered when writing.

    """
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
