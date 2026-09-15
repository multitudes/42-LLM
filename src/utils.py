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

OUTPUT_FILE = "output/function_calling_name.json"
TOOLS_DEFINITION_FILE = "exercise_input/functions_definition.json"
THINK_TAG = "</think>"


def get_functions() -> list[FunctionDefinition]:
    """
    Loads function definitions from the tools definition file.

    Reads raw JSON tool definitions from `TOOLS_DEFINITION_FILE` and converts
    them into a list of `FunctionDefinition` objects.

    Returns:
        List of instantiated `FunctionDefinition` objects.

    Raises:
        RuntimeError: If loading the file, decoding JSON, or instantiating
            `FunctionDefinition` objects fails.

    """
    tools_path = Path(TOOLS_DEFINITION_FILE)
    try:
        with tools_path.open("r", encoding="utf-8") as f:
            functions_raw = json.load(f)

    except (FileNotFoundError,
            json.JSONDecodeError,
            TypeError,
            ValueError) as e:
        msg = f"Error loading functions for tools: {e}"
        raise RuntimeError(msg) from e

    if not isinstance(functions_raw, list):
        actual_type = type(functions_raw).__name__
        msg = (
            f"Error: Expected a list in {tools_path}, "
            f"got {actual_type}."
        )
        raise TypeError(msg)

    try:
        return [FunctionDefinition(**fn) for fn in functions_raw]
    except (TypeError, ValueError) as e:
        msg = f"Error loading functions for tools: {e}"
        raise RuntimeError(msg) from e


def get_input_prompts(file: str | Path) -> list[str]:
    """
    Loads prompts from a JSON file.

    Each prompt in the JSON structure is expected to be under the "prompt" key.

    Args:
        file: Path to the JSON file containing the prompts.

    Returns:
        List of prompt strings.

    Raises:
        RuntimeError: If the file is not found, JSON decoding fails, or
            the expected schema is invalid.

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


def get_tool_list() -> str:
    """
    Converts a list of FunctionDefinition objects to a JSON string.

    Returns:
        A formatted JSON string representation of the tools list.

    Raises:
        RuntimeError: If converting tools to JSON fails.

    """
    tools: list[dict[str, Any]] = []
    functions = get_functions()
    for fn in functions:
        # Build properties for each argument
        properties: dict[str, dict[str, str]] = {}
        for name in fn.args_names:
            arg_type = fn.args_types[name]
            if arg_type == "float":
                type_str = "number"
            elif arg_type == "str":
                type_str = "string"
            elif arg_type == "int":
                type_str = "integer"
            else:
                type_str = "any"
            properties[name] = {"type": type_str}

        # Create a readable description from the function name
        readable_name = fn.fn_name.removeprefix("fn_")
        description = f"{readable_name.replace('_', ' ')} function"
        tool = Tool(
            function=ToolFunction(
                name=fn.fn_name,
                description=description,
                parameters=ToolParameter(
                    properties=properties,
                    required=fn.args_names,
                ),
            ),
        )
        # Use model_dump() for Pydantic v2 (or dict() if using Pydantic v1)
        tools.append(tool.model_dump())

    try:
        return json.dumps(tools, indent=2)
    except (TypeError, ValueError) as e:
        msg = f"Error converting tools to JSON: {e}"
        raise RuntimeError(msg) from e


def enforce_arg_types(fn_name: str,
                      args: dict[str, Any],
                      functions_def: list[dict[str, Any]],
                      ) -> dict[str, Any]:
    """
    Converts argument values to their defined types based on function def.

    Sometimes the LLM returns a float as 1 instead of 1.0. Given a function
    name and its arguments, convert the argument values to the correct types
    based on the tool definitions. If the function name is not found or an
    argument cannot be converted, it is left unchanged.

    Args:
        fn_name: The name of the function to look up.
        args: Dictionary of argument names and their values.
        functions_def: List of function definition dictionaries (tools).

    Returns:
        Dictionary with argument values converted to their expected types.

    """
    fn_def = next(
        (f for f in functions_def if f.get("fn_name") == fn_name), None)
    if not fn_def or "args_types" not in fn_def:
        return args

    args_types = fn_def["args_types"]
    if not isinstance(args_types, dict):
        return args

    for arg_name, arg_type in args_types.items():
        if arg_name in args:
            try:
                if arg_type == "float":
                    # print(f"Converting arg {arg_name} to float")
                    args[arg_name] = float(args[arg_name])
                    # print(f"Converted arg {arg_name}: {args[arg_name]}")
                elif arg_type == "int":
                    args[arg_name] = int(args[arg_name])
                elif arg_type == "str":
                    args[arg_name] = str(args[arg_name])
        # Add more types as needed
            except (ValueError, TypeError):
                pass  # Leave as is if conversion fails

    return args


def extract_json_from_response(
    prompt: str,
    response: str,
) -> SelectedFunction:
    """
    Extracts and parses a JSON object from the model's full output string.

    If the model output does not contain valid JSON or fails to parse, returns
    an empty SelectedFunction object with fn_name set to an empty string.

    Args:
        prompt: The original natural-language request.
        response: The full output string from the model.

    Returns:
        The parsed SelectedFunction object.

    """
    # First get rid of the think block if it exists
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

        # Use Pydantic to validate the dict structure automatically
        # Inject the original prompt into data before validation
        data["prompt"] = prompt
        parsed_fn = SelectedFunction.model_validate(data)

        # Extract fn_name and args from the validated Pydantic model
        fn_name = parsed_fn.fn_name
        if not fn_name:
            return SelectedFunction(prompt=prompt, fn_name="", args={})
        args = parsed_fn.args if isinstance(parsed_fn.args, dict) else {}

        functions_def = get_functions()
        # 3. Use model_dump() (Pydantic v2 standard) instead of .dict()
        functions_def_dicts = [fn.model_dump() for fn in functions_def]

        args = enforce_arg_types(fn_name, args, functions_def_dicts)
        # print(f"\ncheck args {args}")

        return SelectedFunction(prompt=prompt, fn_name=fn_name, args=args)

    # Catches JSON decoding, schema validation (pydantic), and structure errors
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError,
            AttributeError) as e:
        print(f"Error parsing JSON from response: {e}")
        return SelectedFunction(prompt=prompt, fn_name="", args={})


def write_output_to_file(output_to_write_to_file: list[Any]) -> None:
    """
    Writes the processed output list to a JSON file.

    Args:
        output_to_write_to_file: List of objects (e.g., Pydantic models)
        to serialize and save.

    """
    output_path = Path(OUTPUT_FILE)
    # Ensure parent directory exists (replaces os.makedirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([o.model_dump()
                  for o in output_to_write_to_file], f, indent=2)
    print(f"Output corrected and written to {OUTPUT_FILE}")
