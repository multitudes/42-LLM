# src/utils.py
import json
import re

from typing import List
from .schemas import FunctionDefinition, SelectedFunction
from .schemas import Tool, ToolFunction, ToolParameter

PATH_TOOLS_DEFINITION = "exercise_input/functions_definition.json"


def get_functions() -> List[FunctionDefinition]:
    """
    The functions to load are actually the ones defined in the
    PATH_TOOLS_DEFINITION file. Those are converted to
    FunctionDefinition objects and returned as a list.
    Raises: RuntimeError: If there is an error loading the file
        or parsing the JSON because the program cannot continue without
        a valid functions list.
    """
    try:
        with open(PATH_TOOLS_DEFINITION) as f:
            functions_raw = json.load(f)
        functions = [FunctionDefinition(**fn) for fn in functions_raw]
        return functions
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading functions for tools: {e}")


def get_prompts(file: str) -> List[str]:
    """
    Load the prompts from a JSON file.
    Each prompt should be under the "prompt" key.
    Args:
        file (str): Path to the JSON file containing the prompts.
    Returns: List[str]: List of prompt strings.
    Raises: RuntimeError: If there is an error loading the file
        or parsing the JSON.
    """
    try:
        with open(file) as f:
            prompts_raw = json.load(f)
        return [pr["prompt"] for pr in prompts_raw]
    except FileNotFoundError:
        raise RuntimeError(f"Error: {file} not found.")
    except json.JSONDecodeError:
        raise RuntimeError(f"Error: JSON decode failed for {file}.")


def get_tool_list() -> str:
    """
    Convert a list of FunctionDefinition objects to a JSON string
    If the json conversion fails, a RuntimeError is raised because the
    program cannot continue without a valid tools list.
    """
    tools = []
    functions = get_functions()
    for fn in functions:
        # Build properties for each argument
        properties = {}
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
        readable_name = fn.fn_name
        if readable_name.startswith("fn_"):
            readable_name = readable_name[3:]
        description = readable_name.replace('_', ' ') + " function"
        tool = Tool(
            function=ToolFunction(
                name=fn.fn_name,
                description=description,
                parameters=ToolParameter(
                    properties=properties,
                    required=fn.args_names
                )
            )
        )
        tools.append(tool.dict())
    try:
        return json.dumps(tools, indent=2)
    except Exception as e:
        raise RuntimeError(f"Error converting tools to JSON: {e}")


def extract_json_from_response(
        prompt: str,
        response: str
) -> SelectedFunction | None:
    """
    Extracts and parses a JSON object from the model's full output string.
    If the model output does not contain valid JSON, returns an empty
    SelectedFunction object with fn_name as an empty string.

    Args:
        prompt (str): The original natural-language request.
        response (str): The full output string from the model.
    Returns:
        SelectedFunction: The parsed SelectedFunction object.
    """
    # First get rid of the think block if it exists
    think_tag = "</think>"
    if think_tag in response:
        response = response.split(think_tag, 1)[1].strip()
    pattern = r'\{\s*"fn_name":.*?\}\s*\}'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        print("No JSON object found in the response.")
        return SelectedFunction(prompt=prompt, fn_name="", args={})
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        fn_name = data.get("fn_name")
        args = data.get("args", {})
        # print(f"Extracted JSON: fn_name={fn_name}, args={args}")
        return SelectedFunction(prompt=prompt, fn_name=fn_name, args=args)
    except Exception as e:
        print(f"Error parsing JSON from response: {e}")
        return SelectedFunction(prompt=prompt, fn_name="", args={})
