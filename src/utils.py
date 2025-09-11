# src/utils.py
import json
import re

from typing import List
from .schemas import FunctionDef, SelectedFunction
from .schemas import Tool, ToolFunction, ToolParameter


def get_functions() -> List[FunctionDef]:
    try:
        with open("exercise_input/functions_definition.json") as f:
            functions_raw = json.load(f)
        functions = [FunctionDef(**fn) for fn in functions_raw]
        return functions
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading functions: {e}")
        return []


def get_functions_names_for_prompt() -> List[str]:
    try:
        with open("exercise_input/functions_definition.json") as f:
            functions_raw = json.load(f)
        functions = [FunctionDef(**fn) for fn in functions_raw]
        return [fn.fn_name for fn in functions]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading functions names: {e}")
        return []


def get_prompts() -> List[str]:
    try:
        with open("exercise_input/function_calling_tests.json") as f:
            prompts_raw = json.load(f)
        return [pr["prompt"] for pr in prompts_raw]
    except FileNotFoundError:
        print("Error: function_calling_tests.json not found.")
        return []
    except json.JSONDecodeError:
        print("Error: JSON decode failed for function_calling_tests.json.")
        return []


def convert_functions_to_tools(functions: List[FunctionDef]) -> str:
    """
    Convert a list of FunctionDef objects to a JSON string 
    """
    tools = []
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
    return json.dumps(tools, indent=2)


def extract_json_from_response(
        prompt: str,
        response: str
) -> SelectedFunction | None:
    """
    Extracts and parses a JSON object from the model's full output string.

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
        print(f"Extracted JSON: fn_name={fn_name}, args={args}")
        return SelectedFunction(prompt=prompt, fn_name=fn_name, args=args)
    except Exception as e:
        print(f"Error parsing JSON from response: {e}")
        return SelectedFunction(prompt=prompt, fn_name="", args={})
