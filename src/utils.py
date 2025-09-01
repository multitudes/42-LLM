# src/utils.py
from typing import List
from .function_def_classes import FunctionDef
from .tool_classes import Tool, ToolFunction, ToolParameter
import json


def convert_func_to_tools(functions: List[FunctionDef]) -> List[Tool]:
    tools = []
    for fn in functions:
        properties = {name: {
            "type": "number" if fn.args_types[name] == "float"
            else "string" if fn.args_types[name] == "str"
            else "integer" if fn.args_types[name] == "int"
            else "any"} for name in fn.args_names}
        tool = Tool(
            function=ToolFunction(
                name=fn.fn_name,
                description=f"{fn.fn_name} function",
                parameters=ToolParameter(
                    properties=properties,
                    required=fn.args_names
                )
            )
        )
        tools.append(tool.dict())
    return tools


def get_tools() -> List[Tool]:
    with open("exercise_input/functions_definition.json") as f:
        functions_raw = json.load(f)
    functions = [FunctionDef(**fn) for fn in functions_raw]
    tools = convert_func_to_tools(functions)
    return tools
