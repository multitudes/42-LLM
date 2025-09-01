# src/utils.py
from typing import List, Dict
from .function_def_classes import FunctionDef
from .tool_classes import Tool, ToolFunction, ToolParameter


def convert_func_to_tools(functions: List[FunctionDef]) -> List[Dict]:
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
