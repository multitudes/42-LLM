# src/__main__.py
from .function_def_classes import FunctionDef
from .tool_classes import Tool, ToolFunction, ToolParameter
from .utils import convert_func_to_tools
import requests
import json

OLLAMA_URL_API = "http://localhost:11434/api/chat"


def call_ollama_api(data):
    response = requests.post(OLLAMA_URL_API, json=data)
    result = json.loads(response.content.decode())
    return result.get("response", result)

with open("exercise_input/functions_definition.json") as f:
    functions_raw = json.load(f)
functions = [FunctionDef(**fn) for fn in functions_raw]
tools = convert_func_to_tools(functions)

data = {
    "model": "qwen3:0.6b",
    "messages": [
        {
            "role": "user",
            "content": (
                "What is 2 + 2 ? \
                    Reply in JSON choosing from the list of tools and which tool you used. "
            )
        }
    ],
    "tools": tools,
    "stream": False,
    "think": False
}


def main():
    result = call_ollama_api(data)
    print(result)


if __name__ == "__main__":
    main()
