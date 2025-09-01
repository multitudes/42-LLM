import requests
import json

OLLAMA_URL_API = "http://localhost:11434/api/chat"


def call_ollama_api(data):
    response = requests.post(OLLAMA_URL_API, json=data)
    result = json.loads(response.content.decode())
    return result.get("response", result)


def convert_functions_to_tools(functions):
    tools = []
    for fn in functions:
        properties = {name: {"type": "number" if fn["args_types"][name] == "float" else "string" if fn["args_types"]
                             [name] == "str" else "integer" if fn["args_types"][name] == "int" else "any"} for name in fn["args_names"]}
        tool = {
            "type": "function",
            "function": {
                "name": fn["fn_name"],
                "description": f"{fn['fn_name']} function",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": fn["args_names"]
                }
            }
        }
        tools.append(tool)
    return tools


with open("exercise_input/functions_definition.json") as f:
    functions = json.load(f)
tools = convert_functions_to_tools(functions)

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
