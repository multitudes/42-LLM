# src/__main__.py
import requests
import json
import time

from .utils import get_prompts, get_tools
from .ollama_request import OllamaRequest, Message
from pydantic import BaseModel


OLLAMA_URL_API = "http://localhost:11434/api/chat"


def call_ollama_api(data):
    response = requests.post(OLLAMA_URL_API, json=data)
    result = json.loads(response.content.decode())
    return result.get("response", result)


class NameFunctionCall(BaseModel):
    prompt: str
    fn_name: str
    args: dict


def main():
    prompts = get_prompts()
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        messages = [
            Message(
                role="user",
                content=f"{prompt}.  Reply in JSON choosing from the list of tools and which tool you used."
            )
        ]
        tools = get_tools()  # Should return a list of dicts
        data = OllamaRequest(
            model="qwen3:0.6b",
            messages=messages,
            tools=tools,
            stream=False,
            think=False
        )

        result = call_ollama_api(data.dict())

        # print(result)
        tool_calls = result["message"].get("tool_calls")
        if tool_calls and len(tool_calls) > 0:
            tool_call = tool_calls[0]
            fn_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            print(fn_name)
            print(args)
        else:
            content = result["message"].get("content")
            if content:
                try:
                    parsed = json.loads(content)
                    fn_name = parsed["name"]
                    args = parsed["arguments"]
                    print(fn_name)
                    print(args)
                except (TypeError, json.JSONDecodeError):
                    print("Plain text reply from LLM:")
                    print(content)


if __name__ == "__main__":
    main()
