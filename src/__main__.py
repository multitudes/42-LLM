# src/__main__.py
import requests
import json

from .utils import get_tools
from .ollama_request import OllamaRequest, Message


OLLAMA_URL_API = "http://localhost:11434/api/chat"


def call_ollama_api(data):
    response = requests.post(OLLAMA_URL_API, json=data)
    result = json.loads(response.content.decode())
    return result.get("response", result)


def main():
    messages = [
        Message(
            role="user",
            content="What is 2 + 2? Reply in JSON choosing from the list of tools and which tool you used."
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

    print(result)


if __name__ == "__main__":
    main()
