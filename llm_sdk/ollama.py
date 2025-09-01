import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama_api(data):
    response = requests.post(OLLAMA_URL, json=data)
    result = json.loads(response.content.decode())
    return result.get("response", result)
