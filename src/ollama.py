import requests
import json

OLLAMA_URL_API = "http://localhost:11434/api/generate"


def call_ollama_api(data):
    response = requests.post(OLLAMA_URL_API, json=data)
    result = json.loads(response.content.decode())
    return result.get("response", result)
