import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

data = {
    "model": "qwen3:0.6b",
    "prompt": "Hello!",
    "stream": False,
    "think": False
}

response = requests.post(OLLAMA_URL, json=data)

result = json.loads(response.content.decode())

print(result["response"])
