# Prototyping with the Ollama REST API

Early in development, I prototyped my prompt templates and tool-calling logic using a local Ollama server running `qwen3:0.6b`. Interacting with Ollama over HTTP allowed me to observe expected output formatting and establish baseline function-calling schemas before implementing the full PyTorch and Hugging Face Transformers pipeline.

---

## Server Management & Verification

Before making requests, I verify that the local Ollama daemon is active and listening on port `11434`:

```zsh
# Check if port 11434 is in use by Ollama
lsof -i :11434

# Start the Ollama server if it is not running
ollama serve

# Pull the lightweight model variant
ollama pull qwen3:0.6b

```

---

## REST API Prototyping

I tested the generation endpoint using `curl` to send POST requests containing model configuration flags (`-d` / `--data` passes the JSON body):

```sh
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:0.6b",
  "prompt": "Select the function to compute 2+2.",
  "stream": false
}'

```

---

## Python Integration

In my initial prototyping script, I used `requests` to send formatted prompt payloads to the endpoint and inspect the resulting JSON structure:

```python
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen3:0.6b",
    "prompt": "Select the appropriate function to compute 2 + 2.",
    "stream": False,
    "think": False,
}

response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
response.raise_for_status()

data = response.json()
print(data.get("response"))

```

---

## Key Learnings for the Final Pipeline

* **Response Structure:** Prototyping with Ollama helped me design strict regular expressions and Pydantic schemas to strip auxiliary text and extract JSON tool calls cleanly.
* **Transition to Transformers:** While Ollama abstracts model loading and tokenization behind HTTP handlers, my final production system replaces this setup with direct local tokenization and PyTorch inference using Hugging Face `transformers`.
