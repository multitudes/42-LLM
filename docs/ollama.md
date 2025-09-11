
## ollama

At first I did the project with the Ollama api. It is not how it is supposed to be done but it does help to understand how the model should repy, because using the huggingface model with the transformer module is much harder :)

To use the `ollama_chat/qwen3:0.6b` model in your Python code with Ollama, you need to interact with the Ollama server via its REST API. 

example with curl from the docs...
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:0.6b",
  "prompt": "Hello!"
}'
The `-d` option in `curl` stands for "data." It sends the specified data in the body of a POST request to the server.

For example:

```sh
curl http://localhost:11434/api/generate -d '{"model": "qwen3:0.6b", "prompt": "Hello!"}'
```

This sends the JSON payload to the API endpoint as a POST request.  
Without `-d`, `curl` sends a GET request by default.


Here’s a basic example using Python’s `requests` library:

```python
import requests
import json

OLLAMA_URL_API = "http://localhost:11434/api/generate"

data = {
    "model": "qwen3:0.6b",
    "prompt": "Hello!",
    "stream": False,
    "think": False
}

response = requests.post(OLLAMA_URL_API, json=data)

result = json.loads(response.content.decode())

print(result["response"])


```

**Steps:**
1. Make sure Ollama is running locally and the `qwen3:0.6b` model is pulled (`ollama pull qwen3:0.6b`).
2. Install `requests` if needed: `uv pip install requests`
3. Use the code above to send a chat message and get a response.

**Note:**  
- Adjust the endpoint and payload as needed for your use case.
- For more advanced usage, see the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).

to check if the port is free
```
lsof -i :11434
```
If Ollama is not running, you need to start the Ollama server. On macOS, you can usually do this by running:
`ollama serve`

