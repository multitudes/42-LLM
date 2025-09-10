# 42-LLM-test
A project for 42-Global with python and alms


## Common Instructions
General Rules
- Your project must be written in Python 3.11 or later.
- Your project must adhere to the flake8 coding standard. Bonus files are also subject to this standard.
- Your functions should handle exceptions gracefully to avoid crashes. Use try-except
blocks to manage potential errors. If your program crashes due to unhandled exceptions
during the review, it will be considered non-functional.
- All resources (e.g., file handles, network connections) must be properly managed to prevent
leaks.

## Makefile
Include a Makefile in your project to automate common tasks. It must contain the following

rules:
- install: Install project dependencies using pip, uv, pipx, or any other package manager
of your choice.
- run: Execute the main script of your project.
- debug: Run the main script in debug mode using Python’s built-in debugger.
- clean: Remove temporary files or caches to keep the project environment clean.
- lint: Lint your code using flake8 to ensure it meets coding standards.

## Additional Guidelines

- Create test programs to verify project functionality (not submitted or graded).
- Submit your work to the assigned Git repository. Only the content in this repository will
be graded.

If any additional project-specific requirements apply, they will be stated immediately below this
section.

## Additional instructions
- All classes must use pydantic for validation.
- You can use the numpy and json packages.
- The use of dspy (or any similar package) is completely forbidden, including pytorch, hug-
gingface package, transformers etc.
- You can use the following models:
- ollama_chat/qwen3:0.6b (default)
- Feel free to use other models (using the names from the huggingface hub) during the beta and let us know!
- The function to call should be chosen using the LLM, not with heuristics or any other sort of medieval magic.
- It is forbidden to use any private methods or attributes from the LLM_SDK package.
- You should create a virtual environment and install the packages numpy, and pydantic using uv. To use llm_sdk you can copy it in the same directory than the one src is in.
- The evaluators, as well as the moulinette, will just run uv sync.
- Your program must be run using the following command (where src is the folder containing your files):
```
uv run python -m src
```
- All errors should be handled gracefully. It must never crash unexpectedly, and must always provide a clear error message to the user.


## The makefile
To install uv if not present I follow the official uv docs for Linux and mac:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh;
```
Here’s what the curl -LsSf options mean:

-L: Follow redirects (if the URL redirects to another location).
-s: Silent mode (don’t show progress or error messages).
-S: Show errors (used with -s to display errors if they occur).
-f: Fail silently on server errors (don’t output HTML error pages; exit with error code).
Combined, these options make curl quietly download the script, follow redirects, and only show errors if something goes wrong.

## makefile
In a Makefile, the `@` symbol before a command suppresses the command’s echo (it won’t print the command itself, just the output).

- **Use `@` before shell commands:**  
  Example: `@echo "Hello"`

- **Do NOT use `@` before shell control structures (`fi`, `else`, etc.):**  
  These are not commands, but part of the shell syntax.  
  So you write:
  ```
  @if ...; then \
      ... \
  else \
      ... \
  fi
  ```

**Summary:**  
- Use `@` before actual commands to suppress their echo.
- Do not use `@` before shell keywords like `fi`, `else`, `then`.  
- Only the first line of a multi-line shell block needs the `@` to suppress all output.

## ollama
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


## running the code with uv
The difference is:

- `python src/ollama.py` runs your script using the default Python interpreter in your environment (could be system Python or a virtual environment, depending on what’s activated).
- `uv run python src/ollama.py` runs your script using the Python interpreter managed by `uv` in your project’s .venv (virtual environment), ensuring all dependencies from your pyproject.toml or `requirements.txt` are available and isolated.

**Summary:**  
- Use `uv run python ...` for reproducible, isolated environments and dependency management.
- Use `python ...` for quick runs, but you may risk missing dependencies or using the wrong Python version.

The difference is:

- Using `__main__.py` in a package (like src) allows you to run the package as a module with `python -m src`. Python will look for and execute __main__.py automatically. This is the standard way to provide an entry point for a package.

- Calling `main()` directly (e.g., in `ollama.py` or another script) only works if you run that specific file, like `python src/ollama.py`. It does not work with `python -m src` unless you set up `__main__.py` to call it.

**Summary:**  
- `__main__.py` enables `python -m src` and is required for package-style execution.
- Directly calling `main()` is for script-style execution (running a single file).

## llm_sdk
you should place the llm_sdk package (folder or files) in your project’s root directory—at the same level as your src directory.


## debug info message when starting the model
Q
What is this output i get as info when starting
d81485cdf75e47ca/generation_config.json
Generate config GenerationConfig {
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
A:
This output is informational logging from the Hugging Face Transformers library (or a similar LLM SDK) when it loads the model's generation configuration.

Here's what the fields mean:

bos_token_id: The "beginning of sequence" token ID (151643).
eos_token_id: The "end of sequence" token IDs ([151645, 151643]). The model will stop generating when it outputs one of these IDs.
pad_token_id: The "padding" token ID (151643), used to pad sequences to the same length.
do_sample: Whether the model samples tokens (True) or always picks the most likely one (False).
This config controls how the model generates text. It's normal to see this printed when initializing or running the model.

## Resources
https://flake8.pycqa.org/en/latest/index.html  
https://docs.pydantic.dev/1.10/#__tabbed_1_3  
https://pypi.org/project/pydantic/  
https://docs.astral.sh/uv/guides/projects/  